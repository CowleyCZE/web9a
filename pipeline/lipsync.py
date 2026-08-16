from __future__ import annotations

import math
import re
from typing import Any, Iterable


DEFAULT_WORD_TOLERANCE_MS = 120
DEFAULT_LOW_CONFIDENCE = 0.55


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def normalize_word_events(words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, item in enumerate(words):
        text = str(item.get("word", item.get("text", ""))).strip()
        start = max(0.0, _finite(item.get("start", 0.0), 0.0))
        end = max(start, _finite(item.get("end", start), start))
        if not text or end <= start:
            continue
        confidence = min(1.0, max(0.0, _finite(item.get("confidence", item.get("probability", 0.5)), 0.5)))
        result.append({
            "index": index,
            "word": text,
            "start": round(start, 3),
            "end": round(end, 3),
            "start_ms": round(start * 1000),
            "end_ms": round(end * 1000),
            "duration_ms": round((end - start) * 1000),
            "confidence": round(confidence, 4),
            "confidence_source": "provider" if "confidence" in item or "probability" in item else "heuristic",
        })
    return result


def word_to_phonemes(word: str) -> list[str]:
    value = re.sub(r"[^a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]", "", word.lower())
    tokens = []
    index = 0
    while index < len(value):
        pair = value[index:index + 2]
        if pair == "ch":
            tokens.append(pair)
            index += 2
        else:
            tokens.append(value[index])
            index += 1
    return tokens


def build_phoneme_events(word: dict[str, Any]) -> list[dict[str, Any]]:
    phonemes = word_to_phonemes(word["word"])
    if not phonemes:
        return []
    start, end = word["start_ms"], word["end_ms"]
    step = max(1, (end - start) / len(phonemes))
    result = []
    for index, phoneme in enumerate(phonemes):
        phoneme_start = round(start + index * step)
        phoneme_end = round(end if index == len(phonemes) - 1 else start + (index + 1) * step)
        result.append({
            "word_index": word["index"],
            "phoneme_index": index,
            "phoneme": phoneme,
            "start_ms": phoneme_start,
            "end_ms": max(phoneme_start, phoneme_end),
            "confidence": word["confidence"],
        })
    return result


def build_lipsync_manifest(words: Iterable[dict[str, Any]], song_duration: float = 0.0, text_match_score: float | None = None) -> dict[str, Any]:
    normalized = normalize_word_events(words)
    phonemes = [event for word in normalized for event in build_phoneme_events(word)]
    low_confidence = [word for word in normalized if word["confidence"] < DEFAULT_LOW_CONFIDENCE]
    return {
        "schema_version": 1,
        "timebase": "integer_milliseconds",
        "source": "song_alignment.json",
        "song_duration_ms": round(max(0.0, _finite(song_duration, 0.0)) * 1000),
        "text_match_score": text_match_score,
        "words": normalized,
        "phonemes": phonemes,
        "stats": {
            "word_count": len(normalized),
            "phoneme_count": len(phonemes),
            "low_confidence_word_count": len(low_confidence),
            "confidence_mean": round(sum(w["confidence"] for w in normalized) / len(normalized), 4) if normalized else 0.0,
        },
    }


def validate_manifest_against_ranges(manifest: dict[str, Any], ranges: Iterable[dict[str, Any] | tuple[float, float, str]], tolerance_ms: int = DEFAULT_WORD_TOLERANCE_MS) -> dict[str, Any]:
    words = manifest.get("words", []) if isinstance(manifest, dict) else []
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    for raw_range in ranges:
        if isinstance(raw_range, dict):
            start_ms = round(_finite(raw_range.get("start_ms", raw_range.get("start", 0)), 0) if "start_ms" in raw_range else _finite(raw_range.get("start", 0), 0) * 1000)
            end_ms = round(_finite(raw_range.get("end_ms", raw_range.get("end", 0)), 0) if "end_ms" in raw_range else _finite(raw_range.get("end", 0), 0) * 1000)
            label = str(raw_range.get("clip", raw_range.get("asset", "segment")))
        else:
            start, end, label = raw_range
            start_ms, end_ms = round(float(start) * 1000), round(float(end) * 1000)
        if end_ms <= start_ms:
            errors.append(f"{label}: neplatný interval")
            continue
        relevant = [word for word in words if word["start_ms"] < end_ms and word["end_ms"] > start_ms]
        checked += 1
        if not relevant:
            warnings.append(f"{label}: žádná word-level kotva v segmentu")
            continue
        drift_start = abs(relevant[0]["start_ms"] - start_ms)
        drift_end = abs(relevant[-1]["end_ms"] - end_ms)
        low_conf = [word for word in relevant if word["confidence"] < DEFAULT_LOW_CONFIDENCE]
        if drift_start > tolerance_ms or drift_end > tolerance_ms:
            errors.append(f"{label}: word drift start={drift_start}ms end={drift_end}ms")
        if low_conf:
            warnings.append(f"{label}: {len(low_conf)} slov s nízkou confidence")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checked_segments": checked, "tolerance_ms": tolerance_ms}


__all__ = ["DEFAULT_WORD_TOLERANCE_MS", "normalize_word_events", "word_to_phonemes", "build_lipsync_manifest", "validate_manifest_against_ranges"]
