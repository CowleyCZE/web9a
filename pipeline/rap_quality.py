from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

PLOSIVES = {"p", "b", "t", "d", "k", "g"}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def inspect_rap_clip(path: Path, expected_duration: float | None = None) -> dict[str, Any]:
    """Změří metadata rap klipu a vrátí konzervativní quality score bez falešné detekce obličeje."""
    result = {"path": str(path), "valid": False, "duration": 0.0, "width": None, "height": None, "fps": None, "blur_score": None, "face_visibility": None, "mouth_visibility": None, "occlusion_score": None, "issues": []}
    if not path.exists():
        result["issues"].append("missing_media")
        return result
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", str(path)], capture_output=True, text=True, check=False)
        data = json.loads(probe.stdout or "{}")
        streams = data.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), None)
        duration = _finite((data.get("format") or {}).get("duration"), 0.0)
        if video:
            result.update({"duration": duration, "width": int(video.get("width") or 0), "height": int(video.get("height") or 0), "fps": _fps(video.get("r_frame_rate"))})
        result["valid"] = probe.returncode == 0 and bool(video) and duration > 0 and result["width"] >= 160 and result["height"] >= 90
        if not result["valid"]:
            result["issues"].append("invalid_video")
        if expected_duration is not None and abs(duration - expected_duration) > 0.035:
            result["issues"].append("duration_mismatch")
        if result["width"] and result["height"] and min(result["width"], result["height"]) < 360:
            result["issues"].append("low_resolution")
    except (OSError, json.JSONDecodeError, ValueError):
        result["issues"].append("ffprobe_failed")
    score = 1.0 if result["valid"] else 0.0
    score -= 0.2 * len(result["issues"])
    result["quality_score"] = round(max(0.0, min(1.0, score)), 4)
    return result


def _fps(value: Any) -> float | None:
    text = str(value or "")
    if "/" in text:
        num, den = text.split("/", 1)
        denominator = _finite(den, 0)
        return round(_finite(num, 0) / denominator, 3) if denominator else None
    value = _finite(text, 0)
    return round(value, 3) if value else None


def phoneme_locked_qa(manifest: dict[str, Any], ranges: Iterable[dict[str, Any]], tolerance_ms: int = 35, pre_roll_ms: int = 100, post_roll_ms: int = 120) -> dict[str, Any]:
    phonemes = manifest.get("phonemes", []) if isinstance(manifest, dict) else []
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    max_drift = 0
    plosive_drifts = []
    for item in ranges:
        label = str(item.get("clip", item.get("asset", "segment")))
        start = int(round(_finite(item.get("start_ms", 0))))
        end = int(round(_finite(item.get("end_ms", 0))))
        if end <= start:
            errors.append(f"{label}: invalid_range")
            continue
        relevant = [p for p in phonemes if int(p.get("start_ms", 0)) < end and int(p.get("end_ms", 0)) > start]
        if not relevant:
            warnings.append(f"{label}: no_phonemes")
            continue
        checked += 1
        first_drift = int(relevant[0].get("start_ms", 0)) - start
        last_drift = int(relevant[-1].get("end_ms", 0)) - end
        max_drift = max(max_drift, abs(first_drift), abs(last_drift))
        for phoneme in relevant:
            drift = abs(int(phoneme.get("start_ms", 0)) - max(start, int(phoneme.get("start_ms", 0))))
            if str(phoneme.get("phoneme", "")).lower() in PLOSIVES:
                plosive_drifts.append(drift)
        if abs(first_drift) > tolerance_ms or abs(last_drift) > tolerance_ms:
            errors.append(f"{label}: phoneme_drift start={first_drift}ms end={last_drift}ms")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checked_segments": checked, "max_drift_ms": max_drift, "max_plosive_drift_ms": max(plosive_drifts, default=0), "tolerance_ms": tolerance_ms, "pre_roll_ms": pre_roll_ms, "post_roll_ms": post_roll_ms}


def choose_fallback_candidate(candidates: Iterable[dict[str, Any]], used_ids: Iterable[str] = (), minimum_score: float = 0.55) -> dict[str, Any] | None:
    used = set(used_ids)
    ranked = sorted((dict(item) for item in candidates), key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)
    for candidate in ranked:
        identifier = str(candidate.get("id", candidate.get("clip", candidate.get("path", ""))))
        if identifier not in used and float(candidate.get("quality_score", 0.0)) >= minimum_score and not candidate.get("manual_review_required"):
            return candidate
    return None


def rap_continuity_score(previous: dict[str, Any] | None, current: dict[str, Any]) -> float:
    if not previous:
        return 1.0
    score = 1.0
    for key, weight in (("face_scale", 0.25), ("mouth_x", 0.2), ("mouth_y", 0.2), ("lighting", 0.15), ("shot_scale", 0.2)):
        if key in previous and key in current:
            score -= min(1.0, abs(_finite(previous[key]) - _finite(current[key]))) * weight
    return round(max(0.0, min(1.0, score)), 4)


def local_timewarp_plan(source_duration: float, target_duration: float, pre_roll: float = 0.12, post_roll: float = 0.14, minimum: float = 0.85, maximum: float = 1.18) -> dict[str, Any]:
    source_duration = max(0.001, _finite(source_duration, 0.001))
    target_duration = max(0.001, _finite(target_duration, 0.001))
    core_source = max(0.001, source_duration - pre_roll - post_roll)
    core_target = max(0.001, target_duration - pre_roll - post_roll)
    core_speed = max(minimum, min(maximum, core_source / core_target))
    return {"segments": [{"name": "pre_roll", "duration": round(min(pre_roll, target_duration * 0.25), 4), "speed": 1.0}, {"name": "articulation", "duration": round(core_target, 4), "speed": round(core_speed, 4)}, {"name": "post_roll", "duration": round(min(post_roll, target_duration * 0.25), 4), "speed": 1.0}], "clamped": core_speed in (minimum, maximum), "core_speed": round(core_speed, 4)}


def build_rap_qa_summary(clip_reports: Iterable[dict[str, Any]], lipsync: dict[str, Any] | None = None, continuity_scores: Iterable[float] = ()) -> dict[str, Any]:
    reports = list(clip_reports)
    invalid = [item for item in reports if not item.get("valid")]
    continuity = list(continuity_scores)
    result = {"schema_version": 1, "status": "FAIL" if invalid or (lipsync and not lipsync.get("ok")) else "PASS", "clip_count": len(reports), "invalid_clip_count": len(invalid), "max_drift_ms": (lipsync or {}).get("max_drift_ms", 0), "max_plosive_drift_ms": (lipsync or {}).get("max_plosive_drift_ms", 0), "mean_continuity_score": round(sum(continuity) / len(continuity), 4) if continuity else None, "clips": reports, "lipsync": lipsync or {}}
    result["manual_review_required"] = bool(invalid or (lipsync and lipsync.get("errors")))
    return result

__all__ = ["inspect_rap_clip", "phoneme_locked_qa", "choose_fallback_candidate", "rap_continuity_score", "local_timewarp_plan", "build_rap_qa_summary"]
