from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[\wÀ-ž]+", re.UNICODE)


@dataclass(frozen=True)
class ClipMetadata:
    clip_id: str
    group: str = "UNKNOWN"
    duration_sec: float = 0.0
    description: str = ""
    tags: tuple[str, ...] = ()
    location: str = ""
    subject: str = ""
    energy: float = 0.5
    camera_motion: str = ""


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value or "") if len(token) > 2}


def _bounded_energy(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
        if number != number:
            raise ValueError
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return default


def normalize_clip_metadata(clip_id: str, data: dict[str, Any]) -> ClipMetadata:
    description = str(data.get("obsah") or data.get("text") or data.get("description") or "")
    raw_tags = data.get("tags") or data.get("tagy") or []
    if isinstance(raw_tags, str):
        raw_tags = re.split(r"[,;|]", raw_tags)
    tags = tuple(sorted({str(tag).strip().lower() for tag in raw_tags if str(tag).strip()}))
    return ClipMetadata(
        clip_id=clip_id,
        group=str(data.get("group") or "UNKNOWN").upper(),
        duration_sec=max(0.0, float(data.get("duration_sec") or 0.0)),
        description=description,
        tags=tags,
        location=str(data.get("location") or data.get("lokace") or "").strip().lower(),
        subject=str(data.get("subject") or data.get("postava") or "").strip().lower(),
        energy=_bounded_energy(data.get("energy", data.get("energie", 0.5))),
        camera_motion=str(data.get("camera_motion") or data.get("pohyb_kamery") or "").strip().lower(),
    )


def continuity_score(candidate: ClipMetadata, context: str = "", previous: ClipMetadata | None = None, recent_ids: Iterable[str] = ()) -> float:
    context_tokens = _tokens(context)
    candidate_tokens = _tokens(candidate.description) | set(candidate.tags)
    semantic = len(context_tokens & candidate_tokens) / max(1, len(context_tokens)) if context_tokens else 0.0
    score = 0.55 * min(1.0, semantic)
    if previous:
        if candidate.location and previous.location and candidate.location == previous.location:
            score += 0.15
        if candidate.subject and previous.subject and candidate.subject == previous.subject:
            score += 0.15
        score += 0.15 * max(0.0, 1.0 - abs(candidate.energy - previous.energy))
    recent = set(recent_ids)
    if candidate.clip_id in recent:
        score -= 0.35
    return round(score, 6)


def rank_candidates(candidate_ids: Iterable[str], catalog: dict[str, dict[str, Any]], context: str = "", previous_id: str | None = None, recent_ids: Iterable[str] = ()) -> list[str]:
    previous = normalize_clip_metadata(previous_id, catalog[previous_id]) if previous_id and previous_id in catalog else None
    metadata = {clip_id: normalize_clip_metadata(clip_id, catalog[clip_id]) for clip_id in candidate_ids}
    return sorted(
        metadata,
        key=lambda clip_id: (
            continuity_score(metadata[clip_id], context, previous, recent_ids),
            -metadata[clip_id].duration_sec,
            clip_id,
        ),
        reverse=True,
    )


def enrich_beats(times: Iterable[float], bpm: float | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, value in enumerate(times):
        time = round(float(value), 3)
        result.append({
            "time": time,
            "time_ms": int(round(time * 1000)),
            "index": index,
            "bar_index": index // 4,
            "beat_in_bar": index % 4,
            "is_downbeat": index % 4 == 0,
            "phrase_index": index // 16,
            "is_phrase_start": index % 16 == 0,
            "bpm": float(bpm) if bpm is not None else None,
        })
    return result


def nearest_sync_point(time_sec: float, beats: Iterable[dict[str, Any]], prefer_downbeat: bool = False, tolerance_sec: float = 0.20) -> dict[str, Any] | None:
    candidates = [beat for beat in beats if abs(float(beat.get("time", 0.0)) - time_sec) <= tolerance_sec]
    if not candidates:
        return None
    return min(candidates, key=lambda beat: (0 if prefer_downbeat and beat.get("is_downbeat") else 1, abs(float(beat.get("time", 0.0)) - time_sec)))


__all__ = ["ClipMetadata", "normalize_clip_metadata", "continuity_score", "rank_candidates", "enrich_beats", "nearest_sync_point"]
