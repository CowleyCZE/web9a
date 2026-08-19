from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from .models import TimelineEntry
from .broll import validate_source_for_slot


def validate_timeline(
    entries: Iterable[TimelineEntry],
    song_duration: float | None = None,
    known_ids: set[str] | None = None,
    tolerance: float = 0.05,
) -> list[str]:
    errors: list[str] = []
    previous_end = 0.0
    entries = list(entries)

    for index, entry in enumerate(entries, start=1):
        if not math.isfinite(entry.start) or not math.isfinite(entry.end):
            errors.append(f"Řádek {index}: čas není konečné číslo")
            continue
        if entry.start < 0:
            errors.append(f"Řádek {index}: začátek je záporný")
        if entry.end <= entry.start:
            errors.append(f"Řádek {index}: konec musí být větší než začátek")
        if index > 1 and entry.start < previous_end - tolerance:
            errors.append(f"Řádek {index}: překrytí s předchozím klipem")
        if known_ids is not None and entry.clip_id not in known_ids:
            errors.append(f"Řádek {index}: neznámé ID klipu {entry.clip_id}")
        previous_end = max(previous_end, entry.end)

    if song_duration is not None:
        if not math.isfinite(song_duration) or song_duration <= 0:
            errors.append("Délka songu není platné kladné číslo")
        elif entries and entries[-1].end > song_duration + tolerance:
            errors.append(
                f"Timeline přesahuje délku songu ({entries[-1].end:.3f}s > {song_duration:.3f}s)"
            )
    return errors


def validate_media_fit(
    clip_id: str,
    source: Path | str,
    actual_duration: float,
    target_duration: float,
) -> tuple[bool, str]:
    """Validate physical media against a timeline slot using render policy.

    A short ``vid_XX`` is valid because renderer loops it at render time.
    Non-loopable short media remains a mismatch.
    """
    ok, mode = validate_source_for_slot(source, actual_duration, target_duration)
    if ok:
        return True, mode
    return False, f"{clip_id}: {mode}"
