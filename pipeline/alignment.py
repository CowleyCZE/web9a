from __future__ import annotations

import math
from collections.abc import Iterable


def clamp_speed(raw_speed: float, speed_min: float, speed_max: float) -> float:
    """Vrátí konečnou rychlost v povoleném intervalu."""
    speed_min = float(speed_min)
    speed_max = float(speed_max)
    value = float(raw_speed)
    if not all(math.isfinite(x) for x in (value, speed_min, speed_max)):
        raise ValueError("Rychlost i limity musí být konečná čísla")
    if speed_min <= 0 or speed_max < speed_min:
        raise ValueError("Neplatný interval rychlostí")
    return round(max(speed_min, min(speed_max, value)), 4)


def speed_for_slot(source_duration: float, target_duration: float, speed_min: float, speed_max: float) -> float:
    """Spočítá rychlost tak, aby zdrojová délka vyplnila cílový slot."""
    source_duration = float(source_duration)
    target_duration = float(target_duration)
    if source_duration <= 0 or target_duration <= 0:
        raise ValueError("Délky zdroje i slotu musí být kladné")
    return clamp_speed(source_duration / target_duration, speed_min, speed_max)


def distribute_gap(gap_start: float, gap_end: float, source_durations: Iterable[float]) -> list[tuple[float, float]]:
    """Rovnoměrně rozdělí mezeru mezi klipy; odmítne zápornou mezeru a nulový počet klipů."""
    start = float(gap_start)
    end = float(gap_end)
    durations = list(source_durations)
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("Mezera musí být konečný nezáporný interval")
    if not durations:
        return []
    if any(float(duration) < 0 or not math.isfinite(float(duration)) for duration in durations):
        raise ValueError("Délky klipů musí být konečná nezáporná čísla")
    slot = (end - start) / len(durations)
    return [(start + index * slot, start + (index + 1) * slot) for index in range(len(durations))]


def validate_alignment_ranges(ranges: Iterable[tuple[float, float]], tolerance: float = 0.05) -> list[str]:
    errors: list[str] = []
    previous_end = 0.0
    for index, (start, end) in enumerate(ranges, 1):
        if not math.isfinite(start) or not math.isfinite(end):
            errors.append(f"Segment {index}: čas není konečný")
            continue
        if start < 0 or end <= start:
            errors.append(f"Segment {index}: neplatný interval")
        if start < previous_end - tolerance:
            errors.append(f"Segment {index}: překrývá předchozí segment")
        previous_end = max(previous_end, end)
    return errors


__all__ = ["clamp_speed", "speed_for_slot", "distribute_gap", "validate_alignment_ranges"]
