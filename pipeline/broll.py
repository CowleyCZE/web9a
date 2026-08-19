from __future__ import annotations

from pathlib import Path


LOOPABLE_PREFIXES = ("vid_",)


def is_loopable_broll(source: Path | str) -> bool:
    """Return whether a B-roll source may be repeated by the renderer.

    The policy is intentionally based on the media ID/name, not on the current
    duration. A short ``vid_XX`` source is valid: the renderer can repeat it to
    fill its timeline slot. Images have their own loop policy and are handled by
    the image path in the renderer.
    """
    stem = Path(source).stem.lower()
    return stem.startswith(LOOPABLE_PREFIXES)


def fit_mode(source: Path | str, actual_duration: float, target_duration: float) -> str:
    """Return the single source-fitting policy used by validation/render.

    ``loop`` means the source is shorter than the slot and may repeat.
    ``trim`` means the source is longer than the slot and will be cut by ``-t``.
    ``exact`` means no duration adaptation is necessary.
    ``invalid`` means the media has no usable duration.
    """
    if actual_duration <= 0 or target_duration <= 0:
        return "invalid"
    if actual_duration < target_duration:
        return "loop" if is_loopable_broll(source) else "mismatch"
    if actual_duration > target_duration:
        return "trim"
    return "exact"


def validate_source_for_slot(source: Path | str, actual_duration: float, target_duration: float) -> tuple[bool, str]:
    """Validate a source against a timeline slot using the render policy.

    A short ``vid_`` clip is explicitly valid because render-time looping is the
    authoritative fit mechanism. Non-loopable short media remains a mismatch.
    """
    mode = fit_mode(source, actual_duration, target_duration)
    if mode == "invalid":
        return False, "invalid_media"
    if mode == "mismatch":
        return False, "short_non_loopable_source"
    return True, mode


__all__ = ["is_loopable_broll", "fit_mode", "validate_source_for_slot"]
