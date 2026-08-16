from __future__ import annotations

from typing import Any


def motion_style(note: str = "", section: str = "", energy: float = 0.5, is_downbeat: bool = False) -> str:
    text = f"{note} {section}".upper()
    if "[GLITCH]" in text:
        return "glitch"
    if "[WHIPPAN]" in text or "WHIP-PAN" in text or "WHIPPAN" in text:
        return "whippan"
    try:
        value = float(energy)
    except (TypeError, ValueError):
        value = 0.5
    if is_downbeat and value >= 0.7:
        return "impact"
    if "CHORUS" in text or "REFRÉN" in text or "DROP" in text:
        return "impact" if value >= 0.65 else "pulse"
    if "BRIDGE" in text or "BRIDGE" in text:
        return "soft"
    return "clean"


def motion_filters(style: str, duration: float) -> list[str]:
    style = (style or "clean").lower()
    if style == "glitch":
        return ["eq=contrast=1.28:saturation=1.18", "noise=alls=6:allf=t+u"]
    if style == "whippan":
        # Malé periodické naklonění simuluje pohyb kamery bez změny výstupních rozměrů.
        return ["rotate=0.004*sin(2*PI*t):fillcolor=black"]
    if style == "impact":
        return ["eq=contrast=1.30:saturation=1.20", "unsharp=5:5:0.45:5:5:0.0"]
    if style == "pulse":
        return ["eq=contrast=1.18:saturation=1.15"]
    if style == "soft":
        return ["eq=contrast=0.96:saturation=0.92"]
    return []


def transition_plan(segment: dict[str, Any]) -> dict[str, Any]:
    style = motion_style(
        segment.get("note", ""), segment.get("section", ""),
        segment.get("energy", 0.5), bool(segment.get("beat_is_downbeat")),
    )
    return {"style": style, "filters": motion_filters(style, float(segment.get("duration", 0.0))), "reason": "explicit" if style in {"glitch", "whippan"} else "section_energy"}


__all__ = ["motion_style", "motion_filters", "transition_plan"]
