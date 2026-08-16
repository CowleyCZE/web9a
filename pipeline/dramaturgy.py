from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


DEFAULT_PROFILE = {
    "label": "unknown",
    "energy": 0.5,
    "cut_density": 0.45,
    "max_shot_sec": 6.0,
    "preferred_groups": ["VID", "PIC", "CHAR"],
    "transition_style": "cut",
}

PROFILES = {
    "intro": {"energy": 0.35, "cut_density": 0.25, "max_shot_sec": 8.0, "preferred_groups": ["PIC", "VID"], "transition_style": "dissolve"},
    "verse": {"energy": 0.55, "cut_density": 0.45, "max_shot_sec": 6.0, "preferred_groups": ["VID", "CHAR"], "transition_style": "cut"},
    "pre-chorus": {"energy": 0.7, "cut_density": 0.6, "max_shot_sec": 4.5, "preferred_groups": ["VID", "CHAR"], "transition_style": "build"},
    "chorus": {"energy": 0.9, "cut_density": 0.8, "max_shot_sec": 3.5, "preferred_groups": ["VID", "CHAR"], "transition_style": "impact"},
    "drop": {"energy": 1.0, "cut_density": 0.95, "max_shot_sec": 2.5, "preferred_groups": ["VID", "CHAR"], "transition_style": "impact"},
    "bridge": {"energy": 0.45, "cut_density": 0.35, "max_shot_sec": 7.0, "preferred_groups": ["PIC", "VID"], "transition_style": "dissolve"},
    "outro": {"energy": 0.3, "cut_density": 0.2, "max_shot_sec": 8.0, "preferred_groups": ["PIC", "VID"], "transition_style": "fade"},
}


def canonical_section(name: str) -> str:
    raw = unicodedata.normalize("NFKD", (name or "unknown").lower())
    raw = "".join(char for char in raw if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    aliases = {"sloka": "verse", "refrén": "chorus", "refren": "chorus", "predrefrén": "pre-chorus", "predrefren": "pre-chorus", "závěr": "outro", "zaver": "outro"}
    return aliases.get(value, value)


def profile_for_section(name: str) -> dict[str, Any]:
    key = canonical_section(name)
    profile = dict(DEFAULT_PROFILE)
    profile.update(PROFILES.get(key, {}))
    profile["label"] = key
    profile["preferred_groups"] = list(profile["preferred_groups"])
    return profile


def build_dramaturgy_plan(song_sections: Iterable[tuple[str, float, float, str]], beat_events: Iterable[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    events = list(beat_events or [])
    result = []
    for name, start, end, description in song_sections:
        profile = profile_for_section(name)
        downbeats = [event for event in events if event.get("is_downbeat") and start <= float(event.get("time", 0)) < end]
        result.append({
            "name": name,
            "key": profile["label"],
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "start_ms": round(float(start) * 1000),
            "end_ms": round(float(end) * 1000),
            "description": description,
            "energy": profile["energy"],
            "cut_density": profile["cut_density"],
            "max_shot_sec": profile["max_shot_sec"],
            "preferred_groups": profile["preferred_groups"],
            "transition_style": profile["transition_style"],
            "downbeat_count": len(downbeats),
        })
    return result


def section_at_time(plan: Iterable[dict[str, Any]], time_sec: float) -> dict[str, Any]:
    for section in plan:
        if float(section["start"]) <= time_sec < float(section["end"]):
            return section
    return {"key": "unknown", **profile_for_section("unknown")}


__all__ = ["PROFILES", "canonical_section", "profile_for_section", "build_dramaturgy_plan", "section_at_time"]
