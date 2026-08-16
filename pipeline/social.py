from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class SocialProfile:
    name: str
    width: int
    height: int
    bitrate: str = "8M"

PROFILES = {
    "master": SocialProfile("master", 1920, 1080, "12M"),
    "youtube": SocialProfile("youtube", 1920, 1080, "10M"),
    "vertical": SocialProfile("vertical", 1080, 1920, "8M"),
    "shorts": SocialProfile("shorts", 1080, 1920, "8M"),
    "reels": SocialProfile("reels", 1080, 1920, "8M"),
    "square": SocialProfile("square", 1080, 1080, "8M"),
}

def profile_for(name: str) -> SocialProfile:
    key = (name or "youtube").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Neznámý social profil: {name}")
    return PROFILES[key]

def social_export_command(source: Path, output: Path, profile: SocialProfile) -> list[str]:
    vf = f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=increase,crop={profile.width}:{profile.height},setsar=1,fps=30,format=yuv420p"
    return ["ffmpeg", "-hide_banner", "-y", "-i", str(source), "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-b:v", profile.bitrate, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(output)]

def thumbnail_command(source: Path, output: Path, time_sec: float, width: int = 1280, height: int = 720) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-y", "-ss", f"{max(0.0, float(time_sec)):.3f}", "-i", str(source), "-frames:v", "1", "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p", "-q:v", "2", str(output)]

def thumbnail_score(sample: dict[str, Any]) -> float:
    sharpness = max(0.0, min(1.0, float(sample.get("sharpness", 0.5))))
    brightness = max(0.0, min(1.0, float(sample.get("brightness", 0.5))))
    subject = max(0.0, min(1.0, float(sample.get("subject_score", 0.5))))
    black = max(0.0, min(1.0, float(sample.get("black_ratio", 0.0))))
    return round(0.40 * sharpness + 0.25 * subject + 0.20 * (1.0 - abs(brightness - 0.55)) + 0.15 * (1.0 - black), 6)

def rank_thumbnail_candidates(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for sample in samples:
        item = dict(sample)
        item["score"] = thumbnail_score(item)
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-item["score"], float(item.get("time_sec", 0.0))))

__all__ = ["SocialProfile", "PROFILES", "profile_for", "social_export_command", "thumbnail_command", "thumbnail_score", "rank_thumbnail_candidates"]
