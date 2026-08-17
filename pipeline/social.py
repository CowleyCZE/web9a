from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter, ImageStat


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


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def analyze_thumbnail_image(path: Path) -> dict[str, float | bool | str]:
    """Vypočítá deterministické obrazové metriky pro výběr thumbnailu."""
    try:
        with Image.open(path) as original:
            image = original.convert("L").resize((min(640, original.width), min(360, original.height)))
            pixels = list(image.getdata())
            if not pixels:
                raise ValueError("prázdný obraz")
            stats = ImageStat.Stat(image)
            mean = stats.mean[0] / 255.0
            variance = stats.var[0] / (255.0 * 255.0)
            edges = image.filter(ImageFilter.FIND_EDGES)
            edge_mean = ImageStat.Stat(edges).mean[0] / 255.0
            dark_ratio = sum(1 for value in pixels if value < 12) / len(pixels)
            contrast = _bounded((max(pixels) - min(pixels)) / 255.0)
            # Obsahová aktivita je konzervativní proxy pro thumbnail bez tvrzení o detekci osoby.
            activity = _bounded(0.55 * contrast + 0.45 * edge_mean)
            return {"valid": True, "brightness": round(mean, 6), "sharpness": round(_bounded(0.7 * edge_mean + 0.3 * min(1.0, variance * 8)), 6), "black_ratio": round(dark_ratio, 6), "contrast": round(contrast, 6), "subject_score": round(activity, 6)}
    except (OSError, ValueError, TypeError):
        return {"valid": False, "brightness": 0.0, "sharpness": 0.0, "black_ratio": 1.0, "contrast": 0.0, "subject_score": 0.0}


def thumbnail_score(sample: dict[str, Any]) -> float:
    sharpness = _bounded(sample.get("sharpness", 0.0))
    brightness = _bounded(sample.get("brightness", 0.0))
    subject = _bounded(sample.get("subject_score", 0.0))
    black = _bounded(sample.get("black_ratio", 1.0))
    contrast = _bounded(sample.get("contrast", 0.0))
    # Preferuj čitelné, ostré a kontrastní snímky; příliš tmavé nebo přepálené snímky penalizuj.
    exposure = 1.0 - min(1.0, abs(brightness - 0.55) / 0.55)
    return round(0.35 * sharpness + 0.25 * subject + 0.15 * contrast + 0.15 * exposure + 0.10 * (1.0 - black), 6)


def rank_thumbnail_candidates(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for sample in samples:
        item = dict(sample)
        if item.get("path") and not {"brightness", "sharpness", "black_ratio", "contrast", "subject_score"}.issubset(item):
            item.update(analyze_thumbnail_image(Path(str(item["path"]))))
        item["score"] = thumbnail_score(item)
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-item["score"], float(item.get("time_sec", 0.0))))


__all__ = ["SocialProfile", "PROFILES", "profile_for", "social_export_command", "thumbnail_command", "analyze_thumbnail_image", "thumbnail_score", "rank_thumbnail_candidates"]
