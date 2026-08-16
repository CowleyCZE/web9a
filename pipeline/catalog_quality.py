from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

_MEDIA_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp")


def _candidate_path(input_dir: Path, clip_id: str) -> Path | None:
    for suffix in _MEDIA_EXTENSIONS:
        candidate = input_dir / f"{clip_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = [path for path in input_dir.glob(f"{clip_id}.*") if path.suffix.lower() in _MEDIA_EXTENSIONS]
    return sorted(matches)[0] if matches else None


def _average_hash(path: Path) -> str | None:
    try:
        from PIL import Image
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        result = subprocess.run(command, capture_output=True, check=False)
        if not result.stdout:
            return None
        image = Image.open(__import__("io").BytesIO(result.stdout)).convert("L").resize((8, 8))
        pixels = list(image.getdata())
        average = sum(pixels) / len(pixels)
        return "".join("1" if pixel >= average else "0" for pixel in pixels)
    except (ImportError, OSError, ValueError):
        return None


def inspect_media(path: Path) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        data = json.loads(result.stdout or "{}")
    except (OSError, json.JSONDecodeError):
        data = {}
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    fps = None
    rate = str(video.get("r_frame_rate", ""))
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        try:
            fps = round(float(numerator) / float(denominator), 3) if float(denominator) else None
        except ValueError:
            fps = None
    valid = bool(data.get("streams")) and duration > 0 and (not video or (width > 0 and height > 0))
    warnings = []
    if not valid:
        warnings.append("media není platná nebo chybí metadata")
    if video and (width < 320 or height < 180):
        warnings.append("nízké rozlišení")
    if video and fps is not None and fps < 12:
        warnings.append("neobvykle nízké FPS")
    score = 1.0 if valid else 0.0
    if warnings:
        score = max(0.0, score - 0.15 * len(warnings))
    return {
        "path": str(path), "valid": valid, "duration_sec": round(duration, 3),
        "width": width or None, "height": height or None, "fps": fps,
        "aspect_ratio": round(width / height, 4) if width and height else None,
        "phash": _average_hash(path), "quality_score": round(score, 3), "warnings": warnings,
    }


def build_catalog_quality(catalog: dict[str, dict[str, Any]], input_dir: Path) -> dict[str, Any]:
    entries = {}
    for clip_id, metadata in catalog.items():
        path = _candidate_path(input_dir, clip_id)
        if path is None:
            entries[clip_id] = {"valid": False, "quality_score": 0.0, "warnings": ["zdrojový soubor nebyl nalezen"]}
            continue
        entry = inspect_media(path)
        entry["catalog_group"] = metadata.get("group", "")
        entry["catalog_duration_sec"] = metadata.get("duration_sec", 0.0)
        entries[clip_id] = entry
    hash_groups: dict[str, list[str]] = {}
    for clip_id, entry in entries.items():
        if entry.get("phash"):
            hash_groups.setdefault(entry["phash"], []).append(clip_id)
    duplicates = [ids for ids in hash_groups.values() if len(ids) > 1]
    for ids in duplicates:
        for clip_id in ids:
            entries[clip_id].setdefault("warnings", []).append("pravděpodobný perceptuální duplikát")
            entries[clip_id]["quality_score"] = max(0.0, entries[clip_id]["quality_score"] - 0.1)
    return {
        "schema_version": 1, "entries": entries,
        "stats": {
            "catalog_count": len(entries),
            "valid_count": sum(1 for item in entries.values() if item.get("valid")),
            "missing_count": sum(1 for item in entries.values() if "zdrojový soubor nebyl nalezen" in item.get("warnings", [])),
            "duplicate_groups": len(duplicates),
        },
        "duplicate_groups": duplicates,
    }


def write_catalog_quality_report(project_dir: Path, catalog: dict[str, dict[str, Any]]) -> Path:
    report = build_catalog_quality(catalog, project_dir / "INPUT")
    output = project_dir / "EDIT_PROJECT" / "catalog_quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = ["inspect_media", "build_catalog_quality", "write_catalog_quality_report"]
