from __future__ import annotations

import io
import math
import subprocess
from pathlib import Path
from typing import Any


def _sample_times(duration: float, count: int) -> list[float]:
    if duration <= 0 or count <= 0:
        return []
    if count == 1:
        return [duration / 2]
    margin = min(0.25, duration / 20)
    start, end = margin, max(margin, duration - margin)
    return [start + (end - start) * i / (count - 1) for i in range(count)]


def _frame_metrics(frame_bytes: bytes) -> dict[str, float] | None:
    try:
        from PIL import Image, ImageStat
        image = Image.open(io.BytesIO(frame_bytes)).convert("L")
        stat = ImageStat.Stat(image)
        mean = float(stat.mean[0])
        variance = float(stat.var[0])
        dark_ratio = sum(1 for pixel in image.resize((64, 36)).getdata() if pixel <= 8) / (64 * 36)
        return {"mean": mean, "variance": variance, "dark_ratio": dark_ratio}
    except (ImportError, OSError, ValueError):
        return None


def sample_video_frames(path: Path, duration: float, count: int = 12) -> list[dict[str, Any]]:
    samples = []
    for time_sec in _sample_times(duration, count):
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{time_sec:.3f}",
            "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=False)
        except OSError as exc:
            samples.append({"time": time_sec, "error": str(exc)})
            continue
        metrics = _frame_metrics(completed.stdout)
        samples.append({"time": round(time_sec, 3), **(metrics or {"error": "snímek nelze dekódovat"})})
    return samples


def audit_visual_quality(path: Path, duration: float, expected_aspect: float | None = None, sample_count: int = 12) -> dict[str, Any]:
    samples = sample_video_frames(path, duration, sample_count)
    errors: list[str] = []
    warnings: list[str] = []
    valid = [sample for sample in samples if "mean" in sample]
    if not valid:
        return {"ok": False, "severity": "error", "errors": ["nebylo možné dekódovat žádný video snímek"], "warnings": [], "samples": samples}

    black = [sample for sample in valid if sample["dark_ratio"] >= 0.98 and sample["mean"] <= 8]
    if black:
        errors.append(f"detekováno {len(black)} téměř černých vzorků")
    freeze_pairs = []
    for previous, current in zip(valid, valid[1:]):
        if abs(previous["mean"] - current["mean"]) < 0.15 and previous["variance"] < 1.0 and current["variance"] < 1.0:
            freeze_pairs.append((previous["time"], current["time"]))
    if freeze_pairs:
        warnings.append(f"možný freeze frame mezi {freeze_pairs[0][0]:.3f}s a {freeze_pairs[0][1]:.3f}s")
    brightness_jumps = []
    for previous, current in zip(valid, valid[1:]):
        if abs(previous["mean"] - current["mean"]) >= 110:
            brightness_jumps.append((previous["time"], current["time"]))
    if brightness_jumps:
        warnings.append(f"výrazný jasový skok v {len(brightness_jumps)} místě/místech")
    if expected_aspect is not None and expected_aspect > 0:
        # Poměr stran je předán jako metadata z ffprobe; vlastní rozměry se kontrolují v audit_video.
        pass
    return {
        "ok": not errors,
        "severity": "error" if errors else ("warning" if warnings else "ok"),
        "errors": errors,
        "warnings": warnings,
        "samples": samples,
        "sample_count": len(samples),
        "valid_sample_count": len(valid),
    }


def audit_video(path: Path, duration: float, width: int | None = None, height: int | None = None, sample_count: int = 12) -> dict[str, Any]:
    expected_aspect = (width / height) if width and height else None
    result = audit_visual_quality(path, duration, expected_aspect=expected_aspect, sample_count=sample_count)
    result["width"] = width
    result["height"] = height
    result["aspect_ratio"] = round(expected_aspect, 4) if expected_aspect else None
    if width is not None and height is not None and (width <= 0 or height <= 0):
        result["ok"] = False
        result["severity"] = "error"
        result.setdefault("errors", []).append("neplatné rozměry videa")
    return result


__all__ = ["sample_video_frames", "audit_visual_quality", "audit_video"]
