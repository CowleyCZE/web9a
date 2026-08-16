from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RenderProfile:
    name: str
    width: int
    height: int
    fps: int
    crf: int
    preset: str
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    sample_rate: int = 48000
    channels: int = 2
    loudnorm: bool = False

    @property
    def video_encoder_args(self) -> list[str]:
        return [
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format, "-r", str(self.fps),
        ]

    @property
    def audio_encoder_args(self) -> list[str]:
        return [
            "-c:a", self.audio_codec, "-b:a", self.audio_bitrate,
            "-ar", str(self.sample_rate), "-ac", str(self.channels),
        ]


def profile_for(mode: str, hd_mode: str, fps: int) -> RenderProfile:
    sizes = {
        "fullhd": (1920, 1080),
        "hd": (1280, 720),
        "draft": (640, 360),
    }
    width, height = sizes.get(hd_mode, sizes["draft"])
    if mode == "final":
        return RenderProfile("final", width, height, int(fps), 18, "slow", loudnorm=True)
    return RenderProfile("draft", width, height, int(fps), 23, "veryfast", loudnorm=False)


def loudness_filter(profile: RenderProfile) -> str | None:
    if not profile.loudnorm:
        return None
    return "loudnorm=I=-14:TP=-1.5:LRA=11"


def run_loudness_audit(path: Path) -> dict[str, Any]:
    command = [
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"ok": False, "errors": [str(exc)]}
    stderr = completed.stderr or ""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end < start:
        return {"ok": False, "errors": ["ffmpeg nevrátil loudness JSON", stderr[-500:]]}
    try:
        data = json.loads(stderr[start:end + 1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "errors": [f"neplatný loudness JSON: {exc}"]}
    try:
        true_peak = float(data.get("input_tp", "nan"))
        integrated = float(data.get("input_i", "nan"))
    except (TypeError, ValueError):
        true_peak, integrated = math.nan, math.nan
    errors = []
    if not math.isfinite(true_peak) or not math.isfinite(integrated):
        errors.append("loudness audit obsahuje neplatné hodnoty")
    if math.isfinite(true_peak) and true_peak > 0:
        errors.append(f"detekován clipping/true peak nad 0 dBTP: {true_peak:.2f}")
    return {
        "ok": not errors,
        "input_integrated_lufs": integrated if math.isfinite(integrated) else None,
        "input_true_peak_db": true_peak if math.isfinite(true_peak) else None,
        "raw": data,
        "errors": errors,
    }


__all__ = ["RenderProfile", "profile_for", "loudness_filter", "run_loudness_audit"]
