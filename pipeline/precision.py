from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIME_SCALE = 1000
DEFAULT_DRIFT_TOLERANCE_MS = 20
DEFAULT_DURATION_TOLERANCE_MS = 150


def seconds_to_ms(value: Any) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Neplatný čas: {value!r}")
    return int(round(number * TIME_SCALE))


def ms_to_seconds(value: Any) -> float:
    number = int(value)
    if number < 0:
        raise ValueError(f"Neplatný čas v ms: {value!r}")
    return number / TIME_SCALE


def duration_drift_ms(actual_seconds: Any, expected_seconds: Any) -> int:
    return abs(seconds_to_ms(actual_seconds) - seconds_to_ms(expected_seconds))


@dataclass(frozen=True)
class DriftIssue:
    asset: str
    expected_ms: int
    actual_ms: int
    drift_ms: int

    def message(self) -> str:
        return (
            f"{self.asset}: délka {self.actual_ms / TIME_SCALE:.3f}s "
            f"neodpovídá očekávaným {self.expected_ms / TIME_SCALE:.3f}s "
            f"(odchylka {self.drift_ms} ms)."
        )


def validate_duration_drift(asset: str, actual: Any, expected: Any, tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS) -> DriftIssue | None:
    actual_ms = seconds_to_ms(actual)
    expected_ms = seconds_to_ms(expected)
    drift_ms = abs(actual_ms - expected_ms)
    return None if drift_ms <= int(tolerance_ms) else DriftIssue(asset, expected_ms, actual_ms, drift_ms)


def validate_lipsync_manifest(manifest: dict[str, Any], media_durations: dict[str, float], tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS) -> list[str]:
    issues: list[str] = []
    for segment in manifest.get("segments", []):
        asset = str(segment.get("asset", ""))
        expected = segment.get("duration")
        actual = media_durations.get(asset)
        if actual is None:
            continue
        try:
            issue = validate_duration_drift(asset, actual, expected, tolerance_ms)
        except (TypeError, ValueError) as exc:
            issues.append(f"{asset}: neplatná délka v lipsync manifestu ({exc})")
            continue
        if issue:
            issues.append(issue.message())
    return issues


def ffprobe_media_qa(path: Path, expected_duration: float | None = None, tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return {"ok": False, "path": str(path), "errors": [completed.stderr.strip() or "ffprobe selhal"]}
        data = json.loads(completed.stdout or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "path": str(path), "errors": [str(exc)]}

    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    duration = float(fmt.get("duration", 0) or 0)
    errors: list[str] = []
    if not math.isfinite(duration) or duration <= 0:
        errors.append("výstup nemá platnou délku")
    if expected_duration is not None:
        drift = duration_drift_ms(duration, expected_duration)
        if drift > tolerance_ms:
            errors.append(f"délka výstupu se liší o {drift} ms")

    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    if not video:
        errors.append("chybí video stream")
    if not audio:
        errors.append("chybí audio stream")

    return {
        "ok": not errors,
        "path": str(path),
        "duration": duration,
        "duration_ms": seconds_to_ms(duration) if duration > 0 and math.isfinite(duration) else None,
        "video": video,
        "audio": audio,
        "errors": errors,
    }


__all__ = [
    "TIME_SCALE", "DEFAULT_DRIFT_TOLERANCE_MS", "DEFAULT_DURATION_TOLERANCE_MS",
    "seconds_to_ms", "ms_to_seconds", "duration_drift_ms", "DriftIssue",
    "validate_duration_drift", "validate_lipsync_manifest", "ffprobe_media_qa",
]
