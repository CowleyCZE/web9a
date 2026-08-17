from __future__ import annotations

import io
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

DEFAULT_MASK_PROFILE = {
    "character_type": "masked_bird",
    "beak_hue_min": 5,
    "beak_hue_max": 35,
    "beak_saturation_min": 0.25,
    "beak_value_min": 0.18,
    "min_area_ratio": 0.001,
    "max_frame_jump": 0.18,
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _frame(video: Path, time_sec: float) -> Image.Image | None:
    try:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, time_sec):.3f}", "-i", str(video), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
        result = subprocess.run(command, capture_output=True, check=False)
        return Image.open(io.BytesIO(result.stdout)).convert("RGB") if result.returncode == 0 and result.stdout else None
    except (OSError, ValueError):
        return None


def _detect_beak(image: Image.Image, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"visible": False, "confidence": 0.0, "reason": "opencv_unavailable"}
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
    hue_min = int(profile.get("beak_hue_min", 5))
    hue_max = int(profile.get("beak_hue_max", 35))
    sat_min = int(_finite(profile.get("beak_saturation_min", 0.25)) * 255)
    val_min = int(_finite(profile.get("beak_value_min", 0.18)) * 255)
    mask = cv2.inRange(array, np.array([hue_min, sat_min, val_min]), np.array([hue_max, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = array.shape[:2]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * float(profile.get("min_area_ratio", 0.001)):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        score = min(1.0, area / (width * height * 0.08))
        candidates.append((score, contour, x, y, w, h))
    if not candidates:
        return {"visible": False, "confidence": 0.0, "reason": "no_color_component"}
    score, contour, x, y, w, h = max(candidates, key=lambda item: item[0])
    moments = cv2.moments(contour)
    cx = (moments["m10"] / moments["m00"]) / width if moments["m00"] else (x + w / 2) / width
    cy = (moments["m01"] / moments["m00"]) / height if moments["m00"] else (y + h / 2) / height
    points = contour.reshape(-1, 2)
    tip = points[np.argmax(points[:, 0])]
    root = points[np.argmin(points[:, 0])]
    return {"visible": True, "confidence": round(score, 4), "center": [round(cx, 5), round(cy, 5)], "tip": [round(float(tip[0]) / width, 5), round(float(tip[1]) / height, 5)], "root": [round(float(root[0]) / width, 5), round(float(root[1]) / height, 5)], "width_norm": round(w / width, 5), "height_norm": round(h / height, 5), "aspect_ratio": round(w / h, 5), "area_ratio": round(area / (width * height), 6), "axis_angle_deg": round(math.degrees(math.atan2(float(tip[1] - root[1]), float(tip[0] - root[0]))), 3)}


def extract_beak_observations(video: Path, profile: dict[str, Any] | None, sample_times: Iterable[float]) -> list[dict[str, Any]]:
    profile = {**DEFAULT_MASK_PROFILE, **(profile or {})}
    observations = []
    for time_sec in sample_times:
        image = _frame(video, float(time_sec))
        observation = _detect_beak(image, profile) if image else {"visible": False, "confidence": 0.0, "reason": "frame_unavailable"}
        observation["time_ms"] = round(max(0.0, float(time_sec)) * 1000)
        observations.append(observation)
    return observations


def track_beak_motion(observations: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [item for item in observations if item.get("visible") and item.get("tip") and item.get("root")]
    energies = []
    geometry_jumps = []
    previous = None
    for current in visible:
        if previous:
            dx = current["tip"][0] - previous["tip"][0]
            dy = current["tip"][1] - previous["tip"][1]
            root_dx = current["root"][0] - previous["root"][0]
            root_dy = current["root"][1] - previous["root"][1]
            relative = math.hypot(dx - root_dx, dy - root_dy)
            energies.append({"time_ms": current["time_ms"], "motion_energy": round(min(1.0, relative * 8), 5), "relative_tip_motion": round(relative, 6)})
            geometry_jumps.append(abs(current.get("aspect_ratio", 0.0) - previous.get("aspect_ratio", 0.0)))
        previous = current
    peaks = sorted(energies, key=lambda item: item["motion_energy"], reverse=True)
    return {"observations": observations, "visible_ratio": round(len(visible) / len(observations), 4) if observations else 0.0, "motion_energy": energies, "motion_peaks": peaks[: max(1, min(8, len(peaks)))], "max_geometry_jump": round(max(geometry_jumps, default=0.0), 5)}


def align_beak_motion_to_phonemes(motion: dict[str, Any], phonemes: Iterable[dict[str, Any]], tolerance_ms: int = 70) -> dict[str, Any]:
    peaks = motion.get("motion_peaks", [])
    alignments = []
    for phoneme in phonemes:
        onset = int(phoneme.get("start_ms", 0))
        nearest = min(peaks, key=lambda item: abs(int(item["time_ms"]) - onset), default=None)
        drift = abs(int(nearest["time_ms"]) - onset) if nearest else None
        alignments.append({"phoneme": phoneme.get("phoneme"), "audio_ms": onset, "motion_ms": nearest.get("time_ms") if nearest else None, "drift_ms": drift, "ok": drift is not None and drift <= tolerance_ms})
    errors = [item for item in alignments if not item["ok"]]
    return {"ok": not errors, "errors": errors, "alignments": alignments, "max_drift_ms": max((item["drift_ms"] or 0 for item in alignments), default=0), "tolerance_ms": tolerance_ms}


def audit_beak_integrity(motion: dict[str, Any], min_visibility: float = 0.45, max_geometry_jump: float = 0.18) -> dict[str, Any]:
    errors = []
    warnings = []
    visible_ratio = float(motion.get("visible_ratio", 0.0))
    geometry_jump = float(motion.get("max_geometry_jump", 0.0))
    if visible_ratio < min_visibility:
        errors.append(f"beak_visibility_low:{visible_ratio:.3f}")
    if geometry_jump > max_geometry_jump:
        errors.append(f"beak_geometry_jump:{geometry_jump:.3f}")
    if visible_ratio < 0.7:
        warnings.append("partial_profile_or_occlusion")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "beak_visibility": round(visible_ratio, 4), "max_geometry_jump": round(geometry_jump, 5), "deformation_score": round(min(1.0, geometry_jump / max_geometry_jump) if max_geometry_jump else 1.0, 4), "identity_stability": round(max(0.0, visible_ratio * (1.0 - min(1.0, geometry_jump / max_geometry_jump))) if max_geometry_jump else 0.0, 4)}


def build_character_lipsync_qa(motion: dict[str, Any], phoneme_report: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "character_type": "masked_bird", "status": "PASS" if motion.get("visible_ratio", 0.0) >= 0.45 and phoneme_report.get("ok") and integrity.get("ok") else "FAIL", "tracking": motion, "phoneme_alignment": phoneme_report, "integrity": integrity, "manual_review_required": bool(not phoneme_report.get("ok") or not integrity.get("ok"))}

__all__ = ["DEFAULT_MASK_PROFILE", "extract_beak_observations", "track_beak_motion", "align_beak_motion_to_phonemes", "audit_beak_integrity", "build_character_lipsync_qa"]
