from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


def ensure_seed(project_dir: Path, configured: Any = None) -> int:
    edit_dir = project_dir / "EDIT_PROJECT"
    edit_dir.mkdir(parents=True, exist_ok=True)
    seed_path = edit_dir / "seed.json"
    try:
        seed = int(configured) if configured is not None else None
    except (TypeError, ValueError):
        seed = None
    if seed is None:
        if seed_path.exists():
            try:
                seed = int(json.loads(seed_path.read_text(encoding="utf-8")).get("seed"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                seed = None
    if seed is None:
        digest = hashlib.sha256(str(project_dir.resolve()).encode("utf-8")).hexdigest()
        seed = int(digest[:12], 16)
    seed_path.write_text(json.dumps({"seed": seed}, indent=2) + "\n", encoding="utf-8")
    random.seed(seed)
    return seed


def load_segment_locks(project_dir: Path) -> dict[str, dict[str, Any]]:
    path = project_dir / "EDIT_PROJECT" / "segment_locks.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    locks = data.get("locks", data) if isinstance(data, dict) else {}
    return locks if isinstance(locks, dict) else {}


def parse_timeline_summary(timeline_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*)$")
    for line in timeline_text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        time_range, asset, note = match.groups()
        if "-" not in time_range:
            continue
        start_raw, end_raw = [part.strip() for part in time_range.split("-", 1)]
        try:
            start = float(start_raw.replace(":", ".")) if ":" not in start_raw else _timecode(start_raw)
            end = float(end_raw.replace(":", ".")) if ":" not in end_raw else _timecode(end_raw)
        except ValueError:
            continue
        items.append({
            "asset": asset.strip(),
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "note": note.strip(),
            "locked": "[LOCK" in note.upper(),
        })
    return items


def _timecode(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(value)


def build_preview_report(timeline_text: str, qa: dict[str, Any] | None = None, seed: int | None = None, locks: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    segments = parse_timeline_summary(timeline_text)
    assets = [item["asset"] for item in segments]
    counts: dict[str, int] = {}
    for asset in assets:
        counts[asset] = counts.get(asset, 0) + 1
    durations = [item["duration"] for item in segments]
    speeds = []
    for item in segments:
        match = re.search(r"\[SPEED(?:=|\s+)([0-9.]+)", item["note"], re.I)
        if match:
            speeds.append(float(match.group(1)))
    repeated = {asset: count for asset, count in counts.items() if count > 1}
    locks = locks or {}
    locked_assets = sorted({item["asset"] for item in segments if item["locked"]} | set(locks))
    report = {
        "schema_version": 1,
        "seed": seed,
        "segments": len(segments),
        "timeline_duration_sec": round(sum(durations), 3),
        "unique_assets": len(set(assets)),
        "repeated_assets": repeated,
        "locked_assets": locked_assets,
        "locked_segments": sum(1 for item in segments if item["locked"]),
        "min_segment_sec": round(min(durations), 3) if durations else 0.0,
        "max_segment_sec": round(max(durations), 3) if durations else 0.0,
        "average_segment_sec": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "speed_min": min(speeds) if speeds else None,
        "speed_max": max(speeds) if speeds else None,
        "qa_ok": qa.get("ok") if isinstance(qa, dict) else None,
        "qa_errors": qa.get("errors", []) if isinstance(qa, dict) else [],
        "warnings": [],
    }
    if repeated:
        report["warnings"].append(f"Opakované assety: {', '.join(sorted(repeated))}")
    if not segments:
        report["warnings"].append("Timeline neobsahuje žádné parsovatelné segmenty.")
    return report


def write_preview_report(project_dir: Path, qa: dict[str, Any] | None = None, seed: int | None = None) -> Path:
    timeline_path = project_dir / "EDIT_PROJECT" / "timeline.txt"
    text = timeline_path.read_text(encoding="utf-8", errors="ignore") if timeline_path.exists() else ""
    report = build_preview_report(text, qa=qa, seed=seed, locks=load_segment_locks(project_dir))
    path = project_dir / "EDIT_PROJECT" / "preview_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def generate_contact_sheet(paths: Iterable[Path], output: Path, columns: int = 4, thumb_size: tuple[int, int] = (320, 180)) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    images = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb_size)
            canvas = Image.new("RGB", thumb_size, "black")
            canvas.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
            images.append((path.name, canvas))
        except (OSError, ValueError):
            continue
    if not images:
        return None
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 24)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(images):
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + 24)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_size[1] + 3), name[:48], fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return output


__all__ = ["ensure_seed", "load_segment_locks", "parse_timeline_summary", "build_preview_report", "write_preview_report", "generate_contact_sheet"]
