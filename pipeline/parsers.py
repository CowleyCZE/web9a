from __future__ import annotations

import math
import re

from .models import TimelineEntry

KNOWN_SECTIONS = {
    "ANALYZA", "ANALÝZA", "SONG_THEME", "PROJECT_THEME", "PROJECT THEME", "ASSET_PLANNING",
    "VIDEO_STYLE", "RAP_CHARACTER_STYLE", "RAPPER_OUTFIT_PROMPT", "LYRICS_STRUCTURE",
    "VIDEO_PROMPTS", "VERIFIED_VIDEO_ASSETS", "RAPPER_PROMPTS", "CHARACTER_PROMPTS",
    "VERIFIED_RAPPER_ASSETS", "IMAGE_PROMPTS", "VERIFIED_IMAGE_ASSETS",
    "RAPPER_SEGMENT_ALIGNMENT", "MUSIC_VIDEO_TIMELINE", "CURRENT_MUSIC_VIDEO_STRUCTURE",
    "SHOT_ORDER", "EFFECTS", "COLOR_GRADING", "METADATA", "NOVE_POTREBNE_KLIPY",
}
_NORMALIZED_SECTIONS = {name.replace(" ", "_") for name in KNOWN_SECTIONS}
_HEADER_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    content: list[str] = []

    def flush() -> None:
        nonlocal content
        if current is not None:
            sections[current] = "\n".join(content).strip()
        content = []

    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        header = _HEADER_RE.match(stripped) if stripped else None
        raw_name = header.group(1).strip().rstrip(":") if header else stripped.rstrip(":")
        normalized = raw_name.upper().replace(" ", "_")
        plain_header = not header and normalized in _NORMALIZED_SECTIONS
        if header or plain_header:
            flush()
            current = normalized if normalized in _NORMALIZED_SECTIONS else None
            continue
        if current is not None:
            content.append(raw_line)
    flush()
    return sections


def parse_timecode(value: str) -> float:
    raw = str(value or "").strip()
    if not raw or raw.startswith("-"):
        raise ValueError(f"Neplatný časový kód: {value!r}")
    try:
        parts = raw.split(":")
        if len(parts) == 1:
            result = float(parts[0])
        elif len(parts) in (2, 3):
            values = [float(part) for part in parts]
            if any(part < 0 or not math.isfinite(part) for part in values):
                raise ValueError
            if any(part >= 60 for part in values[1:]):
                raise ValueError
            result = sum(part * 60 ** power for power, part in enumerate(reversed(values)))
        else:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"Neplatný časový kód: {value!r}") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Čas musí být konečné nezáporné číslo: {value!r}")
    return result


def format_timecode(seconds: float) -> str:
    seconds = float(seconds)
    if not math.isfinite(seconds):
        raise ValueError("Čas musí být konečné číslo")
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def clean_asset_id(value: str) -> str:
    return str(value or "").strip().strip("`'\" ").strip()


def clean_code_block_text(text: str) -> str:
    return "\n".join(
        raw for raw in str(text or "").splitlines() if not raw.strip().startswith("```")
    ).strip()


def normalize_timeline_text(text: str) -> str:
    lines: list[str] = []
    for raw in clean_code_block_text(text).splitlines():
        line = raw.strip().strip("`")
        if not line:
            continue
        if "|" not in line or "-" not in line:
            lines.append(raw)
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == 2:
            lines.append(f"{parts[0]} | {clean_asset_id(parts[1])} |")
        else:
            parts[1] = clean_asset_id(parts[1])
            lines.append(" | ".join(parts))
    return "\n".join(lines).strip()


def parse_timeline_entries(text: str) -> tuple[list[TimelineEntry], list[str]]:
    entries: list[TimelineEntry] = []
    warnings: list[str] = []
    time_re = re.compile(r"^\[?([^\]-]+)\]?\s*-\s*\[?([^\]|]+)\]?")
    for line_no, raw in enumerate(clean_code_block_text(text).splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) < 2:
            warnings.append(f"Řádek {line_no}: chybí oddělovač nebo ID")
            continue
        match = time_re.match(parts[0])
        if not match:
            warnings.append(f"Řádek {line_no}: neplatný interval {parts[0]!r}")
            continue
        try:
            start = parse_timecode(match.group(1))
            end = parse_timecode(match.group(2))
        except ValueError as exc:
            warnings.append(f"Řádek {line_no}: {exc}")
            continue
        clip_id = clean_asset_id(parts[1])
        if not clip_id:
            warnings.append(f"Řádek {line_no}: prázdné ID klipu")
            continue
        entries.append(TimelineEntry(start, end, clip_id, parts[2].strip() if len(parts) == 3 else ""))
    return entries, warnings


__all__ = [
    "extract_sections", "parse_timecode", "format_timecode", "clean_asset_id",
    "clean_code_block_text", "normalize_timeline_text", "parse_timeline_entries",
]
