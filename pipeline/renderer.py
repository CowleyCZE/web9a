from __future__ import annotations

from pathlib import Path


def video_segment_command(
    source: Path,
    output: Path,
    duration: float,
    video_filter: str,
    is_image: bool = False,
) -> list[str]:
    if duration <= 0:
        raise ValueError("Délka segmentu musí být kladná")
    command = ["ffmpeg", "-hide_banner", "-y"]
    if is_image:
        command += ["-loop", "1"]
    command += ["-t", f"{duration:.3f}", "-i", str(source), "-vf", video_filter, "-an"]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "16" if is_image else "15", str(output)]
    return command


def concat_manifest(parts: list[Path], manifest: Path) -> Path:
    if not parts:
        raise ValueError("Nelze vytvořit concat manifest bez segmentů")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # FFmpeg concat syntax vyžaduje file řádek; cesty jsou absolutní a apostrofy
    # se escapují podle syntaxe concat demuxeru.
    lines = []
    for part in parts:
        escaped = str(part).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def concat_command(manifest: Path, output: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)]


def mux_audio_command(video: Path, audio: Path, output: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium",
        "-crf", "15", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(output),
    ]


def fade_command(video: Path, output: Path, duration: float, fade_duration: float = 1.5) -> list[str]:
    if duration <= 0:
        raise ValueError("Délka videa musí být kladná")
    fade_duration = max(0.05, min(float(fade_duration), duration / 2))
    fade_start = max(0.0, duration - fade_duration)
    vf = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_start:.2f}:d={fade_duration}"
    af = f"afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_start:.2f}:d={fade_duration}"
    return [
        "ffmpeg", "-hide_banner", "-y", "-i", str(video), "-vf", vf, "-af", af,
        "-c:v", "libx264", "-preset", "medium", "-crf", "15",
        "-c:a", "aac", "-b:a", "192k", str(output),
    ]


__all__ = ["video_segment_command", "concat_manifest", "concat_command", "mux_audio_command", "fade_command"]
