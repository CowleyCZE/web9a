from __future__ import annotations

from pathlib import Path
from .output_quality import RenderProfile
from .broll import is_loopable_broll


def video_segment_command(
    source: Path,
    output: Path,
    duration: float,
    video_filter: str,
    is_image: bool = False,
    profile: RenderProfile | None = None,
    loop_short_broll: bool | None = None,
) -> list[str]:
    """Sestaví segment renderu podle centrální B-roll fit politiky.

    Krátké ``vid_`` B-rolly jsou legitimní zdroje: pokud jsou kratší než
    timeline slot, renderer je opakuje. Delší zdroj se naopak ořízne pomocí
    ``-t``. Tím renderer nemusí předem fyzicky měnit zdrojové MP4.
    """
    if duration <= 0:
        raise ValueError("Délka segmentu musí být kladná")
    if loop_short_broll is None:
        loop_short_broll = is_loopable_broll(source) and not is_image

    command = ["ffmpeg", "-hide_banner", "-y"]
    if is_image:
        command += ["-loop", "1"]
    elif loop_short_broll:
        command += ["-stream_loop", "-1"]
    command += ["-t", f"{duration:.3f}", "-i", str(source), "-vf", video_filter, "-an"]
    if profile is None:
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "16" if is_image else "15", "-pix_fmt", "yuv420p"]
    else:
        command += profile.video_encoder_args
    command += [str(output)]
    return command


def concat_manifest(parts: list[Path], manifest: Path) -> Path:
    if not parts:
        raise ValueError("Nelze vytvořit concat manifest bez segmentů")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for part in parts:
        escaped = str(part).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def concat_command(manifest: Path, output: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)]


def mux_audio_command(video: Path, audio: Path, output: Path, profile: RenderProfile | None = None) -> list[str]:
    profile = profile or RenderProfile("default", 0, 0, 30, 15, "medium")
    command = [
        "ffmpeg", "-hide_banner", "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        *profile.video_encoder_args,
        *profile.audio_encoder_args,
        "-shortest", "-movflags", "+faststart",
    ]
    from .output_quality import loudness_filter
    audio_filter = loudness_filter(profile)
    if audio_filter:
        command += ["-af", audio_filter]
    return command + [str(output)]


def fade_command(video: Path, output: Path, duration: float, fade_duration: float = 1.5, profile: RenderProfile | None = None) -> list[str]:
    if duration <= 0:
        raise ValueError("Délka videa musí být kladná")
    fade_duration = max(0.05, min(float(fade_duration), duration / 2))
    fade_start = max(0.0, duration - fade_duration)
    vf = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_start:.2f}:d={fade_duration}"
    af = f"afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_start:.2f}:d={fade_duration}"
    profile = profile or RenderProfile("default", 0, 0, 30, 15, "medium")
    return [
        "ffmpeg", "-hide_banner", "-y", "-i", str(video), "-vf", vf, "-af", af,
        *profile.video_encoder_args, *profile.audio_encoder_args, str(output),
    ]


__all__ = ["video_segment_command", "concat_manifest", "concat_command", "mux_audio_command", "fade_command"]
