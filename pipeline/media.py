from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


HASH_ALGORITHM = "sha256"


def stringify_command(command) -> list[str]:
    return [str(part) for part in command]


def run_cmd(command, check: bool = True, quiet: bool = False):
    command = stringify_command(command)
    if not quiet:
        print(">> " + " ".join(command), flush=True)
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.run(command, check=check, stdout=stdout, stderr=stderr)


def run_ffmpeg(command, quiet: bool = True) -> bool:
    command = stringify_command(command)
    if not quiet:
        print(">> " + " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(bez stderr)"
        print(f"⚠️ FFmpeg selhal (exit {result.returncode}):\n{tail}")
        return False
    return True


def probe_duration(path: Path) -> float:
    """Vrátí konečnou kladnou délku média, jinak 0.0."""
    try:
        output = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        value = float(output)
        if value <= 0 or value != value or value == float("inf"):
            raise ValueError(f"neplatná délka: {output!r}")
        return value
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"⚠️ Nepodařilo se zjistit délku souboru {path}: {exc}")
        return 0.0


def is_valid_media(path: Path, min_size: int = 500) -> bool:
    if not path.exists() or path.stat().st_size < min_size:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-i", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def file_hash(path: Path) -> str:
    """Stabilní obsahový fingerprint pro backup/provenance kontrolu.

    Od této verze je hash explicitně označen algoritmem. Staré neoznačené MD5
    hodnoty se záměrně nepovažují za kompatibilní s novým fingerprintem.
    """
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{HASH_ALGORITHM}:{digest.hexdigest()}"
    except OSError:
        return ""


__all__ = ["HASH_ALGORITHM", "stringify_command", "run_cmd", "run_ffmpeg", "probe_duration", "is_valid_media", "file_hash"]
