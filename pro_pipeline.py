#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pro_pipeline.py
===============
Sjednocený a pokročilý systém pro tvorbu AI videoklipů v prostředí Termux.
Integruje správu projektů, parsování plánů, tvorbu placeholderů,
audio analýzu (Whisper & Librosa), synchronizaci a finální render s efekty.

Použití (Interaktivní menu):
  python3 pro_pipeline.py

Použití (CLI):
  python3 pro_pipeline.py init             # Inicializace projektu
  python3 pro_pipeline.py parse            # Rozparsování full_plan.txt
  python3 pro_pipeline.py placeholders     # Vytvoření placeholderů médií
  python3 pro_pipeline.py analyze-song     # Analýza songu, Whisper, lyrics alignment a beaty
  python3 pro_pipeline.py scenario         # Fáze A: AI scénář (lyrics.txt + mood.txt + postava.txt); lokální/Groq viz 'settings'
  python3 pro_pipeline.py plan-ai          # Fáze B: AI full_plan.txt (scénář + transkripce + klipy.md); VŽDY lokální Ollama
  python3 pro_pipeline.py transcribe-rap   # Transkripce rap klipů a oprava podle lyrics.txt
  python3 pro_pipeline.py resync-rap       # Po ruční úpravě rap_alignment.json znovu vyhledá
                                            # lyrics_window podle lyrics.txt (bez re-transkripce)
  python3 pro_pipeline.py align-rap        # Segmentová úprava rychlosti rap částí
  python3 pro_pipeline.py align-vid        # Zarovná délku vid_xx broll klipů na timeline.txt (backup + speed korekce)
  python3 pro_pipeline.py update-timeline  # Přepočet timeline podle rap_start kotev
  python3 pro_pipeline.py apply-speeds-timeline  # Dopočet rychlostí rap klipů podle časů v timeline.txt (beze změny timeline)
  python3 pro_pipeline.py validate         # Kontrola výstupů a timeline
  python3 pro_pipeline.py prepare-lipsync  # Příprava audio segmentů pro lip-sync
  python3 pro_pipeline.py inject-lipsync   # Vložení segmentů do timeline
  python3 pro_pipeline.py render           # Render videa (s --mode, --res, --fades, --beat-sync)
  python3 pro_pipeline.py settings         # Nastavení (Whisper model, CPU/GPU, FPS, speed limity)
  python3 pro_pipeline.py all              # Kompletní pipeline

České aliasy:
  inicializuj, parsuj, zastupce, analyzuj-song, transkribuj-rap,
  zarovnej-rap, zarovnej-vid, prepocitej-timeline, aplikuj-rychlosti-timeline,
  validuj, priprav-lipsync, renderuj, nastaveni, vse

Volitelné:
  Přidejte --project [Název] pro spuštění na konkrétním projektu z kořenové složky.
"""

import os
import sys
import re
import json
import shutil
import time
import argparse
import subprocess
import requests
import hashlib
import math
from pathlib import Path
import tempfile
import unicodedata
import wave
from difflib import SequenceMatcher

try:
    from pipeline.models import StepResult, TimelineEntry
    from pipeline.validation import validate_timeline
    from pipeline.runtime import project_logger, log_exception
    from pipeline.commands import MAIN_COMMAND_ALIASES
    from pipeline.alignment import clamp_speed, speed_for_slot, distribute_gap, validate_alignment_ranges
    from pipeline.renderer import video_segment_command, concat_manifest, concat_command, mux_audio_command, fade_command
    from pipeline.ai import dispatch_text, normalize_provider, finite_positive_int, finite_temperature
    from pipeline.orchestration import execute_step, execute_sequence
    from pipeline.precision import validate_duration_drift, validate_lipsync_manifest, ffprobe_media_qa, DEFAULT_DURATION_TOLERANCE_MS
    from pipeline.visual_quality import enrich_beats, nearest_sync_point
    from pipeline.output_quality import profile_for, run_loudness_audit
    from pipeline.productivity import ensure_seed, load_segment_locks, write_preview_report, generate_contact_sheet
    from pipeline.dramaturgy import build_dramaturgy_plan, section_at_time
    from pipeline.visual_qa import audit_video
    from pipeline.lipsync import build_lipsync_manifest, validate_manifest_against_ranges, DEFAULT_WORD_TOLERANCE_MS
    from pipeline.catalog_quality import write_catalog_quality_report
    from pipeline.motion import transition_plan, motion_filters
    from pipeline.social import profile_for as social_profile_for, social_export_command, thumbnail_command, rank_thumbnail_candidates
    from pipeline.experiments import write_experiment_manifest
    from pipeline.observability import append_render_event, write_qa_summary, read_render_registry
except ImportError:
    from models import StepResult, TimelineEntry
    from validation import validate_timeline
    from runtime import project_logger, log_exception
    from commands import MAIN_COMMAND_ALIASES
    from alignment import clamp_speed, speed_for_slot, distribute_gap, validate_alignment_ranges
    from renderer import video_segment_command, concat_manifest, concat_command, mux_audio_command, fade_command
    from ai import dispatch_text, normalize_provider, finite_positive_int, finite_temperature
    from orchestration import execute_step, execute_sequence
    from precision import validate_duration_drift, validate_lipsync_manifest, ffprobe_media_qa, DEFAULT_DURATION_TOLERANCE_MS
    from visual_quality import enrich_beats, nearest_sync_point
    from output_quality import profile_for, run_loudness_audit
    from productivity import ensure_seed, load_segment_locks, write_preview_report, generate_contact_sheet
    from dramaturgy import build_dramaturgy_plan, section_at_time
    from visual_qa import audit_video
    from lipsync import build_lipsync_manifest, validate_manifest_against_ranges, DEFAULT_WORD_TOLERANCE_MS
    from catalog_quality import write_catalog_quality_report
    from motion import transition_plan, motion_filters
    from social import profile_for as social_profile_for, social_export_command, thumbnail_command, rank_thumbnail_candidates
    from experiments import write_experiment_manifest
    from observability import append_render_event, write_qa_summary, read_render_registry

# Pokus o import librosa
try:
    import librosa
    # Vyvoláme líné načtení základní funkce, abychom ověřili funkčnost numba/soxr
    _ = librosa.load
    HAS_LIBROSA = True
except (ImportError, RuntimeError, AttributeError):
    HAS_LIBROSA = False

# Pokus o import Groq SDK (cloudová transkripce přes API, viz nastavení)
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

ROOT = Path(__file__).resolve().parent

# Sdílený soubor s Groq API klíčem — stejný soubor používá i video.py,
# pokud leží ve stejné složce. Proměnná prostředí GROQ_API_KEY má přednost.
GROQ_KEY_FILE = ROOT / "groq_api_key.txt"

def load_groq_api_key() -> str:
    """Vrátí Groq API klíč: nejprve z proměnné prostředí GROQ_API_KEY, jinak ze souboru groq_api_key.txt."""
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        if GROQ_KEY_FILE.exists():
            for line in GROQ_KEY_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except Exception:
        pass
    return ""

def ensure_groq_key_file_template():
    """Vytvoří šablonu groq_api_key.txt, pokud ještě neexistuje."""
    if not GROQ_KEY_FILE.exists():
        try:
            GROQ_KEY_FILE.write_text(
                "# Vlož svůj Groq API klíč na řádek níže (bez uvozovek).\n"
                "# Klíč získáš na https://console.groq.com/keys\n"
                "# Tento soubor sdílí video.py i pro_pipeline.py.\n\n",
                encoding="utf-8",
            )
        except Exception:
            pass

# ===== JEDNOTNÁ VRSTVA PRO GROQ CLOUD LLM (textová/scénářová generace) =====
#
# Na rozdíl od _groq_transcribe_file() (přepis audia přes Whisper na Groq) tato
# sekce volá Groq Chat Completions API (LLM) — používá se jako cloudová
# alternativa k lokální Ollamě POUZE pro Fázi A (volba 8a), když v nastavení
# (volba 13) uživatel zvolí text_ai_provider = "groq". Stejně jako u Ollama
# vrstvy platí: při jakémkoli selhání funkce tiše vrátí None a volající musí
# umět reagovat (u 8a hláškou, NE pádem skriptu).
#
# Fáze B (8b, full_plan.txt) Groq NEPOUŽÍVÁ vůbec — běží vždy lokálně přes
# Ollamu (viz _generate_with_text_ai). Důvod: full_plan.txt je dlouhý,
# strukturovaný výstup (kompletní timeline songu) a na Groq free-tier účtu
# (TPM limit, typicky 8000 tokenů/min) se do limitu spolehlivě nevejde —
# zvlášť s českým textem, který se tokenizuje méně efektivně než angličtina.

# Doporučené (aktuální) produkční modely na Groq pro tuto úlohu (scénář/plán,
# čeština, delší strukturovaný výstup). gpt-oss-120b je aktuální vlajkový
# open-weight model na Groq s nejlepší kvalitou pro tento typ generování;
# gpt-oss-20b je rychlejší/levnější náhrada, pokud stačí nižší kvalita.
GROQ_LLM_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_LLM_FAST_MODEL = "openai/gpt-oss-20b"
GROQ_LLM_LAST_ERROR = {"message": None}  # nastaveno při každém selhání groq_chat()


def groq_chat(
    messages,
    model: str = None,
    api_key: str = None,
    temperature: float = 0.7,
    timeout: int = 120,
    max_tokens: int = None,
    reasoning_effort: str = None,
):
    """Zavolá Groq Cloud Chat Completions API (LLM, ne transkripci) a vrátí text
    odpovědi asistenta, nebo None při chybě. Zrcadlí rozhraní ollama_chat(), aby
    šlo v 8a snadno přepínat mezi lokální Ollamou a touto cloudovou variantou
    (Fáze B / 8b Groq nepoužívá — viz komentář výše).

    U reasoning modelů (openai/gpt-oss-20b, openai/gpt-oss-120b) se část
    max_completion_tokens spotřebuje na skryté "přemýšlení" (reasoning), než
    model vůbec začne psát viditelnou odpověď. Pokud je limit příliš nízký (nebo
    není nastavený vůbec a Groq použije nízký výchozí limit), model spotřebuje
    všechny tokeny na přemýšlení a vrátí prázdný `content` s finish_reason
    "length" — proto zde u gpt-oss modelů nastavujeme rozumně vysoký výchozí
    max_completion_tokens a nižší reasoning_effort, pokud volající nezadá jinak.

    Funkce nikdy nevyhazuje výjimku — při jakémkoli problému vrací None a
    volající musí spadnout zpět na chybovou hlášku / lokální variantu."""
    GROQ_LLM_LAST_ERROR["message"] = None
    if not HAS_GROQ:
        GROQ_LLM_LAST_ERROR["message"] = "Balíček `groq` není nainstalován (pip install groq --break-system-packages)."
        return None
    key = api_key or load_groq_api_key()
    if not key:
        GROQ_LLM_LAST_ERROR["message"] = f"Chybí Groq API klíč (nastav GROQ_API_KEY nebo {GROQ_KEY_FILE})."
        return None
    model_name = model or GROQ_LLM_DEFAULT_MODEL
    is_reasoning_model = "gpt-oss" in model_name
    effective_max_tokens = int(max_tokens) if max_tokens else (16000 if is_reasoning_model else None)
    try:
        client = Groq(api_key=key, timeout=timeout)
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        # U gpt-oss modelů dej reasoningu i odpovědi dost prostoru — bez explicitního
        # limitu Groq použije nízký default, do kterého se dlouhý strukturovaný výstup
        # (scénář/full_plan) nevejde a model skončí s prázdným content (finish_reason=length).
        if effective_max_tokens:
            kwargs["max_completion_tokens"] = effective_max_tokens
        if is_reasoning_model:
            kwargs["reasoning_effort"] = reasoning_effort or "low"
        try:
            completion = client.chat.completions.create(**kwargs)
        except TypeError as exc:
            # Starší verze `groq` SDK nemusí znát parametr reasoning_effort — zkus to
            # bez něj, než celý požadavek vzdáme.
            if "reasoning_effort" in kwargs and "reasoning_effort" in str(exc):
                kwargs.pop("reasoning_effort")
                completion = client.chat.completions.create(**kwargs)
            else:
                raise
        choice = completion.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            finish_reason = getattr(choice, "finish_reason", None)
            reasoning = (getattr(choice.message, "reasoning", None) or "").strip()
            if finish_reason == "length":
                GROQ_LLM_LAST_ERROR["message"] = (
                    "Groq API vrátilo prázdnou odpověď — model spotřeboval celý limit tokenů "
                    f"({effective_max_tokens or 'výchozí'}) na skryté 'přemýšlení' (reasoning) a nestihl "
                    "napsat viditelnou odpověď. Zkus zvýšit max_tokens, snížit reasoning_effort na "
                    "'low'/'none', nebo přepnout na jiný (ne-reasoning) Groq model."
                    + (f" [reasoning bylo dlouhé {len(reasoning)} znaků]" if reasoning else "")
                )
            else:
                GROQ_LLM_LAST_ERROR["message"] = f"Groq API vrátilo prázdnou odpověď (finish_reason={finish_reason})."
            return None
        return content
    except Exception as exc:
        GROQ_LLM_LAST_ERROR["message"] = f"{type(exc).__name__}: {exc}"
        return None

# ===== POMOCNÉ FUNKCE PRO FFPROBE A FFMPEG =====

def _stringify_cmd(cmd) -> list[str]:
    return [str(part) for part in cmd]

def run_cmd(cmd, check=True, quiet=False):
    """Spustí příkaz v příkazové řádce s bezpečnou konverzí cest."""
    cmd = _stringify_cmd(cmd)
    if not quiet:
        print(">> " + " ".join(cmd), flush=True)
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.run(cmd, check=check, stdout=stdout, stderr=stderr)

def run_ffmpeg(cmd, quiet=True) -> bool:
    """Spustí FFmpeg a při selhání vypíše konec stderr."""
    cmd = _stringify_cmd(cmd)
    if not quiet:
        print(">> " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_lines = (result.stderr or "").strip().splitlines()
        tail = "\n".join(stderr_lines[-5:]) if stderr_lines else "(bez stderr)"
        print(f"⚠️ FFmpeg selhal (exit {result.returncode}):\n{tail}")
        return False
    return True

def probe_duration(path: Path) -> float:
    """Zjistí délku video nebo audio souboru pomocí ffprobe."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True
        ).strip()
        return float(out)
    except Exception as e:
        print(f"⚠️ Nepodařilo se zjistit délku souboru {path}: {e}")
        return 0.0

def is_valid_media(path: Path) -> bool:
    """Zkontroluje, zda soubor existuje a je čitelný pro FFmpeg."""
    if not path.exists() or path.stat().st_size < 500:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-i", str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except Exception:
        return False

def file_hash(path: Path) -> str:
    """Spočítá MD5 hash obsahu souboru. Používá se k rozpoznání, jestli byl
    soubor (např. gen_vid/vid_14.mp4) mezi dvěma běhy align kroku ručně
    nahrazen novým obsahem, nebo jde stále o výstup z minulého běhu (viz
    OPRAVA v align_vid_clips / align_rap_clips: zálohy v *_original_backup/
    se dřív používaly natrvalo i po nahrazení zdrojového klipu novým obsahem)."""
    if not path.exists():
        return ""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

# Přechodná fasáda: externí media helpery jsou nyní centralizované v pipeline.media.
try:
    from pipeline.media import run_cmd, run_ffmpeg, probe_duration, is_valid_media, file_hash
except ImportError:
    from media import run_cmd, run_ffmpeg, probe_duration, is_valid_media, file_hash

# ===== PARSOVÁNÍ PLÁNU =====

def extract_sections(text):
    """Extrahuje známé sekce a neznámé Markdown nadpisy nepustí do předchozí sekce."""
    main_sections = {
        "ANALYZA", "ANALÝZA", "SONG_THEME", "PROJECT_THEME", "PROJECT THEME", "ASSET_PLANNING",
        "VIDEO_STYLE", "RAP_CHARACTER_STYLE", "RAPPER_OUTFIT_PROMPT", "LYRICS_STRUCTURE",
        "VIDEO_PROMPTS", "VERIFIED_VIDEO_ASSETS", "RAPPER_PROMPTS", "CHARACTER_PROMPTS",
        "VERIFIED_RAPPER_ASSETS", "IMAGE_PROMPTS", "VERIFIED_IMAGE_ASSETS",
        "RAPPER_SEGMENT_ALIGNMENT", "MUSIC_VIDEO_TIMELINE", "CURRENT_MUSIC_VIDEO_STRUCTURE",
        "SHOT_ORDER", "EFFECTS", "COLOR_GRADING", "METADATA", "NOVE_POTREBNE_KLIPY",
    }
    normalized_known = {s.replace(" ", "_") for s in main_sections}
    header_re = re.compile(r"^#{1,6}\s+(.+?)\s*$")
    sections = {}
    current_section = None
    content = []

    def flush():
        nonlocal content
        if current_section is not None:
            sections[current_section] = "\n".join(content).strip()
        content = []

    for line in text.splitlines():
        stripped = line.strip()
        header = header_re.match(stripped) if stripped else None
        raw_name = header.group(1).strip().rstrip(":") if header else stripped.rstrip(":")
        normalized = raw_name.upper().replace(" ", "_")
        is_plain_known_header = not header and normalized in normalized_known
        if header or is_plain_known_header:
            flush()
            current_section = normalized if normalized in normalized_known else None
            continue
        if current_section is not None:
            content.append(line)
    flush()
    return sections

def parse_timecode(value: str) -> float:
    """Převede bezpečně `SS`, `MM:SS` nebo `HH:MM:SS` na sekundy."""
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
                raise ValueError(f"Neplatná minutová/vteřinová složka: {value!r}")
            result = sum(part * 60 ** power for power, part in enumerate(reversed(values)))
        else:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"Neplatný časový kód: {value!r}") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Čas musí být konečné nezáporné číslo: {value!r}")
    return result

def format_timecode(seconds: float) -> str:
    """Naformátuje sekundy do MM:SS.hh."""
    seconds = max(0.0, float(seconds))
    mm = int(seconds // 60)
    ss = seconds - mm * 60
    return f"{mm:02d}:{ss:05.2f}"

def word_count(text: str) -> int:
    """Vrátí počet slov v textu."""
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))

def normalize_text(text: str) -> str:
    """Normalizuje text pro porovnávání."""
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^\w\sáčďéěíňóřšťúůýž]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()

def tokenize(text: str) -> list[str]:
    """Rozdělí text na tokeny bez krátkých filler slov."""
    stop = {"a", "i", "v", "ve", "na", "do", "to", "ten", "ta", "ty", "se", "si", "za", "s", "z", "jsem", "jsi", "jsme", "jste", "je", "jsou"}
    tokens = re.findall(r"\b\w+\b", normalize_text(text), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 2 and t not in stop]

def lyric_words(text: str) -> list[str]:
    """Vrátí slova textu v původním tvaru, bez sekčních značek."""
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return re.findall(r"\b[\wáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]+\b", text, flags=re.UNICODE)

def normalized_words(words: list[str]) -> list[str]:
    """Normalizuje seznam slov pro sekvenční porovnání."""
    return [normalize_text(w) for w in words if normalize_text(w)]

def clean_asset_id(value: str) -> str:
    """Vyčistí ID assetu z markdown zápisu, uvozovek a přebytečných znaků."""
    return value.strip().strip("`'\" ").strip()

def clean_code_block_text(text: str) -> str:
    """Odstraní markdown code fence řádky ze sekce plánu."""
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()

def normalize_timeline_text(text: str) -> str:
    """Znormalizuje řádky timeline, aby měly alespoň čas, asset a poznámku."""
    lines = []
    for raw in clean_code_block_text(text).splitlines():
        line = raw.strip().strip("`")
        if not line:
            continue
        if "|" not in line or "-" not in line:
            lines.append(raw)
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 2:
            lines.append(f"{parts[0]} | {clean_asset_id(parts[1])} |")
        else:
            parts[1] = clean_asset_id(parts[1])
            lines.append(" | ".join(parts))
    return "\n".join(lines).strip()

def parse_timeline_entries(text: str) -> tuple[list[TimelineEntry], list[str]]:
    """Převede textovou timeline na strukturované položky a vrátí syntaktická varování."""
    entries = []
    warnings = []
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
        description = parts[2].strip() if len(parts) == 3 else ""
        entries.append(TimelineEntry(start, end, clip_id, description))
    return entries, warnings

# Přechodná kompatibilní fasáda: autoritativní parsery jsou nyní v pipeline.parsers.
try:
    from pipeline.parsers import (
        extract_sections, parse_timecode, format_timecode, clean_asset_id,
        clean_code_block_text, normalize_timeline_text, parse_timeline_entries,
    )
except ImportError:
    from parsers import (
        extract_sections, parse_timecode, format_timecode, clean_asset_id,
        clean_code_block_text, normalize_timeline_text, parse_timeline_entries,
    )

# ===== JEDNOTNÁ VRSTVA PRO LOKÁLNÍ AI (OLLAMA) =====
#
# Tato sekce poskytuje sdílené, bezstavové funkce pro volání lokálního Ollama
# serveru. Jsou navrženy tak, aby při jakémkoli selhání (server neběží, timeout,
# neplatná odpověď) tiše vrátily None/False a volající kód vždy spadl zpět na
# existující heuristiku — nikdy nesmí dojít k pádu skriptu kvůli nedostupné AI.

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_AVAILABLE_CACHE = {"checked": False, "ok": False, "checked_at": 0.0}
OLLAMA_CACHE_TTL_SEC = 15.0
OLLAMA_LAST_ERROR = {"message": None}  # nastaveno při každém selhání ollama_generate/ollama_chat


def ollama_available(timeout: float = 2.0, force: bool = False) -> bool:
    """Zjistí, zda lokálně běží Ollama server (endpoint /api/tags).

    Výsledek je krátce cachovaný v procesu, aby se kontrola neopakovala
    zbytečně mnohokrát v rámci jednoho běhu skriptu. Použij force=True pro
    vynucení nové kontroly (např. po změně nastavení v menu)."""
    now = time.monotonic()
    cache_fresh = now - OLLAMA_AVAILABLE_CACHE.get("checked_at", 0.0) < OLLAMA_CACHE_TTL_SEC
    if not force and OLLAMA_AVAILABLE_CACHE["checked"] and cache_fresh:
        return OLLAMA_AVAILABLE_CACHE["ok"]
    ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        ok = r.status_code == 200
    except (requests.exceptions.RequestException, OSError):
        ok = False
    OLLAMA_AVAILABLE_CACHE.update({"checked": True, "ok": ok, "checked_at": now})
    return ok


def ollama_generate(
    prompt: str,
    model: str = None,
    format: str = "json",
    temperature: float = 0.2,
    timeout: int = 30,
    system: str = None,
    num_ctx: int = None,
):
    """Zavolá lokální Ollama (/api/generate) a vrátí syrový text odpovědi, nebo None při chybě.

    - `format="json"` požádá Ollamu, aby vynutila validní JSON výstup (podporováno
      novějšími verzemi Ollama). Volající by měl i tak vždy výstup validovat.
      `format=None`/jiná hodnota nechá odpověď jako volný text (pro prózu/scénáře).
    - `num_ctx` explicitně nastaví velikost kontextového okna modelu. Ollama má
      defaultně jen 2048 tokenů bez ohledu na to, co model umí — pro delší prompty
      (scénář, produkční plán) je nutné ho zvýšit, jinak Ollama TICHO ořízne vstup.
    - Funkce nikdy nevyhazuje výjimku — při jakémkoli problému vrací None a
      volající musí spadnout zpět na heuristiku.
    """
    if not ollama_available():
        return None
    model_name = model or OLLAMA_DEFAULT_MODEL
    options = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system:
        payload["system"] = system
    if format == "json":
        payload["format"] = "json"
    OLLAMA_LAST_ERROR["message"] = None
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
        if r.status_code != 200:
            OLLAMA_LAST_ERROR["message"] = f"HTTP {r.status_code}: {truncate_for_prompt(r.text, 300)}"
            return None
        data = r.json()
        response = data.get("response", "")
        if not response:
            OLLAMA_LAST_ERROR["message"] = f"Ollama vrátila prázdnou odpověď. Raw: {truncate_for_prompt(str(data), 300)}"
            return None
        return response
    except requests.exceptions.Timeout:
        OLLAMA_LAST_ERROR["message"] = f"Timeout po {timeout}s — model generuje déle, než je nastavený limit."
        return None
    except requests.exceptions.ConnectionError as exc:
        OLLAMA_LAST_ERROR["message"] = f"Nelze se připojit k Ollama serveru: {exc}"
        return None
    except Exception as exc:
        OLLAMA_LAST_ERROR["message"] = f"{type(exc).__name__}: {exc}"
        return None


def ollama_chat(
    messages,
    model: str = None,
    format: str = "json",
    temperature: float = 0.2,
    timeout: int = 30,
    num_ctx: int = None,
):
    """Zavolá lokální Ollama (/api/chat) pro vícezprávové konverzace. Vrací text odpovědi
    asistenta, nebo None při chybě. Používá se tam, kde dává smysl chat formát
    (např. delší kontext s rolí systému) namísto jednoho promptu.
    `num_ctx` viz poznámka u ollama_generate() — nutné pro delší prompty."""
    if not ollama_available():
        return None
    model_name = model or OLLAMA_DEFAULT_MODEL
    options = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if format == "json":
        payload["format"] = "json"
    OLLAMA_LAST_ERROR["message"] = None
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
        if r.status_code != 200:
            OLLAMA_LAST_ERROR["message"] = f"HTTP {r.status_code}: {truncate_for_prompt(r.text, 300)}"
            return None
        data = r.json()
        content = (data.get("message", {}) or {}).get("content", "")
        if not content:
            OLLAMA_LAST_ERROR["message"] = f"Ollama vrátila prázdnou odpověď. Raw: {truncate_for_prompt(str(data), 300)}"
            return None
        return content
    except requests.exceptions.Timeout:
        OLLAMA_LAST_ERROR["message"] = f"Timeout po {timeout}s — model generuje déle, než je nastavený limit."
        return None
    except requests.exceptions.ConnectionError as exc:
        OLLAMA_LAST_ERROR["message"] = f"Nelze se připojit k Ollama serveru: {exc}"
        return None
    except Exception as exc:
        OLLAMA_LAST_ERROR["message"] = f"{type(exc).__name__}: {exc}"
        return None


def ollama_chat_stream(
    messages,
    model: str = None,
    temperature: float = 0.2,
    num_ctx: int = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 600.0,
    max_total_seconds: float = 3600.0,
    progress_every: float = 20.0,
):
    """Zavolá lokální Ollama (/api/chat) STREAMOVANĚ a vrátí kompletní text odpovědi,
    nebo None při chybě.

    Na rozdíl od ollama_chat() (jeden pevný `timeout` na CELOU generaci) tady hlídáme
    jen `read_timeout` MEZI jednotlivými přijatými chunky (tokeny). Na slabém CPU
    hardwaru (bez GPU, málo RAM) může generace delšího strukturovaného výstupu (např.
    full_plan.txt pro celou písničku) klidně trvat desítky minut — dokud model průběžně
    produkuje tokeny, běh nezabijeme jen proto, že celkový čas přesáhl nějaké pevné číslo
    (to dřív způsobovalo `Timeout po 1800s` i když Ollama ve skutečnosti pracovala dál).

    DŮLEŽITÉ: `read_timeout` musí pokrýt i tzv. prefill (zpracování celého vstupního
    promptu modelem), NE jen mezery mezi tokeny při psaní odpovědi. Na CPU s velkým
    kontextem (`num_ctx`) může prefill u 7B modelu sám o sobě trvat i několik minut,
    než padne úplně první token streamu — proto je defaultní hodnota vyšší (600s), ne
    jen pár desítek sekund; příliš nízká hodnota tu dřív hlásila falešný "zaseknutý
    model", i když Ollama jen ještě zpracovávala prompt.
    `max_total_seconds` je jen krajní pojistka proti opravdu zaseknutému/nekonečnému běhu.

    Funkce nikdy nevyhazuje výjimku — při jakémkoli problému vrací None a nastaví
    OLLAMA_LAST_ERROR, volající musí spadnout zpět na chybovou hlášku."""
    OLLAMA_LAST_ERROR["message"] = None
    if not ollama_available():
        OLLAMA_LAST_ERROR["message"] = f"Nelze se připojit k Ollama serveru ({OLLAMA_BASE_URL})."
        return None
    model_name = model or OLLAMA_DEFAULT_MODEL
    options = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    start = time.time()
    last_progress = start
    content_parts = []
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as r:
            if r.status_code != 200:
                OLLAMA_LAST_ERROR["message"] = f"HTTP {r.status_code}: {truncate_for_prompt(r.text, 300)}"
                return None
            for line in r.iter_lines():
                if not line:
                    continue
                now = time.time()
                if now - start > max_total_seconds:
                    OLLAMA_LAST_ERROR["message"] = (
                        f"Generování překročilo celkovou pojistku {int(max_total_seconds)}s, i když model "
                        "stále průběžně produkoval tokeny (nešlo o zaseknutí). Zvyš 'ollama_stream_max_total_sec' "
                        "v Nastavení (volba 13), nebo zkus menší/rychlejší model (ollama_plan_model)."
                    )
                    return None
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("error"):
                    OLLAMA_LAST_ERROR["message"] = f"Ollama vrátila chybu: {obj['error']}"
                    return None
                chunk = (obj.get("message", {}) or {}).get("content", "")
                if chunk:
                    content_parts.append(chunk)
                if now - last_progress >= progress_every:
                    elapsed = int(now - start)
                    print(f"   ⏳ Ollama stále generuje... ({elapsed}s, zatím {sum(len(c) for c in content_parts)} znaků)")
                    last_progress = now
                if obj.get("done"):
                    break
        content = "".join(content_parts).strip()
        if not content:
            OLLAMA_LAST_ERROR["message"] = "Ollama vrátila prázdnou odpověď (stream)."
            return None
        return content
    except requests.exceptions.ReadTimeout:
        elapsed = int(time.time() - start)
        OLLAMA_LAST_ERROR["message"] = (
            f"Ollama {read_timeout:.0f}s neposlala žádný token (po {elapsed}s celkem). "
            "Pokud se to stalo hned na začátku (bez jakéhokoli průběžného '⏳' výpisu předtím), "
            "je nejpravděpodobnější příčinou pomalé zpracování promptu (prefill) na CPU se "
            "slabším hardwarem u velkého kontextu — ne nutně zaseknutý server. Zkus zvýšit "
            "read_timeout (Nastavení, volba 13) nebo použít menší model (ollama_plan_model)."
        )
        return None
    except requests.exceptions.ConnectionError as exc:
        OLLAMA_LAST_ERROR["message"] = f"Nelze se připojit k Ollama serveru: {exc}"
        return None
    except Exception as exc:
        OLLAMA_LAST_ERROR["message"] = f"{type(exc).__name__}: {exc}"
        return None


def extract_json_from_text(text: str):
    """Bezpečně extrahuje první validní JSON objekt z textové odpovědi modelu.

    Modely (zvlášť menší lokální) občas přidají markdown code-fence, preambuli
    nebo text za JSON objektem. Tato funkce se pokusí najít nejvnořenější blok
    `{...}` a naparsovat ho; při selhání vrátí None (volající musí použít fallback)."""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()
    # Odstraníme markdown code-fence, pokud je přítomný
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Nejprve zkusíme přímý parse celého textu
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: najdeme první { a odpovídající uzavírací } (počítáním závorek)
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def clamp_confidence(value, default: float = 0.0) -> float:
    """Ořeže hodnotu confidence do rozsahu 0.0-1.0, s bezpečným fallbackem."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(0.0, min(1.0, v))


def truncate_for_prompt(text: str, max_chars: int = 400) -> str:
    """Ořeže text na rozumnou délku pro vložení do promptu (šetří kontext/čas)."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ===== PROMPT TEMPLATES PRO AI-GENEROVANÝ STŘIHOVÝ PLÁN (VOLBA 8 / Fáze A+B) =====
#
# Fáze A (generate_scenario_ai): Prompt 1 — Režisér a Scenárista. Vstupy: text písně,
# volitelný popis nálady/žánru, textový popis hlavní postavy (místo obrázku — viz
# rozhodnutí kvůli 8GB RAM bez GPU). Výstup: scénář (Prompts/scenario.txt).
#
# Fáze B (generate_full_plan_ai): Prompt 2 — Video Pipeline Architect. Vstupy: scénář
# z Fáze A, transkripce písně, a DOSTUPNÉ EXISTUJÍCÍ KLIPY z INPUT/klipy.md. Výstup:
# kompletní full_plan.txt — s preferencí znovupoužití existujících klipů.

SCENARISTA_SYSTEM_PROMPT = """Jsi špičkový filmový režisér, vizuální umělec a scenárista specializující se na hudební videoklipy (český rap). Máš výjimečný cit pro atmosféru, storytelling, rytmus a estetiku. Dokážeš analyzovat emoce v hudbě a textu a přetavit je do silných, nezapomenutelných obrazů."""

SCENARISTA_PROMPT_TEMPLATE = """# ÚKOL
Na základě dodaného textu písně, popisu nálady/žánru, textového popisu hlavní postavy a seznamu JIŽ EXISTUJÍCÍCH natočených/vygenerovaných klipů vygeneruj kompletní, profesionálně strukturovaný scénář pro hudební videoklip.

# POKYNY PRO TVORBU SCÉNÁŘE

1. Analýza Vstupů:
   * Text: Analyzuj téma, emoce, příběh a dynamiku skladby (sloky vs. refrény).
   * Popis Postavy (Klíčové): Postava popsaná níže (přesně tak, jak je popsána) MUSÍ být protagonistou videoklipu. Její vzhled a styl musí zůstat konzistentní v celém scénáři (pokud scénář záměrně nepočítá s proměnou).
   * Atmosféra: Propoj náladu textu s estetikou popisu postavy.
2. Existující klipy (viz sekce "DOSTUPNÉ EXISTUJÍCÍ KLIPY" níže) — POUŽIJ, KDE TO SEDÍ:
   * Než vymyslíš scénu, zkontroluj, jestli obsahem (prostředí, akce, nálada) neodpovídá už existujícímu klipu z níže uvedeného seznamu.
   * Existující klipy nejsou povinné — piš scénář především podle textu a nálady, ale kdykoliv se existující klip do dané pasáže obsahově hodí, napiš tu scénu tak, aby mu odpovídala (stejné prostředí/akce/rekvizity), ať ho lze v Fázi B znovu použít místo generování nového.
   * U klipů rap_XX je uveden jejich SKUTEČNÝ odrapovaný text — pokud ho použiješ, umísti danou scénu do té části scénáře, kde se v textu písně skutečně tento úsek zpívá/rapuje.
   * Klidně vymýšlej i zcela nové scény, které žádný existující klip nepokrývá — přesné rozhodnutí, co se znovupoužije a co je nutné nově vytvořit, se řeší až technicky ve Fázi B.
3. Koncept a Storytelling: Vymysli ústřední vizuální koncept (lineární příběh, metafora, nebo série atmosférických obrazů). Postava musí být středobodem všeho dění.
4. Formátování a Struktura: Používej standardní scénáristický formát, strukturovaný podle částí písně (Intro, Sloka, Refrén, atd.):
   * ZÁHLAVÍ SCÉNY: (Např. EXT. OPUŠTĚNÁ TOVÁRNA - NOC).
   * Popis akce: Co se děje. Buď velmi popisný. Odkazuj na detaily z popisu postavy (např. "Postava, přesně podle popisu – v černé kožené bundě a s unaveným výrazem...").
   * Vizuální poznámky: Poznámky k osvětlení, kameře, barvám.
   * Text písně: Vlož odpovídající část textu k dané akci.
5. Dynamika a Gradace: Zajisti, aby vizuální stránka gradovala spolu s hudbou. Refrény musí být vizuálně nejvýraznější.
6. Zahrň na začátek scénáře i sekci "ANALÝZA PROTAGONISTY" (fyzické detaily postavy vytažené z popisu níže) a na konec sekci "LYRICS_STRUCTURE" — seznam hudebních sekcí s přibližnými časy ve formátu `nazev_sekce: M:SS - M:SS | krátký popis`, pokud je z textu/kontextu odhadnutelný.

# VSTUPNÍ DATA

## Text písně:
{lyrics}

## Popis nálady/žánru:
{mood}

## Popis hlavní postavy (protagonista, musí být použit přesně takto):
{character_description}

## DOSTUPNÉ EXISTUJÍCÍ KLIPY (INPUT/klipy.md) — použij je ve scénáři, kde obsahově sedí:
{existing_clips}

# POŽADOVANÝ VÝSTUP
Kompletní scénář (NÁZEV, KONCEPT, ANALÝZA PROTAGONISTY, strukturovaný rozpis scén po částech písně, LYRICS_STRUCTURE na konci). Piš pouze čistý text, žádné markdown code-fence bloky."""


# Fáze B běží ve DVOU navazujících voláních Ollamy (místo jednoho), aby seznam existujících
# klipů (Pravidlo č. 1) nemusel soutěžit o místo v kontextu se vším ostatním najednou:
#   ČÁST 1/2 (FULL_PLAN_PART1_*) — kreativní/asset sekce. Vstup: scénář + CELÝ seznam
#       existujících klipů (bez transkripce — ta zde není potřeba).
#   ČÁST 2/2 (FULL_PLAN_PART2_*) — časová osa. Vstup: transkripce + CELÝ seznam existujících
#       klipů + kompaktní shrnutí nově naplánovaných klipů z části 1/2 (bez scénáře).
# Výsledky obou volání se v generate_full_plan_ai() slučují do finálního full_plan.txt
# v pevně daném pořadí sekcí (viz canonical_order), takže nezáleží na tom, v jakém pořadí
# je model skutečně vypsal.

FULL_PLAN_PART1_SYSTEM_PROMPT = """Jsi špičkový AI Video Pipeline Architect, technický režisér a střihač hudebních videoklipů. V této části (1/2) se soustředíš výhradně na kreativní a asset plánování — analýzu, styl, a rozhodnutí, které klipy se znovupoužijí a které je nutné nově vytvořit. Časovou osu (timeline) řeší až druhá část, nezabývej se jí zde. Jsi posedlý přesností a konzistencí postavy napříč všemi prompty."""

FULL_PLAN_PART1_TEMPLATE = """# TVŮJ ÚKOL (ČÁST 1/2 — KREATIVNÍ A ASSET PLÁNOVÁNÍ)
Na základě dodaného scénáře a seznamu JIŽ EXISTUJÍCÍCH natočených/vygenerovaných klipů rozhodni, které klipy se znovupoužijí a pro které je nutné vytvořit nové prompty. Časovou osu (MUSIC_VIDEO_TIMELINE, SHOT_ORDER) v této části NEŘEŠ — ta se generuje samostatně v části 2/2 na základě tvého výstupu odsud.

---

# PRAVIDLO Č. 1 (NEJDŮLEŽITĚJŠÍ) — ZNOVUPOUŽITÍ EXISTUJÍCÍCH KLIPŮ
Níže v sekci "DOSTUPNÉ EXISTUJÍCÍ KLIPY" je seznam klipů, které už byly vytvořeny a čekají ve složkách projektu. Než napíšeš JAKÝKOLI nový vid_XX/rap_XX/pic_XX prompt:
1. Zkontroluj, zda existující klip svým obsahem (a u rap_XX i skutečným textem, který se v něm rapuje) odpovídá potřebě nějaké scény/pasáže ze scénáře.
2. Pokud ANO — tento klip NEPIŠ znovu do VIDEO_PROMPTS/RAPPER_PROMPTS/IMAGE_PROMPTS (už existuje, nemá se generovat znovu). Započítej ho jako "znovupoužitý" v ASSET_PLANNING.
3. Pokud NE (žádný existující klip nesedí) — teprve pak vytvoř nový prompt s NOVÝM ID (pokračuj v číslování za nejvyšším existujícím ID dané kategorie, např. pokud existuje vid_24, nový je vid_25).
4. Na konec dokumentu (sekce NOVE_POTREBNE_KLIPY) přidej seznam ID, která jsi musel nově vymyslet.

Poznámka: přesné zařazení ID (existujících i nových) do časové osy řeší až část 2/2 — tvým úkolem zde je jen rozhodnout, co se znovupoužije, a vytvořit prompty pro to, co chybí.

---

# PRAVIDLA PRO ANALÝZU SCÉNÁŘE A TVORBU PROMPTŮ (jen pro NOVĚ potřebné klipy)
1. Analýza Protagonisty: Ve scénáři najdi sekci "ANALÝZA PROTAGONISTY". Z ní vytvoř ultimátní RAPPER_OUTFIT_PROMPT v angličtině. Tento vzhled musí být konzistentní!
2. Tvorba Video Promptů (vid_XX): Přelož vizuální poznámky a popisy akcí do úderných anglických promptů (kamera, barvy, osvětlení, pohyb). Vždy uveď [Duration: 8.0s].
3. Tvorba Rap Promptů (rap_XX): Pro lip-sync pasáže vytvoř prompty začínající `Omni Flash. Animate 2D character...`. Zvol délku 4.0s, 6.0s nebo 8.0s podle textu a uveď ji jako [Duration: X.0s].

---

# STRUKTURA VÝSTUPU
Vygeneruj POUZE čistý text formátovaný přesně podle této šablony (nepoužívej markdown code bloky, žádné ```):

### ANALYZA
[Stručné shrnutí projektu, délka v sekundách, použitá technika na základě scénáře.]

### SONG_THEME
[Jedna věta vystihující ústřední děj/téma]

### ASSET_PLANNING
* Existující klipy znovupoužité: [počet] | Nově potřebné klipy: [počet]
* Veo 3.1 Lite (B-roll): vid_01 až vid_XX, délka 8,0 s.
* Omni Flash (Rap & Foto): rap_01 až rap_XX (4/6/8s), pic_01...

### VIDEO_STYLE
[Vizuální styl v angličtině]

### RAP_CHARACTER_STYLE
[Fyzický popis postavy česky, podle scénáře]

### RAPPER_OUTFIT_PROMPT
[Detailní prompt anglicky pro NOVĚ generované klipy postavy]

### VIDEO_PROMPTS
vid_XX|[Duration: 8.0s] [Anglický prompt] — POUZE pro nově potřebné klipy

### RAPPER_PROMPTS
rap_XX|[Duration: X.0s] Omni Flash. [Anglický prompt] — POUZE pro nově potřebné klipy

### IMAGE_PROMPTS
pic_XX | Zdrojové médium: [název] | [Duration: 8.0s] [prompt] — POUZE pro nově potřebné klipy

### EFFECTS
- [postprodukční efekty ze scénáře]

### METADATA
* Total Duration: [song_duration] seconds (MM:SS.ms)
* BPM: [odhad]
* Key: [odhad]
* Visual Frame Rate: 24fps
* Output Aspect Ratio: 16:9 widescreen
* AI Models Utilized: Veo 3.1 Lite, Omni Flash

### NOVE_POTREBNE_KLIPY
- [seznam ID nově vymyšlených klipů, nebo "žádné — vše pokryto existujícími klipy"]

---

# VSTUPNÍ DATA

## Scénář (z Fáze A):
{scenario}

## Celková délka písně (song_duration):
{song_duration}

## DOSTUPNÉ EXISTUJÍCÍ KLIPY (INPUT/klipy.md) — POUŽÍVEJ PŘEDNOSTNĚ, CELÝ SEZNAM JE ZÁVAZNÝ:
{existing_clips}

## ZÁVAZNÉ ČÍSLOVÁNÍ NOVÝCH ID (spočítáno mechanicky, NEPOČÍTEJ si vlastní):
Pokud pro danou kategorii vytváříš JAKÝKOLI nový klip, MUSÍ jeho ID začínat přesně zde uvedeným číslem
(a další nová ID v téže kategorii pokračují +1, +2, ...). NIKDY nepoužij ID, které je uvedené výše v
seznamu DOSTUPNÉ EXISTUJÍCÍ KLIPY jako existující — i kdyby se ti zdálo, že by "sedělo" obsahově líp.
{next_free_ids}

Zpracuj dodaná data a vygeneruj dokument přesně podle šablony výše, s důrazem na Pravidlo č. 1.

# DŮLEŽITÉ — ÚPLNOST VÝSTUPU
Musíš vygenerovat VŠECH 12 sekcí přesně v tomto pořadí: ### ANALYZA, ### SONG_THEME,
### ASSET_PLANNING, ### VIDEO_STYLE, ### RAP_CHARACTER_STYLE, ### RAPPER_OUTFIT_PROMPT,
### VIDEO_PROMPTS, ### RAPPER_PROMPTS, ### IMAGE_PROMPTS, ### EFFECTS, ### METADATA,
### NOVE_POTREBNE_KLIPY — každou přesně jednou. Žádnou sekci NEOPAKUJ. NEPIŠ sekce
MUSIC_VIDEO_TIMELINE ani SHOT_ORDER, ty se generují samostatně v části 2/2. Po dopsání
sekce NOVE_POTREBNE_KLIPY okamžitě skonči — bez shrnutí, bez závěrečného odstavce."""


FULL_PLAN_PART2_SYSTEM_PROMPT = """Jsi špičkový AI Video Pipeline Architect a střihač hudebních videoklipů. V této části (2/2) je tvým jediným úkolem sestavit na frame přesnou časovou osu (MUSIC_VIDEO_TIMELINE a SHOT_ORDER) z klipů, které jsou ti k dispozici — existujících i nově naplánovaných v části 1/2. Časová osa se generuje po menších navazujících úsecích (kvůli délce písně), ty se teď staráš jen o jeden konkrétní úsek. Jsi posedlý přesným timingem a perfektní synchronizací zvuku a obrazu (lip-sync)."""

FULL_PLAN_PART2_TEMPLATE = """# TVŮJ ÚKOL (ČÁST 2/2 — ČASOVÁ OSA, ÚSEK {segment_index}/{segment_total})
V části 1/2 už bylo rozhodnuto, které existující klipy se znovupoužijí a jaké nové klipy (s jakým ID a délkou) byly naplánovány. Tvým úkolem TEĎ je rozmístit tyto klipy (existující i nově naplánované) do PŘESNĚ VYMEZENÉHO ČASOVÉHO ÚSEKU písně — od {segment_start} do {segment_end}. Ostatní úseky písně generují samostatná navazující volání (úsek {segment_index} z celkových {segment_total}) — NEZABÝVEJ SE jimi a svým výstupem nepřesahuj hranice tohoto úseku.

---

# PRAVIDLO Č. 1 (NEJDŮLEŽITĚJŠÍ) — SPRÁVNÉ UMÍSTĚNÍ KLIPŮ
Níže v sekci "DOSTUPNÉ EXISTUJÍCÍ KLIPY" je seznam VŠECH klipů, které už existují v celém projektu (ne jen těch, co se hodí do tohoto úseku). Klipy rap_XX z tohoto seznamu mají uvedený svůj skutečný odrapovaný text — přednostně je umísti do timeline na místo, kam se tento text v písni skutečně hodí/odpovídá transkripci TOHOTO úseku. Nevymýšlej pro existující klipy nové ID ani nový obsah — použij přesně ID ze seznamu. Klip z tohoto seznamu, který obsahově patří jinam (mimo tento úsek), sem nezařazuj — objeví se ve svém vlastním úseku.

V sekci "NOVĚ NAPLÁNOVANÉ KLIPY (z části 1/2)" je seznam VŠECH nově vytvořených ID z celého projektu. Zařaď sem jen ta, která obsahově/textově patří do TOHOTO úseku; ostatní se zařadí v jiném navazujícím volání.

---

# PRAVIDLA PRO ČASOVÁNÍ A PEVNÉ DÉLKY KLIPŮ (KRITICKÉ!)
Transkripce tohoto úseku obsahuje přesné časy v sekundách. Rozděl ji tak, aby:
1. KAŽDÝ čas v tvém výstupu MUSÍ ležet MEZI {segment_start} A {segment_end} — nikdy dřív, nikdy později.
   Tenhle úsek pokrývá jen pár desítek vteřin písně, ne celou píseň ani žádnou jinou její část.
2. Formát je MM:SS.ms, kde SS (vteřinová složka) je VŽDY v rozsahu 00–59 — sekundy nikdy nepřeteč do
   minut jako syrové celkové číslo. Převod: minuty = celkové_sekundy // 60 (celá část), vteřiny =
   celkové_sekundy % 60 (zbytek). Např. 65.54 s -> 01:05.54 (NE 00:65.54). Pokud si nejsi jistý/á
   přesným součtem, radši zaokrouhli v rámci tohoto úseku, ale NIKDY nezapisuj vteřiny >= 60.
3. Rapové klipy (rap_XX, ať existující nebo nové) MUSÍ MÍT DÉLKU PŘESNĚ 4.0s, 6.0s nebo 8.0s — použij jejich uvedenou délku, drobný nesoulad se dořeší speed-rampingem.
4. B-roll klipy (vid_XX, pic_XX), ať existující nebo nové, MUSÍ MÍT DÉLKU PŘESNĚ 8.0s v timeline (existující klipy kratší než 8s lze zopakovat/prodloužit smyčkou, uveď to v poznámce).
5. Kontinuita UVNITŘ tohoto úseku: konec jednoho klipu = začátek dalšího, bez mezer a překryvů.
6. PRVNÍ klip tohoto úseku MUSÍ začínat PŘESNĚ na {segment_start}.
7. POSLEDNÍ klip tohoto úseku MUSÍ končit PŘESNĚ na {segment_end} — bez ohledu na to, jestli jde o konec celé písně, nebo jen o hranici s dalším navazujícím úsekem.
8. Použij jen ID, která do tohoto úseku obsahově patří; totéž ID nepoužívej vícekrát, pokud to nedává smysl (např. záměrné opakování motivu).

---

# STRUKTURA VÝSTUPU
Vygeneruj POUZE čistý text formátovaný přesně podle této šablony, POUZE za tento úsek (nepoužívej markdown code bloky, žádné ```). Časy v příkladu níže jsou schematické — MÍSTO NICH VŽDY POUŽIJ SKUTEČNÉ ČASY z rozsahu {segment_start}–{segment_end}, nikdy je needopisuj doslova:

### MUSIC_VIDEO_TIMELINE
MM:SS.ms-MM:SS.ms | id_klipu (existující NEBO nově naplánovaný z části 1/2) | Popis / text písně
[...pokračuj celým tímto úsekem, používej ID kdykoliv to obsahově sedí. DŮLEŽITÉ: čas piš PŘESNĚ takto —
MM:SS.ms-MM:SS.ms | id | popis — BEZ hranatých závorek kolem času, pomlčka spojuje časy bez mezer,
každý pár musí ležet uvnitř {segment_start}–{segment_end} a vteřiny musí být 00–59.]

### SHOT_ORDER
MM:SS.ms-MM:SS.ms | id_klipu |
[...kompletní seznam bez textu za tento úsek, stejný formát času jako výše, bez závorek...]

---

# VSTUPNÍ DATA

## Transkripce písně — POUZE tento úsek ({segment_start}–{segment_end}), segmenty s časy v sekundách:
{transcription}

## Celková délka písně (song_duration, jen pro kontext, netýká se hranic tohoto úseku):
{song_duration}

## DOSTUPNÉ EXISTUJÍCÍ KLIPY (INPUT/klipy.md) — CELÝ seznam z celého projektu, POUŽÍVEJ PŘEDNOSTNĚ:
{existing_clips}

## NOVĚ NAPLÁNOVANÉ KLIPY (z části 1/2) — CELÝ seznam z celého projektu, zařaď jen ta, co patří do tohoto úseku:
{new_clips_summary}

Zpracuj dodaná data a vygeneruj dokument přesně podle šablony výše, s důrazem na Pravidlo č. 1 a na přesné hranice úseku {segment_start}–{segment_end}.

# DŮLEŽITÉ — ÚPLNOST VÝSTUPU
Musíš vygenerovat PŘESNĚ 2 sekce v tomto pořadí: ### MUSIC_VIDEO_TIMELINE, ### SHOT_ORDER —
každou přesně jednou, a to POUZE za tento úsek ({segment_start}–{segment_end}). Žádnou sekci
NEOPAKUJ a nepiš žádné jiné sekce ani jiné úseky písně. Po dopsání SHOT_ORDER okamžitě skonči —
bez shrnutí, bez závěrečného odstavce, bez opakování pravidel."""


# ===== HLAVNÍ TŘÍDA PIPELINE =====

class TemagenPipeline:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.input_dir = project_dir / "INPUT"
        self.prompts_dir = project_dir / "Prompts"
        self.edit_dir = project_dir / "EDIT_PROJECT"
        self.gen_vid = project_dir / "gen_vid"
        self.gen_rap = project_dir / "gen_rap"
        self.gen_char = project_dir / "gen_char"
        self.gen_pic = project_dir / "gen_pic"
        self.export_dir = project_dir / "EXPORT"

        self.full_plan = self.prompts_dir / "full_plan.txt"
        self.timeline_file = self.edit_dir / "timeline.txt"
        self.settings_file = self.edit_dir / "settings.json"
        self.logger = project_logger(self.project_dir)

    # ===== NASTAVENÍ PROJEKTU (Whisper model, CPU/GPU, FPS, speed limity) =====

    DEFAULT_SETTINGS = {
        "whisper_model": "small",   # tiny / base / small / medium / large (pro lokální Whisper)
        "device": "cpu",            # cpu / gpu (pro lokální Whisper)
        "transcription_provider": "local",  # local (openai-whisper CLI) / groq (Groq Cloud API)
        "groq_model": "whisper-large-v3-turbo",  # nebo "whisper-large-v3" (přesnější, pomalejší)
        "fps_override": None,       # None = odvodit z rozlišení, jinak konkrétní FPS
        "speed_min": 0.5,
        "speed_max": 2.0,
        "ollama_enabled": True,               # Zapnout/vypnout použití lokální AI (Ollama)
        "ollama_model": OLLAMA_DEFAULT_MODEL,  # Model použitý pro Ollama (musí být `ollama pull`nutý)
        "ollama_scenario_model": OLLAMA_DEFAULT_MODEL,   # Model pro Fázi A (generování scénáře)
        "ollama_plan_model": "qwen2.5:7b-instruct",       # Model pro Fázi B (generování full_plan.txt) — větší, kvalitnější, ale pomalejší
        "ollama_plan_num_ctx": 8192,        # Fáze B (8b): velikost kontextového okna (num_ctx) pro Ollamu — PROMPT+ODPOVĚĎ dohromady.
                                             # Vyšší hodnota = méně ořezávání existing_clips_block a méně úseků v Části 2/2, ale výrazně
                                             # víc RAM a pomalejší prefill na CPU. Viz volba 13 → kategorie "Ollama pokročilé".
        "ollama_plan_max_chunk_seconds": 60.0,  # Fáze B (8b), Část 2/2: horní strop na délku ČASOVÉHO okna jednoho úseku
                                             # časové osy, NEZÁVISLE na num_ctx/znakovém rozpočtu. Vyšší num_ctx totiž umí
                                             # nacpat do jednoho volání klidně celou píseň (méně švů mezi úseky), ale slabší
                                             # lokální model pak snadno "ztratí nit" při přesném sčítání času přes několik
                                             # minut najednou (viz reálný případ: první řádek úseku dostal konec = konec
                                             # celé písně místo pár vteřin). Tenhle strop drží úseky kratší i při velkém
                                             # num_ctx — katalog existujících klipů se posílá celý v každém úseku bez ohledu
                                             # na tohle nastavení, takže se tím Pravidlo č. 1 neoslabuje.
        "ollama_stream_read_timeout_sec": 600,   # Fáze B (8b): max. čekání MEZI tokeny (vč. prvního — zahrnuje prefill) na CPU, viz ollama_chat_stream()
        "ollama_stream_max_total_sec": 7200,     # Fáze B (8b): celková pojistka na běh (i když model stabilně produkuje tokeny) — na slabém CPU může celý full_plan.txt trvat přes hodinu
        "text_ai_provider": "local",          # local (lokální Ollama) / groq (Groq Cloud API — LLM chat, POUZE pro Fázi A / volba 8a)
        "groq_scenario_model": GROQ_LLM_DEFAULT_MODEL,   # Groq LLM model pro Fázi A (generování scénáře), použije se když text_ai_provider="groq"
        "groq_scenario_max_tokens": 3000,     # Fáze A (8a) na Groq: horní strop na DÉLKU ODPOVĚDI (max_tokens). Groq
                                             # free-tier TPM limit (typicky 8000 tokenů/min) se počítá ze SOUČTU
                                             # vstupu i požadovaného výstupu — starý default 16000 byl sám o sobě
                                             # skoro celý limit, i pár set slov vstupu pak stačilo na 413
                                             # rate_limit_exceeded. Scénář reálně potřebuje řádově níž.
        # Fáze B (8b, full_plan.txt) běží vždy lokálně přes Ollamu (ollama_plan_model) — Groq se
        # pro ni nepoužívá, viz _generate_with_text_ai(). groq_plan_model proto záměrně chybí.
        "last_render_mode": "draft",          # Zapamatovaný režim z posledního renderu (volby 10/R) — jen jako výchozí předvyplnění
        "last_render_res": "draft",           # Zapamatované rozlišení z posledního renderu
        "last_render_beat_sync": True,        # Zapamatované beat-sync z posledního renderu
        "last_render_fades": True,            # Zapamatované stmívačky z posledního renderu
    }

    @classmethod
    def _normalize_settings(cls, settings: dict) -> dict:
        """Normalizuje a validuje hodnoty načtené z JSON před použitím v pipeline."""
        normalized = dict(cls.DEFAULT_SETTINGS)
        normalized.update({k: v for k, v in settings.items() if k in cls.DEFAULT_SETTINGS})
        try:
            normalized["speed_min"] = float(normalized["speed_min"])
            normalized["speed_max"] = float(normalized["speed_max"])
            if not (math.isfinite(normalized["speed_min"]) and math.isfinite(normalized["speed_max"])):
                raise ValueError
            if normalized["speed_min"] <= 0 or normalized["speed_max"] < normalized["speed_min"]:
                raise ValueError
        except (TypeError, ValueError):
            normalized["speed_min"] = float(cls.DEFAULT_SETTINGS["speed_min"])
            normalized["speed_max"] = float(cls.DEFAULT_SETTINGS["speed_max"])
        for key in ("ollama_plan_num_ctx", "ollama_stream_read_timeout_sec", "ollama_stream_max_total_sec"):
            try:
                value = int(normalized[key])
                if value <= 0:
                    raise ValueError
                normalized[key] = value
            except (TypeError, ValueError):
                normalized[key] = int(cls.DEFAULT_SETTINGS[key])
        normalized["ollama_enabled"] = bool(normalized.get("ollama_enabled", True))
        return normalized

    def load_settings(self) -> dict:
        """Načte nastavení projektu, doplní výchozí hodnoty a ověří rozsahy."""
        stored = self._load_json(self.settings_file, {})
        return self._normalize_settings(stored if isinstance(stored, dict) else {})

    def save_settings(self, settings: dict) -> None:
        """Uloží validovaná nastavení projektu do EDIT_PROJECT/settings.json."""
        self._write_json(self.settings_file, self._normalize_settings(settings))

    def _whisper_device_args(self, settings: dict = None) -> list[str]:
        """Vrátí argumenty pro whisper CLI podle nastaveného zařízení (CPU/GPU)."""
        settings = settings or self.load_settings()
        device = str(settings.get("device", "cpu")).lower()
        if device in ("gpu", "cuda"):
            return ["--device", "cuda"]
        return ["--device", "cpu"]

    def _groq_ready(self, verbose: bool = True) -> str:
        """Zkontroluje, že je Groq SDK nainstalovaný a je dostupný API klíč. Vrátí API klíč nebo ''."""
        if not HAS_GROQ:
            if verbose:
                print("❌ Balíček `groq` není nainstalován. Nainstaluj: pip install groq --break-system-packages")
            return ""
        api_key = load_groq_api_key()
        if not api_key:
            ensure_groq_key_file_template()
            if verbose:
                print("❌ Chybí Groq API klíč.")
                print(f"   Získej klíč na https://console.groq.com/keys a vlož ho do souboru: {GROQ_KEY_FILE}")
                print("   (nebo nastav proměnnou prostředí GROQ_API_KEY)")
            return ""
        return api_key

    def _groq_transcribe_file(self, media_path: Path, settings: dict, api_key: str) -> dict:
        """Pošle audio/video soubor na Groq Cloud API a vrátí Whisper-kompatibilní JSON (verbose_json).
        Groq API přímo podporuje mp4/mp3/wav/m4a/ogg/webm, není třeba extrahovat zvuk přes ffmpeg.
        Limit velikosti souboru je cca 19.5MB (dle Groq Playground)."""
        max_mb = 19.5
        size_mb = media_path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            print(f"❌ Soubor {media_path.name} má {size_mb:.1f}MB, limit Groq API je {max_mb}MB.")
            print("   Zkomprimuj audio, nebo použij lokální Whisper pro tento soubor.")
            return {}

        model = settings.get("groq_model", "whisper-large-v3-turbo")
        try:
            client = Groq(api_key=api_key)
            with open(media_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(media_path.name, f.read()),
                    model=model,
                    temperature=0,
                    language="cs",
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )
            return transcription.model_dump() if hasattr(transcription, "model_dump") else dict(transcription)
        except Exception as e:
            print(f"❌ Transkripce {media_path.name} přes Groq API selhala: {e}")
            return {}

    # ===== JEDNOTNÁ VRSTVA PRO LOKÁLNÍ AI (OLLAMA) — metody instance =====

    def _ollama_ready(self, settings: dict = None) -> bool:
        """Vrátí True, pokud je Ollama v nastavení projektu povolená A skutečně běží."""
        settings = settings or self.load_settings()
        if not settings.get("ollama_enabled", True):
            return False
        return ollama_available()

    def _ollama_model(self, settings: dict = None) -> str:
        settings = settings or self.load_settings()
        return str(settings.get("ollama_model") or OLLAMA_DEFAULT_MODEL)

    def _ollama_scenario_model(self, settings: dict = None) -> str:
        settings = settings or self.load_settings()
        return str(settings.get("ollama_scenario_model") or OLLAMA_DEFAULT_MODEL)

    def _ollama_plan_model(self, settings: dict = None) -> str:
        settings = settings or self.load_settings()
        return str(settings.get("ollama_plan_model") or OLLAMA_DEFAULT_MODEL)

    # ===== VÝBĚR POSKYTOVATELE PRO FÁZI A (8a): LOKÁLNÍ OLLAMA vs GROQ CLOUD LLM =====
    # (Fáze B / 8b vždy lokálně — viz _generate_with_text_ai)

    def _text_ai_provider(self, settings: dict = None) -> str:
        """Vrátí zvoleného poskytovatele AI pro generování scénáře (Fáze A, volba
        8a): 'local' (Ollama) nebo 'groq' (Groq Cloud LLM). Fáze B (8b) toto
        nastavení ignoruje a běží vždy lokálně."""
        settings = settings or self.load_settings()
        provider = str(settings.get("text_ai_provider", "local")).lower()
        return provider if provider in ("local", "groq") else "local"

    def _text_ai_ready(self, settings: dict = None) -> bool:
        """Zjistí, zda je aktuálně zvolený poskytovatel textové AI pro Fázi A (8a) dostupný.
        Pro Fázi B (8b) se místo toho používá přímo _ollama_ready()."""
        settings = settings or self.load_settings()
        if self._text_ai_provider(settings) == "groq":
            return HAS_GROQ and bool(load_groq_api_key())
        return self._ollama_ready(settings)

    def _groq_scenario_model(self, settings: dict = None) -> str:
        settings = settings or self.load_settings()
        return str(settings.get("groq_scenario_model") or GROQ_LLM_DEFAULT_MODEL)

    def _generate_with_text_ai(
        self,
        messages,
        phase: str,
        settings: dict = None,
        temperature: float = 0.7,
        timeout: int = 900,
        num_ctx: int = 8192,
        max_tokens: int = None,
    ):
        """Jednotný dispatcher pro Fázi A/B (8a='scenario', 8b='plan').

        Fáze A (scénář) respektuje nastavení 'text_ai_provider' (volba 13) —
        lokální Ollama, nebo Groq Cloud LLM (kratší výstup, do Groq free-tier
        TPM limitu se vejde snáz).

        Fáze B (full_plan.txt) běží VŽDY lokálně přes Ollamu, bez ohledu na
        'text_ai_provider'. Důvod: full_plan.txt je dlouhý, silně strukturovaný
        výstup (kompletní timeline songu) a na Groq free-tier účtu (TPM limit,
        typicky 8000 tokenů/min) se do limitu spolehlivě nevejde — buď selže
        rovnou na vstupu (413 rate_limit_exceeded), nebo se generování při
        streamu výstupu samo zadusí. Lokální Ollama žádný TPM limit nemá, jen
        je (na CPU bez GPU) pomalejší — proto se navíc volá streamovaně
        (ollama_chat_stream), aby dlouhé generování nezabilo jeden pevný
        timeout, dokud model průběžně produkuje tokeny.

        Vrací dvojici (text_odpovědi_nebo_None, jméno_použitého_modelu,
        chybová_hláška_nebo_None)."""
        settings = settings or self.load_settings()
        result = dispatch_text(
            messages=messages,
            phase=phase,
            settings=settings,
            local_model=(self._ollama_scenario_model(settings) if phase == "scenario" else self._ollama_plan_model(settings)),
            groq_model=self._groq_scenario_model(settings),
            local_call=ollama_chat,
            local_stream_call=ollama_chat_stream,
            groq_call=groq_chat,
            temperature=temperature,
            timeout=timeout,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
        )
        if result.text:
            return result.text, result.model, None
        if result.provider == "groq":
            error = GROQ_LLM_LAST_ERROR.get("message") or result.error or "neznámá chyba Groq"
        else:
            error = OLLAMA_LAST_ERROR.get("message") or result.error or "neznámá chyba Ollamy"
        return None, result.model, error

    # ===== STAV PROJEKTU A NASTAVENÍ RENDERU PRO INTERAKTIVNÍ MENU =====

    def status_summary_line(self, settings: dict = None) -> str:
        """Jednořádkové shrnutí aktuálně zvolených AI poskytovatelů — zobrazuje se
        v hlavičce hlavního menu, ať uživatel nemusí kvůli tomu chodit do volby 13."""
        settings = settings or self.load_settings()
        trans_provider = str(settings.get("transcription_provider", "local")).lower()
        trans_label = f"groq/{settings.get('groq_model', 'whisper-large-v3-turbo')}" if trans_provider == "groq" else f"local/{settings.get('whisper_model', 'small')}"
        text_provider = self._text_ai_provider(settings)
        if text_provider == "groq":
            scenario_label = f"groq/{self._groq_scenario_model(settings)}"
        else:
            scenario_label = f"local/{self._ollama_scenario_model(settings)}"
        # Fáze B (8b, full_plan.txt) běží vždy lokálně přes Ollamu — viz _generate_with_text_ai().
        plan_label = f"local/{self._ollama_plan_model(settings)}"
        return f"Transkripce: {trans_label}  |  AI scénář (8a): {scenario_label}  |  AI plán (8b): {plan_label}"

    def project_progress_checklist(self) -> str:
        """Krátký přehled, které klíčové mezisoubory pipeline už existují — pomáhá
        rychle poznat, na kterém kroku je projekt rozpracovaný, bez nutnosti
        procházet jednotlivé volby menu."""
        checks = [
            ("full_plan.txt", self.full_plan.exists() and self.full_plan.stat().st_size > 0),
            ("song_alignment.json (4)", (self.edit_dir / "song_alignment.json").exists()),
            ("rap_alignment.json (5)", (self.edit_dir / "rap_alignment.json").exists()),
            ("timeline.txt (7)", self.timeline_file.exists() and self.timeline_file.stat().st_size > 0),
        ]
        parts = [f"{'✅' if ok else '⬜'} {label}" for label, ok in checks]
        return "  ".join(parts)

    def _prompt_render_settings(self, settings: dict = None) -> dict:
        """Zeptá se na parametry renderu (mód / rozlišení / beat-sync / stmívačky).
        Výchozí hodnoty předvyplní z posledního renderu tohoto projektu (uloženo
        v EDIT_PROJECT/settings.json), aby se u opakovaného renderu nemuselo
        pokaždé přeťukávat totéž. Sdíleno volbami 10 (Render videa) a R (Render
        bez rapu)."""
        settings = settings if settings is not None else self.load_settings()
        last_mode = settings.get("last_render_mode", "draft")
        last_res = settings.get("last_render_res", "draft")
        last_beat = settings.get("last_render_beat_sync", True)
        last_fades = settings.get("last_render_fades", True)

        mode_default = "2" if last_mode == "final" else "1"
        m_choice = input(f"Režim (1 - draft [rychlý], 2 - final [cinematic]) [Enter = {mode_default}]: ").strip() or mode_default
        mode = "final" if m_choice == "2" else "draft"

        res_default = {"draft": "1", "hd": "2", "fullhd": "3"}.get(last_res, "1")
        r_choice = input(f"Rozlišení (1 - draft [640x360], 2 - hd [1280x720], 3 - fullhd [1920x1080]) "
                         f"[Enter = {res_default}]: ").strip() or res_default
        res = "fullhd" if r_choice == "3" else ("hd" if r_choice == "2" else "draft")

        beat_default = "a" if last_beat else "n"
        beat_choice = (input(f"Synchronizovat na beaty a přidat efekty? (a/n) [Enter = {beat_default}]: ").strip().lower()
                       or beat_default)
        beat_sync = beat_choice != 'n'

        fade_default = "a" if last_fades else "n"
        fade_choice = (input(f"Přidat stmívačky na začátek a konec? (a/n) [Enter = {fade_default}]: ").strip().lower()
                       or fade_default)
        fades = fade_choice != 'n'

        settings["last_render_mode"] = mode
        settings["last_render_res"] = res
        settings["last_render_beat_sync"] = beat_sync
        settings["last_render_fades"] = fades
        self.save_settings(settings)

        return {"mode": mode, "hd_mode": res, "beat_sync": beat_sync, "fades": fades}

    def run_render_flow(self, no_rap: bool = False):
        """Sdílený průběh voleb 10 (Render videa) a R (Render bez rapu): zeptá se
        na parametry (s předvyplněnými posledně použitými hodnotami), zvaliduje
        projekt a spustí render; pokud draft validace selže, nabídne vynucené
        pokračování (final vždy vyžaduje opravu chyb)."""
        label = " BEZ RAPU" if no_rap else ""
        print(f"\n⚙️  NASTAVENÍ RENDERU{label}:")
        settings = self.load_settings()
        opts = self._prompt_render_settings(settings)

        print(f"\n🚀 Spouštím renderování{label} (režim: {opts['mode']}, rozlišení: {opts['hd_mode']})...")
        if self.validate_project(final=(opts["mode"] == "final"), no_rap=no_rap):
            self.render_video(mode=opts["mode"], hd_mode=opts["hd_mode"],
                               use_fades=opts["fades"], use_beat_sync=opts["beat_sync"])
        elif opts["mode"] == "draft":
            force_choice = input("\n⚠️  Validace našla problémy (viz výše). "
                                  "Pokračovat v DRAFT renderu i přesto? (a/N): ").strip().lower()
            if force_choice == 'a':
                print("⚠️  Pokračuji přes nalezené problémy (jen draft — pro final je nutné je opravit).")
                self.render_video(mode=opts["mode"], hd_mode=opts["hd_mode"],
                                   use_fades=opts["fades"], use_beat_sync=opts["beat_sync"])
        else:
            print("❌ Validace pro FINAL render selhala — oprav chyby výše a zkus to znovu.")

    def run_all_flow(self, no_rap: bool = False):
        """Sdílený průběh voleb 11 (Kompletní pipeline) a C (Kompletní pipeline
        bez rapu): zeptá se na režim a u draftu nabídne volbu 'force' (stejnou,
        jakou umí CLI přepínač --force), a spustí run_all()."""
        label = " BEZ RAPU" if no_rap else ""
        print(f"\n🚀 Spouštím kompletní pipeline{label}...")
        m_choice = input("Režim (1 - draft, 2 - final): ").strip()
        mode = "final" if m_choice == "2" else "draft"
        print(f"   Použit režim: {mode}.")
        force = False
        if mode == "draft":
            force_choice = input("   Pokračovat i při selhání validace? (a/N): ").strip().lower()
            force = force_choice == 'a'
        self.run_all(mode=mode, no_rap=no_rap, force=force)

    def _llm_choose_best_candidate(
        self,
        task_description: str,
        candidates: list[dict],
        extra_rules: str = "",
        settings: dict = None,
    ) -> dict | None:
        """Obecná pomocná funkce: pošle Ollamě 3-5 kandidátů (např. lyric okna, pozice
        v songu, nebo výskyty klipu v songu) a nechá ji vybrat nejlepší + volitelně
        opravit drobné chyby transkripce.

        `candidates` je seznam dictů s klíči libovolnými, ale musí obsahovat "index"
        (0-based pozice v seznamu, kterou model má vrátit) a nějaký "text" popisující
        kandidáta pro model.

        Vrací dict {"index": int, "corrected_text": str|None, "is_valid_match": bool,
        "confidence": float, "reason": str} nebo None, pokud AI není dostupná nebo
        vrátila neplatnou odpověď — volající MUSÍ v takovém případě použít stávající
        heuristiku beze změny."""
        if not candidates:
            return None
        if not self._ollama_ready(settings):
            return None

        settings = settings or self.load_settings()
        model = self._ollama_model(settings)

        candidate_lines = []
        for c in candidates:
            candidate_lines.append(
                f'{c["index"]}: "{truncate_for_prompt(c.get("text", ""), 160)}" '
                f'(heuristické skóre: {c.get("score", 0):.2f})'
            )
        candidates_block = "\n".join(candidate_lines)

        prompt = f"""Jsi asistent pro post-processing přepisu českého rapu / textu písně.
Úkol: {task_description}

Kandidáti (vyber index toho nejlepšího):
{candidates_block}

{extra_rules}

Odpověz VÝHRADNĚ jedním JSON objektem v tomto přesném tvaru, bez jakéhokoli
dalšího textu, komentáře nebo markdown:
{{"index": <int, index vybraného kandidáta>, "corrected_text": "<oprava textu pokud je potřeba, jinak stejný text>", "is_valid_match": <true/false, zda shoda vůbec dává smysl>, "confidence": <float 0.0-1.0>, "reason": "<krátké zdůvodnění, max 1 věta>"}}"""

        raw = ollama_generate(prompt, model=model, format="json", temperature=0.2, timeout=30)
        parsed = extract_json_from_text(raw)
        if not parsed:
            return None

        try:
            idx = int(parsed.get("index"))
        except (TypeError, ValueError):
            return None
        valid_indices = {c["index"] for c in candidates}
        if idx not in valid_indices:
            return None

        return {
            "index": idx,
            "corrected_text": parsed.get("corrected_text") or None,
            "is_valid_match": bool(parsed.get("is_valid_match", True)),
            "confidence": clamp_confidence(parsed.get("confidence"), default=0.5),
            "reason": truncate_for_prompt(str(parsed.get("reason", "")), 200),
        }

    def _llm_choose_broll_clip(
        self,
        section_name: str,
        t_start: float,
        t_end: float,
        lyrics_context: str,
        available_clips: dict,
        settings: dict = None,
    ) -> dict | None:
        """Nechá Ollamu vybrat nejvhodnější B-roll klip pro daný časový úsek/sekci.

        `available_clips` je dict {clip_id: description}. Vrací
        {"chosen_clip": str, "reason": str, "confidence": float} nebo None při selhání
        (nedostupná AI, neplatná odpověď, nebo neexistující clip_id) — volající musí
        použít stávající fallback (sémantický matcher / cyklické řazení)."""
        if not available_clips:
            return None
        if not self._ollama_ready(settings):
            return None
        settings = settings or self.load_settings()
        model = self._ollama_model(settings)

        section_lower = (section_name or "").lower()
        style_rule = "Neutrální / obecná nálada."
        if "intro" in section_lower:
            style_rule = "INTRO — preferuj atmosférické, ustavující záběry."
        elif "outro" in section_lower:
            style_rule = "OUTRO — preferuj klidné, konkluzivní záběry."
        elif "chorus" in section_lower or "refr" in section_lower or "drop" in section_lower:
            style_rule = "CHORUS/DROP — preferuj dynamické, energické záběry."
        elif "transition" in section_lower or "prechod" in section_lower:
            style_rule = "TRANSITION — krátký, jednoduchý přechodový záběr."

        clip_list = "\n".join(f'- {cid}: {truncate_for_prompt(desc, 140)}' for cid, desc in available_clips.items())

        prompt = f"""Jsi profesionální střihač hudebních videoklipů (český rap).
Vyber nejvhodnější B-roll klip pro tento moment videa.

Sekce písně: {section_name} (čas {t_start:.1f}s - {t_end:.1f}s)
Styl sekce: {style_rule}

Text písně v tomto úseku (čeština): "{truncate_for_prompt(lyrics_context, 250)}"

Dostupné klipy:
{clip_list}

Pravidla:
- Vyber klip, jehož vizuál nejlépe odpovídá náladě a významu textu.
- Dodržuj styl sekce uvedený výše.
- Preferuj vizuální rozmanitost (nevybírej pořád stejné téma).

Odpověz VÝHRADNĚ jedním JSON objektem, bez dalšího textu:
{{"chosen_clip": "<id klipu ze seznamu výše>", "reason": "<krátké zdůvodnění>", "confidence": <float 0.0-1.0>}}"""

        raw = ollama_generate(prompt, model=model, format="json", temperature=0.2, timeout=30)
        parsed = extract_json_from_text(raw)
        if not parsed:
            return None

        chosen = clean_asset_id(str(parsed.get("chosen_clip", "")))
        if chosen not in available_clips:
            # Model si mohl vymyslet ID nebo přidat text navíc — zkusíme najít shodu uvnitř.
            match = re.search(r'\b(?:vid|rap|char)_\d+\b', chosen)
            if match and match.group(0) in available_clips:
                chosen = match.group(0)
            else:
                return None

        return {
            "chosen_clip": chosen,
            "reason": truncate_for_prompt(str(parsed.get("reason", "")), 200),
            "confidence": clamp_confidence(parsed.get("confidence"), default=0.5),
        }

    def find_audio(self) -> Path:
        """Najde audio soubor v INPUT složce (deterministicky první dle názvu)."""
        if not self.input_dir.exists():
            return None
        found_all = []
        for ext in ("*.mp3", "*.wav", "*.ogg", "*.m4a"):
            found_all.extend(self.input_dir.glob(ext))
        if not found_all:
            return None
        found_all = sorted(set(found_all), key=lambda p: p.name.lower())
        if len(found_all) > 1:
            names = ", ".join(p.name for p in found_all)
            print(f"⚠️ V INPUT/ je více audio souborů ({names}). Použiji: {found_all[0].name}")
        return found_all[0]

    def init_project(self):
        """Vytvoří adresářovou strukturu projektu. Bezpečné spustit opakovaně —
        existující soubory/složky se nepřepisují (mkdir/touch s exist_ok=True)."""
        folders = [self.input_dir, self.prompts_dir, self.edit_dir, self.gen_vid, self.gen_rap, self.gen_char, self.gen_pic, self.export_dir]
        already_existed = self.input_dir.exists() and self.prompts_dir.exists()
        for f in folders:
            f.mkdir(parents=True, exist_ok=True)

        (self.prompts_dir / "full_plan.txt").touch(exist_ok=True)
        (self.input_dir / "lyrics.txt").touch(exist_ok=True)
        if already_existed:
            print(f"✅ Projekt '{self.project_dir.name}' už byl inicializovaný — struktura zkontrolována, "
                  f"nic nebylo přepsáno:\n   {self.project_dir}")
        else:
            print(f"✅ Projekt '{self.project_dir.name}' inicializován na:\n   {self.project_dir}")

    def parse_plan(self):
        """Rozparsuje full_plan.txt a rozdistribuuje data."""
        self.logger.info("Začíná parsování plánu: %s", self.full_plan)
        if not self.full_plan.exists() or self.full_plan.stat().st_size == 0:
            self.logger.error("Plán chybí nebo je prázdný: %s", self.full_plan)
            print(f"❌ Soubor {self.full_plan} chybí nebo je prázdný. Nejprve do něj vložte kreativní plán songu.")
            return

        text = self.full_plan.read_text(encoding="utf-8")
        sections = extract_sections(text)

        # Uložení rozparsovaných promptů
        (self.prompts_dir / "video_prompts.txt").write_text(sections.get("VERIFIED_VIDEO_ASSETS", sections.get("VIDEO_PROMPTS", "")), encoding="utf-8")
        rapper_prompts_content = sections.get("CHARACTER_PROMPTS", sections.get("VERIFIED_RAPPER_ASSETS", sections.get("RAPPER_PROMPTS", "")))
        (self.prompts_dir / "rapper_prompts.txt").write_text(rapper_prompts_content, encoding="utf-8")
        if "CHARACTER_PROMPTS" in sections:
            (self.prompts_dir / "character_prompts.txt").write_text(sections["CHARACTER_PROMPTS"], encoding="utf-8")
        (self.prompts_dir / "pic_prompts.txt").write_text(sections.get("VERIFIED_IMAGE_ASSETS", sections.get("IMAGE_PROMPTS", "")), encoding="utf-8")

        # Analýza a Styl
        (self.prompts_dir / "analysis.txt").write_text(sections.get("ANALYZA", sections.get("ANALÝZA", "")), encoding="utf-8")
        (self.prompts_dir / "theme.txt").write_text(sections.get("PROJECT_THEME", sections.get("SONG_THEME", "")), encoding="utf-8")
        (self.prompts_dir / "video_style.txt").write_text(sections.get("VIDEO_STYLE", ""), encoding="utf-8")
        (self.prompts_dir / "rapper_style.txt").write_text(sections.get("RAP_CHARACTER_STYLE", ""), encoding="utf-8")

        # Editace a Střih
        shot_order_text = normalize_timeline_text(sections.get("SHOT_ORDER", ""))
        timeline_text = normalize_timeline_text(
            sections.get("CURRENT_MUSIC_VIDEO_STRUCTURE", sections.get("MUSIC_VIDEO_TIMELINE", ""))
        )
        if not timeline_text:
            timeline_text = shot_order_text
        if not shot_order_text and timeline_text:
            shot_order_text = timeline_text

        timeline_entries, timeline_warnings = parse_timeline_entries(timeline_text)
        audio_path = self.find_audio()
        song_duration = probe_duration(audio_path) if audio_path else None
        # Média mohou být v době parsování teprve plánovaná; existenci ID
        # proto ověřuje až pozdější validační/render fáze.
        timeline_errors = validate_timeline(timeline_entries, song_duration=song_duration)
        for warning in timeline_warnings:
            print(f"⚠️ Timeline: {warning}")
        for error in timeline_errors:
            print(f"❌ Timeline: {error}")
        if timeline_errors:
            self.logger.error("Validace timeline selhala: %s", "; ".join(timeline_errors))
            print("❌ Plán nebyl zapsán jako validní timeline. Oprav full_plan.txt a spusť parse znovu.")
            return False

        (self.timeline_file).write_text(timeline_text, encoding="utf-8")
        (self.edit_dir / "shot_order.txt").write_text(shot_order_text, encoding="utf-8")
        (self.edit_dir / "effects.txt").write_text(sections.get("EFFECTS", ""), encoding="utf-8")
        (self.edit_dir / "color_grading.txt").write_text(sections.get("COLOR_GRADING", ""), encoding="utf-8")
        (self.edit_dir / "metadata.txt").write_text(sections.get("METADATA", ""), encoding="utf-8")

        if "RAPPER_SEGMENT_ALIGNMENT" in sections:
            (self.edit_dir / "rapper_segment_alignment.txt").write_text(sections["RAPPER_SEGMENT_ALIGNMENT"], encoding="utf-8")

        def _ids(folder, prefix, suffix):
            if not folder.exists():
                return []
            return sorted(p.stem for p in folder.glob(f"{prefix}_*{suffix}") if p.stat().st_size > 500)

        metadata = {
            "song_length": round(probe_duration(self.find_audio()), 3) if self.find_audio() else 0.0,
            "rap_clips": _ids(self.gen_rap, "rap", ".mp4"),
            "char_clips": _ids(self.gen_char, "char", ".mp4"),
            "broll_clips": _ids(self.gen_vid, "vid", ".mp4"),
            "images": sorted(p.stem for p in self.gen_pic.glob("pic_*.*")) if self.gen_pic.exists() else [],
            "shot_order": [],
            "original_timestamps": {},
        }
        if self.timeline_file.exists():
            for raw in self.timeline_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "|" not in raw or "-" not in raw:
                    continue
                parts = [p.strip() for p in raw.split("|")]
                if len(parts) < 2:
                    continue
                clip = parts[1]
                metadata["shot_order"].append(clip)
                if clip.startswith("rap_"):
                    try:
                        start, end = [parse_timecode(x.strip()) for x in parts[0].split("-", 1)]
                        metadata["original_timestamps"][clip] = {"start": round(start, 3), "end": round(end, 3)}
                    except Exception:
                        pass
        self._write_json(self.edit_dir / "metadata.json", metadata)

        self.logger.info("Plán úspěšně rozparsován do Prompts/ a EDIT_PROJECT/")
        print(f"✅ Plán full_plan.txt úspěšně rozparsován do složek Prompts/ a EDIT_PROJECT/")

    def create_placeholders(self):
        """Vytvoří prázdné placeholdery pro chybějící média popsaná v plánu."""
        if not self.full_plan.exists():
            print(f"❌ Chybí plán: {self.full_plan}")
            return

        text = self.full_plan.read_text(encoding="utf-8")
        sections = extract_sections(text)

        prompt_map = {
            "VIDEO_PROMPTS": (self.gen_vid, ".mp4"),
            "VERIFIED_VIDEO_ASSETS": (self.gen_vid, ".mp4"),
            "CHARACTER_PROMPTS": (self.gen_char, ".mp4"),
            "RAPPER_PROMPTS": (self.gen_rap, ".mp4"),
            "VERIFIED_RAPPER_ASSETS": (self.gen_rap, ".mp4"),
            "IMAGE_PROMPTS": (self.gen_pic, ".png"),
            "VERIFIED_IMAGE_ASSETS": (self.gen_pic, ".png"),
        }

        total = 0
        for sec, (folder, ext) in prompt_map.items():
            if sec not in sections:
                continue
            lines = [l.strip() for l in sections[sec].splitlines() if l.strip()]
            for line in lines:
                # Přeskočit oddělovací/formátovací řádky (např. "---", "|---|---|", markdown fence)
                if re.fullmatch(r"[-|:\s`*]+", line):
                    continue
                delimiter = '|' if '|' in line else ':'
                parts = line.split(delimiter, 1)
                asset_id = clean_asset_id(parts[0])
                if not asset_id or len(asset_id) > 20 or " " in asset_id:
                    continue

                # Pro gen_pic (IMAGE_PROMPTS) zkontrolujeme více přípon
                exists = False
                if folder == self.gen_pic:
                    for test_ext in (".mp4", ".png", ".jpg", ".jpeg"):
                        if (folder / f"{asset_id}{test_ext}").exists():
                            exists = True
                            break
                else:
                    if (folder / f"{asset_id}{ext}").exists():
                        exists = True

                if not exists:
                    file_path = folder / f"{asset_id}{ext}"
                    file_path.touch(exist_ok=True)
                    print(f"  + Placeholder: {file_path.relative_to(self.project_dir)}")
                    total += 1
        print(f"✅ Vytvořeno {total} nových placeholderů.")

    def analyze_audio(self):
        """Spustí analýzu beatu pomocí Librosa a volitelně Whisper transkripci."""
        audio_path = self.find_audio()
        if not audio_path:
            print("❌ V INPUT složce nebylo nalezeno žádné audio (MP3/WAV/M4A).")
            return

        print(f"🎵 Analyzuji audio: {audio_path.name}")

        # 1. Librosa Beat Detection
        if HAS_LIBROSA:
            print("🥁 Zjišťuji tempo a beaty pomocí Librosa...")
            y, sr = librosa.load(str(audio_path))
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Bezpečné získání BPM (Librosa 0.10+ vrací pole, starší verze float)
            try:
                bpm_val = float(tempo)
            except TypeError:
                bpm_val = float(tempo[0])

            times = librosa.frames_to_time(beats, sr=sr).tolist()
            enriched_beats = enrich_beats(times, bpm=bpm_val)

            beats_file = self.edit_dir / "beats.json"
            with open(beats_file, "w", encoding="utf-8") as f:
                json.dump({
                    "bpm": bpm_val,
                    "beats": [round(t, 3) for t in times],
                    "beat_events": enriched_beats,
                    "schema_version": 2,
                }, f, indent=2)
            print(f"  ✅ BPM: {bpm_val:.1f}. Beaty uloženy do: {beats_file.relative_to(self.project_dir)}")
        else:
            print("⚠️ Librosa není nainstalována (použijte: pip install librosa). Detekce beatů přeskočena.")

        # 2. Transkripce (lokální Whisper, nebo Groq Cloud API)
        trans_file = self.input_dir / "transcription.json"
        if trans_file.exists():
            print(f"✅ Transkripce již existuje v souboru: {trans_file.relative_to(self.project_dir)}")
            return

        settings = self.load_settings()
        provider = str(settings.get("transcription_provider", "local")).lower()

        if provider == "groq":
            self._transcribe_song_with_groq(audio_path, trans_file, settings)
        else:
            self._transcribe_song_with_local_whisper(audio_path, trans_file, settings)

    def _transcribe_song_with_groq(self, audio_path: Path, trans_file: Path, settings: dict):
        """Transkripce songu přes Groq Cloud API (whisper-large-v3 / whisper-large-v3-turbo)."""
        api_key = self._groq_ready()
        if not api_key:
            print(f"   Nebo vlož transkripci ručně do: {trans_file}")
            return

        model = settings.get("groq_model", "whisper-large-v3-turbo")
        print(f"🗣️  Spouštím transkripci songu přes Groq Cloud API (model: {model})...")
        data = self._groq_transcribe_file(audio_path, settings, api_key)
        if not data:
            print(f"   Transkripci můžeš dodat ručně do: {trans_file}")
            return
        self._write_json(trans_file, data)
        print(f"\n  ✅ Transkripce (Groq) dokončena a uložena do: {trans_file.relative_to(self.project_dir)}")

    def _transcribe_song_with_local_whisper(self, audio_path: Path, trans_file: Path, settings: dict):
        """Transkripce songu přes lokálně nainstalovaný openai-whisper CLI."""
        whisper_bin = shutil.which("whisper") or shutil.which("whisper-cli")
        if not whisper_bin:
            print("❌ Nebyl nalezen příkaz `whisper` ani `whisper-cli` v PATH.")
            print(f"   Nainstaluj: pip install openai-whisper")
            print(f"   Nebo přepni v Nastavení na Groq Cloud API, nebo vlož transkripci ručně do: {trans_file}")
            return
        if Path(whisper_bin).name == "whisper-cli":
            print("❌ Nalezen jen `whisper-cli` (whisper.cpp) — nepoužívá stejnou syntaxi CLI jako")
            print("   openai-whisper (--model/--word_timestamps/--output_format), transkripce by selhala.")
            print(f"   Nainstaluj: pip install openai-whisper --break-system-packages")
            print(f"   Nebo přepni v Nastavení na Groq Cloud API, nebo vlož transkripci ručně do: {trans_file}")
            return

        model = settings.get("whisper_model", "small")
        print(f"🗣️  Spouštím Whisper transkripci (model: {model}, zařízení: {settings.get('device', 'cpu')})...")
        cmd = [
            whisper_bin, str(audio_path),
            "--model", model,
            *self._whisper_device_args(settings),
            "--language", "cs",
            "--task", "transcribe",
            "--word_timestamps", "True",
            "--output_format", "json",
            "--output_dir", str(self.input_dir),
        ]
        print(f"  Příkaz: {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)

            whisper_gen = self.input_dir / f"{audio_path.stem}.json"
            if whisper_gen.exists() and whisper_gen != trans_file:
                shutil.move(str(whisper_gen), str(trans_file))

            for ext in (".txt", ".srt", ".vtt", ".tsv"):
                extra_file = self.input_dir / f"{audio_path.stem}{ext}"
                if extra_file.exists():
                    extra_file.unlink()

            print(f"\n  ✅ Transkripce dokončena a uložena do: {trans_file.relative_to(self.project_dir)}")
        except subprocess.CalledProcessError as e:
            if e.returncode == -9:
                print(f"\n❌ Whisper byl zabit systémem (SIGKILL / OOM) — došla RAM pro model '{model}'.")
                print(f"   Model '{model}' potřebuje více paměti, než je aktuálně volné.")
                print(f"   Řešení: přepni v Nastavení (volba 13) na menší model — 'small' nebo 'base' — a zkus to znovu.")
            else:
                print(f"\n❌ Whisper skončil s chybou (návratový kód {e.returncode}).")
            print(f"   Transkripci můžeš dodat ručně do: {trans_file}")
        except Exception as e:
            print(f"\n⚠️  Spuštění Whisper selhalo: {e}")
            print(f"   Transkripci můžeš dodat ručně do: {trans_file}")

    def sync_timeline(self):
        """Auto-generuje timeline.txt spárováním Whisper transkripce a rapper klipů."""
        # ── Primární cesta: použij MUSIC_VIDEO_TIMELINE z full_plan.txt (pokud existuje) ──
        if self.full_plan.exists():
            plan_text = self.full_plan.read_text(encoding="utf-8")
            plan_sections = extract_sections(plan_text)
            mvt = plan_sections.get("MUSIC_VIDEO_TIMELINE", "").strip()
            if mvt:
                # Parsujeme pouze platné řádky timeline (přeskočíme dekorativní hlavičky)
                timeline_lines = []
                for line in mvt.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("---") or stripped.startswith("==="):
                        continue
                    if "|" not in stripped:
                        continue
                    parts = [x.strip() for x in stripped.split("|")]
                    if len(parts) < 2:
                        continue
                    # Ověříme, že první část vypadá jako časový rozsah (NNN.NN - NNN.NN)
                    time_part = parts[0].strip("[] \t")
                    if "-" not in time_part:
                        continue
                    try:
                        t_start_raw, t_end_raw = [x.strip().strip("[] ") for x in time_part.split("-", 1)]
                        # parse_timecode zvládá jak "65.54", tak "01:05.54" — narozdíl od
                        # holého float(), který na formátu MM:SS.ms selhává.
                        parse_timecode(t_start_raw)
                        parse_timecode(t_end_raw)
                    except (ValueError, IndexError):
                        continue
                    # Platný řádek — přidáme
                    timeline_lines.append(stripped)

                if timeline_lines:
                    self.edit_dir.mkdir(parents=True, exist_ok=True)
                    self.timeline_file.write_text("\n".join(timeline_lines), encoding="utf-8")
                    print(f"✅ Timeline načtena ze sekce MUSIC_VIDEO_TIMELINE v full_plan.txt ({len(timeline_lines)} segmentů)")
                    print(f"   → {self.timeline_file.relative_to(self.project_dir)}")
                    return
                else:
                    print("⚠️  Sekce MUSIC_VIDEO_TIMELINE v full_plan.txt je prázdná nebo neobsahuje platné řádky.")
                    print("    Pokračuji automatickým generováním...")
        # ── Konec primární cesty ──────────────────────────────────────────────

        trans_file = self.input_dir / "transcription.json"
        align_file = self.edit_dir / "rapper_segment_alignment.txt"

        if not trans_file.exists():
            print(f"❌ Soubor s transkripcí {trans_file} neexistuje. Nejprve spusťte 'analyze-song' nebo vytvořte transkripci.")
            return

        audio_path = self.find_audio()
        audio_duration = probe_duration(audio_path) if audio_path else 180.0

        with open(trans_file, "r", encoding="utf-8") as f:
            trans_data = json.load(f)

        segments = trans_data.get("segments", [])
        if not segments:
            print("❌ V transkripci nebyly nalezeny žádné segmenty.")
            return

        # ── Oprava textu podle lyrics.txt (song_alignment.json z volby 4 / analyze-song) ──
        song_alignment_words = []
        song_alignment_path = self.edit_dir / "song_alignment.json"
        if song_alignment_path.exists():
            song_alignment_words = self._load_json(song_alignment_path, {}).get("words", [])
        if song_alignment_words:
            print(f"✅ Používám opravený text z lyrics.txt ({song_alignment_path.name}, {len(song_alignment_words)} slov)")
        else:
            print("⚠️  song_alignment.json nenalezen/prázdný — použije se surový text z Whisperu.")
            print("    Pro opravu textu podle lyrics.txt spusť nejprve volbu 4 (analyze-song).")

        rap_time_alignments = []
        rap_map = {}

        def _parse_alignment_time(t_str):
            parts = t_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])

        if align_file.exists():
            print(f"ℹ️  Načítám časové zarovnání rappera z: {align_file.name}")
            for line in align_file.read_text(encoding="utf-8").splitlines():
                if not line.strip() or "|" not in line:
                    continue
                parts = [x.strip() for x in line.split("|", 1)]
                clip_or_keyword, time_or_clip = parts[0], parts[1]
                if "-" in time_or_clip and any(c.isdigit() for c in time_or_clip):
                    try:
                        start_str, end_str = time_or_clip.split("-", 1)
                        t_start = _parse_alignment_time(start_str)
                        t_end = _parse_alignment_time(end_str)
                        rap_time_alignments.append((t_start, t_end, clip_or_keyword))
                    except Exception as e:
                        print(f"⚠️  Chyba při parsování řádku alignmentu '{line}': {e}")
                else:
                    rap_map[clip_or_keyword.lower()] = time_or_clip
        else:
            print("⚠️ Chybí rapper_segment_alignment.txt — rap párování přes klíčová slova nebude dostupné.")

        brolls = sorted([f.stem for f in self.gen_vid.glob("*.mp4") if f.stem.startswith("vid_")])
        if not brolls:
            brolls = ["vid_01", "vid_02", "vid_03"]

        # ── Sémantické párování B-roll klipů ──
        broll_descriptions = {}
        descriptions_file = self.edit_dir / "broll_descriptions.json"
        use_semantic = False
        _choose_best_broll = None
        if descriptions_file.exists():
            try:
                broll_descriptions = self._load_json(descriptions_file, {})
                from semantic_broll_match import choose_best_broll as _cbf
                _choose_best_broll = _cbf
                use_semantic = bool(broll_descriptions)
                if use_semantic:
                    print(f"🧠 Sémantické párování B-roll aktivováno — {len(broll_descriptions)} popisů")
            except ImportError:
                print("⚠️  semantic_broll_match.py nenalezen, použiji cyklické řazení.")
            except Exception as exc:
                print(f"⚠️  Sémantické párování nedostupné ({exc}), použiji cyklické řazení.")
        # ────────────────────────────────────────────────────────────────────────────

        # ── Ollama jako druhá volba, pokud semantic_broll_match.py chybí/selže ──
        settings = self.load_settings()
        use_ollama_broll = (not use_semantic) and self._ollama_ready(settings) and bool(broll_descriptions)
        if use_ollama_broll:
            print(f"🧠 Sémantický modul nedostupný, použiji lokální Ollama pro výběr B-rollu (model: {self._ollama_model(settings)}).")
        # ────────────────────────────────────────────────────────────────────────────

        timeline_entries = []
        broll_idx = 0
        used_semantic = set()
        used_ollama_clips = set()
        total_segs = len(segments)

        for seg_i, seg in enumerate(segments):
            start = round(seg["start"], 3)
            end = round(seg["end"], 3)
            raw_text = seg["text"].strip()

            if song_alignment_words:
                seg_words = [
                    w["word"] for w in song_alignment_words
                    if start - 0.05 <= (float(w["start"]) + float(w["end"])) / 2.0 <= end + 0.05
                ]
                text = " ".join(seg_words).strip() or raw_text
            else:
                text = raw_text

            matched_clip = None

            # 1. Zkusíme časové zarovnání (pokud existuje)
            seg_mid = (start + end) / 2.0
            for t_start, t_end, clip_name in rap_time_alignments:
                if t_start - 0.05 <= seg_mid <= t_end + 0.05:
                    matched_clip = clip_name
                    break

            # 2. Pokud nemáme časové zarovnání, zkusíme klíčová slova
            if not matched_clip:
                for key, clip in rap_map.items():
                    if key in text.lower():
                        matched_clip = clip
                        break

            if not matched_clip:
                if use_semantic and text and broll_descriptions and _choose_best_broll:
                    try:
                        print(f"  🧠 [{seg_i+1}/{total_segs}] Páruji: \"{text[:50]}\"", flush=True)
                        matched_clip = _choose_best_broll(text, broll_descriptions, used_semantic)
                        used_semantic.add(matched_clip)
                    except Exception as e:
                        matched_clip = brolls[broll_idx % len(brolls)]
                        broll_idx += 1
                elif use_ollama_broll and text:
                    available = {k: v for k, v in broll_descriptions.items() if k not in used_ollama_clips}
                    if not available or len(available) < 2:
                        used_ollama_clips.clear()
                        available = dict(broll_descriptions)
                    result = self._llm_choose_broll_clip(
                        section_name="",
                        t_start=start,
                        t_end=end,
                        lyrics_context=text,
                        available_clips=available,
                        settings=settings,
                    )
                    if result:
                        matched_clip = result["chosen_clip"]
                        used_ollama_clips.add(matched_clip)
                        print(f"  🧠 [{seg_i+1}/{total_segs}] \"{text[:50]}\" → {matched_clip} "
                              f"(confidence {result['confidence']:.2f})")
                    else:
                        matched_clip = brolls[broll_idx % len(brolls)]
                        broll_idx += 1
                else:
                    matched_clip = brolls[broll_idx % len(brolls)]
                    broll_idx += 1

            timeline_entries.append(f"{start:.2f} - {end:.2f} | {matched_clip} | {text}")

        self.timeline_file.write_text("\n".join(timeline_entries), encoding="utf-8")
        print(f"✅ Timeline vygenerována a zapsána do: {self.timeline_file.relative_to(self.project_dir)}")

    def _load_song_segments(self) -> list[dict]:
        """Načte segmenty song transkripce z INPUT/transcription.json."""
        trans_file = self.input_dir / "transcription.json"
        if not trans_file.exists():
            return []
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = data.get("segments", [])
            return [s for s in segments if str(s.get("text", "")).strip()]
        except Exception as exc:
            print(f"⚠️ Nepodařilo se načíst song transkripci: {exc}")
            return []

    def _song_text_for_range(self, song_segments: list[dict], start: float, end: float) -> str:
        """Vrátí text písně pro daný časový rozsah."""
        texts = []
        for seg in song_segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            if seg_end > start and seg_start < end:
                text = str(seg.get("text", "")).strip()
                if text:
                    texts.append(text)
        return " ".join(texts).strip()

    def _load_lyrics_reference(self) -> list[str]:
        """Načte čisté lyric řádky jako referenci pro opravu transkripce."""
        lyrics_path = self.input_dir / "lyrics.txt"
        if not lyrics_path.exists():
            return []

        lines = []
        for raw in lyrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            lines.append(line)
        return lines

    def _load_lyrics_text(self) -> str:
        """Načte lyrics.txt jako jeden čistý text."""
        return "\n".join(self._load_lyrics_reference()).strip()

    def _load_text_file(self, path: Path) -> str:
        """Bezpečně načte libovolný textový soubor; vrátí '' pokud neexistuje."""
        if not path or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _find_character_description(self) -> str:
        """Načte textový popis hlavní postavy z INPUT/postava.txt (preferováno) nebo
        INPUT/character.txt. Používá se místo obrázku (viz rozhodnutí kvůli HW bez GPU
        a jen 8GB RAM — vision model by byl na tomto stroji rizikový)."""
        for name in ("postava.txt", "character.txt"):
            text = self._load_text_file(self.input_dir / name)
            if text:
                return text
        return ""

    def _find_mood_description(self) -> str:
        """Načte volitelný popis nálady/žánru z INPUT/mood.txt."""
        return self._load_text_file(self.input_dir / "mood.txt")

    def _parse_mmss_to_seconds(self, value: str) -> float:
        """Převede 'MM:SS' (formát délek v klipy.md) na sekundy."""
        value = (value or "").strip()
        try:
            return parse_timecode(value)
        except Exception:
            return 0.0

    def _load_klipy_md(self) -> dict:
        """Naparsuje INPUT/klipy.md — seznam už existujících/vygenerovaných klipů.

        Očekávaný formát (skupiny CHAR/PIC/RAP/VID oddělené '### Skupina X' a '---',
        každý klip jako '* **Název:** ...', '* **Délka:** MM:SS', '* **Obsah:** ...',
        '* **Text:** ...'). Parser je tolerantní: chybějící pole nezpůsobí pád, jen se
        u daného klipu vynechají. Vrací dict {clip_id: {"group", "duration_sec", "obsah", "text"}}.
        """
        klipy_path = self.input_dir / "klipy.md"
        if not klipy_path.exists():
            return {}
        text = self._load_text_file(klipy_path)
        if not text:
            return {}

        clips = {}
        current_group = ""
        current = {}

        def _flush():
            nonlocal current
            name = current.get("name")
            if name:
                clips[name] = {
                    "group": current_group,
                    "duration_sec": self._parse_mmss_to_seconds(current.get("duration", "0:00")),
                    "obsah": current.get("obsah", "").strip(),
                    "text": current.get("text", "").strip(),
                }
            current = {}

        group_re = re.compile(r"^#{1,4}\s*Skupina\s+(\w+)", re.IGNORECASE)
        field_re = re.compile(r"^\*?\s*\*\*(Název|Délka|Obsah|Text):?\*\*:?\s*(.*)$", re.IGNORECASE)

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            gmatch = group_re.match(line)
            if gmatch:
                _flush()
                current_group = gmatch.group(1).strip().upper()
                continue
            if line.startswith("---"):
                _flush()
                continue
            fmatch = field_re.match(line)
            if fmatch:
                field = fmatch.group(1).strip().lower()
                value = fmatch.group(2).strip()
                if field == "název":
                    _flush()
                    current["name"] = clean_asset_id(value)
                elif field == "délka":
                    current["duration"] = value
                elif field == "obsah":
                    current["obsah"] = value
                elif field == "text":
                    current["text"] = value
        _flush()
        return clips

    def _format_existing_clips_for_prompt(self, klipy: dict) -> str:
        """Sestaví čitelný textový blok existujících klipů pro vložení do Prompt 2,
        seskupený podle typu, s reálným rapovaným textem u rap_XX klipů (klíčové pro
        přesné napasování na transkripci)."""
        if not klipy:
            return "(žádné existující klipy nenalezeny — všechny budou muset být nově vygenerované)"

        groups = {}
        for cid, data in klipy.items():
            groups.setdefault(data.get("group", "OSTATNÍ"), []).append((cid, data))

        lines = []
        for group_name in ("RAP", "VID", "PIC", "CHAR"):
            items = groups.get(group_name)
            if not items:
                continue
            items.sort(key=lambda x: x[0])
            lines.append(f"-- Skupina {group_name} --")
            for cid, data in items:
                dur = data.get("duration_sec", 0.0)
                obsah = truncate_for_prompt(data.get("obsah", ""), 200)
                if group_name == "RAP" and data.get("text") and "neobsahuje" not in data.get("text", "").lower():
                    real_text = truncate_for_prompt(data.get("text", ""), 200)
                    lines.append(f"{cid} | délka {dur:.0f}s | obsah: {obsah} | ODRAPOVANÝ TEXT: \"{real_text}\"")
                else:
                    lines.append(f"{cid} | délka {dur:.0f}s | obsah: {obsah}")
            lines.append("")
        # Zbylé skupiny mimo standardní čtveřici
        for group_name, items in groups.items():
            if group_name in ("RAP", "VID", "PIC", "CHAR"):
                continue
            for cid, data in sorted(items):
                lines.append(f"{cid} | délka {data.get('duration_sec', 0.0):.0f}s | obsah: {truncate_for_prompt(data.get('obsah', ''), 200)}")

        return "\n".join(lines).strip()

    def _summarize_new_clips_for_timeline(self, sections1: dict) -> str:
        """Sestaví kompaktní shrnutí klipů nově naplánovaných v Části 1/2 (ID + délka +
        zkrácený obsah) pro Část 2/2 (časovou osu) — tak nemusíme do druhého volání posílat
        znovu celé anglické prompty a šetříme tím místo v kontextu pro transkripci a
        existující klipy."""
        lines = []
        # Formát řádků z Části 1/2: "id|[Duration: X.Ys] prompt..." nebo
        # "pic_XX | Zdrojové médium: ... | [Duration: X.Ys] prompt...".
        id_pattern = re.compile(r'^([a-zA-Z]+_\d+)\s*\|\s*(?:[^|]*\|\s*)?\[Duration:\s*([\d.]+)s\]\s*(.*)$')
        fallback_pattern = re.compile(r'^([a-zA-Z]+_\d+)\s*\|\s*(.*)$')
        for section_name in ("VIDEO_PROMPTS", "RAPPER_PROMPTS", "IMAGE_PROMPTS"):
            block = sections1.get(section_name, "")
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line or "|" not in line:
                    continue
                m = id_pattern.match(line)
                if m:
                    cid, dur, prompt = m.groups()
                    try:
                        dur_val = float(dur)
                    except ValueError:
                        dur_val = 0.0
                    lines.append(f"{cid} | délka {dur_val:.0f}s | {truncate_for_prompt(prompt, 150)}")
                    continue
                fm = fallback_pattern.match(line)
                if fm:
                    cid, rest = fm.groups()
                    lines.append(f"{cid} | {truncate_for_prompt(rest, 150)}")
        if not lines:
            return "(část 1/2 nehlásí žádné nově naplánované klipy — do timeline patří jen existující klipy ze seznamu výše)"
        return "\n".join(lines)

    def _split_transcription_into_chunks(self, transcription_lines: list, transcription_spans: list,
                                          song_duration: float, max_chars_per_chunk: int,
                                          max_chunk_seconds: float = None):
        """Rozdělí řádky transkripce (a jejich časové rozsahy) na po sobě jdoucí ÚSEKY tak, aby
        žádný úsek nepřekročil `max_chars_per_chunk` znaků ANI (pokud je zadáno) `max_chunk_seconds`
        vteřin časového rozpětí písně. Na rozdíl od katalogu existujících klipů je transkripce
        přirozeně sekvenční, takže dělení podle času nezpůsobuje ztrátu kontextu potřebného pro
        Pravidlo č. 1 (to zajišťuje volající tím, že do KAŽDÉHO úseku posílá CELÝ existing_clips_block).

        `max_chunk_seconds` je záměrně NEZÁVISLÝ na `max_chars_per_chunk`/num_ctx: vyšší num_ctx
        umí nacpat do jednoho volání znakově klidně celou píseň (0 švů mezi úseky), ale slabší
        lokální model pak snáz "ztratí nit" při přesném sčítání času přes několik minut najednou
        (reálně pozorováno: řádek uprostřed/na začátku dlouhého úseku dostal koncový čas =
        konec CELÉ písně místo pár vteřin, protože si spletl "konec úseku" s "konec tohoto řádku").
        Tenhle strop proto úsek rozseká i tehdy, když by se znakově vešel celý najednou.

        Vrací (chunks_text, chunk_bounds):
          chunks_text  — seznam textových bloků (spojené řádky transkripce daného úseku)
          chunk_bounds — seznam dvojic (segment_start, segment_end) v sekundách, SPOJITÝCH
                         (konec úseku i = začátek úseku i+1), první začíná na 0.0 a poslední
                         končí přesně na song_duration.
        """
        if not transcription_lines:
            return ["(transkripce nedostupná)"], [(0.0, song_duration)]

        groups = []  # [(start_idx, end_idx), ...] inkluzivní indexy do transcription_lines
        current_start_idx = 0
        current_len = 0
        for idx, line in enumerate(transcription_lines):
            line_len = len(line) + 1  # + newline
            starts_new_group = idx > current_start_idx and current_len + line_len > max_chars_per_chunk
            if not starts_new_group and max_chunk_seconds and idx > current_start_idx:
                # Časový strop: kdyby tenhle řádek do úseku přibyl, jak dlouhé by časové
                # rozpětí úseku bylo? (od začátku prvního řádku skupiny po začátek TOHOTO řádku,
                # jako konzervativní odhad — reálný konec úseku se dopočítá až níž).
                group_span = transcription_spans[idx][0] - transcription_spans[current_start_idx][0]
                if group_span > max_chunk_seconds:
                    starts_new_group = True
            if starts_new_group:
                groups.append((current_start_idx, idx - 1))
                current_start_idx = idx
                current_len = 0
            current_len += line_len
        groups.append((current_start_idx, len(transcription_lines) - 1))

        chunks_text = []
        chunk_bounds = []
        prev_end = 0.0
        for i, (start_idx, end_idx) in enumerate(groups):
            chunks_text.append("\n".join(transcription_lines[start_idx:end_idx + 1]))
            if i == len(groups) - 1:
                seg_end = song_duration
            else:
                # Konec úseku = začátek prvního slova NÁSLEDUJÍCÍHO úseku — hranice tak na
                # sebe navazují bez mezer, i když mezi slovy byla krátká pauza (ticho).
                next_start_idx = groups[i + 1][0]
                seg_end = transcription_spans[next_start_idx][0]
            chunk_bounds.append((prev_end, seg_end))
            prev_end = seg_end
        return chunks_text, chunk_bounds

    def _force_align_timeline_chunk_boundaries(self, timeline_text: str, segment_start: float,
                                                segment_end: float, max_auto_fix: float = 2.0) -> str:
        """Mechanicky sjednotí časovou značku PRVNÍHO a POSLEDNÍHO řádku úseku
        MUSIC_VIDEO_TIMELINE na přesné hranice segmentu (segment_start/segment_end). Model
        dostal instrukci hranice dodržet, ale drobná nepřesnost (řádově desetiny sekundy) by
        se při spojování více úseků za sebou sečetla do slyšitelné mezery/přeskoku — proto se
        opravuje jen ČASOVÝ ÚDAJ, ID klipu a popis zůstávají beze změny. Pokud je odchylka
        větší než `max_auto_fix` sekund, oprava proběhne také (jinak by spojený timeline měl
        díru), ale vypíše se varování, ať se dá zkontrolovat ručně."""
        time_line_re = re.compile(r'^(\[?)(\d{1,2}:\d{2}\.\d+)(\]?)\s*-\s*(\[?)(\d{1,2}:\d{2}\.\d+)(\]?)(\s*\|.*)$')

        def _fmt(sec: float) -> str:
            sec = max(0.0, sec)
            m = int(sec // 60)
            s = sec - m * 60
            return f"{m:02d}:{s:05.2f}"

        def _rebuild(line: str, new_start=None, new_end=None) -> str:
            mm = time_line_re.match(line.strip())
            if not mm:
                return line
            b1, start_s, b1c, b2, end_s, b2c, rest = mm.groups()
            start_txt = _fmt(new_start) if new_start is not None else start_s
            end_txt = _fmt(new_end) if new_end is not None else end_s
            return f"{b1}{start_txt}{b1c}-{b2}{end_txt}{b2c}{rest}"

        lines = timeline_text.splitlines()
        idxs = [i for i, l in enumerate(lines) if time_line_re.match(l.strip())]
        if not idxs:
            return timeline_text

        first_i, last_i = idxs[0], idxs[-1]
        fm = time_line_re.match(lines[first_i].strip())
        try:
            cur_start = parse_timecode(fm.group(2))
        except ValueError:
            # Neplatný časový kód (např. '18:80.30') — nemá smysl počítat odchylku,
            # rovnou přepíšeme na očekávaný začátek úseku.
            lines[first_i] = _rebuild(lines[first_i], new_start=segment_start)
            cur_start = segment_start
        if abs(cur_start - segment_start) > 0.02:
            if abs(cur_start - segment_start) > max_auto_fix:
                print(f"⚠️  Úsek začíná na {cur_start:.2f}s místo očekávaných {segment_start:.2f}s "
                      "(rozdíl > 2s) — opravuji časovou značku, ale zkontroluj návaznost ručně.")
            lines[first_i] = _rebuild(lines[first_i], new_start=segment_start)

        lm = time_line_re.match(lines[last_i].strip())
        try:
            cur_end = parse_timecode(lm.group(5))
        except ValueError:
            lines[last_i] = _rebuild(lines[last_i], new_end=segment_end)
            cur_end = segment_end
        if abs(cur_end - segment_end) > 0.02:
            if abs(cur_end - segment_end) > max_auto_fix:
                print(f"⚠️  Úsek končí na {cur_end:.2f}s místo očekávaných {segment_end:.2f}s "
                      "(rozdíl > 2s) — opravuji časovou značku, ale zkontroluj návaznost ručně.")
            lines[last_i] = _rebuild(lines[last_i], new_end=segment_end)

        return "\n".join(lines)

    def _validate_transcription_vs_duration(self, words: list, song_duration: float, tolerance: float = 1.5) -> list[str]:
        """Zkontroluje, jestli časy slov v transcription.json nepřesahují skutečnou délku
        audia (song_duration ze zkusídla souboru). Bez tohohle se do Části 2/2 posílal
        text transkripce s časy až za koncem MP3 (viz konkrétní případ: 17 slov v Outru
        s 'start' 234–246s u 233,66s dlouhé písně), zatímco poslední úsek timeline se
        mechanicky nutil končit přesně na song_duration — model dostal dvě vzájemně si
        odporující informace o tom, kde píseň končí, a to je přesně situace, kdy si
        v odpovědi 'vymyslel' vlastní časování (viz minulý rozbitý MUSIC_VIDEO_TIMELINE)."""
        if not words or song_duration <= 0:
            return []
        overruns = [w for w in words if w["start"] > song_duration + tolerance]
        if not overruns:
            return []
        first = overruns[0]
        return [
            f"transcription.json obsahuje {len(overruns)} slov s časem za koncem skutečné "
            f"délky audia ({song_duration:.2f}s) — první z nich: '{first['word']}' na {first['start']:.2f}s. "
            "Tahle slova se do Části 2/2 nepošlou (viz _clamp_words_to_duration), ale zkontroluj "
            "prosím transcription.json ručně (např. znovu spustit volbu 4 — Analyzovat song) — "
            "je možné, že Whisper/Groq transkripce obsahuje navíc kus outra, který v mixu už není."
        ]

    def _clamp_words_to_duration(self, words: list, song_duration: float) -> tuple[list, int]:
        """Ořízne slova z transkripce na skutečnou délku audia: slova začínající až po
        konci audia se zahodí (nemůžou mít v timeline reálné místo), slova přesahující
        přes konec se zkrátí. Vrací (ořezaná_slova, počet_zahozených_slov)."""
        if song_duration <= 0:
            return words, 0
        clamped = []
        dropped = 0
        for w in words:
            if w["start"] >= song_duration:
                dropped += 1
                continue
            w2 = dict(w)
            if w2["end"] > song_duration:
                w2["end"] = song_duration
            clamped.append(w2)
        return clamped, dropped

    def _highest_clip_index(self, klipy: dict, prefix: str) -> int:
        """Vrátí nejvyšší číselný index existujícího ID daného prefixu (vid_/rap_/pic_),
        aby nově generovaná ID nekolidovala s existujícími."""
        highest = 0
        for cid in klipy.keys():
            m = re.match(rf"^{re.escape(prefix)}(\d+)$", cid)
            if m:
                highest = max(highest, int(m.group(1)))
        return highest

    def _format_next_free_ids_for_prompt(self, klipy: dict) -> str:
        """Sestaví explicitní blok 'další volné ID' pro každý prefix (vid_/rap_/pic_/char_)
        a vloží ho do promptu Části 1/2 — MODEL SE NESMÍ SPOLÉHAT na to, že si správně
        spočítá nejvyšší existující ID sám z dlouhého seznamu (u velkých katalogů a
        slabších/lokálních modelů to spolehlivě selhává — model pak nová ID očísluje
        znovu od 01 a koliduje s existujícími klipy, viz _validate_no_id_collisions).
        Číslování počítáme mechanicky v Pythonu, ne v hlavě modelu."""
        prefixes = [("vid", "VID"), ("rap", "RAP"), ("pic", "PIC"), ("char", "CHAR")]
        lines = []
        for prefix, group_name in prefixes:
            highest = self._highest_clip_index(klipy, f"{prefix}_")
            if highest == 0:
                lines.append(f"- {prefix}_XX: žádný existující klip této skupiny → první nové ID je {prefix}_01")
            else:
                lines.append(f"- {prefix}_XX: nejvyšší existující je {prefix}_{highest:02d} → PRVNÍ NOVÉ ID musí být {prefix}_{highest + 1:02d} (a dál +1)")
        return "\n".join(lines)

    def _validate_no_id_collisions(self, sections1: dict, klipy: dict) -> list[str]:
        """Zkontroluje, že žádný 'nově potřebný' klip z Části 1/2 (VIDEO_PROMPTS/
        RAPPER_PROMPTS/IMAGE_PROMPTS) nepoužívá ID, které už patří existujícímu klipu
        v INPUT/klipy.md. Bez této kontroly model občas (zvlášť u velkých katalogů)
        znovu očísluje 'nové' klipy od 01 a přepíše/zdvojí existující assety pod stejným
        ID — což po dalších krocích (placeholders/parse) potichu poškodí projekt.
        Vrací seznam lidsky čitelných hlášek o kolizích (prázdný seznam = v pořádku)."""
        id_line_re = re.compile(r'^([a-zA-Z]+_\d+)\s*\|')
        collisions = []
        for section_name in ("VIDEO_PROMPTS", "RAPPER_PROMPTS", "IMAGE_PROMPTS"):
            block = sections1.get(section_name, "")
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                m = id_line_re.match(line)
                if not m:
                    continue
                cid = m.group(1)
                if cid in klipy:
                    existing_obsah = truncate_for_prompt(klipy[cid].get("obsah", ""), 100)
                    collisions.append(
                        f"{cid} je v {section_name} označen jako NOVÝ klip, ale toto ID už patří "
                        f"existujícímu klipu v klipy.md (\"{existing_obsah}\") — kolize ID."
                    )
        return collisions

    def _validate_timeline_ids(self, timeline_text: str, shot_order_text: str, known_ids: set) -> list[str]:
        """Zkontroluje, že každé ID použité v MUSIC_VIDEO_TIMELINE/SHOT_ORDER existuje
        buď v katalogu existujících klipů, nebo mezi nově naplánovanými klipy z Části 1/2.
        Bez této kontroly se do finálního full_plan.txt dřív mohla dostat ID, která
        neodkazují na žádný reálný ani nově vytvořený prompt (typicky když model v Části
        2/2 'vymyslel' vlastní ID mimo katalog, nebo naopak katalog v kontextu ořízl)."""
        id_ref_re = re.compile(r'\|\s*([a-zA-Z]+_\d+)\s*\|')
        unknown = set()
        for text in (timeline_text, shot_order_text):
            for line in text.splitlines():
                m = id_ref_re.search(line.strip())
                if m and m.group(1) not in known_ids:
                    unknown.add(m.group(1))
        return [f"ID '{cid}' použité v timeline/shot_order neexistuje ani v klipy.md, ani mezi nově naplánovanými klipy z Části 1/2." for cid in sorted(unknown)]

    def _validate_timeline_monotonic(self, timeline_text: str, song_duration: float,
                                      max_gap_or_overlap: float = 2.0) -> list[str]:
        """Zkontroluje, že řádky MUSIC_VIDEO_TIMELINE jdou v čase vzestupně a na sebe
        navazují (žádné velké skoky zpátky ani díry), a že poslední řádek končí zhruba
        na song_duration. Bez tohohle se sloučené úseky z Části 2/2 mohly v praxi
        zacyklit/přeskočit (např. konec jednoho úseku v minutách, začátek dalšího zpátky
        v sekundách) a chyba se odhalila až při renderu."""
        time_line_re = re.compile(r'^\[?(\d{1,2}:\d{2}\.\d+)\]?\s*-\s*\[?(\d{1,2}:\d{2}\.\d+)\]?\s*\|')
        problems = []
        prev_end = None
        prev_line_no = 0
        for line_no, raw_line in enumerate(timeline_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            m = time_line_re.match(line)
            if not m:
                continue
            try:
                start = parse_timecode(m.group(1))
                end = parse_timecode(m.group(2))
            except ValueError as e:
                problems.append(f"Řádek {line_no}: {e}")
                continue
            if end < start:
                problems.append(f"Řádek {line_no}: konec ({m.group(2)}) je před začátkem ({m.group(1)}).")
            if prev_end is not None and abs(start - prev_end) > max_gap_or_overlap:
                problems.append(
                    f"Řádek {line_no}: začíná na {m.group(1)}, ale předchozí řádek (č. {prev_line_no}) "
                    f"končil na {prev_end:.2f}s — rozdíl {abs(start - prev_end):.2f}s (mezera/díra/přeskok "
                    f"> {max_gap_or_overlap}s)."
                )
            prev_end = end
            prev_line_no = line_no
        if prev_end is not None and song_duration > 0 and abs(prev_end - song_duration) > max_gap_or_overlap:
            problems.append(
                f"Poslední řádek timeline končí na {prev_end:.2f}s, ale song_duration je "
                f"{song_duration:.2f}s — rozdíl {abs(prev_end - song_duration):.2f}s."
            )
        return problems

    def _validate_timeline_chunk_bounds(self, timeline_text: str, segment_start: float,
                                         segment_end: float, tolerance: float = 5.0) -> list[str]:
        """Zkontroluje, že VŠECHNY řádky MUSIC_VIDEO_TIMELINE vygenerované pro TENTO
        dílčí úsek (segment_start–segment_end) mají časy uvnitř očekávaného okna
        (+/- `tolerance` s), a že žádný řádek neobsahuje vyloženě neplatný časový kód
        (minutová/vteřinová složka >= 60, např. '18:80.30').

        Na rozdíl od `_force_align_timeline_chunk_boundaries` (ta mechanicky opraví
        jen ČASOVOU ZNAČKU prvního a posledního řádku, aby úseky na sebe navazovaly)
        tahle kontrola hlídá i VNITŘNÍ řádky úseku a volá se hned po vygenerování
        úseku — ne až po sloučení všech úseků na konci (`_validate_timeline_monotonic`).
        Slabší lokální model umí uvnitř 21vteřinového úseku vygenerovat řádky s časy
        v řádu minut (viz reálný případ: 01:20.30, 04:50.70, 08:20.30... místo pár
        vteřin) — bez včasné kontroly se to odhalí až po sloučení VŠECH úseků, tedy
        po zbytečném čekání na generování zbylých (často mnohaminutových) úseků."""
        time_line_re = re.compile(
            r'^\[?(\d{1,2}):(\d{2}(?:\.\d+)?)\]?\s*-\s*\[?(\d{1,2}):(\d{2}(?:\.\d+)?)\]?\s*\|'
        )
        problems = []
        lo = segment_start - tolerance
        hi = segment_end + tolerance
        for line_no, raw_line in enumerate(timeline_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            m = time_line_re.match(line)
            if not m:
                continue
            start_mm, start_ss, end_mm, end_ss = m.groups()
            invalid = False
            for label, ss_txt in (("začátku", start_ss), ("konce", end_ss)):
                if float(ss_txt) >= 60:
                    problems.append(
                        f"Řádek {line_no} úseku: neplatný časový kód u {label} — vteřinová "
                        f"složka '{ss_txt}' je >= 60 (model si spletl formát MM:SS)."
                    )
                    invalid = True
            if invalid:
                continue
            start = parse_timecode(f"{start_mm}:{start_ss}")
            end = parse_timecode(f"{end_mm}:{end_ss}")
            if start < lo or start > hi:
                problems.append(
                    f"Řádek {line_no} úseku: začátek {start_mm}:{start_ss} ({start:.2f}s) je mimo "
                    f"očekávané okno tohoto úseku ({segment_start:.2f}–{segment_end:.2f}s, "
                    f"tolerance {tolerance:.0f}s)."
                )
            if end < lo or end > hi:
                problems.append(
                    f"Řádek {line_no} úseku: konec {end_mm}:{end_ss} ({end:.2f}s) je mimo "
                    f"očekávané okno tohoto úseku ({segment_start:.2f}–{segment_end:.2f}s, "
                    f"tolerance {tolerance:.0f}s)."
                )
        return problems

    def _build_expected_clip_durations(self, klipy: dict, sections1: dict) -> dict:
        """Sestaví {clip_id: očekávaná_délka_v_sekundách} ze DVOU zdrojů — existujícího
        katalogu (klipy.md, `_load_klipy_md`) a nově naplánovaných klipů z Části 1/2
        (VIDEO_PROMPTS/RAPPER_PROMPTS/IMAGE_PROMPTS, tag `[Duration: X.Ys]`). Používá se
        v `_validate_timeline_row_durations`, aby šlo každý řádek MUSIC_VIDEO_TIMELINE
        zkontrolovat proti SKUTEČNÉ deklarované délce daného klipu, ne proti obecnému
        pevnému výčtu (rap klipy z Části 1/2 v praxi nemají vždy jen 4/6/8s, jen jednu
        z těch tří hodnot preferují — viz FULL_PLAN_PART1_SYSTEM_PROMPT)."""
        expected = {}
        for cid, data in klipy.items():
            dur = data.get("duration_sec", 0.0)
            if dur and dur > 0:
                expected[cid] = float(dur)

        id_pattern = re.compile(r'^([a-zA-Z]+_\d+)\s*\|\s*(?:[^|]*\|\s*)?\[Duration:\s*([\d.]+)s\]')
        for section_name in ("VIDEO_PROMPTS", "RAPPER_PROMPTS", "IMAGE_PROMPTS"):
            block = sections1.get(section_name, "")
            for raw_line in block.splitlines():
                line = raw_line.strip()
                m = id_pattern.match(line)
                if m:
                    cid, dur = m.groups()
                    try:
                        expected[cid] = float(dur)
                    except ValueError:
                        pass
        return expected

    def _validate_timeline_row_durations(self, timeline_text: str, expected_durations: dict,
                                          tolerance: float = 0.6, max_unknown_duration: float = 12.0) -> list[str]:
        """Zkontroluje, že délka (konec - začátek) KAŽDÉHO řádku MUSIC_VIDEO_TIMELINE zhruba
        odpovídá SKUTEČNÉ deklarované délce daného klipu (viz `_build_expected_clip_durations`).

        Tohle chytá konkrétní reálně pozorovanou halucinaci slabšího lokálního modelu: řádek
        (typicky první v úseku) dostane koncový čas nastavený na konec CELÉHO úseku/písně
        místo pár vteřin skutečné délky klipu (např. `rap_01 | 00:00.00-03:53.66` pro klip,
        který ve skutečnosti trvá pár vteřin). `_validate_timeline_chunk_bounds` tohle
        nezachytí, pokud úsek pokrývá celou píseň (233.66s je "uvnitř okna" 0–233.66s) —
        tahle kontrola jde nezávisle na hranicích úseku, dívá se jen na délku každého řádku.

        Pro ID, které v `expected_durations` není (neznámý/neregistrovaný klip), se místo
        přesné kontroly použije jen hrubá pojistka `max_unknown_duration` (klip v hudebním
        videu jednoduše nemá důvod trvat desítky vteřin/minuty)."""
        time_line_re = re.compile(
            r'^\[?(\d{1,2}:\d{2}\.\d+)\]?\s*-\s*\[?(\d{1,2}:\d{2}\.\d+)\]?\s*\|\s*([a-zA-Z]+_\d+)\s*\|'
        )
        problems = []
        for line_no, raw_line in enumerate(timeline_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            m = time_line_re.match(line)
            if not m:
                continue
            try:
                start = parse_timecode(m.group(1))
                end = parse_timecode(m.group(2))
            except ValueError:
                continue  # neplatný časový kód řeší _validate_timeline_monotonic/_validate_timeline_chunk_bounds
            cid = m.group(3)
            actual_dur = end - start
            expected_dur = expected_durations.get(cid)
            if expected_dur is not None:
                if abs(actual_dur - expected_dur) > tolerance:
                    problems.append(
                        f"Řádek {line_no}: klip '{cid}' má v timeline délku {actual_dur:.2f}s "
                        f"({m.group(1)}–{m.group(2)}), ale jeho skutečná/deklarovaná délka je "
                        f"{expected_dur:.2f}s (rozdíl {abs(actual_dur - expected_dur):.2f}s > "
                        f"tolerance {tolerance:.1f}s). Typický příznak: model omylem nastavil "
                        "koncový čas na konec úseku/písně místo skutečné délky klipu."
                    )
            elif actual_dur > max_unknown_duration:
                problems.append(
                    f"Řádek {line_no}: klip '{cid}' (neznámá/neregistrovaná délka) má v timeline "
                    f"{actual_dur:.2f}s ({m.group(1)}–{m.group(2)}) — podezřele dlouhé pro jeden "
                    f"záběr hudebního videa (strop {max_unknown_duration:.0f}s)."
                )
        return problems

    def _whisper_segments_to_words(self, data: dict) -> list[dict]:
        """Převede Whisper/Groq JSON na word-level mapu; chybějící word timestampy dopočítá.

        Podporuje tři formáty:
        - lokální openai-whisper CLI: words vnořené v každém segmentu (seg["words"])
        - Groq Cloud API (verbose_json): words jako samostatné top-level pole (data["words"])
        - vlastní/ruční formát po sekcích: {"song_duration": "MM:SS.mmm",
          "transcript": [{"section": ..., "words": [{"word","start","end"}]}]},
          kde start/end jsou textové časové kódy (MM:SS.mmm), ne float sekundy.
        """
        transcript_sections = data.get("transcript")
        if isinstance(transcript_sections, list):
            words = []
            for section in transcript_sections:
                for item in (section.get("words") or []):
                    raw_word = str(item.get("word", "")).strip()
                    # Anotace typu [Music]/[Smích]/[Výkřik] nejsou zpívaný text —
                    # přeskočit, aby se nepletly do zarovnávání na lyrics.txt.
                    if not raw_word or re.fullmatch(r"\[[^\]]*\]", raw_word):
                        continue
                    try:
                        start = parse_timecode(str(item.get("start", "0")))
                        end = parse_timecode(str(item.get("end", item.get("start", "0"))))
                    except ValueError:
                        continue
                    words.append({"word": raw_word, "start": start, "end": max(start, end)})
            if words:
                return words

        top_level_words = data.get("words")
        if top_level_words:
            words = []
            for item in top_level_words:
                word = str(item.get("word", "")).strip()
                if word:
                    words.append({
                        "word": word,
                        "start": float(item.get("start", 0.0)),
                        "end": float(item.get("end", item.get("start", 0.0))),
                    })
            if words:
                return words

        words = []
        for seg in data.get("segments", []):
            seg_words = seg.get("words")
            if seg_words:
                for item in seg_words:
                    word = str(item.get("word", "")).strip()
                    if word:
                        words.append({
                            "word": word,
                            "start": float(item.get("start", seg.get("start", 0.0))),
                            "end": float(item.get("end", seg.get("end", seg.get("start", 0.0)))),
                        })
                continue

            text_words = lyric_words(str(seg.get("text", "")))
            if not text_words:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            dur = max(0.001, end - start)
            step = dur / len(text_words)
            for i, word in enumerate(text_words):
                words.append({
                    "word": word,
                    "start": round(start + i * step, 3),
                    "end": round(start + (i + 1) * step, 3),
                })
        return words

    def _align_words_to_lyrics(self, raw_words: list[dict], official_text: str) -> list[dict]:
        """Nahradí text slov oficiálními lyrics, ale zachová časové značky z Whisperu.

        Pozn. k LLM post-processingu: tato funkce pouze interpoluje časové značky mezi
        opcode bloky (SequenceMatcher) a je čistě deterministická — LLM zde záměrně
        NENÍ použita, protože by mohla ovlivnit přesné časování slov (word timestamps),
        které musí zůstat 1:1 odvozené z Whisperu. Volba/oprava TEXTU (který lyrics okno
        použít) už prochází přes LLM o úroveň výš v `_best_lyrics_window_scored()` —
        tahle funkce jen namapuje výsledný text na časy, beze změny."""
        official_words = lyric_words(official_text)
        if not raw_words or not official_words:
            return raw_words

        # Pozor: raw_norm musí mít STEJNOU délku a pořadí jako raw_words, protože
        # opcode indexy (i1, i2) ze SequenceMatcheru se používají přímo pro
        # raw_words[i1:i2]. normalized_words() by slova, co se normalizují na
        # prázdný řetězec (např. anotace "[Music]"), z výsledku VYNECHALA, čímž by
        # posunula indexy vůči raw_words a rozbila mapování časů. Proto se tu
        # normalizuje 1:1 bez filtrování — prázdný řetězec u anotací je v pořádku,
        # official_norm (skutečná slova z lyrics.txt) prázdný nikdy není, takže se
        # s takovým raw prvkem beztak nikdy nespáruje jako "equal".
        raw_norm = [normalize_text(w["word"]) for w in raw_words]
        official_norm = normalized_words(official_words)
        matcher = SequenceMatcher(None, raw_norm, official_norm, autojunk=False)
        aligned = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            raw_slice = raw_words[i1:i2]
            official_slice = official_words[j1:j2]
            if tag == "delete":
                continue
            if not official_slice:
                continue

            if tag == "equal":
                # Přesná 1:1 shoda — použij originální (Whisperovy/zdrojové) časy
                # každého slova přímo, žádná interpolace není potřeba ani žádoucí.
                for r, word in zip(raw_slice, official_slice):
                    aligned.append({
                        "word": word,
                        "start": round(float(r["start"]), 3),
                        "end": round(float(r["end"]), 3),
                    })
                continue

            if raw_slice:
                start = float(raw_slice[0]["start"])
                end = float(raw_slice[-1]["end"])
            else:
                # Vložená official lyrics slova nemají vlastní Whisper čas,
                # proto je rozprostřeme mezi sousední rozpoznaná slova.
                start = float(raw_words[i1 - 1]["end"]) if i1 > 0 else float(raw_words[0]["start"])
                end = float(raw_words[i1]["start"]) if i1 < len(raw_words) else start + 0.2

            if end <= start:
                end = start + 0.001 * len(official_slice)
            step = (end - start) / max(1, len(official_slice))

            for idx, word in enumerate(official_slice):
                aligned.append({
                    "word": word,
                    "start": round(start + idx * step, 3),
                    "end": round(start + (idx + 1) * step, 3),
                })

        return aligned if aligned else raw_words

    def _best_lyrics_window(self, raw_words: list[dict], official_text: str, hint_text: str = "") -> str:
        """Najde krátký úsek lyrics.txt, který nejlépe odpovídá raw transkripci klipu."""
        text, _score = self._best_lyrics_window_scored(raw_words, official_text, hint_text)
        return text

    def _best_lyrics_window_scored(self, raw_words: list[dict], official_text: str, hint_text: str = "") -> tuple[str, float]:
        """Jako _best_lyrics_window, ale navíc vrací skóre shody (0.0-1.0) pro účely validace
        ('Vazba na text': ověření, že text klipu se skutečně nachází v lyrics.txt).

        Pokud je dostupná lokální Ollama, nechá ji vybrat mezi několika (max 5) nejlepšími
        kandidátními okny podle heuristiky — model může opravit drobné transkripční chyby
        a potvrdit/zamítnout, že jde o reálnou shodu. Při jakémkoli selhání AI se použije
        čistě heuristický výsledek beze změny."""
        official_words = lyric_words(official_text)
        raw_norm = normalized_words([w["word"] for w in raw_words])
        official_norm = normalized_words(official_words)
        if not raw_norm or not official_norm:
            return official_text, 0.0

        # Slovník všech (významových) slov, která se v lyrics.txt opravdu vyskytují —
        # použije se níž k ověření, že případná "oprava" od LLM není vymyšlená.
        official_vocab = set(tokenize(official_text))

        hint_norm = normalized_words(lyric_words(hint_text))
        target_len = max(1, len(raw_norm))
        min_len = max(1, int(target_len * 0.65))
        max_len = min(len(official_norm), max(target_len + 6, int(target_len * 1.45)))
        best = {"score": -1.0, "raw_score": 0.0, "start": 0, "end": min(len(official_words), target_len)}
        top_candidates = []  # udržujeme top-N (podle score) pro případný LLM post-processing

        for start in range(0, max(1, len(official_norm) - min_len + 1)):
            for win_len in range(min_len, max_len + 1):
                end = start + win_len
                if end > len(official_norm):
                    continue
                window_norm = official_norm[start:end]
                raw_score = SequenceMatcher(None, raw_norm, window_norm, autojunk=False).ratio()
                score = raw_score
                if hint_norm:
                    hint_score = SequenceMatcher(None, hint_norm, window_norm, autojunk=False).ratio()
                    score = (score * 0.75) + (hint_score * 0.25)
                # Jemný tie-break ve prospěch oken blízkých délce skutečně rozpoznaného
                # (Whisper) textu. Bez tohohle můžou dvě okna vyjít na téměř identické
                # skóre a vyhrát to kratší, které "uřízne" okrajové slovo, i když ve
                # skutečnosti do klipu patřilo (viz bug: chybějící první/poslední slovo
                # v lyrics_window, přestože sedělo). Váha je malá (max ~3 %), aby
                # nepřebila skutečně lepší shodu, jen rozhodne mezi vyrovnanými kandidáty.
                length_diff_ratio = abs(win_len - target_len) / target_len
                score = score * (1 - min(0.03, 0.03 * length_diff_ratio))
                if score > best["score"]:
                    best = {"score": score, "raw_score": raw_score, "start": start, "end": end}
                # Udržujeme malý rolling pool nejlepších kandidátů (pro LLM výběr) —
                # jen omezený počet, aby to nezatěžovalo paměť u dlouhých textů.
                if score > 0.15:
                    top_candidates.append({"score": score, "raw_score": raw_score, "start": start, "end": end})
                    if len(top_candidates) > 200:
                        top_candidates.sort(key=lambda c: c["score"], reverse=True)
                        top_candidates = top_candidates[:20]

        text = " ".join(official_words[best["start"]:best["end"]]).strip()
        raw_score_final = round(max(0.0, best["raw_score"]), 4)

        # ── Volitelný LLM post-processing mezi top-5 kandidáty ──
        if self._ollama_ready() and top_candidates:
            top_candidates.sort(key=lambda c: c["score"], reverse=True)
            # Odstraníme téměř identická okna (stejný start), necháme rozmanité kandidáty
            seen_starts = set()
            distinct = []
            for c in top_candidates:
                if c["start"] in seen_starts:
                    continue
                seen_starts.add(c["start"])
                distinct.append(c)
                if len(distinct) >= 5:
                    break
            if len(distinct) >= 2:
                llm_candidates = []
                for i, c in enumerate(distinct):
                    window_text = " ".join(official_words[c["start"]:c["end"]]).strip()
                    llm_candidates.append({"index": i, "text": window_text, "score": c["raw_score"]})
                raw_clip_text = " ".join(w["word"] for w in raw_words)
                result = self._llm_choose_best_candidate(
                    task_description=(
                        "Vyber, které z níže uvedených okusů oficiálního textu (lyrics.txt) "
                        "nejlépe odpovídá tomu, co bylo skutečně rozpoznáno v audio klipu, "
                        f"a případně oprav drobné chyby. Rozpoznaný (nepřesný) text klipu: "
                        f'"{truncate_for_prompt(raw_clip_text, 160)}"'
                    ),
                    candidates=llm_candidates,
                    extra_rules="Pokud žádný kandidát rozumně neodpovídá, nastav is_valid_match na false.",
                )
                if result and result["is_valid_match"]:
                    chosen = distinct[result["index"]]
                    chosen_window_text = " ".join(official_words[chosen["start"]:chosen["end"]]).strip()
                    corrected = result.get("corrected_text")
                    final_text = chosen_window_text

                    if corrected and corrected.strip():
                        # Bezpečnostní kotva č. 1: "corrected_text" od (malého lokálního) LLM
                        # přijmeme JEN pokud jsou jeho (významová) slova skutečně součástí
                        # lyrics.txt. Bez téhle kontroly se může stát, že model úkol nesplní
                        # a jen zopakuje nepřesný Whisper přepis zpátky jako "opravu" — a ta
                        # by se pak bez kontroly vydávala za text ze skutečné písně, i když
                        # obsahuje slova, která v lyrics.txt vůbec nejsou (např. přeslechnutá
                        # vlastní jména). Čistě znaková podobnost na tohle nestačí, protože
                        # zbytek věty bývá stejný a pár chybných slov ratio moc nesníží.
                        corrected_tokens = tokenize(corrected)
                        if corrected_tokens:
                            in_vocab_ratio = sum(1 for t in corrected_tokens if t in official_vocab) / len(corrected_tokens)
                        else:
                            in_vocab_ratio = 0.0

                        # Bezpečnostní kotva č. 2: "oprava" nesmí text nepřiměřeně ZKRÁTIT.
                        # I když je in_vocab_ratio 100 % (model jen ubral slova z vybraného
                        # okna, každé z nich samo o sobě v lyrics.txt existuje), výsledek by
                        # klidně mohl potichu uříznout první/poslední slovo, které do klipu
                        # reálně patřilo — to je přesně bug, kdy transcript_fixed/lyrics_window
                        # skončí kratší než skutečně odrapovaný text. Proto porovnáme počet
                        # slov korekce s počtem slov jak zvoleného okna, tak Whisperova
                        # rozpoznaného textu (raw_words) — korekce smí být kratší nejvýš
                        # o 1 slovo a nesmí být o víc než 2 slova delší.
                        window_len = chosen["end"] - chosen["start"]
                        len_reference = max(window_len, len(raw_words))
                        length_ok = len(corrected_tokens) >= len_reference - 1 and len(corrected_tokens) <= len_reference + 2

                        if in_vocab_ratio >= 0.9 and length_ok:
                            final_text = corrected.strip()
                        elif in_vocab_ratio >= 0.9 and not length_ok:
                            print(f"⚠️  LLM korekce textu zamítnuta (nevhodná délka: {len(corrected_tokens)} "
                                  f"slov vs. očekávaných ~{len_reference}) — použito heuristické okno beze změny.")

                    if final_text:
                        text = final_text
                        raw_score_final = round(max(raw_score_final, clamp_confidence(chosen["raw_score"])), 4)

        return text, raw_score_final

    def _load_json(self, path: Path, default):
        """Bezpečně načte JSON."""
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            try:
                rel = path.relative_to(self.project_dir)
            except ValueError:
                rel = path
            print(f"⚠️ Neplatný JSON v {rel}: {exc}")
            return default
        except Exception as exc:
            try:
                rel = path.relative_to(self.project_dir)
            except ValueError:
                rel = path
            print(f"⚠️ Nepodařilo se načíst JSON {rel}: {exc}")
            return default

    def _write_json(self, path: Path, data) -> None:
        """Zapíše JSON v čitelné podobě."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_song_position_after(
        self,
        text: str,
        song_words: list[dict],
        after_time: float,
        used_positions: set,
    ) -> float | None:
        """Najde další výskyt textu v songu za daným časem (pro opakované refrény)."""
        query = normalized_words(lyric_words(text))
        song_norm = normalized_words([w["word"] for w in song_words])
        if not query or not song_norm:
            return None

        n = len(query)
        min_len = max(1, int(n * 0.7))
        max_len = min(len(song_norm), max(n + 4, int(n * 1.35)))
        candidates = []
        for start in range(0, max(1, len(song_norm) - min_len + 1)):
            ss = float(song_words[start]["start"])
            if ss < after_time - 0.5:
                continue
            for win_len in range(min_len, max_len + 1):
                end = start + win_len
                if end > len(song_norm):
                    continue
                score = SequenceMatcher(None, query, song_norm[start:end], autojunk=False).ratio()
                if score >= 0.40:
                    candidates.append({"score": score, "song_start": ss, "start": start, "end": end})

        if not candidates:
            return None

        available = []
        for c in candidates:
            taken = any(abs(c["song_start"] - used) < 2.0 for used in used_positions)
            if not taken:
                available.append(c)

        pool = available if available else candidates
        best = max(pool, key=lambda c: c["score"])

        # ── Volitelný LLM post-processing, pokud je více blízkých kandidátů (ambiguita) ──
        if self._ollama_ready() and len(pool) >= 2:
            pool_sorted = sorted(pool, key=lambda c: c["score"], reverse=True)[:5]
            # LLM zapojíme jen když jsou top kandidáti opravdu blízko sebe (heuristika sama neví, co je lepší)
            if pool_sorted[0]["score"] - pool_sorted[-1]["score"] < 0.25:
                llm_candidates = []
                for i, c in enumerate(pool_sorted):
                    window_text = " ".join(w["word"] for w in song_words[c["start"]:c["end"]])
                    llm_candidates.append({"index": i, "text": f'[{c["song_start"]:.1f}s] {window_text}', "score": c["score"]})
                result = self._llm_choose_best_candidate(
                    task_description=(
                        f'Hledáme opakovaný výskyt textu "{truncate_for_prompt(text, 120)}" v písni '
                        f"po čase {after_time:.1f}s (např. opakovaný refrén). Vyber nejpravděpodobnější výskyt."
                    ),
                    candidates=llm_candidates,
                )
                if result and result["is_valid_match"]:
                    best = pool_sorted[result["index"]]

        return best["song_start"]

    def _find_unique_song_position(self, text: str, song_words: list[dict], preferred_start: float, used_positions: set) -> float | None:
        """Najde výskyt textu v songu blízko preferred_start, ale vylučuje již zabrané pozice."""
        query = normalized_words(lyric_words(text))
        song_norm = normalized_words([w["word"] for w in song_words])
        if not query or not song_norm:
            return None

        n = len(query)
        min_len = max(1, int(n * 0.7))
        max_len = min(len(song_norm), max(n + 4, int(n * 1.35)))

        # Sesbíráme VŠECHNY dobré shody
        candidates = []
        for start in range(0, max(1, len(song_norm) - min_len + 1)):
            for win_len in range(min_len, max_len + 1):
                end = start + win_len
                if end > len(song_norm):
                    continue
                score = SequenceMatcher(None, query, song_norm[start:end], autojunk=False).ratio()
                if score >= 0.40:
                    ss = float(song_words[start]["start"])
                    candidates.append({
                        "score": score,
                        "song_start": ss,
                        "song_index": start,
                    })

        if not candidates:
            return None

        # Odfiltrujeme už zabrané pozice (s tolerancí 2s)
        available = []
        for c in candidates:
            taken = False
            for used in used_positions:
                if abs(c["song_start"] - used) < 2.0:
                    taken = True
                    break
            if not taken:
                available.append(c)

        # Pokud nic volného, vezmeme i zabrané (raději duplicita než ztráta klipu)
        pool = available if available else candidates

        # Z dostupných vybereme nejbližší k preferred_start s penalizací za vzdálenost
        best = None
        best_rank = -999.0
        for c in pool:
            distance = abs(c["song_start"] - preferred_start)
            rank = c["score"] - min(0.35, distance / 240.0)
            c["rank"] = rank
            if rank > best_rank:
                best_rank = rank
                best = c

        # ── Volitelný LLM post-processing, pokud je více blízkých kandidátů (ambiguita) ──
        if best and self._ollama_ready() and len(pool) >= 2:
            pool_sorted = sorted(pool, key=lambda c: c["rank"], reverse=True)[:5]
            if pool_sorted[0]["rank"] - pool_sorted[-1]["rank"] < 0.20:
                llm_candidates = []
                for i, c in enumerate(pool_sorted):
                    window_text = " ".join(w["word"] for w in song_words[c["song_index"]:c["song_index"] + max(1, int(len(query) * 1.1))])
                    llm_candidates.append({"index": i, "text": f'[{c["song_start"]:.1f}s] {window_text}', "score": c["score"]})
                result = self._llm_choose_best_candidate(
                    task_description=(
                        f'Hledáme výskyt textu "{truncate_for_prompt(text, 120)}" v písni, nejblíže '
                        f"preferovanému času {preferred_start:.1f}s. Vyber nejpravděpodobnější výskyt "
                        "(nemusí být nutně nejblíže časově, pokud textově sedí výrazně lépe)."
                    ),
                    candidates=llm_candidates,
                )
                if result and result["is_valid_match"]:
                    best = pool_sorted[result["index"]]

        return best["song_start"] if best else None

    def _load_timeline_rap_ranges(self) -> dict[str, tuple[float, float]]:
        """Načte časové rozsahy rap klipů z timeline.txt."""
        timeline_path = self.timeline_file
        if not timeline_path.exists():
            return {}

        ranges = {}
        for raw in timeline_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            time_range = parts[0]
            clip_name = parts[1]
            if not clip_name.startswith("rap_") or "-" not in time_range:
                continue
            try:
                start_raw, end_raw = [x.strip() for x in time_range.split("-", 1)]
                start = parse_timecode(start_raw)
                end = parse_timecode(end_raw)
                ranges[clip_name] = (start, end)
            except Exception:
                continue
        return ranges

    def _select_lyric_candidates(self, timeline_text: str, lyric_lines: list[str], limit: int = 8) -> list[str]:
        """Vybere nejpravděpodobnější lyric řádky pro daný časový úsek."""
        if not lyric_lines:
            return []

        timeline_tokens = set(tokenize(timeline_text))
        scored = []
        for line in lyric_lines:
            line_tokens = set(tokenize(line))
            if not line_tokens:
                continue
            overlap = len(timeline_tokens & line_tokens)
            seq_score = SequenceMatcher(None, normalize_text(timeline_text), normalize_text(line)).ratio()
            score = (overlap * 0.7) + (seq_score * 0.3)
            scored.append((score, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [line for score, line in scored[:limit] if score > 0]

    def _correct_transcription_by_lyrics(self, transcript: str, lyric_lines: list[str]) -> str:
        """Vrátí nejpravděpodobnější lyric text pro danou transkripci."""
        if not lyric_lines:
            return transcript.strip()

        transcript_clean = normalize_text(transcript)
        transcript_tokens = set(tokenize(transcript))

        best_line = ""
        best_score = 0.0

        for line in lyric_lines:
            line_clean = normalize_text(line)
            line_tokens = set(tokenize(line))
            if not line_tokens:
                continue

            token_overlap = len(transcript_tokens & line_tokens) / max(1, min(len(transcript_tokens), len(line_tokens)))
            seq_score = SequenceMatcher(None, transcript_clean, line_clean).ratio()
            score = (token_overlap * 0.7) + (seq_score * 0.3)

            if score > best_score:
                best_score = score
                best_line = line.strip()

        if best_line and best_score >= 0.18:
            return best_line
        return transcript.strip()

    def _transcribe_rap_clip(self, clip_path: Path, whisper_bin: str, tmpdir: Path) -> str:
        """Transkribuje jeden rap klip přes Whisper CLI a vrátí čistý text."""
        base = tmpdir / clip_path.stem
        wav_path = base.with_suffix(".wav")
        json_path = base.with_suffix(".json")

        extract_cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", str(clip_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path)
        ]
        subprocess.run(extract_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wav_path.exists():
            return ""

        settings = self.load_settings()
        transcribe_cmd = [
            whisper_bin,
            str(wav_path),
            "--model", settings["whisper_model"],
            *self._whisper_device_args(settings),
            "--word_timestamps", "True",
            "--output_format", "json",
            "--output_dir", str(tmpdir),
        ]
        try:
            subprocess.run(transcribe_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return ""

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
                segments = data.get("segments", [])
                text = " ".join(str(s.get("text", "")).strip() for s in segments if str(s.get("text", "")).strip()).strip()
                json_path.unlink(missing_ok=True)
                return text
            except Exception:
                return ""
        return ""

    def _adjust_rap_clip_speed(self, input_path: Path, output_path: Path, speed: float) -> bool:
        """Vyrenderuje klip na požadovanou rychlost bez audia."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if abs(speed - 1.0) < 0.02:
            shutil.copy2(str(input_path), str(output_path))
            return True

        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", str(input_path),
            "-filter:v", f"setpts=PTS/{speed}",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        result = run_ffmpeg(cmd)
        return result

    def analyze_song(self) -> bool:
        """Vytvoří word-level mapu master skladby zarovnanou na lyrics.txt."""
        self.analyze_audio()
        trans_file = self.input_dir / "transcription.json"
        if not trans_file.exists():
            print("❌ Nelze vytvořit song_alignment.json bez INPUT/transcription.json.")
            return False

        lyrics_text = self._load_lyrics_text()
        if not lyrics_text:
            print("❌ Nelze vytvořit song_alignment.json bez INPUT/lyrics.txt.")
            return False

        data = self._load_json(trans_file, {})
        raw_words = self._whisper_segments_to_words(data)
        if not raw_words:
            print(f"❌ {trans_file.relative_to(self.project_dir)} neobsahuje žádná rozpoznaná slova "
                  f"(poškozený/nekompatibilní soubor?). song_alignment.json se NEVYTVOŘÍ.")
            print("   Smaž transcription.json a spusť volbu 4 znovu, nebo transkripci dodej ručně.")
            return False

        aligned_words = self._align_words_to_lyrics(raw_words, lyrics_text)
        if not aligned_words:
            print("❌ Zarovnání transkriptu na lyrics.txt selhalo. song_alignment.json se NEVYTVOŘÍ.")
            return False

        # Orientační kontrola textové shody — nízké skóre často znamená, že lyrics.txt
        # používá zkratky pro opakování (refrén 2x apod.), což časové značky po tomto
        # místě rozhodí pro zbytek písně.
        raw_norm = normalized_words([w["word"] for w in raw_words])
        official_norm = normalized_words(lyric_words(lyrics_text))
        match_ratio = SequenceMatcher(None, raw_norm, official_norm, autojunk=False).ratio()

        audio_path = self.find_audio()
        song_duration = probe_duration(audio_path) if audio_path else 0.0

        out = {
            "source": str(trans_file.relative_to(self.project_dir)),
            "lyrics_source": "INPUT/lyrics.txt",
            "song_duration": round(song_duration, 3),
            "text_match_score": round(match_ratio, 4),
            "text": " ".join(w["word"] for w in aligned_words),
            "words": aligned_words,
        }
        self._write_json(self.edit_dir / "song_alignment.json", out)
        lipsync_manifest = build_lipsync_manifest(
            aligned_words, song_duration=song_duration, text_match_score=match_ratio
        )
        self._write_json(self.edit_dir / "word_phoneme_alignment.json", lipsync_manifest)
        print(f"✅ Vytvořena word-level mapa songu: {(self.edit_dir / 'song_alignment.json').relative_to(self.project_dir)}")
        print(f"✅ Vytvořen word/foném lipsync manifest: {(self.edit_dir / 'word_phoneme_alignment.json').relative_to(self.project_dir)}")
        if match_ratio < 0.5:
            print(f"⚠️  Nízká shoda transkriptu s lyrics.txt ({match_ratio:.2f}). Zkontroluj, jestli lyrics.txt")
            print("   nepíše opakované pasáže (refrén apod.) jen jednou — časové značky po takovém")
            print("   místě pak mohou být pro zbytek písně nepřesné.")
        return True

    def _transcribe_media_json(self, media_path: Path, whisper_bin: str, tmpdir: Path) -> dict:
        """Transkribuje media soubor (např. rap_xx.mp4) a vrátí Whisper-kompatibilní JSON.
        Použije Groq Cloud API nebo lokální Whisper podle nastavení 'transcription_provider'."""
        settings = self.load_settings()
        provider = str(settings.get("transcription_provider", "local")).lower()

        if provider == "groq":
            api_key = self._groq_ready()
            if not api_key:
                return {}
            # Groq umí mp4/mp3/wav/m4a/... přímo, extrakce přes ffmpeg není potřeba.
            return self._groq_transcribe_file(media_path, settings, api_key)

        wav_path = tmpdir / f"{media_path.stem}.wav"
        out_base = tmpdir / media_path.stem
        json_path = out_base.with_suffix(".json")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-y", "-i", str(media_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wav_path.exists():
            return {}
        try:
            subprocess.run([
                whisper_bin, str(wav_path), "--model", settings["whisper_model"],
                *self._whisper_device_args(settings),
                "--word_timestamps", "True",
                "--output_format", "json", "--output_dir", str(tmpdir)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return {}
        return self._load_json(json_path, {})

    def _best_song_match(self, clip_words: list[dict], song_words: list[dict], preferred_start: float = None, exclude_positions: set = None) -> dict:
        """Najde nejlepší výskyt textu klipu v word mapě songu s volitelnou preferencí pozice.

        Pokud je dostupná Ollama, u nejednoznačných shod (top kandidáti blízko sebe skóre)
        nechá ji vybrat mezi top-5 kandidáty a posoudit, zda jde vůbec o reálnou shodu,
        nebo o falešný pozitiv (viz `is_valid_match` v odpovědi). Heuristika zůstává jediným
        zdrojem pravdy, pokud AI není dostupná nebo vrátí neplatnou odpověď."""
        clip_norm = normalized_words([w["word"] for w in clip_words])
        song_norm = normalized_words([w["word"] for w in song_words])
        if not clip_norm or not song_norm:
            return {"score": 0.0, "song_start": None, "song_end": None, "song_index": None}

        n = len(clip_norm)
        best = {"rank": -1.0, "score": 0.0, "song_start": None, "song_end": None, "song_index": None}
        min_len = max(1, int(n * 0.7))
        max_len = min(len(song_norm), max(n + 4, int(n * 1.3)))
        pool = []  # rolling top-N kandidátů podle rank (pro LLM post-processing)

        # Pokud máme velmi krátký klip (1-2 slova), zvýšíme min_len
        if n <= 2:
            min_len = n

        for start in range(0, max(1, len(song_norm) - min_len + 1)):
            ss = float(song_words[start]["start"])

            # Pokud je pozice v exclude listu (s tolerancí), penalizujeme ji drasticky
            exclude_penalty = 0.0
            if exclude_positions:
                for used in exclude_positions:
                    if abs(ss - used) < 1.0:
                        exclude_penalty = 0.5
                        break

            for win_len in range(min_len, max_len + 1):
                end = start + win_len
                if end > len(song_norm):
                    continue
                score = SequenceMatcher(None, clip_norm, song_norm[start:end], autojunk=False).ratio()

                # Zabráníme extrémně krátkým shodám (méně než 0.4s) a extrémním nepoměrům trvání
                song_dur = float(song_words[end - 1]["end"]) - ss
                rap_dur = clip_words[-1]["end"] - clip_words[0]["start"]
                if song_dur < 0.4 and n > 1:
                    continue

                # Pokud je song_dur víc než 5x delší nebo 5x kratší než rap_dur, je to pravděpodobně falešná shoda
                if n > 3 and (song_dur > rap_dur * 5 or song_dur < rap_dur / 5):
                    continue

                rank = score - exclude_penalty
                if preferred_start is not None:
                    # Penalizace za vzdálenost (výraznější)
                    distance = abs(ss - preferred_start)
                    rank -= min(0.35, distance / 300.0)

                candidate = {
                    "rank": rank,
                    "score": round(score, 4),
                    "song_start": ss,
                    "song_end": song_words[end - 1]["end"],
                    "song_index": start,
                }
                if rank > best["rank"]:
                    best = candidate
                if rank > 0.15:
                    pool.append(candidate)
                    if len(pool) > 100:
                        pool.sort(key=lambda c: c["rank"], reverse=True)
                        pool = pool[:15]

        # ── Volitelný LLM post-processing pro nejednoznačné shody ──
        if best["song_index"] is not None and self._ollama_ready() and len(pool) >= 2:
            pool_sorted = sorted(pool, key=lambda c: c["rank"], reverse=True)[:5]
            if pool_sorted[0]["rank"] - pool_sorted[-1]["rank"] < 0.20:
                clip_text = " ".join(w["word"] for w in clip_words)
                llm_candidates = []
                for i, c in enumerate(pool_sorted):
                    window_text = " ".join(w["word"] for w in song_words[c["song_index"]:c["song_index"] + max(1, n)])
                    llm_candidates.append({"index": i, "text": f'[{c["song_start"]:.1f}s] {window_text}', "score": c["score"]})
                result = self._llm_choose_best_candidate(
                    task_description=(
                        f'Rozpoznaný text rap klipu (nepřesný přepis): "{truncate_for_prompt(clip_text, 160)}". '
                        "Vyber, které místo v písni tomuto klipu skutečně odpovídá. Pokud žádné z míst "
                        "textově opravdu neodpovídá (falešná shoda), nastav is_valid_match na false."
                    ),
                    candidates=llm_candidates,
                )
                if result:
                    if result["is_valid_match"]:
                        best = pool_sorted[result["index"]]
                    # is_valid_match == False → heuristický výsledek necháváme beze změny
                    # (nechceme AI umožnit shodu úplně zrušit, jen upřesnit výběr mezi kandidáty).

        best.pop("rank", None)
        return best

    def _best_song_match_for_text_near(self, text: str, song_words: list[dict], preferred_start: float | None = None) -> dict:
        """Najde výskyt textu v songu, při opakování preferuje pozici blízkou původní timeline."""
        query = normalized_words(lyric_words(text))
        song_norm = normalized_words([w["word"] for w in song_words])
        if not query or not song_norm:
            return {"score": 0.0, "song_start": None, "song_end": None, "song_index": None}

        n = len(query)
        min_len = max(1, int(n * 0.7))
        max_len = min(len(song_norm), max(n + 4, int(n * 1.35)))
        best = {"rank": -1.0, "score": 0.0, "song_start": None, "song_end": None, "song_index": None}

        for start in range(0, max(1, len(song_norm) - min_len + 1)):
            for win_len in range(min_len, max_len + 1):
                end = start + win_len
                if end > len(song_norm):
                    continue
                score = SequenceMatcher(None, query, song_norm[start:end], autojunk=False).ratio()
                rank = score
                if preferred_start is not None:
                    distance = abs(float(song_words[start]["start"]) - preferred_start)
                    rank -= min(0.35, distance / 240.0)
                if rank > best["rank"]:
                    best = {
                        "rank": rank,
                        "score": round(score, 4),
                        "song_start": song_words[start]["start"],
                        "song_end": song_words[end - 1]["end"],
                        "song_index": start,
                    }
        best.pop("rank", None)
        return best

    def transcribe_rap_clips(self):
        """Transkribuje rap klipy a vytvoří EDIT_PROJECT/rap_alignment.json."""
        settings = self.load_settings()
        provider = str(settings.get("transcription_provider", "local")).lower()
        whisper_bin = None
        if provider == "groq":
            if not self._groq_ready():
                return
            print(f"🗣️  Transkripce rap klipů přes Groq Cloud API (model: {settings.get('groq_model', 'whisper-large-v3-turbo')})...")
        else:
            whisper_bin = shutil.which("whisper") or shutil.which("whisper-cli")
            if not whisper_bin:
                print("❌ Nebyl nalezen příkaz `whisper` ani `whisper-cli` v PATH.")
                print("   Nebo přepni v Nastavení na Groq Cloud API.")
                return
            if Path(whisper_bin).name == "whisper-cli":
                print("❌ Nalezen jen `whisper-cli` (whisper.cpp) — ten používá jinou syntaxi CLI")
                print("   argumentů než tento pipeline očekává (openai-whisper), takže by transkripce")
                print("   tiše selhala u každého klipu. Nainstaluj `pip install openai-whisper")
                print("   --break-system-packages`, nebo přepni v Nastavení transcription_provider na 'groq'.")
                return

        song_alignment = self._load_json(self.edit_dir / "song_alignment.json", {})
        song_words = song_alignment.get("words", [])
        lyrics_text = self._load_lyrics_text()
        timeline_ranges = self._load_timeline_rap_ranges()
        song_segments = self._load_song_segments()
        rap_clips = sorted(self.gen_rap.glob("rap_*.mp4"))
        if not rap_clips:
            print("❌ Ve složce gen_rap nebyly nalezeny žádné rap_*.mp4 klipy.")
            return

        backup_dir = self.project_dir / "gen_rap_original_backup"
        results = {}
        used_song_matches = set()
        failed_clips = []
        with tempfile.TemporaryDirectory(prefix="rap_words_", dir=str(self.project_dir)) as tmp:
            tmpdir = Path(tmp)
            for idx, clip_path in enumerate(rap_clips, 1):
                try:
                    source_path = backup_dir / clip_path.name if (backup_dir / clip_path.name).exists() else clip_path
                    data = self._transcribe_media_json(source_path, whisper_bin, tmpdir)
                    raw_words = self._whisper_segments_to_words(data)

                    if not raw_words:
                        print(f"  [{idx:02d}/{len(rap_clips):02d}] ⚠️ {clip_path.stem}: transkripce nevrátila žádná "
                              f"slova (selhání whisperu/ffmpeg, nebo ticho) — klip bude označen jako neúplný.")
                        failed_clips.append(clip_path.stem)

                    hint_text = ""
                    clip_range = timeline_ranges.get(clip_path.stem)
                    if clip_range and song_segments:
                        hint_text = self._song_text_for_range(song_segments, clip_range[0], clip_range[1])
                    clip_lyrics, lyrics_match_score = self._best_lyrics_window_scored(raw_words, lyrics_text, hint_text)
                    fixed_words = self._align_words_to_lyrics(raw_words, clip_lyrics)
                    rap_start = fixed_words[0]["start"] if fixed_words else 0.0
                    rap_end = fixed_words[-1]["end"] if fixed_words else probe_duration(clip_path)

                    # Použijeme preferovaný start z timeline, pokud ho máme
                    preferred_start = clip_range[0] if clip_range else None
                    match = self._best_song_match(fixed_words, song_words, preferred_start, used_song_matches) if song_words else {}
                    if match.get("song_start") is not None:
                        used_song_matches.add(round(match["song_start"], 1))
                    results[clip_path.stem] = {
                        "clip": clip_path.name,
                        "source": str(source_path.relative_to(self.project_dir)),
                        "clip_duration": round(probe_duration(source_path), 3),
                        "rap_start": round(rap_start, 3),
                        "rap_end": round(rap_end, 3),
                        "rap_duration": round(max(0.001, rap_end - rap_start), 3),
                        "transcript_raw": " ".join(w["word"] for w in raw_words),
                        "transcript_empty": not bool(raw_words),
                        "lyrics_window": clip_lyrics,
                        "lyrics_match_score": lyrics_match_score,
                        "transcript_fixed": " ".join(w["word"] for w in fixed_words),
                        "words": fixed_words,
                        # Syrová (Whisper) slova vč. časů, uložená zvlášť od "words" (což jsou
                        # už slova opravená podle lyrics.txt). Díky tomu lze později — po ruční
                        # úpravě "transcript_raw" a/nebo "rap_start"/"rap_end" v tomto souboru —
                        # znovu spustit jen vyhledání lyrics_window (resync-rap), BEZ nutnosti
                        # znovu pouštět Whisper transkripci.
                        "words_raw": raw_words,
                        "song_match": match,
                    }
                    print(f"  [{idx:02d}/{len(rap_clips):02d}] {clip_path.stem}: rap {rap_start:.2f}-{rap_end:.2f}s, "
                          f"shoda se songem {match.get('score', 0):.2f}, shoda s lyrics.txt {lyrics_match_score:.2f}")
                except Exception as e:
                    print(f"  [{idx:02d}/{len(rap_clips):02d}] ⚠️ {clip_path.stem}: zpracování selhalo ({e}), přeskočeno.")
                    failed_clips.append(clip_path.stem)
                finally:
                    # Průběžný zápis — pád/chyba u jednoho klipu nezahodí výsledky ostatních.
                    self._write_json(self.edit_dir / "rap_alignment.json", results)

        print(f"✅ Vytvořen report: {(self.edit_dir / 'rap_alignment.json').relative_to(self.project_dir)}")
        if failed_clips:
            print(f"⚠️  {len(failed_clips)} klip(ů) bez validní transkripce (viz 'transcript_empty' v reportu): {', '.join(failed_clips)}")

    def _reconcile_raw_words_for_resync(self, entry: dict) -> list[dict]:
        """Pomocná funkce pro resync_rap_alignment_from_lyrics().

        Z aktuálního obsahu jednoho záznamu v rap_alignment.json (tak, jak ho
        uživatel případně ručně upravil) sestaví seznam "syrových" slov se
        start/end časy, který lze poslat do _best_lyrics_window_scored() /
        _align_words_to_lyrics() — beze změny už uloženou 'words_raw' (raw
        Whisper výstup), pokud ji uživatel fakticky nezměnil.

        Rozlišuje dvě situace, které uživatel v souboru může udělat:
          1) Upraví text 'transcript_raw' (např. opraví přeslechnuté slovo
             "LAV" → "love") beze změny 'rap_start'/'rap_end' → počet/pořadí
             slov v 'words_raw' se neshoduje s novým textem, takže se časy
             nově rozprostřou rovnoměrně do PŮVODNÍHO časového rozsahu
             (start prvního, end posledního uloženého slova).
          2) Upraví jen časování 'rap_start'/'rap_end' (např. zjistí, že rap
             ve videu začíná/končí jinde) beze změny textu → text zůstává
             stejný, ale existující slova ('words_raw') se PROPORCIONÁLNĚ
             přeškálují do nového časového rozsahu.
        Obojí lze samozřejmě upravit najednou.
        """
        transcript_raw = (entry.get("transcript_raw") or "").strip()
        text_words = transcript_raw.split()
        stored = entry.get("words_raw") or entry.get("words") or []
        rap_start = entry.get("rap_start")
        rap_end = entry.get("rap_end")

        if not text_words:
            return [dict(w) for w in stored]

        same_text = (
            len(stored) == len(text_words)
            and all(str(sw.get("word", "")) == tw for sw, tw in zip(stored, text_words))
        )

        if same_text:
            span_start = float(stored[0]["start"])
            span_end = float(stored[-1]["end"])
            timing_changed = (
                rap_start is not None and rap_end is not None
                and rap_end > rap_start
                and (abs(span_start - float(rap_start)) > 0.01 or abs(span_end - float(rap_end)) > 0.01)
            )
            if timing_changed:
                old_span = max(0.001, span_end - span_start)
                new_span = float(rap_end) - float(rap_start)
                scale = new_span / old_span
                return [
                    {
                        "word": w["word"],
                        "start": round(float(rap_start) + (float(w["start"]) - span_start) * scale, 3),
                        "end": round(float(rap_start) + (float(w["end"]) - span_start) * scale, 3),
                    }
                    for w in stored
                ]
            return [dict(w) for w in stored]

        # Text se změnil (jiná slova a/nebo jejich počet) — časy nelze převzít
        # 1:1, rozprostřeme je rovnoměrně do dostupného časového rozsahu:
        # přednostně 'rap_start'/'rap_end' (pokud je uživatel zadal/upravil),
        # jinak původní rozsah z 'words_raw', jinak odhad podle délky textu.
        if rap_start is not None and rap_end is not None and float(rap_end) > float(rap_start):
            start, end = float(rap_start), float(rap_end)
        elif stored:
            start, end = float(stored[0]["start"]), float(stored[-1]["end"])
        else:
            start, end = 0.0, max(len(text_words) * 0.3, 1.0)
        if end <= start:
            end = start + max(len(text_words) * 0.3, 0.5)
        step = (end - start) / len(text_words)
        return [
            {"word": w, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3)}
            for i, w in enumerate(text_words)
        ]

    def resync_rap_alignment_from_lyrics(self):
        """Znovu vyhledá lyrics_window/transcript_fixed (a dopočítá navazující pole)
        pro všechny klipy v EDIT_PROJECT/rap_alignment.json — BEZ nové Whisper
        transkripce. Určeno pro použití PO ruční úpravě souboru: pokud v něm
        opravíš 'transcript_raw' (přeslechnuté slovo) a/nebo 'rap_start'/'rap_end'
        (posun časování), tato volba podle toho přepočítá zbytek záznamu
        (lyrics_window, lyrics_match_score, transcript_fixed, words, song_match).

        'clip', 'source', 'clip_duration' a 'transcript_empty' se nemění.
        """
        rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})
        if not rap_alignment:
            print("❌ Chybí EDIT_PROJECT/rap_alignment.json. Nejprve spusť transcribe-rap (volba 5 / 4 v klipy.py).")
            return False

        lyrics_text = self._load_lyrics_text()
        if not lyrics_text.strip():
            print("❌ INPUT/lyrics.txt je prázdný nebo neexistuje — bez něj nelze lyrics_window dohledat.")
            return False

        timeline_ranges = self._load_timeline_rap_ranges()
        song_segments = self._load_song_segments()
        song_alignment = self._load_json(self.edit_dir / "song_alignment.json", {})
        song_words = song_alignment.get("words", [])

        used_song_matches = set()
        # Klipy, které už mají platnou pozici v songu (nezměněné), zabereme jako
        # obsazené jako první, aby si nově přepočítávané klipy nezabraly stejné
        # místo v songu jako klip, který zrovna neupravujeme.
        for name, data in rap_alignment.items():
            sm = (data.get("song_match") or {}).get("song_start")
            if sm is not None:
                used_song_matches.add(round(sm, 1))

        changed = 0
        for name, data in rap_alignment.items():
            raw_words = self._reconcile_raw_words_for_resync(data)
            if not raw_words:
                print(f"  ⚠️  {name}: chybí 'transcript_raw'/'words_raw' — přeskočeno.")
                continue

            hint_text = ""
            clip_range = timeline_ranges.get(name)
            if clip_range and song_segments:
                hint_text = self._song_text_for_range(song_segments, clip_range[0], clip_range[1])

            clip_lyrics, lyrics_match_score = self._best_lyrics_window_scored(raw_words, lyrics_text, hint_text)
            fixed_words = self._align_words_to_lyrics(raw_words, clip_lyrics)
            rap_start = fixed_words[0]["start"] if fixed_words else raw_words[0]["start"]
            rap_end = fixed_words[-1]["end"] if fixed_words else raw_words[-1]["end"]

            old_song_start = (data.get("song_match") or {}).get("song_start")
            if old_song_start is not None:
                used_song_matches.discard(round(old_song_start, 1))
            preferred_start = clip_range[0] if clip_range else None
            match = self._best_song_match(fixed_words, song_words, preferred_start, used_song_matches) if song_words else {}
            if match.get("song_start") is not None:
                used_song_matches.add(round(match["song_start"], 1))

            old_window = data.get("lyrics_window", "")
            data["lyrics_window"] = clip_lyrics
            data["lyrics_match_score"] = lyrics_match_score
            data["transcript_fixed"] = " ".join(w["word"] for w in fixed_words)
            data["rap_start"] = round(rap_start, 3)
            data["rap_end"] = round(rap_end, 3)
            data["rap_duration"] = round(max(0.001, rap_end - rap_start), 3)
            data["words"] = fixed_words
            data["words_raw"] = raw_words
            data["song_match"] = match

            changed += 1
            marker = "🔄" if clip_lyrics != old_window else "· "
            print(f"  {marker} {name}: lyrics_window = \"{clip_lyrics}\" (shoda {lyrics_match_score:.2f})")

        self._write_json(self.edit_dir / "rap_alignment.json", rap_alignment)
        print(f"✅ Přepočítáno {changed}/{len(rap_alignment)} klipů v rap_alignment.json (bez re-transkripce).")
        print("   👉 Pokud ses spokojen(a) s výsledkem, pokračuj volbou pro dolazení tempa (align-rap) "
              "a/nebo přepočet timeline (update-timeline).")
        return True

    def _adjust_rap_clip_segments(self, input_path: Path, output_path: Path, rap_start: float, rap_end: float, speed: float) -> bool:
        """Upraví jen rap segment klipu, části před a po rapu nechá rychlostí 1.0."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clip_dur = probe_duration(input_path)
        rap_start = max(0.0, min(rap_start, clip_dur))
        rap_end = max(rap_start, min(rap_end, clip_dur))
        with tempfile.TemporaryDirectory(prefix="rap_segments_", dir="/tmp") as tmp:
            tmpdir = Path(tmp)
            parts = []
            specs = [
                ("pre", 0.0, rap_start, 1.0),
                ("rap", rap_start, rap_end, speed),
                ("post", rap_end, clip_dur, 1.0),
            ]
            for name, start, end, spd in specs:
                dur = end - start
                if dur <= 0.03:
                    continue
                part = tmpdir / f"{name}.mp4"
                filters = ["setpts=PTS"] if abs(spd - 1.0) < 0.001 else [f"setpts=PTS/{spd}"]
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                    "-i", str(input_path), "-vf", ",".join(filters), "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(part)
                ]
                if run_ffmpeg(cmd):
                    parts.append(part)

            if not parts:
                return False
            if len(parts) == 1:
                shutil.copy2(str(parts[0]), str(output_path))
                return True
            concat_file = tmpdir / "concat.txt"
            concat_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8")
            return run_ffmpeg([
                "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_file), "-c", "copy", str(output_path)
            ])

    def _adjust_rap_clip_nonlinear(self, input_path: Path, output_path: Path, anchors: list, duration: float, speed_min=0.5, speed_max=2.0) -> bool:
        """Nelineární úprava rychlosti klipu podle kotevních bodů (clip_t, song_t)."""
        if not anchors:
            return False

        # Přidáme začátek a konec pro plynulost
        full_anchors = []
        first_clip_t, first_song_t = anchors[0]
        if first_clip_t > 0.05:
            full_anchors.append((0.0, first_song_t - first_clip_t))
        full_anchors.extend(anchors)
        last_clip_t, last_song_t = full_anchors[-1]
        if last_clip_t < duration - 0.05:
            full_anchors.append((duration, last_song_t + (duration - last_clip_t)))

        parts = []
        for i in range(len(full_anchors) - 1):
            t1, s1 = full_anchors[i]
            t2, s2 = full_anchors[i+1]
            dt = t2 - t1
            ds = s2 - s1
            if dt <= 0.01: continue

            # Ochrana proti dělení nulou a nesmyslným rychlostem
            if ds <= 0.001:
                speed = speed_max
            else:
                speed = round(max(speed_min, min(speed_max, dt / ds)), 4)

            parts.append({"start": t1, "end": t2, "speed": speed})

        if not parts: return False

        with tempfile.TemporaryDirectory(prefix="nonlinear_rap_", dir=str(self.project_dir)) as tmp:
            tmp_path = Path(tmp)
            concat_list = []
            for i, p in enumerate(parts):
                seg_path = tmp_path / f"seg_{i:03d}.mp4"
                dur = p["end"] - p["start"]
                if dur < 0.01: continue
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-ss", f"{p['start']:.3f}", "-t", f"{dur:.3f}",
                    "-i", str(input_path), "-vf", f"setpts=PTS/{p['speed']:.4f},scale=1280:720,fps=30",
                    "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", str(seg_path)
                ]
                if run_ffmpeg(cmd):
                    concat_list.append(seg_path)

            if not concat_list: return False
            concat_file = tmp_path / "concat.txt"
            concat_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in concat_list), encoding="utf-8")
            return run_ffmpeg(["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)])

    def align_rap_clips(self):
        """
        Spočítá přesnou rychlost rapové části každého klipu tak, aby délka rapu ve
        videu odpovídala délce odpovídajícího úseku v cílovém songu (song_match).

        Klip se dělí na tři části: Before (0..rap_start), Rap (rap_start..rap_end)
        a After (rap_end..clip_duration). Rychlost se mění výhradně u prostřední
        (Rap) části — Before i After zůstávají beze změny, takže začátek i konec
        klipu jsou časově zachované. Po přepočtu se navíc synchronizuje časování
        jednotlivých slov v poli `words` a do rap_alignment.json se zapíšou
        rozšířená metadata (before_duration, after_duration, target_rap_duration,
        speed_factor, new_clip_duration).
        """
        rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})
        if not rap_alignment:
            print("❌ Chybí EDIT_PROJECT/rap_alignment.json. Nejprve spusť transcribe-rap.")
            return

        backup_dir = self.project_dir / "gen_rap_original_backup"
        adjusted_dir = self.project_dir / "gen_rap_adjusted"
        backup_dir.mkdir(parents=True, exist_ok=True)
        adjusted_dir.mkdir(parents=True, exist_ok=True)

        settings = self.load_settings()
        speed_min = float(settings.get("speed_min", 0.5))
        speed_max = float(settings.get("speed_max", 2.0))

        # OPRAVA: stejná pojistka jako v align_vid_clips — pozná podle hashe posledního
        # vlastního výstupu, jestli uživatel gen_rap/{name}.mp4 mezitím ručně nahradil
        # novým obsahem, a pokud ano, osvěží zálohu místo slepého použití staré.
        prev_corrections = self._load_json(self.edit_dir / "rap_alignment.json", {})

        corrections = {}
        for name, data in rap_alignment.items():
            src_current = self.gen_rap / f"{name}.mp4"
            src_backup = backup_dir / f"{name}.mp4"
            current_hash = file_hash(src_current) if src_current.exists() else ""
            prev_output_hash = (prev_corrections.get(name) or {}).get("output_hash")
            if not src_backup.exists():
                if src_current.exists():
                    shutil.copy2(str(src_current), str(src_backup))
            elif prev_output_hash and current_hash and current_hash != prev_output_hash:
                print(f"  {name}: ℹ️  zjištěn nový obsah v gen_rap/ — osvěžuji zdrojovou zálohu.")
                shutil.copy2(str(src_current), str(src_backup))
            src = src_backup if src_backup.exists() else src_current

            match = data.get("song_match", {}) or {}
            song_start = match.get("song_start")
            song_end = match.get("song_end")

            if song_start is None or song_end is None or not src.exists():
                corrections[name] = {"status": "no_match", "speed_factor": 1.0, "new_duration": float(data.get("clip_duration", 0.0))}
                print(f"  {name}: speed=1.0000, status=no_match")
                continue

            try:
                # Vždy přepočítat ze skutečného souboru — nikdy nedůvěřovat cachované
                # clip_duration, protože tu align_rap_clips sám o pár řádků níž přepisuje
                # na "novou" délku po přepočtu. Bez re-probe by druhý běh nad stejným
                # (nedotčeným) backup souborem počítal s už jednou upravenou délkou.
                clip_duration = probe_duration(src)
                rap_start = float(data.get("rap_start", 0.0))
                rap_end = float(data.get("rap_end", clip_duration))

                before_duration = max(0.0, rap_start)
                source_rap_duration = max(0.001, rap_end - rap_start)
                after_duration = max(0.0, clip_duration - rap_end)
                target_rap_duration = max(0.001, float(song_end) - float(song_start))

                raw_speed = source_rap_duration / target_rap_duration
                speed = round(max(speed_min, min(speed_max, raw_speed)), 4)
                was_clamped = abs(speed - raw_speed) > 0.0005

                out = adjusted_dir / f"{name}.mp4"
                ok = self._adjust_rap_clip_segments(src, out, rap_start, rap_end, speed)

                # Skutečně dosažená délka rap segmentu po (případně oříznuté) rychlosti —
                # ne teoretický cíl, který se při clampingu speed_min/speed_max nedosáhne.
                achieved_rap_duration = round(source_rap_duration / speed, 4)
                new_clip_duration = round(before_duration + achieved_rap_duration + after_duration, 4)

                if ok:
                    shutil.copy2(str(out), str(self.gen_rap / f"{name}.mp4"))

                    # Synchronizace časování slov: lineární transformace vzhledem k rap_start
                    words = data.get("words", [])
                    new_words = []
                    for w in words:
                        nw = dict(w)
                        old_start = float(w.get("start", rap_start))
                        old_end = float(w.get("end", old_start))
                        nw["start"] = round(rap_start + (old_start - rap_start) / speed, 4)
                        nw["end"] = round(rap_start + (old_end - rap_start) / speed, 4)
                        new_words.append(nw)
                    data["words"] = new_words

                    data["rap_duration"] = achieved_rap_duration
                    data["clip_duration"] = new_clip_duration
                    data["before_duration"] = round(before_duration, 4)
                    data["after_duration"] = round(after_duration, 4)
                    data["target_rap_duration"] = round(target_rap_duration, 4)
                    data["speed_factor"] = speed
                    data["new_clip_duration"] = new_clip_duration
                    # OPRAVA: hash našeho vlastního výstupu — příští běh podle něj pozná,
                    # jestli uživatel gen_rap/{name}.mp4 mezitím ručně nahradil.
                    data["output_hash"] = file_hash(self.gen_rap / f"{name}.mp4")

                    status = "clamped" if was_clamped else "ok"
                    if was_clamped:
                        print(f"  ⚠️ {name}: požadovaná rychlost {raw_speed:.3f}x přesáhla limit "
                              f"[{speed_min},{speed_max}], reálná délka rapu {achieved_rap_duration:.3f}s "
                              f"≠ cíl {target_rap_duration:.3f}s")
                else:
                    status = "failed"

                corrections[name] = {
                    "status": status,
                    "speed_factor": speed,
                    "new_duration": new_clip_duration if ok else round(clip_duration, 4),
                }
                print(f"  {name}: speed={speed:.4f}, status={status}  (rap {source_rap_duration:.3f}s → {achieved_rap_duration:.3f}s)")
            except Exception as e:
                print(f"  ⚠️ {name}: zpracování selhalo ({e}), přeskočeno.")
                corrections[name] = {"status": "error", "speed_factor": 1.0, "new_duration": float(data.get("clip_duration", 0.0))}

            # Průběžný zápis po každém klipu — pád na jednom souboru už nezahodí
            # výsledky pro všechny ostatní klipy zpracované v tomto běhu.
            self._write_json(self.edit_dir / "rap_alignment.json", rap_alignment)
            self._write_json(self.edit_dir / "speed_corrections.json", corrections)

        print("✅ Rap klipy synchronizovány na délku odpovídajícího úseku v songu (rap_alignment.json aktualizován).")

    def align_vid_clips(self) -> dict:
        """
        Zkontroluje skutečnou (fyzickou) délku každého vid_xx broll klipu v gen_vid/
        oproti požadované délce jeho úseku v timeline.txt a klip fyzicky zrychlí
        nebo zpomalí (ffmpeg setpts), aby jeho délka přesně odpovídala přiřazenému
        časovému slotu.

        Před jakoukoliv úpravou se originál klipu vždy nejprve zazálohuje do
        gen_vid_original_backup/ (idempotentně — pokud záloha už existuje, znovu
        se nepřepisuje a další přepočty vycházejí vždy ze zálohovaného originálu,
        nikdy z už jednou přepočítaného souboru, aby se chyba speedu nekumulovala).
        Výsledek se uloží do EDIT_PROJECT/vid_speed_corrections.json.
        """
        if not self.timeline_file.exists():
            print("❌ Chybí EDIT_PROJECT/timeline.txt. Nejprve vygeneruj/aktualizuj timeline (volba 8 / update-timeline).")
            return {}

        # --- Načtení požadovaných (cílových) délek vid_xx klipů z timeline.txt ---
        target_durations = {}
        duplicate_names = set()
        for raw in self.timeline_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            time_range = parts[0]
            clip_name = clean_asset_id(parts[1])
            if not clip_name.startswith("vid_") or "-" not in time_range:
                continue
            try:
                start_raw, end_raw = [x.strip() for x in time_range.split("-", 1)]
                start = parse_timecode(start_raw)
                end = parse_timecode(end_raw)
            except Exception:
                continue
            duration = max(0.001, end - start)
            if clip_name in target_durations:
                duplicate_names.add(clip_name)
            target_durations[clip_name] = duration

        if not target_durations:
            print("❌ V timeline.txt nebyly nalezeny žádné vid_xx klipy.")
            return {}

        if duplicate_names:
            print(
                "⚠️  Tyto vid klipy se v timeline.txt objevují vícekrát — použije se jejich poslední "
                f"výskyt: {', '.join(sorted(duplicate_names))}"
            )

        backup_dir = self.project_dir / "gen_vid_original_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        settings = self.load_settings()
        speed_min = float(settings.get("speed_min", 0.5))
        speed_max = float(settings.get("speed_max", 2.0))

        # OPRAVA: načteme předchozí výsledky, abychom pomocí uloženého hashe výstupu
        # poznali, jestli uživatel mezi běhy gen_vid/vid_XX.mp4 ručně nahradil novým
        # obsahem (pak se musí záloha v gen_vid_original_backup/ osvěžit), nebo jde
        # stále o náš vlastní výstup z minulého běhu (pak se použije stará záloha beze
        # změny, aby se rychlostní korekce nekumulovaly).
        prev_corrections = self._load_json(self.edit_dir / "vid_speed_corrections.json", {})

        corrections = {}
        print(f"🎞️ Kontroluji délku {len(target_durations)} vid_xx klipů oproti timeline.txt...")

        for clip_name, target_duration in sorted(target_durations.items()):
            try:
                src_current = self.gen_vid / f"{clip_name}.mp4"
                src_backup = backup_dir / f"{clip_name}.mp4"

                if not src_current.exists() and not src_backup.exists():
                    corrections[clip_name] = {
                        "status": "missing",
                        "speed_factor": 1.0,
                        "target_duration": round(target_duration, 4),
                    }
                    print(f"  {clip_name}: ❌ chybí soubor v gen_vid/")
                    continue

                # OPRAVA (bylo: záloha se dělala jen jednou a pak se natrvalo používala,
                # i když uživatel gen_vid/vid_XX.mp4 mezitím nahradil novým obsahem —
                # výsledkem bylo, že se nový klip při dalším align-vid přepsal zpět
                # starým obsahem ze zálohy, jen rychlostně upraveným).
                # Nyní: pokud aktuální soubor NEODPOVídá hashi našeho posledního
                # vlastního výstupu (uloženému v corrections['output_hash']), znamená
                # to, že ho uživatel mezi běhy ručně nahradil — zálohu proto osvěžíme
                # aktuálním (novým) obsahem, aby se z ní opravdu vycházelo.
                current_hash = file_hash(src_current) if src_current.exists() else ""
                prev_output_hash = (prev_corrections.get(clip_name) or {}).get("output_hash")

                if not src_backup.exists():
                    if src_current.exists():
                        shutil.copy2(str(src_current), str(src_backup))
                elif prev_output_hash and current_hash and current_hash != prev_output_hash:
                    print(f"  {clip_name}: ℹ️  zjištěn nový obsah v gen_vid/ (liší se od minulého výstupu) — osvěžuji zdrojovou zálohu.")
                    shutil.copy2(str(src_current), str(src_backup))
                src = src_backup if src_backup.exists() else src_current

                actual_duration = probe_duration(src)
                if actual_duration <= 0:
                    corrections[clip_name] = {
                        "status": "invalid_media",
                        "speed_factor": 1.0,
                        "target_duration": round(target_duration, 4),
                    }
                    print(f"  {clip_name}: ❌ neplatné médium, nelze zjistit délku")
                    continue

                # Bezpečnostní rezerva: cílíme mírně NAD skutečnou požadovanou délku,
                # aby výsledný klip byl vždy o pár desítek ms DELŠÍ než potřeba (kvůli
                # zaokrouhlení na celé snímky). Render krok si přebytek ořízne pomocí
                # -t, takže je to vždy bezpečnější než klip, který by vyšel kratší.
                safety_margin = 0.05
                effective_target = target_duration + safety_margin

                raw_speed = actual_duration / effective_target
                speed = round(max(speed_min, min(speed_max, raw_speed)), 4)

                ok = self._adjust_rap_clip_speed(src, src_current, speed)
                new_duration = probe_duration(src_current) if ok else actual_duration
                status = "ok" if ok else "failed"

                corrections[clip_name] = {
                    "status": status,
                    "speed_factor": speed,
                    "actual_duration": round(actual_duration, 4),
                    "target_duration": round(target_duration, 4),
                    "new_duration": round(new_duration, 4),
                    # OPRAVA: hash našeho vlastního výstupu — příští běh podle něj pozná,
                    # jestli uživatel gen_vid/{clip_name}.mp4 mezitím ručně nahradil.
                    "output_hash": file_hash(src_current) if ok else prev_output_hash,
                }

                clamp_note = ""
                if not (speed_min <= raw_speed <= speed_max):
                    clamp_note = f"  ⚠️ výpočtem vyšla rychlost {raw_speed:.4f}x, oříznuto na limit {speed_min}-{speed_max}"

                print(
                    f"  {clip_name}: speed={speed:.4f}, status={status}  "
                    f"({actual_duration:.3f}s → {new_duration:.3f}s, cíl {target_duration:.3f}s){clamp_note}"
                )
            except Exception as e:
                print(f"  {clip_name}: ⚠️ zpracování selhalo ({e}), přeskočeno.")
                corrections[clip_name] = {"status": "error", "speed_factor": 1.0, "target_duration": round(target_duration, 4)}
            finally:
                # Průběžný zápis — pád na jednom klipu nezahodí výsledky ostatních.
                self._write_json(self.edit_dir / "vid_speed_corrections.json", corrections)

        ok_count = sum(1 for c in corrections.values() if c.get("status") == "ok")
        print(f"\n✅ Hotovo — {ok_count}/{len(corrections)} vid_xx klipů přepočítáno na požadovanou délku z timeline.txt.")
        print(f"   Originály jsou zálohované v: {backup_dir.relative_to(self.project_dir)}")
        return corrections

    def update_timeline_from_alignment(self):
        """Přepočítá timeline: ignoruje časy z plánu, ukotví rap_xx na pozice v songu, vyplní mezery B-rolly."""
        corrections = self._load_json(self.edit_dir / "speed_corrections.json", {})
        if not corrections:
            print("❌ Chybí speed_corrections.json. Spusť align-rap."); return

        # rap_start (offset uvnitř klipu, kde reálně začíná verš) a song_start (fallback
        # pozice v songu ze song_match) se do speed_corrections.json NIKDY nezapisují —
        # tato data žijí jen v rap_alignment.json. Bez něj by corr.get("rap_start"/"song_start")
        # vždy spadl na default a "ukotvení podle rap_start" by fakticky nikdy neproběhlo.
        rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})

        if not self.timeline_file.exists():
            if self.full_plan.exists(): self.parse_plan()
            else: print("❌ Chybí timeline."); return

        song_alignment = self._load_json(self.edit_dir / "song_alignment.json", {})
        song_words = song_alignment.get("words", [])
        audio_dur = probe_duration(self.find_audio()) if self.find_audio() else 180.0

        clip_max_durations = {}
        for folder, patterns in [(self.gen_rap, ["*.mp4"]), (self.gen_char, ["*.mp4"]), (self.gen_vid, ["*.mp4"]), (self.gen_pic, ["*.mp4", "*.png", "*.jpg", "*.jpeg"])]:
            if folder.exists():
                for p in patterns:
                    for f in folder.glob(p):
                        clip_max_durations[f.stem] = 5.0 if f.suffix.lower() in (".png", ".jpg", ".jpeg") else probe_duration(f)

        entries = []
        for raw in self.timeline_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "|" not in line:
                entries.append({"type": "text", "raw": raw}); continue
            parts = [p.strip() for p in line.split("|")]
            clip_name = clean_asset_id(parts[1])
            try: old_start = parse_timecode(parts[0].split("-")[0].strip())
            except: old_start = 0.0
            entries.append({
                "type": "clip", "clip": clip_name, "is_rap": clip_name.startswith("rap_"),
                "note": " | ".join(parts[2:]), "old_start": old_start,
                "max_dur": clip_max_durations.get(clip_name, 8.0), "new_start": None, "new_end": None
            })

        used_song_positions = set()
        for entry in entries:
            if entry["type"] != "clip" or not entry["is_rap"]: continue
            corr = corrections.get(entry["clip"], {})
            if corr.get("status") not in ("ok", "clamped"): continue
            align_data = rap_alignment.get(entry["clip"], {})
            rap_start_offset = float(align_data.get("rap_start", 0.0))
            fallback_song_start = (align_data.get("song_match", {}) or {}).get("song_start")
            note_text = re.sub(r"\[[^\]]+\]", " ", entry["note"])
            song_start = self._find_unique_song_position(note_text, song_words, entry["old_start"], used_song_positions)
            if song_start is None:
                song_start = self._find_song_position_after(note_text, song_words, max(used_song_positions or [0]), used_song_positions)
            if song_start is None:
                song_start = float(fallback_song_start) if fallback_song_start is not None else entry["old_start"]
            used_song_positions.add(round(song_start, 1))
            entry["new_start"] = max(0.0, song_start - rap_start_offset)
            entry["new_end"] = entry["new_start"] + float(corr.get("new_duration", entry["max_dur"]))
            # Timeline už referencuje jen nový (fyzicky přepočítaný) klip — žádné
            # [SPEED x] / [SPEED_APPLIED] tagy se do poznámky nezapisují.
            entry["note"] = re.sub(r"\[SPEED.*?\]", "", entry["note"], flags=re.I).strip()

        current_time = 0.0; idx = 0
        while idx < len(entries):
            entry = entries[idx]
            if entry["type"] != "clip": idx += 1; continue
            if entry["is_rap"] and entry["new_start"] is not None:
                if entry["new_start"] < current_time - 0.01:
                    back = idx - 1
                    while back >= 0:
                        if entries[back]["type"] == "clip":
                            entries[back]["new_end"] = entry["new_start"]
                            if entries[back]["new_start"] is not None:
                                entries[back]["new_start"] = min(entries[back]["new_start"], entries[back]["new_end"])
                                if entries[back]["is_rap"] and entries[back]["new_end"] <= entries[back]["new_start"] + 0.01:
                                    print(f"⚠️  Kolize: {entries[back]['clip']} byl kvůli překryvu s {entry['clip']} zkrácen na ~0s a bude z timeline VYNECHÁN.")
                            if entries[back]["is_rap"]: break
                        back -= 1
                current_time = entry["new_end"]; idx += 1
            else:
                block = []; j = idx; gap_end = audio_dur
                while j < len(entries):
                    if entries[j]["type"] == "clip" and entries[j]["new_start"] is not None and (entries[j]["is_rap"] or j > idx):
                        gap_end = entries[j]["new_start"]; break
                    if entries[j]["type"] == "clip": block.append(j)
                    j += 1
                gap_dur = max(0.0, gap_end - current_time)
                if block:
                    each = gap_dur / len(block); t = current_time
                    for b_idx in block:
                        entries[b_idx]["new_start"] = t; entries[b_idx]["new_end"] = t + each; t += each
                current_time = gap_end; idx = j

        rebuilt = []
        for e in entries:
            if e["type"] == "text": rebuilt.append(e["raw"])
            elif e["new_start"] is not None and e["new_end"] > e["new_start"] + 0.01:
                rebuilt.append(f"{format_timecode(e['new_start'])} - {format_timecode(e['new_end'])} | {e['clip']} | {e['note']}")
        self.timeline_file.write_text("\n".join(rebuilt), encoding="utf-8")
        print(f"✅ Timeline přepočítána (Sequential Reflow pass v3).")

    def apply_speeds_from_timeline(self):
        """
        Dopočítá rychlost rap klipů podle časových slotů, které už jsou napevno
        zapsané v EDIT_PROJECT/timeline.txt — beze změny pořadí, časů nebo počtu řádků.

        Na rozdíl od align-rap + update-timeline (které přepočítají POZICI rapu podle
        songu a pak přerovnají celou timeline), tahle funkce bere timeline.txt jako
        zdroj pravdy a jen dopočítá, jak moc je potřeba zrychlit/zpomalit "rapovanou"
        střední část každého rap_xx klipu, aby přesně vyplnila svůj slot. Ticho
        před/po rapu zůstává na rychlosti 1.0x (řeší _adjust_rap_clip_segments).
        """
        if not self.timeline_file.exists():
            print("❌ Chybí EDIT_PROJECT/timeline.txt.")
            return

        rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})
        if not rap_alignment:
            print("ℹ️  Chybí EDIT_PROJECT/rap_alignment.json — nejprve spouštím transkripci rap klipů (transcribe-rap)...")
            self.transcribe_rap_clips()
            rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})
            if not rap_alignment:
                print("❌ Transkripce se nezdařila, nelze pokračovat.")
                return

        settings = self.load_settings()
        speed_min = float(settings.get("speed_min", 0.5))
        speed_max = float(settings.get("speed_max", 2.0))

        backup_dir = self.project_dir / "gen_rap_original_backup"
        adjusted_dir = self.project_dir / "gen_rap_adjusted"
        backup_dir.mkdir(parents=True, exist_ok=True)
        adjusted_dir.mkdir(parents=True, exist_ok=True)

        corrections = self._load_json(self.edit_dir / "speed_corrections.json", {})

        raw_lines = self.timeline_file.read_text(encoding="utf-8").splitlines()
        rebuilt = []
        processed = 0

        print("🎚️  Přepočítávám rychlosti rap klipů podle timeline.txt (časy a pořadí zůstávají beze změny)...")

        for raw in raw_lines:
            line = raw.strip()
            if not line or line.startswith("#") or "|" not in line:
                rebuilt.append(raw)
                continue

            parts = [p.strip() for p in line.split("|")]
            clip_name = clean_asset_id(parts[1]) if len(parts) > 1 else ""
            if not clip_name.startswith("rap_"):
                rebuilt.append(raw)
                continue

            try:
                bounds = parts[0].split("-")
                t_start = parse_timecode(bounds[0].strip())
                t_end = parse_timecode(bounds[1].strip())
            except Exception:
                rebuilt.append(raw)
                continue
            target_dur = max(0.05, t_end - t_start)

            src_current = self.gen_rap / f"{clip_name}.mp4"
            src_backup = backup_dir / f"{clip_name}.mp4"
            if not src_current.exists() and not src_backup.exists():
                print(f"  ⚠️  {clip_name}: soubor klipu nenalezen (gen_rap), řádek beze změny.")
                rebuilt.append(raw)
                continue
            if src_current.exists() and not src_backup.exists():
                shutil.copy2(str(src_current), str(src_backup))
            src = src_backup if src_backup.exists() else src_current

            data = rap_alignment.get(clip_name)
            clip_dur = probe_duration(src)
            if data:
                rap_start = min(clip_dur, max(0.0, float(data.get("rap_start", 0.0))))
                rap_end = min(clip_dur, max(rap_start, float(data.get("rap_end", clip_dur))))
            else:
                print(f"  ⚠️  {clip_name}: chybí v rap_alignment.json — použiji celou délku klipu jako 'rap' část.")
                rap_start = 0.0
                rap_end = clip_dur

            rap_dur = max(0.05, rap_end - rap_start)
            try:
                speed = speed_for_slot(rap_dur, target_dur, speed_min, speed_max)
            except ValueError as exc:
                print(f"  ⚠️  {clip_name}: nelze vypočítat rychlost ({exc}), řádek beze změny.")
                rebuilt.append(raw)
                continue

            out = adjusted_dir / f"{clip_name}.mp4"
            ok = self._adjust_rap_clip_segments(src, out, rap_start, rap_end, speed)

            note_text = " | ".join(parts[2:]) if len(parts) > 2 else ""
            if ok:
                shutil.copy2(str(out), str(self.gen_rap / f"{clip_name}.mp4"))
                new_duration = probe_duration(out)
                corrections[clip_name] = {
                    "status": "ok",
                    "speed_factor": speed,
                    "new_duration": round(new_duration, 3),
                }
                processed += 1
                # Timeline používá už jen nový (fyzicky přepočítaný) klip —
                # žádné [SPEED x] / [SPEED_APPLIED] tagy se do poznámky nezapisují.
                note = re.sub(r"\[SPEED.*?\]", "", note_text, flags=re.I).strip()
                rebuilt.append(f"{parts[0].strip()} | {clip_name} | {note}")
                print(f"  ✅ {clip_name}: speed={speed:.4f}x  (slot {target_dur:.2f}s, rap {rap_dur:.2f}s)")
            else:
                corrections[clip_name] = {"status": "failed", "speed_factor": speed, "new_duration": 0.0}
                print(f"  ❌ {clip_name}: úprava rychlosti selhala (viz chyba FFmpeg výše), řádek beze změny.")
                rebuilt.append(raw)

        self._write_json(self.edit_dir / "speed_corrections.json", corrections)
        self.timeline_file.write_text("\n".join(rebuilt), encoding="utf-8")
        print(f"\n✅ Hotovo — upraveno {processed} rap klipů podle timeline.txt.")
        print("   Pořadí, časy i počet řádků v timeline.txt zůstaly beze změny.")
        print("   ⚠️  Krok 'update-timeline' (7) teď NESPOUŠTĚJ — přepsal by tohle časování.")
        print("   Pokračuj rovnou volbou 9 (validace) → 10 (render).")

    def contains_broken_encoding(self, text: str) -> bool:
        """Rozpozná typické rozbití UTF-8 / ASCII escape artefakty."""
        if not text:
            return True

        bad_patterns = (
            r"\d{3,}[a-fA-F]\d{2,}",   # M011bln00edk apod.
            r"00e[0-9a-fA-F]",        # častý hex artefakt
            r"\\u[0-9a-fA-F]{4}",     # nevyřešený unicode escape
        )
        return any(re.search(pattern, text) for pattern in bad_patterns)

    def sanitize_utf8_text(self, text: str) -> str:
        """Normalizuje skutečně validní Unicode, ale nepředstírá opravu dat."""
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize("NFC", text).replace("\ufeff", "").strip()

    def validate_transcription_integrity(self) -> bool:
        """Zastaví lip-sync pipeline, pokud je transkripce poškozená."""
        transfile = self.input_dir / "transcription.json"
        data = self._load_json(transfile, {})

        full_text = self.sanitize_utf8_text(str(data.get("text", "")))
        segments = data.get("segments", [])

        if not full_text or not segments:
            print("Transkripce neobsahuje text nebo segmenty.")
            return False

        if self.contains_broken_encoding(full_text):
            print("Transkripce má poškozené kódování. Spusťte novou českou transkripci.")
            return False

        top_level_words = data.get("words", [])
        if top_level_words:
            word_count = len(top_level_words)
        else:
            word_count = sum(len(seg.get("words", [])) for seg in segments)
        if word_count < 20:
            print("Transkripce nemá dostatek word-level timestampů.")
            return False

        return True

    def transcribe_song_czech(self):
        """
        Vytvoří novou transkripci s vynucenou češtinou.
        Nepoužívá existující transcription.json, pokud je poškozený.
        """
        audiopath = self.find_audio()
        if not audiopath:
            print("Audio nebylo nalezeno.")
            return False

        transfile = self.input_dir / "transcription.json"
        settings = self.load_settings()
        provider = str(settings.get("transcription_provider", "local")).lower()

        if provider == "groq":
            api_key = self._groq_ready()
            if not api_key:
                return False
            model = settings.get("groq_model", "whisper-large-v3-turbo")
            print(f"Vytvářím novou českou transkripci přes Groq Cloud API (model: {model})...")
            data = self._groq_transcribe_file(audiopath, settings, api_key)
            if not data:
                return False

            if self.contains_broken_encoding(str(data.get("text", ""))):
                print("Nová transkripce stále obsahuje poškozené znaky.")
                return False

            self._write_json(transfile, data)
            print(f"Česká transkripce (Groq) připravena: {transfile}")
            return True

        cmd = [
            "whisper", str(audiopath),
            "--model", settings.get("whisper_model", "large-v3"),
            *self._whisper_device_args(settings),
            "--language", "cs",
            "--task", "transcribe",
            "--word_timestamps", "True",
            "--output_format", "json",
            "--output_dir", str(self.input_dir),
        ]

        print("Vytvářím novou českou transkripci s word timestamps...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(result.stderr[-1500:])
            return False

        generated = self.input_dir / f"{audiopath.stem}.json"
        if not generated.exists():
            print("Whisper nevytvořil očekávaný JSON soubor.")
            return False

        raw = generated.read_text(encoding="utf-8", errors="strict")
        data = json.loads(raw)

        if self.contains_broken_encoding(str(data.get("text", ""))):
            print("Nová transkripce stále obsahuje poškozené znaky.")
            return False

        self._write_json(transfile, data)

        if generated != transfile and generated.exists():
            generated.unlink()

        print(f"Česká transkripce připravena: {transfile}")
        return True

    def export_lipsync_audio_segments(self):
        """Exportuje WAV pouze pro rapové položky v timeline."""
        if not self.validate_transcription_integrity():
            return False

        audio_path = self.find_audio()
        if not audio_path:
            print("Audio nebylo nalezeno.")
            return False

        if not self.timeline_file.exists():
            print("Chybí EDIT_PROJECT/timeline.txt.")
            return False

        output_dir = self.project_dir / "LIPSYNC_AUDIO"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Odstraní staré WAVy, aby se nemíchaly se starým manifestem.
        for old_wav in output_dir.glob("*.wav"):
            old_wav.unlink()

        manifest = []
        occurrences = {}

        for raw in self.timeline_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():
            line = raw.strip()

            if not line or line.startswith("#") or "|" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 2:
                continue

            time_range = parts[0]
            source_asset = clean_asset_id(parts[1])

            # Pouze původní rapové položky: rap01, rap02, rap10...
            if not re.fullmatch(r"rap\d{2}", source_asset, flags=re.IGNORECASE):
                continue

            try:
                start_text, end_text = [
                    value.strip()
                    for value in time_range.split("-", 1)
                ]
                start = parse_timecode(start_text)
                end = parse_timecode(end_text)
            except Exception:
                print(f"Neplatný čas v timeline: {line}")
                continue

            if end <= start:
                print(f"Neplatný rozsah v timeline: {line}")
                continue

            occurrences[source_asset] = occurrences.get(source_asset, 0) + 1
            occurrence = occurrences[source_asset]

            # Každý výskyt dostane vlastní asset.
            asset = f"{source_asset}_{occurrence:02d}"
            output_audio = output_dir / f"{asset}.wav"

            lyrics = " | ".join(parts[2:]).strip()

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", str(audio_path),
                "-ac", "1",
                "-ar", "48000",
                "-c:a", "pcm_s16le",
                str(output_audio),
            ]

            if not run_ffmpeg(cmd):
                print(f"Nepodařilo se vytvořit: {output_audio.name}")
                continue

            manifest.append({
                "asset": asset,
                "source_asset": source_asset,
                "occurrence": occurrence,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "duration_ms": round((end - start) * 1000),
                "audio": str(output_audio.relative_to(self.project_dir)),
                "lyrics": lyrics,
                "generation_rule": (
                    "Use this exact attached audio. Do not invent, replace, "
                    "translate, speed-change or paraphrase the spoken lyric."
                ),
            })

        if not manifest:
            print("V timeline nebyly nalezeny žádné rapové položky.")
            return False

        self._write_json(output_dir / "manifest.json", {
            "source_audio": str(audio_path.relative_to(self.project_dir)),
            "segments": manifest,
        })

        print(
            f"Exportováno {len(manifest)} WAV segmentů "
            "pouze pro rapové položky timeline."
        )
        return True

    def inject_lipsync_segments_into_timeline(self):
        """Nahradí rapové řádky v timeline přesnými segmenty z manifestu."""
        manifest_path = self.project_dir / "LIPSYNC_AUDIO" / "manifest.json"
        if not manifest_path.exists():
            print("❌ Manifest lip-sync segmentů nebyl nalezen. Spusť nejdříve prepare-lipsync.")
            return False

        manifest = self._load_json(manifest_path, {})
        segments = manifest.get("segments", [])

        if not segments:
            print("❌ Manifest lip-sync segmentů je prázdný.")
            return False

        original_lines = []
        if self.timeline_file.exists():
            original_lines = self.timeline_file.read_text(encoding="utf-8", errors="ignore").splitlines()

        manifest_assets = {str(segment.get("asset", "")) for segment in segments}

        # Odstraníme jak původní draftové rap01, tak i dříve vložené rap01_01 segmenty.
        kept = []
        for raw in original_lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                kept.append(raw)
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                asset_id = clean_asset_id(parts[1])
                # Mažeme rapXX i rapXX_YY
                if re.fullmatch(r"rap\d{2}(_\d{2})?", asset_id, flags=re.IGNORECASE):
                    continue
                # Odstraní také všechny segmenty definované v aktuálním manifestu pro aktualizaci
                if asset_id in manifest_assets:
                    continue
            kept.append(raw)

        lipsync_lines = []
        for segment in segments:
            asset = segment["asset"]
            start = float(segment["start"])
            end = float(segment["end"])
            lyric = str(segment.get("lyrics", "")).replace("|", " ")

            lipsync_lines.append(
                f"{format_timecode(start)} - {format_timecode(end)} | "
                f"{asset} | LIPSYNC EXACT | {lyric}"
            )

        merged = kept + lipsync_lines

        def timeline_start(line_text):
            try:
                if "|" not in line_text: return 10**9
                time_part = line_text.split("|", 1)[0].split("-", 1)[0].strip()
                return parse_timecode(time_part)
            except Exception:
                return 10**9

        merged.sort(key=timeline_start)

        self.timeline_file.write_text("\n".join(merged) + "\n", encoding="utf-8")
        print(f"✅ Timeline doplněna o {len(lipsync_lines)} segmentů z manifestu.")
        return True

    def validate_lipsync_assets(self) -> list[str]:
        """Ověří, že každý segment z manifestu má hotový odpovídající klip v gen_rap."""
        issues = []
        manifest_path = self.project_dir / "LIPSYNC_AUDIO" / "manifest.json"
        if not manifest_path.exists():
            return [
                "Chybí LIPSYNC_AUDIO/manifest.json. "
                "Spusťte nejdříve prepare-lipsync."
            ]
        manifest = self._load_json(manifest_path, {})
        segments = manifest.get("segments", [])

        if not segments:
            return ["Chybí LIPSYNC_AUDIO/manifest.json nebo nemá žádné segmenty."]

        for segment in segments:
            asset = segment.get("asset", "")
            expected = self.gen_rap / f"{asset}.mp4"

            if not is_valid_media(expected):
                issues.append(f"Chybí nebo je neplatný lip-sync klip: {asset}.mp4")
                continue

            actual = probe_duration(expected)
            target = segment.get("duration_ms")
            if target is None:
                target = segment.get("duration", 0)
            try:
                if "duration_ms" in segment:
                    actual_ms = round(actual * 1000)
                    drift_ms = abs(actual_ms - int(target))
                    if drift_ms > DEFAULT_DURATION_TOLERANCE_MS:
                        issues.append(
                            f"{asset}: délka {actual_ms / 1000:.3f}s neodpovídá "
                            f"očekávaným {int(target) / 1000:.3f}s "
                            f"(odchylka {drift_ms} ms)."
                        )
                else:
                    drift_issue = validate_duration_drift(
                        asset, actual, target, DEFAULT_DURATION_TOLERANCE_MS
                    )
                    if drift_issue:
                        issues.append(drift_issue.message())
            except (TypeError, ValueError) as exc:
                issues.append(f"{asset}: neplatná délka v manifestu ({exc})")

        timeline_text = self.timeline_file.read_text(encoding="utf-8", errors="ignore") if self.timeline_file.exists() else ""
        for segment in segments:
            asset = str(segment.get("asset", ""))
            if not re.search(rf"(?m)^\s*[^|\n]+\|\s*{re.escape(asset)}\s*\|", timeline_text):
                issues.append(f"{asset} není v timeline.txt. Spusťte inject-lipsync.")

        return issues

    def validate_project(self, final: bool = False, no_rap: bool = False) -> bool:
        """Provede validační bránu projektu podle new_pipeline.txt."""
        self.logger.info("Validuji projekt; final=%s, no_rap=%s", final, no_rap)
        issues = []

        if self.timeline_file.exists() and self.timeline_file.stat().st_size > 0:
            timeline_text = self.timeline_file.read_text(encoding="utf-8", errors="replace")
            if final and "[ODHAD DÉLKY]" in timeline_text:
                issues.append("Timeline obsahuje odhadovanou délku klipu; final render vyžaduje změřené délky")
            entries, parse_warnings = parse_timeline_entries(timeline_text)
            issues.extend(f"Timeline: {warning}" for warning in parse_warnings)
            audio_path = self.find_audio()
            song_duration = probe_duration(audio_path) if audio_path else None
            issues.extend(validate_timeline(entries, song_duration=song_duration))

        if final:
            if not no_rap:
                issues.extend(self.validate_lipsync_assets())
            if not self.validate_transcription_integrity():
                issues.append(
                    "Transcription.json je neplatný pro lip-sync "
                    "(poškozené znaky nebo chybějící word timestamps)."
                )
        else:
            if not no_rap:
                settings = self.load_settings()
                speed_min = float(settings.get("speed_min", 0.5))
                speed_max = float(settings.get("speed_max", 2.0))

                corrections = self._load_json(self.edit_dir / "speed_corrections.json", {})
                for name, corr in corrections.items():
                    speed = float(corr.get("speed_factor", 1.0))
                    # Pozn.: speed_factor != 1.0 už NENÍ problém — architektura vždy
                    # generuje fyzicky přepočítaný klip (before+adjusted_rap+after),
                    # takže odchylka od 1.0 je očekávaná a žádané chování.
                    if not speed_min <= speed <= speed_max:
                        issues.append(f"{name}: speed_factor mimo limit {speed_min}-{speed_max} ({speed})")
                    if corr.get("status") != "ok":
                        issues.append(f"{name}: korekce není OK ({corr.get('status')})")

                # --- Vazba na text: rapové texty z klipů se musí nacházet v lyrics.txt ---
                rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})
                MIN_LYRICS_MATCH = 0.35
                for name, data in rap_alignment.items():
                    score = data.get("lyrics_match_score")
                    if score is None:
                        continue  # starší report bez skóre, nelze ověřit
                    if score < MIN_LYRICS_MATCH:
                        issues.append(
                            f"{name}: text klipu nebyl s dostatečnou jistotou nalezen v lyrics.txt "
                            f"(shoda {score:.2f} < {MIN_LYRICS_MATCH})"
                        )

        for path in [self.input_dir / "lyrics.txt", self.timeline_file]:
            if not path.exists() or path.stat().st_size == 0:
                issues.append(f"Chybí nebo je prázdný soubor: {path.relative_to(self.project_dir)}")

        # Společné kontroly překryvů a délek
        ranges = []
        if self.timeline_file.exists():
            for raw in self.timeline_file.read_text(encoding="utf-8").splitlines():
                if "|" not in raw or "-" not in raw:
                    continue
                try:
                    left = raw.split("|", 1)[0]
                    start, end = [parse_timecode(x.strip()) for x in left.split("-", 1)]
                    ranges.append((start, end, raw.strip()))
                except Exception:
                    issues.append(f"Neplatný řádek timeline: {raw.strip()}")
        ranges.sort()
        for prev, cur in zip(ranges, ranges[1:]):
            if cur[0] < prev[1] - 0.05:
                issues.append(f"Překryv timeline: '{prev[2]}' / '{cur[2]}'")
            if cur[0] > prev[1] + 0.5:
                issues.append(f"Hluché místo v timeline: {prev[1]:.2f}s - {cur[0]:.2f}s")

        # Kontrola celkové délky timeline vs audio
        audio_path = self.find_audio()
        if audio_path and ranges:
            audio_dur = probe_duration(audio_path)
            timeline_dur = ranges[-1][1]
            if abs(timeline_dur - audio_dur) > 1.0:
                issues.append(f"Celková délka timeline ({timeline_dur:.2f}s) neodpovídá délce audia ({audio_dur:.2f}s)")

        # Kontrola Character klipů v no_rap módu
        if no_rap and self.timeline_file.exists():
            for raw in self.timeline_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "|" not in raw:
                    continue
                parts = [p.strip() for p in raw.split("|")]
                if len(parts) >= 2:
                    asset_id = clean_asset_id(parts[1])
                    if asset_id.startswith("char_"):
                        char_path = self.gen_char / f"{asset_id}.mp4"
                        if not is_valid_media(char_path):
                            issues.append(f"Chybí nebo je neplatný character klip: {asset_id}.mp4 (nahrajte do gen_char/)")

        if not final:
            corrections = self._load_json(self.edit_dir / "speed_corrections.json", {})
            ok_rap_clips = {name for name, corr in corrections.items() if corr.get("status") in ("ok", "clamped")}

            # Načteme shot_order, abychom věděli, které klipy jsou VÁŽNĚ očekávány
            expected_clips = set()
            shot_order_file = self.edit_dir / "shot_order.txt"
            if shot_order_file.exists():
                for line in shot_order_file.read_text(encoding="utf-8").splitlines():
                    if "|" in line:
                        expected_clips.add(clean_asset_id(line.split("|")[1]))
            else:
                # Fallback na metadata
                metadata = self._load_json(self.edit_dir / "metadata.json", {})
                expected_clips = set(metadata.get("shot_order", []))

            if ok_rap_clips and self.timeline_file.exists():
                timeline_rap_lines = {}
                for raw in self.timeline_file.read_text(encoding="utf-8").splitlines():
                    if "|" not in raw:
                        continue
                    parts = [p.strip() for p in raw.split("|")]
                    if len(parts) < 2:
                        continue
                    clip_name = clean_asset_id(parts[1])
                    if clip_name.startswith("rap_") and clip_name in ok_rap_clips:
                        timeline_rap_lines.setdefault(clip_name, []).append(raw)

                for clip_name in ok_rap_clips:
                    # Validujeme JEN klipy, které jsou v plánu (shot_order)
                    if clip_name not in expected_clips:
                        continue

                    lines_for_clip = timeline_rap_lines.get(clip_name, [])
                    if not lines_for_clip:
                        issues.append(f"{clip_name}: chybí v timeline.txt (očekáván v plánu)")
                        continue

                    # Místo dřívější kontroly textového tagu [SPEED_APPLIED] ověříme
                    # přímo fyzický klip v gen_rap/ — musí odpovídat přepočítané délce.
                    expected_dur = float(corrections.get(clip_name, {}).get("new_duration", 0.0))
                    clip_path = self.gen_rap / f"{clip_name}.mp4"
                    if expected_dur > 0:
                        if not is_valid_media(clip_path):
                            issues.append(f"{clip_name}: chybí nebo je neplatný přepočítaný klip v gen_rap/.")
                        else:
                            actual_dur = probe_duration(clip_path)
                            if abs(actual_dur - expected_dur) > 0.2:
                                issues.append(
                                    f"{clip_name}: klip v gen_rap/ ({actual_dur:.2f}s) neodpovídá "
                                    f"přepočítané délce ({expected_dur:.2f}s) — spusť znovu align-rap / apply-speeds-timeline."
                                )

        report = {"ok": not issues, "issues": issues}
        self._write_json(self.edit_dir / "validation_report.json", report)

        # ── Volitelný AI audit (jen čitelný souhrn, nikdy nerozhoduje o pass/fail) ──
        self._ai_validation_audit(report)

        if issues:
            print(f"❌ Validace našla {len(issues)} problémů. Viz EDIT_PROJECT/validation_report.json")
            return False
        print("✅ Validace projektu prošla.")
        return True

    def _ai_validation_audit(self, report: dict) -> None:
        """Volitelný AI audit nad výstupy validace — vytvoří čitelný souhrn nejkritičtějších
        problémů, podezřelých klipů a priorit oprav. Nikdy sám nerozhoduje o pass/fail
        (to zůstává čistě deterministické, viz `report["ok"]` výše) a při nedostupné
        Ollamě se jednoduše přeskočí beze změny chování."""
        settings = self.load_settings()
        if not self._ollama_ready(settings):
            return

        rap_alignment = self._load_json(self.edit_dir / "rap_alignment.json", {})
        song_alignment = self._load_json(self.edit_dir / "song_alignment.json", {})
        timeline_text = self.timeline_file.read_text(encoding="utf-8", errors="ignore") if self.timeline_file.exists() else ""

        # Sestavíme kompaktní kontext, ať prompt nenafoukneme
        issues_block = "\n".join(f"- {i}" for i in report.get("issues", [])) or "(žádné)"
        rap_summary_lines = []
        for name, data in list(rap_alignment.items())[:40]:
            score = data.get("lyrics_match_score")
            rap_summary_lines.append(f"- {name}: lyrics_match_score={score}")
        rap_summary = "\n".join(rap_summary_lines) or "(žádná rap_alignment.json data)"
        timeline_preview = truncate_for_prompt(timeline_text, 1500)
        song_alignment_note = f"song_alignment.json obsahuje {len(song_alignment.get('words', []))} slov." if song_alignment else "song_alignment.json chybí."

        model = self._ollama_model(settings)
        prompt = f"""Jsi technický auditor pipeline pro tvorbu hudebních videoklipů. Na základě
níže uvedených výstupů vytvoř STRUČNÝ, čitelný souhrn problémů. Nerozhoduj o tom,
zda projekt jako celek projde nebo ne — to už bylo určeno jinde. Tvým úkolem je
jen pomoct člověku rychle pochopit, na co se zaměřit.

Nalezené problémy (deterministická validace):
{issues_block}

Souhrn rap_alignment.json (skóre shody textu s lyrics.txt):
{rap_summary}

{song_alignment_note}

Ukázka timeline.txt (může být oříznutá):
{timeline_preview}

Odpověz VÝHRADNĚ jedním JSON objektem, bez dalšího textu:
{{"critical_issues": ["...", "..."], "suspicious_clips": ["...", "..."], "priority_fixes": ["...", "..."]}}"""

        raw = ollama_generate(prompt, model=model, format="json", temperature=0.2, timeout=45)
        parsed = extract_json_from_text(raw)
        if not parsed:
            return  # AI audit selhal/nedostupný — ticho, beze změny chování

        audit = {
            "critical_issues": parsed.get("critical_issues") or [],
            "suspicious_clips": parsed.get("suspicious_clips") or [],
            "priority_fixes": parsed.get("priority_fixes") or [],
        }
        # Sanitizace — jen seznamy stringů, ořezané na rozumnou délku
        for key in list(audit.keys()):
            if not isinstance(audit[key], list):
                audit[key] = []
            else:
                audit[key] = [truncate_for_prompt(str(x), 200) for x in audit[key][:15]]

        self._write_json(self.edit_dir / "ai_validation_summary.json", audit)

        if audit["critical_issues"] or audit["suspicious_clips"] or audit["priority_fixes"]:
            print("\n🧠 AI audit (doplňkový, nerozhoduje o pass/fail):")
            if audit["critical_issues"]:
                print("   Nejkritičtější problémy:")
                for x in audit["critical_issues"][:5]:
                    print(f"     - {x}")
            if audit["suspicious_clips"]:
                print("   Podezřelé klipy:")
                for x in audit["suspicious_clips"][:5]:
                    print(f"     - {x}")
            if audit["priority_fixes"]:
                print("   Priorita oprav:")
                for x in audit["priority_fixes"][:5]:
                    print(f"     - {x}")
            print(f"   → Detaily v EDIT_PROJECT/ai_validation_summary.json")

    def _resolve_timeline_overlaps(self, entries, overlap_tolerance: float = 0.05) -> list[dict]:
        """Posune sousedící klipy v pořadí plánu, aby se nepřekrývaly. Rap klipy mají prioritu."""
        fixes = []
        prev = None
        for entry in entries:
            if entry["type"] != "clip":
                continue
            if entry.get("new_start") is None or entry.get("new_end") is None:
                continue
            if prev is not None and entry["new_start"] < prev["new_end"] - overlap_tolerance:
                if entry["is_rap"]:
                    # Rap klip je kotva, nesmíme s ním hýbat! Zkrátíme raději předchozí klip.
                    old_prev_end = prev["new_end"]
                    prev["new_end"] = entry["new_start"]
                    fixes.append({
                        "clip": prev["clip"],
                        "old_end": round(old_prev_end, 3),
                        "new_end": round(prev["new_end"], 3),
                        "reason": "shortened_to_fix_overlap_with_rap_anchor",
                    })
                else:
                    duration = entry["new_end"] - entry["new_start"]
                    old_start = entry["new_start"]
                    entry["new_start"] = prev["new_end"]
                    entry["new_end"] = entry["new_start"] + duration
                    fixes.append({
                        "clip": entry["clip"],
                        "old_start": round(old_start, 3),
                        "new_start": round(entry["new_start"], 3),
                        "new_end": round(entry["new_end"], 3),
                        "reason": "overlap_with_previous",
                    })
            prev = entry
        return fixes

    def _settings_summary_lines(self, settings: dict) -> list[str]:
        """Vrátí přehledný souhrn aktuálního nastavení po kategoriích (pro tisk v menu)."""
        num_ctx = settings.get("ollama_plan_num_ctx", 8192)
        return [
            "  [1] Transkripce      : "
            f"provider={settings['transcription_provider']}, whisper_model={settings['whisper_model']}, "
            f"zařízení={settings['device']}, groq_model={settings['groq_model']}",
            "  [2] Render           : "
            f"fps_override={settings['fps_override'] or 'auto'}, speed={settings['speed_min']}-{settings['speed_max']}",
            "  [3] Ollama základní  : "
            f"ollama_enabled={settings.get('ollama_enabled', True)}, "
            f"ollama_model={settings.get('ollama_model', OLLAMA_DEFAULT_MODEL)}, "
            f"scenario_model={settings.get('ollama_scenario_model', OLLAMA_DEFAULT_MODEL)}",
            "  [4] Ollama Fáze B    : "
            f"ollama_plan_model={settings.get('ollama_plan_model', 'qwen2.5:7b-instruct')}, "
            f"num_ctx={num_ctx}, read_timeout={settings.get('ollama_stream_read_timeout_sec', 600)}s, "
            f"max_total={settings.get('ollama_stream_max_total_sec', 7200)}s",
            "  [5] AI provider 8a   : "
            f"text_ai_provider={settings.get('text_ai_provider', 'local')}, "
            f"groq_scenario_model={settings.get('groq_scenario_model', GROQ_LLM_DEFAULT_MODEL)} "
            "(Fáze B/8b běží vždy lokálně, tohle se týká jen scénáře 8a)",
        ]

    def _settings_edit_transcription(self, settings: dict) -> None:
        print("\n— TRANSKRIPCE —")
        print("Poskytovatel transkripce (1 - lokální Whisper, 2 - Groq Cloud API) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice == "1":
            settings["transcription_provider"] = "local"
        elif choice == "2":
            settings["transcription_provider"] = "groq"
            if not HAS_GROQ:
                print("  ⚠️ Balíček `groq` není nainstalován: pip install groq --break-system-packages")
            if not load_groq_api_key():
                ensure_groq_key_file_template()
                print(f"  ⚠️ Chybí Groq API klíč. Vlož ho do souboru: {GROQ_KEY_FILE}")
                print("     (nebo nastav proměnnou prostředí GROQ_API_KEY)")

        groq_models = ["whisper-large-v3", "whisper-large-v3-turbo"]
        print(f"Groq model ({'/'.join(groq_models)}) [Enter = beze změny]: ", end="")
        choice = input().strip().lower()
        if choice in groq_models:
            settings["groq_model"] = choice

        models = ["tiny", "base", "small", "medium", "large"]
        print(f"Lokální Whisper model ({'/'.join(models)}) [Enter = beze změny]: ", end="")
        choice = input().strip().lower()
        if choice in models:
            settings["whisper_model"] = choice

        print("Zařízení pro lokální Whisper (1 - CPU, 2 - GPU) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice == "1":
            settings["device"] = "cpu"
        elif choice == "2":
            settings["device"] = "gpu"

    def _settings_edit_render(self, settings: dict) -> None:
        print("\n— RENDER —")
        print("FPS pro render (Enter = automaticky podle rozlišení, 0 = automaticky): ", end="")
        choice = input().strip()
        if choice:
            try:
                val = int(choice)
                settings["fps_override"] = val if val > 0 else None
            except ValueError:
                print("  ⚠️ Neplatná hodnota FPS, ponechávám beze změny.")

        print(f"Minimální speed_factor (aktuálně {settings['speed_min']}) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            try:
                settings["speed_min"] = float(choice)
            except ValueError:
                print("  ⚠️ Neplatná hodnota, ponechávám beze změny.")

        print(f"Maximální speed_factor (aktuálně {settings['speed_max']}) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            try:
                settings["speed_max"] = float(choice)
            except ValueError:
                print("  ⚠️ Neplatná hodnota, ponechávám beze změny.")

        if settings["speed_min"] > settings["speed_max"]:
            print("  ⚠️ speed_min > speed_max, hodnoty prohazuji.")
            settings["speed_min"], settings["speed_max"] = settings["speed_max"], settings["speed_min"]

    def _settings_edit_ollama_basic(self, settings: dict) -> None:
        print("\n— OLLAMA (ZÁKLADNÍ) —")
        print(f"Používat lokální AI (Ollama) pro sémantické rozhodování "
              f"(aktuálně {'zapnuto' if settings.get('ollama_enabled', True) else 'vypnuto'})? "
              f"(1 - zapnout, 2 - vypnout) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice == "1":
            settings["ollama_enabled"] = True
        elif choice == "2":
            settings["ollama_enabled"] = False

        print(f"Ollama model (aktuálně {settings.get('ollama_model', OLLAMA_DEFAULT_MODEL)}, "
              f"musí být předem stažený přes `ollama pull <model>`) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            settings["ollama_model"] = choice

        print(f"Ollama model pro Fázi A / scénář (aktuálně "
              f"{settings.get('ollama_scenario_model', OLLAMA_DEFAULT_MODEL)}) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            settings["ollama_scenario_model"] = choice

        if settings.get("ollama_enabled", True):
            if ollama_available(force=True):
                print(f"  ✅ Ollama server je dostupný na {OLLAMA_BASE_URL}.")
            else:
                print(f"  ⚠️ Ollama server neběží na {OLLAMA_BASE_URL} — skript automaticky "
                      "spadne zpět na heuristiku, dokud Ollama nepoběží.")

    def _settings_edit_ollama_phase_b(self, settings: dict) -> None:
        print("\n— OLLAMA / FÁZE B (full_plan.txt, volba 8b) —")
        print(f"Ollama model pro Fázi B / full_plan.txt (aktuálně "
              f"{settings.get('ollama_plan_model', 'qwen2.5:7b-instruct')}, doporučeno větší model "
              f"pro kvalitnější strukturovaný výstup) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            settings["ollama_plan_model"] = choice

        current_num_ctx = int(settings.get("ollama_plan_num_ctx", 8192))
        print(f"Kontextové okno num_ctx (aktuálně {current_num_ctx}). Tvrdý strop na PROMPT+ODPOVĚĎ "
              "dohromady u Fáze B — vyšší hodnota = méně ořezávaný seznam existujících klipů "
              "(existing_clips_block) a méně navazujících úseků v Části 2/2, ale výrazně víc RAM "
              "a pomalejší prefill na CPU. Typické hodnoty: 8192 (výchozí), 16384, 32768. "
              "[Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            try:
                new_num_ctx = int(choice)
                if new_num_ctx < 2048:
                    print("  ⚠️ num_ctx < 2048 by prompt vůbec nepustilo dovnitř, ponechávám původní hodnotu.")
                else:
                    settings["ollama_plan_num_ctx"] = new_num_ctx
            except ValueError:
                print("  ⚠️ Neplatné číslo, ponechávám původní hodnotu.")

        current_max_chunk_sec = settings.get("ollama_plan_max_chunk_seconds", 60.0)
        print(f"Max. časové rozpětí ČÁSTI 2/2 na jeden úsek v sekundách (aktuálně "
              f"{current_max_chunk_sec}s). NEZÁVISLÉ na num_ctx — drží jednotlivá volání kratší, "
              "i když by se znakově vešla celá píseň najednou. Vyšší num_ctx bez tohohle stropu "
              "umí u slabšího modelu vést k tomu, že si model splete 'konec úseku' s 'koncem "
              "jednoho konkrétního klipu' (řádek pak dostane koncový čas rovný konci celé písně "
              "místo pár vteřin). Nižší hodnota = víc (kratších, spolehlivějších) úseků, ale "
              "pomalejší běh (katalog klipů se posílá znovu v každém). Typické hodnoty: 30–90. "
              "0 nebo prázdné = bez stropu (jen podle num_ctx, původní chování). "
              "[Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            try:
                new_max_chunk_sec = float(choice)
                settings["ollama_plan_max_chunk_seconds"] = new_max_chunk_sec if new_max_chunk_sec > 0 else None
            except ValueError:
                print("  ⚠️ Neplatné číslo, ponechávám původní hodnotu.")

        print(f"Max. čekání mezi tokeny v sekundách, VČETNĚ prvního tokenu / prefillu "
              f"(aktuálně {settings.get('ollama_stream_read_timeout_sec', 600)}s). Na slabším CPU "
              f"(bez GPU, málo RAM) může prefill velkého kontextu sám trvat i několik minut — pokud "
              f"8b hlásí 'neposlala žádný token' hned na začátku, zkus tuhle hodnotu zvýšit (např. "
              f"na 900 nebo 1200) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            try:
                settings["ollama_stream_read_timeout_sec"] = max(30, int(choice))
            except ValueError:
                print("  ⚠️ Neplatné číslo, ponechávám původní hodnotu.")

        print(f"Celková časová pojistka na celý běh v sekundách, i když model stabilně produkuje "
              f"tokeny (aktuálně {settings.get('ollama_stream_max_total_sec', 7200)}s "
              f"= {settings.get('ollama_stream_max_total_sec', 7200) / 60:.0f} min). Na slabém CPU (bez GPU) "
              f"může kompletní full_plan.txt na celou písničku legitimně trvat přes hodinu — pokud 8b hlásí "
              f"'překročilo celkovou pojistku' a přitom průběžně přibývaly znaky, zvyš tuhle hodnotu (např. "
              f"na 10800 = 3h) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice:
            try:
                settings["ollama_stream_max_total_sec"] = max(60, int(choice))
            except ValueError:
                print("  ⚠️ Neplatné číslo, ponechávám původní hodnotu.")

    def _settings_edit_ai_provider_8a(self, settings: dict) -> None:
        print("\n— AI PROVIDER PRO FÁZI A (scénář, volba 8a) —")
        print("  ℹ️  Fáze B (8b, full_plan.txt) běží VŽDY lokálně přes Ollamu — je to dlouhý strukturovaný "
              "výstup, který se na Groq free-tier TPM limitu spolehlivě nevejde. Tohle nastavení se týká "
              "jen Fáze A.")
        print(f"Poskytovatel AI pro Fázi A (aktuálně: {settings.get('text_ai_provider', 'local')}) "
              f"(1 - lokální Ollama, 2 - Groq Cloud LLM) [Enter = beze změny]: ", end="")
        choice = input().strip()
        if choice == "1":
            settings["text_ai_provider"] = "local"
        elif choice == "2":
            settings["text_ai_provider"] = "groq"
            if not HAS_GROQ:
                print("  ⚠️ Balíček `groq` není nainstalován: pip install groq --break-system-packages")
            if not load_groq_api_key():
                ensure_groq_key_file_template()
                print(f"  ⚠️ Chybí Groq API klíč. Vlož ho do souboru: {GROQ_KEY_FILE}")
                print("     (nebo nastav proměnnou prostředí GROQ_API_KEY)")

        if settings.get("text_ai_provider", "local") == "groq":
            print(f"Groq LLM model pro Fázi A / scénář (aktuálně "
                  f"{settings.get('groq_scenario_model', GROQ_LLM_DEFAULT_MODEL)}). "
                  f"Doporučeno: {GROQ_LLM_DEFAULT_MODEL} (nejkvalitnější) nebo {GROQ_LLM_FAST_MODEL} "
                  f"(rychlejší/levnější). [Enter = beze změny]: ", end="")
            choice = input().strip()
            if choice:
                settings["groq_scenario_model"] = choice

            if HAS_GROQ and load_groq_api_key():
                print("  ✅ Groq Cloud LLM je nakonfigurováno (balíček i API klíč jsou v pořádku) — pro Fázi A.")
            else:
                print("  ⚠️ Groq Cloud LLM zatím není plně funkční (chybí balíček `groq` nebo API klíč) — "
                      "volba 8a spadne na chybovou hlášku, dokud to nedoplníš.")

    def configure_settings(self):
        """Interaktivní nastavení projektu, rozdělené do kategorií (přehlednější než jeden dlouhý
        sled otázek): transkripce, render, Ollama základní/Fáze B (vč. num_ctx), AI provider pro
        Fázi A. Fáze B (8b) provider vždy ignoruje a běží vždy lokálně přes Ollamu."""
        settings = self.load_settings()
        editors = {
            "1": ("Transkripce", self._settings_edit_transcription),
            "2": ("Render", self._settings_edit_render),
            "3": ("Ollama základní", self._settings_edit_ollama_basic),
            "4": ("Ollama Fáze B (vč. num_ctx)", self._settings_edit_ollama_phase_b),
            "5": ("AI provider 8a", self._settings_edit_ai_provider_8a),
        }

        while True:
            print("\n⚙️  NASTAVENÍ PROJEKTU")
            for line in self._settings_summary_lines(settings):
                print(line)
            print("  [0] Uložit a skončit")
            print("Kterou kategorii chceš upravit? (0-5, nebo 'a' = projít všechny popořadě): ", end="")
            choice = input().strip().lower()

            if choice in ("", "0", "q", "konec"):
                break
            elif choice == "a":
                for _, editor in editors.values():
                    editor(settings)
            elif choice in editors:
                editors[choice][1](settings)
            else:
                print("  ⚠️ Neplatná volba, zkus znovu.")
                continue

            self.save_settings(settings)
            print(f"✅ Průběžně uloženo do {self.settings_file.relative_to(self.project_dir)}")

        self.save_settings(settings)
        print(f"✅ Nastavení uloženo do {self.settings_file.relative_to(self.project_dir)}")

    def run_all(self, mode="draft", hd_mode="draft", no_rap=False, force=False):
        """Spustí kompletní pipeline podle new_pipeline.txt."""
        setup = execute_sequence([
            ("init_project", self.init_project),
            ("parse_plan", self.parse_plan),
            ("create_placeholders", self.create_placeholders),
        ])
        if not setup.ok:
            self.logger.error("Pipeline zastavena: %s", "; ".join(setup.errors))
            for error in setup.errors:
                print(f"❌ {error}")
            return False

        if mode == "final":
            transcribed = execute_step("transcribe_song_czech", self.transcribe_song_czech)
            if not transcribed.ok:
                self.logger.error("Final pipeline zastavena: %s", "; ".join(transcribed.errors))
                for error in transcribed.errors:
                    print(f"❌ {error}")
                return False
            self.analyze_song()
            exported = execute_step("export_lipsync_audio_segments", self.export_lipsync_audio_segments)
            if not exported.ok:
                self.logger.error("Export lipsync segmentů selhal: %s", "; ".join(exported.errors))
                for error in exported.errors:
                    print(f"❌ {error}")
                return False
            print(
                "\n✅ Audio segmenty jsou připravené v LIPSYNC_AUDIO/.\n"
                "👉 Vygenerujte rap_001.mp4, rap_002.mp4 atd. z těchto přesně odpovídajících WAV souborů.\n"
                "👉 Vložte je do gen_rap/ a teprve potom spusťte 'inject-lipsync', 'validate' a 'render'."
            )
            return True

        self.analyze_song()
        if not no_rap:
            rap_steps = execute_sequence([
                ("transcribe_rap_clips", self.transcribe_rap_clips),
                ("align_rap_clips", self.align_rap_clips),
            ])
            if not rap_steps.ok:
                self.logger.error("Rap pipeline zastavena: %s", "; ".join(rap_steps.errors))
                for error in rap_steps.errors:
                    print(f"❌ {error}")
                return False
        self.update_timeline_from_alignment()
        validated = self.validate_project(final=False, no_rap=no_rap)
        if validated:
            print("\n✅ Validace v pořádku. Pipeline (body 1–9) dokončena — render (bod 10) se nespouští automaticky, spusť ho ručně.")
        elif force:
            print("⚠️ Validace selhala, ale force=True bylo zadáno. Render se nespouští automaticky; spusť jej ručně.")
        else:
            print("⚠️ Validace našla problémy. Oprav je a render spusť ručně.")
        return bool(validated)

    def generate_scenario_ai(self) -> bool:
        """FÁZE A: Nechá lokální AI (Ollama) vymyslet scénář videoklipu na základě
        textu písně, volitelné nálady/žánru, textového popisu hlavní postavy a
        seznamu už existujících klipů (pokud jsou k dispozici).

        Vstupy (INPUT/):
          - lyrics.txt           (povinné — text písně)
          - postava.txt / character.txt  (silně doporučené — popis hlavní postavy)
          - mood.txt              (volitelné — nálada/žánr)
          - klipy.md               (volitelné — existující klipy; pokud jsou, AI je
                                     do scénáře zapracuje tam, kde obsahově sedí)

        Výstup: Prompts/scenario.txt (vstup pro Fázi B — generate_full_plan_ai).

        Poskytovatel AI (lokální Ollama / Groq Cloud LLM) se řídí nastavením
        'text_ai_provider' (volba 13 → Nastavení)."""
        settings = self.load_settings()
        provider = self._text_ai_provider(settings)
        if not self._text_ai_ready(settings):
            if provider == "groq":
                print("❌ Groq Cloud LLM není dostupné (chybí balíček `groq` nebo API klíč — nastavení 'text_ai_provider', volba 13).")
                print(f"   Vlož Groq API klíč do {GROQ_KEY_FILE} nebo nastav proměnnou GROQ_API_KEY.")
            else:
                print("❌ Ollama není dostupná/vypnutá (nastavení 'ollama_enabled'/'text_ai_provider', volba 13).")
                print("   Fázi A nelze provést bez AI — zkontroluj, že `ollama serve` běží, nebo přepni na Groq Cloud LLM v nastavení.")
            return False

        lyrics = self._load_lyrics_text()
        if not lyrics:
            print(f"❌ Chybí {self.input_dir / 'lyrics.txt'} — bez textu písně nelze scénář vygenerovat.")
            return False

        character_description = self._find_character_description()
        if not character_description:
            print("⚠️  Nenalezen INPUT/postava.txt (ani character.txt) — scénář bude vygenerován "
                  "BEZ konzistentního popisu postavy. Doporučuji soubor doplnit a spustit znovu.")
            character_description = "(popis postavy nebyl dodán — použij obecnou postavu podle kontextu textu písně)"

        mood = self._find_mood_description() or "(nebylo zadáno — odvoď náladu z textu písně)"

        klipy = self._load_klipy_md()
        if klipy:
            print(f"📼 Načteno {len(klipy)} existujících klipů z klipy.md — AI je zohlední při psaní scénáře "
                  f"({sum(1 for v in klipy.values() if v['group']=='RAP')} rap, "
                  f"{sum(1 for v in klipy.values() if v['group']=='VID')} vid, "
                  f"{sum(1 for v in klipy.values() if v['group']=='PIC')} pic, "
                  f"{sum(1 for v in klipy.values() if v['group']=='CHAR')} char).")
        else:
            print("ℹ️  INPUT/klipy.md nenalezen nebo prázdný — scénář bude vygenerován bez ohledu na "
                  "existující klipy (běžné u prvního běhu / nového projektu).")
        existing_clips_block = self._format_existing_clips_for_prompt(klipy)

        provider_label = "Groq Cloud LLM" if provider == "groq" else "lokální AI (Ollama)"

        # Groq free-tier účet má TPM limit (typicky 8000 tokenů/min) počítaný ze
        # SOUČTU vstupu i požadovaného výstupu (max_tokens) — s rostoucím
        # katalogem klipů (INPUT/klipy.md) se vstup časem přehoupne přes limit
        # a Groq vrátí 413 rate_limit_exceeded (reálně zachyceno: 20636 tokenů
        # požadováno vs. limit 8000 — z toho ~16000 byl jen samotný max_tokens).
        # Pro Ollamu žádný TPM limit není, takže tam necháváme původní štědré
        # ořezávání beze změny.
        lyrics_limit = 6000
        clips_limit = 6000
        max_tokens = None
        if provider == "groq":
            max_tokens = int(settings.get("groq_scenario_max_tokens", 3000) or 3000)
            lyrics_limit = 3000
            clips_limit = 3000
            existing_clips_block = truncate_for_prompt(existing_clips_block, clips_limit)

        prompt = SCENARISTA_PROMPT_TEMPLATE.format(
            lyrics=truncate_for_prompt(lyrics, lyrics_limit),
            mood=mood,
            character_description=character_description,
            existing_clips=existing_clips_block,
        )
        messages = [
            {"role": "system", "content": SCENARISTA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if provider == "groq":
            # Bezpečnostní pojistka navíc: pokud by i po výše uvedeném ořezání
            # (např. u výjimečně velkého klipy.md se spoustou klipů) odhad
            # vstup+max_tokens pořád přesahoval bezpečnou rezervu pod TPM
            # limitem, dál zkracujeme seznam existujících klipů (nejobjemnější
            # část promptu), dokud se odhad nevejde nebo nedojdeme na rozumné
            # minimum.
            def _est_tokens(text: str) -> int:
                return max(1, len(text) // 4)  # hrubý odhad (~4 znaky/token)

            safety_budget = 7500  # rezerva pod Groq free-tier TPM limitem (8000)
            attempts = 0
            while attempts < 4:
                estimated = _est_tokens(SCENARISTA_SYSTEM_PROMPT + prompt) + max_tokens
                if estimated <= safety_budget or clips_limit <= 500:
                    break
                clips_limit = max(500, clips_limit // 2)
                existing_clips_block = truncate_for_prompt(
                    self._format_existing_clips_for_prompt(klipy), clips_limit
                )
                prompt = SCENARISTA_PROMPT_TEMPLATE.format(
                    lyrics=truncate_for_prompt(lyrics, lyrics_limit),
                    mood=mood,
                    character_description=character_description,
                    existing_clips=existing_clips_block,
                )
                messages[1]["content"] = prompt
                attempts += 1
            if attempts:
                print(f"ℹ️  Seznam existujících klipů zkrácen na ~{clips_limit} znaků, "
                      "aby se požadavek vešel do Groq TPM limitu.")

        model_preview = self._groq_scenario_model(settings) if provider == "groq" else self._ollama_scenario_model(settings)
        print(f"🎬 Generuji scénář pomocí {provider_label} (model: {model_preview})...")
        if provider != "groq":
            print("   (na CPU bez GPU to může u prvního běhu trvat i několik minut — model se musí "
                  "nejdřív načíst do paměti, buď trpělivý/á)")
        else:
            print(f"   (max_tokens={max_tokens}, nastavitelné v 'groq_scenario_max_tokens' — volba 13)")
        raw, model, error = self._generate_with_text_ai(
            messages, phase="scenario", settings=settings, temperature=0.7, timeout=900, num_ctx=8192,
            max_tokens=max_tokens,
        )

        if not raw or not raw.strip():
            print(f"❌ Generování scénáře selhalo: {error}")
            if provider == "groq":
                print("   Zkontroluj Groq API klíč a jméno modelu v nastavení (volba 13 → groq_scenario_model).")
            else:
                print(f"   Zkontroluj: `ollama pull {model}`, že server neběží na jinou náročnou úlohu zároveň,")
                print("   a případně otestuj přímo: "
                      f'curl {OLLAMA_BASE_URL}/api/chat -d \'{{"model":"{model}","messages":'
                      '[{"role":"user","content":"ahoj"}],"stream":false}\'')
            return False

        scenario_text = clean_code_block_text(raw.strip())
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = self.prompts_dir / "scenario.txt"
        header = (
            f"# AI-GENEROVANÝ SCÉNÁŘ — poskytovatel: {provider_label}, model: {model}, vygenerováno automaticky (Fáze A / volba 8a)\n"
            f"# Zkontroluj a uprav ručně, pokud je potřeba, než spustíš Fázi B (volba 8b).\n\n"
        )
        scenario_path.write_text(header + scenario_text, encoding="utf-8")
        print(f"✅ Scénář uložen do {scenario_path.relative_to(self.project_dir)} ({len(scenario_text)} znaků).")
        print("   👉 Zkontroluj/uprav scénář, pak spusť Fázi B (volba 8b) pro vygenerování full_plan.txt.")
        return True

    def generate_full_plan_ai(self) -> bool:
        """FÁZE B: Nechá lokální AI (Ollama) vygenerovat kompletní full_plan.txt na
        základě scénáře z Fáze A, transkripce písně a seznamu existujících klipů
        (INPUT/klipy.md) — s preferencí znovupoužití existujících klipů před
        generováním nových.

        Fáze B běží VŽDY lokálně přes Ollamu (viz _generate_with_text_ai) —
        nastavení 'text_ai_provider' (volba 13) se týká jen Fáze A (scénář).
        full_plan.txt je dlouhý strukturovaný výstup, který se na Groq
        free-tier TPM limitu spolehlivě nevejde, proto pro Fázi B Groq
        nepoužíváme."""
        settings = self.load_settings()
        provider = "local"
        if not self._ollama_ready(settings):
            print("❌ Ollama není dostupná/vypnutá (nastavení 'ollama_enabled', volba 13) — Fáze B potřebuje lokální Ollamu.")
            print(f"   Zkontroluj, že běží `ollama serve` a je stažený model (`ollama pull {self._ollama_plan_model(settings)}`).")
            return False

        scenario_path = self.prompts_dir / "scenario.txt"
        scenario_text = self._load_text_file(scenario_path)
        if not scenario_text:
            print(f"❌ Chybí {scenario_path.relative_to(self.project_dir)}.")
            choice = input("   Spustit teď Fázi A (generování scénáře) automaticky? (a/N): ").strip().lower()
            if choice == "a":
                if not self.generate_scenario_ai():
                    return False
                scenario_text = self._load_text_file(scenario_path)
            else:
                return False

        # Transkripce písně (segmenty, ne slova — kvůli velikosti kontextu lokálního modelu)
        # Používáme sdílenou _whisper_segments_to_words(), protože ta jako jediná podporuje
        # všechny 3 vstupní formáty (Whisper CLI, Groq verbose_json, i vlastní formát po sekcích
        # s "transcript"/"words" a MM:SS.ms časy) — narozdíl od dřívějšího přímého čtení
        # trans_data["segments"], které selhávalo na jiných formátech, i když byly validní.
        trans_file = self.input_dir / "transcription.json"
        trans_data = self._load_json(trans_file, {})
        words = self._whisper_segments_to_words(trans_data) if isinstance(trans_data, dict) else []
        if not words:
            print(f"⚠️  {trans_file.relative_to(self.project_dir)} chybí nebo neobsahuje rozpoznatelná slova. "
                  "Spusť nejprve volbu 4 (Analyzovat song). Pokračuji bez přesné transkripce — "
                  "výsledný timing bude méně přesný.")

        # song_duration se počítá PŘED sestavením transcription_lines/spans, protože podle
        # něj ořezáváme slova, jejichž časy jsou za skutečným koncem audia (viz
        # _validate_transcription_vs_duration / _clamp_words_to_duration níže) — jinak by se
        # do Části 2/2 poslal text s časy, které si odporují s nucenou hranicí posledního úseku.
        audio_path = self.find_audio()
        song_duration = probe_duration(audio_path) if audio_path else (words[-1]["end"] if words else 0.0)

        duration_problems = self._validate_transcription_vs_duration(words, song_duration)
        if duration_problems:
            for p in duration_problems:
                print(f"⚠️  {p}")
        words, dropped_count = self._clamp_words_to_duration(words, song_duration)
        if dropped_count:
            print(f"   → {dropped_count} slov(o) mimo skutečnou délku audia bylo z transkripce pro Část 2/2 vynecháno.")

        # Slova seskupíme do krátkých úseků (podle mezer mezi slovy > 0.6s = předěl) —
        # výsledek je stejně čitelný pro AI jako "segmenty", jen sestavený ze slov.
        # transcription_spans drží (start, end) pro každý řádek — beze slov už "text"
        # rozeznat čas nejde, ale AI generuje časovou osu z {transcription} textu, takže
        # nám jde jen o hranice pro pozdější řezání na úseky (viz _split_transcription_into_chunks).
        transcription_lines = []
        transcription_spans = []
        current_words: list[dict] = []

        def _flush():
            if not current_words:
                return
            start = current_words[0]["start"]
            end = current_words[-1]["end"]
            text = " ".join(w["word"] for w in current_words)
            transcription_lines.append(f"{start:.2f}-{end:.2f}s: {text}")
            transcription_spans.append((start, end))

        prev_end = None
        for w in words:
            if prev_end is not None and w["start"] - prev_end > 0.6 and current_words:
                _flush()
                current_words = []
            current_words.append(w)
            prev_end = w["end"]
        _flush()

        transcription_block = "\n".join(transcription_lines) if transcription_lines else "(transkripce nedostupná)"

        klipy = self._load_klipy_md()
        if not klipy:
            print(f"⚠️  {self.input_dir / 'klipy.md'} nenalezen nebo prázdný — AI nebude mít žádné "
                  "existující klipy k opětovnému použití a vygeneruje kompletně nový seznam.")
        else:
            print(f"📼 Načteno {len(klipy)} existujících klipů z klipy.md "
                  f"({sum(1 for v in klipy.values() if v['group']=='RAP')} rap, "
                  f"{sum(1 for v in klipy.values() if v['group']=='VID')} vid, "
                  f"{sum(1 for v in klipy.values() if v['group']=='PIC')} pic, "
                  f"{sum(1 for v in klipy.values() if v['group']=='CHAR')} char).")
        existing_clips_block = self._format_existing_clips_for_prompt(klipy)

        provider_label = "lokální AI (Ollama)"
        model_preview = self._ollama_plan_model(settings)
        print(f"🧠 Generuji full_plan.txt pomocí {provider_label} (model: {model_preview}) — u větších modelů "
              "to na CPU může trvat i desítky minut. Fáze B běží ve DVOU krocích: (1/2) kreativní/asset "
              "sekce jedním voláním, (2/2) časová osa v N navazujících úsecích podle času písně — v obojím "
              "dostává celý seznam existujících klipů přednost, aby se vešel bez ořezávání. Generuje se "
              "streamovaně, takže dokud model produkuje tokeny (i pomalu), běh se sám neukončí — jen se "
              "občas vypíše průběžný stav, buď trpělivý/á...")

        # Ollama 'num_ctx' je tvrdý strop na PROMPT + ODPOVĚĎ dohromady. Pokud by pevné znakové
        # limity spolu s fixní šablonou přelezly num_ctx, Ollama potichu ořízne ZAČÁTEK promptu —
        # tam je přitom systémový prompt i celá sekce STRUKTURA VÝSTUPU. Model pak nevidí, co má
        # vlastně vygenerovat, a výsledek je neúplný/duplicitní/halucinovaný. Proto Fázi B rozdělujeme
        # na dvě volání (viz komentář u FULL_PLAN_PART1_*/PART2_* výše) a limity počítáme dynamicky
        # podle skutečné velikosti šablony. Navíc dáváme existing_clips_block PŘEDNOST před scénářem/
        # transkripcí — Pravidlo č. 1 je nejdůležitější, takže se ořízne až jako poslední možnost.
        num_ctx = int(settings.get("ollama_plan_num_ctx", 8192))
        chars_per_token = 2.2  # konzervativní odhad pro češtinu s diakritikou (CPU/Ollama tokenizer)

        def _budget(system_prompt: str, template: str, placeholders: list, response_budget_tokens: int) -> int:
            fixed_chars = len(system_prompt) + len(template)
            for ph in placeholders:
                fixed_chars -= len("{" + ph + "}")
            available_tokens = max(500, num_ctx - response_budget_tokens - (fixed_chars / chars_per_token))
            return int(available_tokens * chars_per_token)

        def _prioritize_clips(available_chars: int, clips_text: str, other_min_chars: int = 800):
            """Dá seznamu existujících klipů přednost — ořízne ho až tehdy, když by jinak
            nezbylo ani `other_min_chars` znaků pro zbývající vstup (scénář/transkripci)."""
            clips_cap = min(len(clips_text), max(0, available_chars - other_min_chars))
            other_cap = max(other_min_chars, available_chars - clips_cap)
            return clips_cap, other_cap

        # ── ČÁST 1/2 — kreativní a asset sekce (scénář + CELÝ seznam existujících klipů) ──
        next_free_ids_block = self._format_next_free_ids_for_prompt(klipy)
        available1 = _budget(
            FULL_PLAN_PART1_SYSTEM_PROMPT, FULL_PLAN_PART1_TEMPLATE,
            ["scenario", "song_duration", "existing_clips", "next_free_ids"], response_budget_tokens=3000,
        )
        available1 -= len(next_free_ids_block)  # tenhle blok je krátký a nesmí se ořezávat, jde mimo rozpočet
        clips_cap1, scenario_cap1 = _prioritize_clips(available1, existing_clips_block, other_min_chars=800)
        if len(existing_clips_block) > clips_cap1:
            print(f"ℹ️  Část 1/2: seznam existujících klipů ({len(existing_clips_block)} znaků) i po prioritizaci "
                  f"nadále přesahuje dostupný kontext (num_ctx={num_ctx}) — ořezávám na {clips_cap1} znaků. "
                  "Zvaž zvýšení num_ctx (volba 13) nebo zkrácení INPUT/klipy.md.")
        if len(scenario_text) > scenario_cap1:
            print(f"ℹ️  Část 1/2: scénář ořezán na {scenario_cap1} znaků, aby prošel do num_ctx={num_ctx} "
                  "spolu s celým seznamem existujících klipů.")

        prompt1 = FULL_PLAN_PART1_TEMPLATE.format(
            scenario=truncate_for_prompt(scenario_text, scenario_cap1),
            song_duration=f"{song_duration:.2f}",
            existing_clips=truncate_for_prompt(existing_clips_block, clips_cap1),
            next_free_ids=next_free_ids_block,
        )
        messages1 = [
            {"role": "system", "content": FULL_PLAN_PART1_SYSTEM_PROMPT},
            {"role": "user", "content": prompt1},
        ]
        print("🧠 [1/2] Generuji kreativní/asset sekce...")
        # timeout zde je jen dolní hranice pro celkovou pojistku (_generate_with_text_ai
        # pro fázi "plan" reálně použije max(timeout, ollama_stream_max_total_sec) a hlídá
        # hlavně mezery mezi jednotlivými tokeny, ne celkovou dobu běhu — viz ollama_chat_stream()).
        raw1, model, error1 = self._generate_with_text_ai(
            messages1, phase="plan", settings=settings, temperature=0.3, timeout=3600, num_ctx=num_ctx,
        )
        if not raw1 or not raw1.strip():
            print(f"❌ Generování full_plan.txt selhalo (část 1/2): {error1}")
            print(f"   Zkus: `ollama pull {model}` — nebo v Nastavení (volba 13) zvol menší ollama_plan_model "
                  "(např. qwen2.5:3b). Pokud hláška zmiňuje 'celkovou pojistku' (a znaky přitom průběžně "
                  "přibývaly), zvyš v Nastavení 'ollama_stream_max_total_sec'. Pokud zmiňuje 'neposlala "
                  "žádný token' hned na začátku, zvyš 'ollama_stream_read_timeout_sec' (prefill promptu "
                  "na CPU) — a zkontroluj, že `ollama serve` běží.")
            return False

        plan_text1 = clean_code_block_text(raw1.strip())
        sections1 = extract_sections(plan_text1)
        required1 = ["VIDEO_PROMPTS", "RAPPER_PROMPTS", "ASSET_PLANNING", "METADATA"]
        missing1 = [s for s in required1 if s not in sections1 or not sections1[s].strip()]

        draft1_path = self.prompts_dir / "full_plan_ai_draft_part1.txt"
        draft1_path.write_text(plan_text1, encoding="utf-8")

        if missing1:
            print(f"⚠️  Část 1/2 AI výstupu postrádá povinné sekce: {', '.join(missing1)}.")
            print(f"   Výstup jsem uložil ke kontrole do {draft1_path.relative_to(self.project_dir)}, "
                  "full_plan.txt jsem NEPŘEPSAL.")
            print("   Zkontroluj draft ručně, případně zkus jiný/větší model (ollama_plan_model, volba 13) a spusť volbu 8b znovu.")
            return False

        # ── Kontrola kolizí ID: 'nový' klip nesmí použít ID, které už patří existujícímu klipu ──
        # (viz _validate_no_id_collisions — dřív se stávalo, že model přeočísloval nové klipy
        # znovu od 01 a kolidoval s existujícím katalogem; teď navíc dostal next_free_ids_block,
        # ale i tak si to ověříme mechanicky, ne jen spoléháním na to, že to dodržel.)
        collisions1 = self._validate_no_id_collisions(sections1, klipy)
        if collisions1:
            print(f"❌ Část 1/2: {len(collisions1)} nově naplánovaných klipů koliduje ID s existujícím katalogem (klipy.md):")
            for c in collisions1:
                print(f"   - {c}")
            print(f"   Výstup jsem uložil ke kontrole do {draft1_path.relative_to(self.project_dir)}, full_plan.txt jsem NEPŘEPSAL.")
            print("   Zkus jiný/větší model (ollama_plan_model, volba 13) a spusť volbu 8b znovu — "
                  "model dostal přesná volná ID k použití (next_free_ids), ale nedodržel je.")
            return False

        # ── ČÁST 2/2 — časová osa, rozdělená na N SEKVENČNÍCH volání podle ČASU písně ──
        # Na rozdíl od katalogu existujících klipů (ten zůstává v každém volání CELÝ, jinak by
        # Pravidlo č. 1 nefungovalo — model by neviděl klipy z "jiné části" katalogu) je
        # transkripce přirozeně sekvenční, takže ji lze rozsekat na po sobě jdoucí časové úseky
        # beze ztráty kontextu. Volání běží sekvenčně (ne paralelně), protože každý navazující
        # úsek musí přesně navazovat na konec toho předchozího.
        new_clips_summary = self._summarize_new_clips_for_timeline(sections1)
        expected_clip_durations = self._build_expected_clip_durations(klipy, sections1)

        response_budget_tokens_chunk = 2200
        available_chunk = _budget(
            FULL_PLAN_PART2_SYSTEM_PROMPT, FULL_PLAN_PART2_TEMPLATE,
            ["transcription", "song_duration", "existing_clips", "new_clips_summary",
             "segment_index", "segment_total", "segment_start", "segment_end"],
            response_budget_tokens=response_budget_tokens_chunk,
        )
        existing_clips_for_chunks = existing_clips_block
        max_chars_transcription_per_chunk = available_chunk - len(existing_clips_block) - len(new_clips_summary)
        if max_chars_transcription_per_chunk < 300:
            # I s rozsekáním transkripce na úseky se samotný katalog klipů + shrnutí nových
            # klipů nevejde pohodlně vedle sebe do jednoho volání — ořízneme klipy jen jako
            # POSLEDNÍ možnost (stejně jako u Části 1/2), a nahlas o tom informujeme.
            clips_cap_chunk, _ = _prioritize_clips(available_chunk - len(new_clips_summary), existing_clips_block, other_min_chars=300)
            print(f"ℹ️  Část 2/2: i po rozdělení transkripce na úseky se celý seznam existujících "
                  f"klipů ({len(existing_clips_block)} znaků) nevejde pohodlně do num_ctx={num_ctx} "
                  f"spolu se shrnutím nových klipů — ořezávám na {clips_cap_chunk} znaků v KAŽDÉM "
                  "dílčím volání. Zvaž zvýšení num_ctx (volba 13) nebo zkrácení INPUT/klipy.md.")
            existing_clips_for_chunks = truncate_for_prompt(existing_clips_block, clips_cap_chunk)
            max_chars_transcription_per_chunk = max(300, available_chunk - clips_cap_chunk - len(new_clips_summary))

        # Časový strop na úsek (viz ollama_plan_max_chunk_seconds) je NEZÁVISLÝ na num_ctx/znakovém
        # rozpočtu výše — drží jednotlivá volání kratší i při velkém num_ctx, protože slabší lokální
        # model snáz "ztratí nit" při přesném sčítání času přes několik minut najednou v jednom volání.
        max_chunk_seconds = settings.get("ollama_plan_max_chunk_seconds")
        try:
            max_chunk_seconds = float(max_chunk_seconds) if max_chunk_seconds else None
        except (TypeError, ValueError):
            max_chunk_seconds = None

        transcription_chunks, chunk_bounds = self._split_transcription_into_chunks(
            transcription_lines, transcription_spans, song_duration, max_chars_transcription_per_chunk,
            max_chunk_seconds=max_chunk_seconds,
        )
        n_chunks = len(transcription_chunks)
        print(f"🧠 [2/2] Časová osa se generuje ve {n_chunks} navazujících úsecích — celý seznam "
              f"existujících klipů je posílán v KAŽDÉM úseku (Pravidlo č. 1).")
        if max_chunk_seconds:
            print(f"   (strop max. {max_chunk_seconds:.0f}s časového rozpětí na úsek — "
                  "ollama_plan_max_chunk_seconds, volba 13 — nezávisle na num_ctx.)")
        if n_chunks > 12:
            print(f"⚠️  {n_chunks} úseků je docela dost — každý úsek znovu posílá celý seznam "
                  "existujících klipů, takže na CPU to bude výrazně pomalejší než jedno velké "
                  "volání. Zvaž zvýšení num_ctx nebo ollama_plan_max_chunk_seconds (volba 13), ať je úseků méně.")

        timeline_parts = []
        shot_order_parts = []
        draft2_path = self.prompts_dir / "full_plan_ai_draft_part2.txt"
        for i, (chunk_text, (seg_start, seg_end)) in enumerate(zip(transcription_chunks, chunk_bounds), start=1):
            seg_start_fmt = f"{int(seg_start // 60):02d}:{seg_start - int(seg_start // 60) * 60:05.2f}"
            seg_end_fmt = f"{int(seg_end // 60):02d}:{seg_end - int(seg_end // 60) * 60:05.2f}"
            print(f"🧠 [2/2] Úsek {i}/{n_chunks} ({seg_start_fmt}–{seg_end_fmt})...")

            prompt2 = FULL_PLAN_PART2_TEMPLATE.format(
                transcription=chunk_text,
                song_duration=f"{song_duration:.2f}",
                existing_clips=existing_clips_for_chunks,
                new_clips_summary=new_clips_summary,
                segment_index=i,
                segment_total=n_chunks,
                segment_start=seg_start_fmt,
                segment_end=seg_end_fmt,
            )
            messages2 = [
                {"role": "system", "content": FULL_PLAN_PART2_SYSTEM_PROMPT},
                {"role": "user", "content": prompt2},
            ]
            raw2, model, error2 = self._generate_with_text_ai(
                messages2, phase="plan", settings=settings, temperature=0.3, timeout=3600, num_ctx=num_ctx,
            )
            if not raw2 or not raw2.strip():
                print(f"❌ Generování full_plan.txt selhalo (část 2/2, úsek {i}/{n_chunks}): {error2}")
                if timeline_parts:
                    partial = "### MUSIC_VIDEO_TIMELINE\n" + "\n".join(timeline_parts) + "\n\n### SHOT_ORDER\n" + "\n".join(shot_order_parts) + "\n"
                    draft2_path.write_text(partial, encoding="utf-8")
                    print(f"   Úspěšně dokončené úseky (1 až {i - 1}) jsem uložil do "
                          f"{draft2_path.relative_to(self.project_dir)} ke kontrole.")
                print(f"   Kreativní/asset sekce z části 1/2 zůstávají uložené v "
                      f"{draft1_path.relative_to(self.project_dir)} — full_plan.txt jsem NEPŘEPSAL.")
                print(f"   Zkus: `ollama pull {model}` — nebo v Nastavení (volba 13) zvol menší ollama_plan_model.")
                return False

            chunk_plan_text = clean_code_block_text(raw2.strip())
            chunk_sections = extract_sections(chunk_plan_text)
            chunk_timeline = chunk_sections.get("MUSIC_VIDEO_TIMELINE", "").strip()
            chunk_shot_order = chunk_sections.get("SHOT_ORDER", "").strip()

            if not chunk_timeline or not chunk_shot_order:
                print(f"⚠️  Úsek {i}/{n_chunks} AI výstupu postrádá MUSIC_VIDEO_TIMELINE a/nebo SHOT_ORDER.")
                if timeline_parts:
                    partial = "### MUSIC_VIDEO_TIMELINE\n" + "\n".join(timeline_parts) + "\n\n### SHOT_ORDER\n" + "\n".join(shot_order_parts) + "\n"
                    draft2_path.write_text(partial, encoding="utf-8")
                    print(f"   Úspěšně dokončené úseky (1 až {i - 1}) jsem uložil do "
                          f"{draft2_path.relative_to(self.project_dir)} ke kontrole.")
                print("   full_plan.txt jsem NEPŘEPSAL.")
                return False

            timeline_ok = bool(re.search(r"\[?\d{1,2}:\d{2}\.\d+\]?\s*-\s*\[?\d{1,2}:\d{2}\.\d+\]?\s*\|", chunk_timeline))
            if not timeline_ok:
                print(f"⚠️  Úsek {i}/{n_chunks}: MUSIC_VIDEO_TIMELINE nemá očekávaný formát časů "
                      "(MM:SS.ms - MM:SS.ms | id | ...).")
                if timeline_parts:
                    partial = "### MUSIC_VIDEO_TIMELINE\n" + "\n".join(timeline_parts) + "\n\n### SHOT_ORDER\n" + "\n".join(shot_order_parts) + "\n"
                    draft2_path.write_text(partial, encoding="utf-8")
                    print(f"   Úspěšně dokončené úseky (1 až {i - 1}) jsem uložil do "
                          f"{draft2_path.relative_to(self.project_dir)} ke kontrole.")
                print("   full_plan.txt jsem NEPŘEPSAL.")
                return False

            # ── Kontrola, že se model 'nerozutekl' mimo časové okno TOHOTO úseku ──
            # Bez tohohle se chyba (viz reálný případ: vnitřní řádky s časy v řádu
            # minut místo pár vteřin, navíc jeden vyloženě neplatný časový kód typu
            # '18:80.30') odhalí až po sloučení VŠECH úseků na úplném konci — tedy
            # po zbytečném vygenerování i zbylých, často mnohem delších úseků.
            bounds_problems = self._validate_timeline_chunk_bounds(chunk_timeline, seg_start, seg_end)
            if bounds_problems:
                print(f"❌ Úsek {i}/{n_chunks} ({seg_start_fmt}–{seg_end_fmt}): model vygeneroval "
                      f"časy mimo očekávané okno tohoto úseku ({len(bounds_problems)} problémů):")
                for p in bounds_problems[:10]:
                    print(f"   - {p}")
                if len(bounds_problems) > 10:
                    print(f"   ... a dalších {len(bounds_problems) - 10}.")
                if timeline_parts:
                    partial = "### MUSIC_VIDEO_TIMELINE\n" + "\n".join(timeline_parts) + "\n\n### SHOT_ORDER\n" + "\n".join(shot_order_parts) + "\n"
                    draft2_path.write_text(partial, encoding="utf-8")
                    print(f"   Úspěšně dokončené úseky (1 až {i - 1}) jsem uložil do "
                          f"{draft2_path.relative_to(self.project_dir)} ke kontrole.")
                print("   full_plan.txt jsem NEPŘEPSAL.")
                print("   Zkus jiný/větší model (ollama_plan_model, volba 13) — u tohohle úseku "
                      "se lokální model spletl v časování, nejde o problém se zdrojovými daty.")
                return False

            # ── Kontrola, že délka KAŽDÉHO řádku odpovídá skutečné/deklarované délce klipu ──
            # Bez tohohle projde i nesmysl typu "rap_01 | 00:00.00-03:53.66" (klip natažený
            # na délku celého úseku/písně místo svých pár vteřin) — `_validate_timeline_chunk_bounds`
            # výše to nezachytí, protože 03:53.66 leží uvnitř okna tohoto úseku, pokud úsek
            # pokrývá celou píseň. Chytá se to tady, hned po vygenerování úseku, s jasnou hláškou
            # KTERÝ klip a JAKÝ rozdíl oproti jeho skutečné délce — místo matoucí "díry v čase"
            # o desítky/stovky vteřin, která se jinak projeví až u navazujícího řádku/úseku.
            duration_problems = self._validate_timeline_row_durations(chunk_timeline, expected_clip_durations)
            if duration_problems:
                print(f"❌ Úsek {i}/{n_chunks} ({seg_start_fmt}–{seg_end_fmt}): {len(duration_problems)} "
                      "řádků má délku neodpovídající skutečné délce klipu:")
                for p in duration_problems[:10]:
                    print(f"   - {p}")
                if len(duration_problems) > 10:
                    print(f"   ... a dalších {len(duration_problems) - 10}.")
                if timeline_parts:
                    partial = "### MUSIC_VIDEO_TIMELINE\n" + "\n".join(timeline_parts) + "\n\n### SHOT_ORDER\n" + "\n".join(shot_order_parts) + "\n"
                    draft2_path.write_text(partial, encoding="utf-8")
                    print(f"   Úspěšně dokončené úseky (1 až {i - 1}) jsem uložil do "
                          f"{draft2_path.relative_to(self.project_dir)} ke kontrole.")
                print("   full_plan.txt jsem NEPŘEPSAL.")
                print("   Zkus jiný/větší model (ollama_plan_model, volba 13) nebo sniž "
                      "ollama_plan_max_chunk_seconds (volba 13) — u tohohle úseku se model "
                      "spletl v délce konkrétního klipu, nejde o problém se zdrojovými daty.")
                return False

            # Mechanicky sjednotíme hranice tohoto úseku, ať navazující úseky sedí na sebe
            # přesně (viz docstring _force_align_timeline_chunk_boundaries).
            chunk_timeline = self._force_align_timeline_chunk_boundaries(chunk_timeline, seg_start, seg_end)

            timeline_parts.append(chunk_timeline)
            shot_order_parts.append(chunk_shot_order)

        plan_text2 = "### MUSIC_VIDEO_TIMELINE\n" + "\n".join(timeline_parts) + "\n\n### SHOT_ORDER\n" + "\n".join(shot_order_parts) + "\n"
        sections2 = extract_sections(plan_text2)
        draft2_path.write_text(plan_text2, encoding="utf-8")

        # ── Sloučení obou částí do finálního full_plan.txt, v pevném kanonickém pořadí ──
        # (nezávisí na tom, v jakém pořadí model sekce skutečně vypsal v rámci každého volání)
        sections_all = {**sections1, **sections2}
        canonical_order = [
            "ANALYZA", "SONG_THEME", "ASSET_PLANNING", "VIDEO_STYLE", "RAP_CHARACTER_STYLE",
            "RAPPER_OUTFIT_PROMPT", "VIDEO_PROMPTS", "RAPPER_PROMPTS", "IMAGE_PROMPTS",
            "MUSIC_VIDEO_TIMELINE", "SHOT_ORDER", "EFFECTS", "METADATA", "NOVE_POTREBNE_KLIPY",
        ]
        plan_text = "\n\n".join(
            f"### {name}\n{sections_all.get(name, '').strip()}" for name in canonical_order
        ).strip() + "\n"

        # ── Validace sloučeného výstupu: musí obsahovat klíčové sekce, jinak NEPŘEPISUJEME full_plan.txt ──
        sections = extract_sections(plan_text)
        required = ["VIDEO_PROMPTS", "RAPPER_PROMPTS", "MUSIC_VIDEO_TIMELINE", "SHOT_ORDER", "METADATA"]
        missing = [s for s in required if s not in sections or not sections[s].strip()]

        draft_path = self.prompts_dir / "full_plan_ai_draft.txt"
        draft_path.write_text(plan_text, encoding="utf-8")

        if missing:
            print(f"⚠️  Sloučený AI výstup postrádá povinné sekce: {', '.join(missing)}.")
            print(f"   Výstup jsem NEPŘEPSAL do full_plan.txt, jen uložil ke kontrole do "
                  f"{draft_path.relative_to(self.project_dir)}.")
            return False

        # ── Kontrola ID použitých v timeline/shot_order proti katalogu + nově naplánovaným klipům ──
        # (bez tohohle se do full_plan.txt dřív mohla dostat ID, na která neodkazuje žádný
        # existující ani nově vytvořený prompt — timeline pak odkazuje "do prázdna")
        known_ids = set(klipy.keys()) | {
            m.group(1) for section_name in ("VIDEO_PROMPTS", "RAPPER_PROMPTS", "IMAGE_PROMPTS")
            for m in re.finditer(r'^([a-zA-Z]+_\d+)\s*\|', sections1.get(section_name, ""), re.MULTILINE)
        }
        id_problems = self._validate_timeline_ids(
            sections.get("MUSIC_VIDEO_TIMELINE", ""), sections.get("SHOT_ORDER", ""), known_ids,
        )
        if id_problems:
            print(f"❌ MUSIC_VIDEO_TIMELINE/SHOT_ORDER odkazuje na {len(id_problems)} neexistujících ID:")
            for p in id_problems:
                print(f"   - {p}")
            print(f"   Výstup jsem NEPŘEPSAL do full_plan.txt, jen uložil ke kontrole do "
                  f"{draft_path.relative_to(self.project_dir)}.")
            return False

        # ── Kontrola návaznosti/monotónnosti časové osy napříč všemi sloučenými úseky ──
        time_problems = self._validate_timeline_monotonic(sections.get("MUSIC_VIDEO_TIMELINE", ""), song_duration)
        if time_problems:
            print(f"❌ MUSIC_VIDEO_TIMELINE má {len(time_problems)} problémů s návazností času:")
            for p in time_problems:
                print(f"   - {p}")
            print(f"   Výstup jsem NEPŘEPSAL do full_plan.txt, jen uložil ke kontrole do "
                  f"{draft_path.relative_to(self.project_dir)}.")
            print("   Zkus zvýšit num_ctx (volba 13) — méně úseků v Části 2/2 = méně švů, kde k tomuhle může dojít.")
            return False

        # Vše v pořádku → zálohuj starý plán a zapiš nový
        if self.full_plan.exists() and self.full_plan.stat().st_size > 0:
            backup_path = self.prompts_dir / "full_plan.txt.bak"
            backup_path.write_text(self.full_plan.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            print(f"💾 Původní full_plan.txt zazálohován do {backup_path.relative_to(self.project_dir)}.")

        self.full_plan.write_text(plan_text, encoding="utf-8")
        new_clips_section = sections.get("NOVE_POTREBNE_KLIPY", "").strip()
        print(f"✅ full_plan.txt vygenerován AI ve 2 voláních a uložen ({len(plan_text)} znaků).")
        if new_clips_section:
            print("📋 Nově potřebné klipy (je třeba je ještě vygenerovat/natočit):")
            print("   " + new_clips_section.replace("\n", "\n   "))
        else:
            print("📋 AI nehlásí žádné nově potřebné klipy — vše pokryto existujícími klipy z klipy.md.")
        print("   👉 Doporučuji zkontrolovat full_plan.txt, pak pokračuj volbou 2 (Rozparsovat full_plan.txt).")
        return True

    def generate_video_plan(self):
        """Automaticky generuje kvalitní MUSIC_VIDEO_TIMELINE sekci v full_plan.txt.

        Kombinuje RAPPER_SEGMENT_ALIGNMENT s inteligentním sémantickým párováním
        B-roll klipů pomocí LLM, respektuje hudební strukturu (intro/verse/chorus/drop/outro).
        """
        if not self.full_plan.exists() or self.full_plan.stat().st_size == 0:
            print("❌ Soubor full_plan.txt chybí nebo je prázdný.")
            return

        plan_text = self.full_plan.read_text(encoding="utf-8")
        sections = extract_sections(plan_text)

        # ── 1. Načtení hudební struktury ──
        lyrics_structure = sections.get("LYRICS_STRUCTURE", "")
        song_sections = []  # [(name, start_sec, end_sec), ...]
        for line in lyrics_structure.splitlines():
            line = line.strip().lstrip("- ")
            if not line or ":" not in line:
                continue
            # Formát: "intro: 0:00 - 0:09 | popis"
            parts = line.split(":", 1)
            section_name = parts[0].strip()
            rest = parts[1].strip()
            time_match = re.match(r'(\d+:\d+)\s*-\s*(\d+:\d+)', rest)
            if time_match:
                t_start = parse_timecode(time_match.group(1))
                t_end = parse_timecode(time_match.group(2))
                desc_part = rest[time_match.end():].strip().lstrip("| ").strip()
                song_sections.append((section_name, t_start, t_end, desc_part))

        if not song_sections:
            print("⚠️  LYRICS_STRUCTURE nebyla nalezena nebo je prázdná. Použiji jednoduchou strukturu.")
            # Fallback: celý song = 1 sekce
            audio_path = self.find_audio()
            dur = probe_duration(audio_path) if audio_path else 180.0
            song_sections = [("song", 0.0, dur, "")]

        audio_path = self.find_audio()
        audio_duration = probe_duration(audio_path) if audio_path else song_sections[-1][2]

        beat_events = []
        beats_file = self.edit_dir / "beats.json"
        if beats_file.exists():
            try:
                beat_data = json.loads(beats_file.read_text(encoding="utf-8"))
                beat_events = beat_data.get("beat_events", [])
            except (OSError, json.JSONDecodeError):
                beat_events = []
        dramaturgy_plan = build_dramaturgy_plan(song_sections, beat_events)
        self._write_json(self.edit_dir / "dramaturgy.json", {
            "schema_version": 1,
            "sections": dramaturgy_plan,
        })
        print(f"🎭 Dramaturgie: {len(dramaturgy_plan)} hudebních sekcí zapsáno do EDIT_PROJECT/dramaturgy.json")

        # ── 2. Načtení rapper segment alignment ──
        rap_alignment = []  # [(start, end, clip_name), ...]
        alignment_text = sections.get("RAPPER_SEGMENT_ALIGNMENT", "")
        if not alignment_text:
            align_file = self.edit_dir / "rapper_segment_alignment.txt"
            if align_file.exists():
                alignment_text = align_file.read_text(encoding="utf-8")

        for line in alignment_text.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [x.strip() for x in line.split("|", 1)]
            clip_name = clean_asset_id(parts[0])
            time_range = parts[1]
            if "-" in time_range:
                try:
                    t_start_str, t_end_str = time_range.split("-", 1)
                    t_start = parse_timecode(t_start_str)
                    t_end = parse_timecode(t_end_str)
                    rap_alignment.append((t_start, t_end, clip_name))
                except Exception:
                    pass

        rap_alignment.sort(key=lambda x: x[0])
        if not rap_alignment:
            rap_json = self._load_json(self.edit_dir / "rap_alignment.json", {})
            skipped_invalid = 0
            for clip_name, data in rap_json.items():
                match = data.get("song_match", {})
                if match.get("song_start") is None or match.get("song_end") is None:
                    continue
                t_start = float(match["song_start"])
                t_end = float(match["song_end"])
                clip_duration = float(data.get("clip_duration", 0.0) or 0.0)
                max_reasonable = max(18.0, clip_duration * 3.0)
                if t_end <= t_start or (t_end - t_start) > max_reasonable:
                    skipped_invalid += 1
                    continue
                rap_alignment.append((t_start, t_end, clip_name))
            rap_alignment.sort(key=lambda x: x[0])
            if skipped_invalid and not rap_alignment:
                print("❌ rap_alignment.json obsahuje neplatné dlouhé shody. Spusť znovu volbu 5 a 6 po aktuální opravě skriptu.")
                return
        print(f"🎤 Načteno {len(rap_alignment)} rapper segmentů z alignmentu.")

        # ── 3. Načtení B-roll popisů (z broll_descriptions.json nebo VIDEO_PROMPTS) ──
        broll_descriptions = {}
        desc_file = self.edit_dir / "broll_descriptions.json"
        if desc_file.exists():
            with open(desc_file, "r", encoding="utf-8") as f:
                broll_descriptions = json.load(f)
            print(f"📝 Načteno {len(broll_descriptions)} B-roll popisů z broll_descriptions.json")
        else:
            # Fallback: extrahuj z VIDEO_PROMPTS
            vp = sections.get("VIDEO_PROMPTS", "") or sections.get("VERIFIED_VIDEO_ASSETS", "")
            if not vp and (self.prompts_dir / "video_prompts.txt").exists():
                vp = (self.prompts_dir / "video_prompts.txt").read_text(encoding="utf-8", errors="ignore")
            for line in vp.splitlines():
                line = line.strip().strip("`")
                if not line or "|" not in line:
                    continue
                parts = line.split("|", 1)
                clip_id = clean_asset_id(parts[0])
                desc = parts[1].strip().strip("`") if len(parts) > 1 else ""
                if clip_id.startswith("vid_"):
                    broll_descriptions[clip_id] = desc
            print(f"📝 Extrahováno {len(broll_descriptions)} B-roll popisů z VIDEO_PROMPTS")

        # Ověříme, které vid_XX soubory reálně existují
        existing_vids = set()
        if self.gen_vid.exists():
            existing_vids = {f.stem for f in self.gen_vid.glob("*.mp4") if f.stat().st_size > 500}
        # Filtrujeme popisy na existující soubory
        if existing_vids:
            broll_descriptions = {k: v for k, v in broll_descriptions.items() if k in existing_vids}
            for vid in sorted(existing_vids):
                broll_descriptions.setdefault(vid, vid)

        if not broll_descriptions:
            print("❌ Žádné B-roll popisy ani soubory nebyly nalezeny.")
            return

        # Načtení rapper promptů pro popis rapper segmentů v timeline
        rapper_descriptions = {}
        rp = sections.get("RAPPER_PROMPTS", "") or sections.get("VERIFIED_RAPPER_ASSETS", "")
        if not rp and (self.prompts_dir / "rapper_prompts.txt").exists():
            rp = (self.prompts_dir / "rapper_prompts.txt").read_text(encoding="utf-8", errors="ignore")
        for line in rp.splitlines():
            line = line.strip().strip("`")
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            clip_id = clean_asset_id(parts[0])
            desc = parts[1].strip().strip("`") if len(parts) > 1 else ""
            if clip_id.startswith("rap_"):
                # Extrahuj krátký popis (poslední věta s rapping:)
                rap_match = re.search(r'rapping[^:]*:\s*["\'](.+?)["\']', desc)
                rapper_descriptions[clip_id] = rap_match.group(1)[:80] if rap_match else desc[:80]

        # ── 4. Sestavení timeline: rap segmenty + mezery pro B-roll ──
        timeline_segments = []  # [(start, end, clip, trim, description), ...]

        # Přidáme rap segmenty
        for t_start, t_end, clip_name in rap_alignment:
            desc = rapper_descriptions.get(clip_name, clip_name)
            timeline_segments.append((t_start, t_end, clip_name, 0.0, desc))

        # Identifikujeme mezery
        timeline_segments.sort(key=lambda x: x[0])
        gaps = []  # [(gap_start, gap_end), ...]

        # Mezera na začátku (před prvním rap segmentem)
        if timeline_segments and timeline_segments[0][0] > 0.5:
            gaps.append((0.0, timeline_segments[0][0]))
        elif not timeline_segments:
            gaps.append((0.0, audio_duration))

        # Mezery mezi rap segmenty
        for i in range(len(timeline_segments) - 1):
            gap_start = timeline_segments[i][1]
            gap_end = timeline_segments[i + 1][0]
            if gap_end - gap_start > 0.5:  # Minimálně 0.5s mezera
                gaps.append((gap_start, gap_end))

        # Mezera na konci (po posledním rap segmentem)
        if timeline_segments:
            last_end = timeline_segments[-1][1]
            if audio_duration - last_end > 0.5:
                gaps.append((last_end, audio_duration))

        print(f"🔍 Nalezeno {len(gaps)} mezer k vyplnění B-rollem.")

        # ── 5. Sémantické párování B-roll klipů pro mezery ──
        # Načteme transkripci pro kontext
        trans_file = self.input_dir / "transcription.json"
        transcription_segments = []
        if trans_file.exists():
            with open(trans_file, "r", encoding="utf-8") as f:
                transcription_segments = json.load(f).get("segments", [])

        def get_lyrics_for_range(t_start, t_end):
            """Vrátí text lyrics pro daný časový rozsah."""
            texts = []
            for seg in transcription_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                # Překryv
                if seg_start < t_end and seg_end > t_start and seg.get("text", "").strip():
                    texts.append(seg["text"].strip())
            return " ".join(texts)

        def get_section_name(t):
            """Vrátí název hudební sekce pro daný čas."""
            for name, s_start, s_end, _ in song_sections:
                if s_start <= t < s_end:
                    return name
            return "unknown"

        # Kontrola Ollama (jednotná vrstva — respektuje settings.json ollama_enabled/ollama_model)
        settings = self.load_settings()
        use_llm = self._ollama_ready(settings)
        if use_llm:
            print(f"🧠 Ollama LLM aktivována pro sémantický výběr B-rollu (model: {self._ollama_model(settings)}).")
        else:
            print("⚠️  Ollama není dostupná/vypnutá. Použiji heuristické (cyklické) párování.")

        used_brolls = set()
        broll_segments = []

        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            lyrics_context = get_lyrics_for_range(gap_start, gap_end)
            section_name = get_section_name((gap_start + gap_end) / 2)

            # Hustota střihu je řízena dramaturgickým profilem sekce.
            section_profile = section_at_time(dramaturgy_plan, (gap_start + gap_end) / 2)
            MAX_BROLL_DUR = float(section_profile.get("max_shot_sec", 7.0))
            sub_starts = []
            t = gap_start
            while t < gap_end - 0.5:
                sub_end = min(t + MAX_BROLL_DUR, gap_end)
                sub_starts.append((t, sub_end))
                t = sub_end

            for sub_start, sub_end in sub_starts:
                sub_lyrics = get_lyrics_for_range(sub_start, sub_end)
                if not sub_lyrics:
                    sub_lyrics = lyrics_context  # Fallback na celkový kontext mezery

                chosen_clip = None
                chosen_reason = ""
                chosen_confidence = None

                if use_llm and broll_descriptions:
                    # Paměť použitých klipů — nabídneme přednostně nepoužité, ale
                    # když dojdou, uvolníme celou paměť (raději opakování než prázdný seznam).
                    available = {k: v for k, v in broll_descriptions.items() if k not in used_brolls}
                    if not available or len(available) < 2:
                        used_brolls.clear()
                        available = dict(broll_descriptions)

                    result = self._llm_choose_broll_clip(
                        section_name=section_name,
                        t_start=sub_start,
                        t_end=sub_end,
                        lyrics_context=sub_lyrics,
                        available_clips=available,
                        settings=settings,
                    )
                    if result:
                        chosen_clip = result["chosen_clip"]
                        chosen_reason = result["reason"]
                        chosen_confidence = result["confidence"]

                if not chosen_clip:
                    # Heuristický fallback: cyklicky přiřazujeme (beze změny původního chování)
                    available_list = [k for k in broll_descriptions.keys() if k not in used_brolls]
                    if not available_list:
                        used_brolls.clear()
                        available_list = list(broll_descriptions.keys())
                    chosen_clip = available_list[0]

                used_brolls.add(chosen_clip)
                desc = broll_descriptions.get(chosen_clip, "")[:80]
                desc = (
                    f"{desc} [SECTION={section_profile.get('key', 'unknown')}]"
                    f" [ENERGY={float(section_profile.get('energy', 0.5)):.2f}]"
                    f" [CUT_DENSITY={float(section_profile.get('cut_density', 0.45)):.2f}]"
                )
                broll_segments.append((sub_start, sub_end, chosen_clip, 0.0, desc))
                if chosen_confidence is not None:
                    print(f"  🧠 {sub_start:.2f}-{sub_end:.2f}s [{section_name}] → {chosen_clip} "
                          f"(confidence {chosen_confidence:.2f}: {chosen_reason})")
                print(f"  🎬 {sub_start:.2f}-{sub_end:.2f}s [{section_name}] → {chosen_clip}")

        # ── 6. Sloučení a seřazení všech segmentů ──
        all_segments = timeline_segments + broll_segments
        all_segments.sort(key=lambda x: x[0])

        # ── 7. Formátování MUSIC_VIDEO_TIMELINE ──
        timeline_lines = []
        timeline_lines.append("=" * 78)
        timeline_lines.append("")

        # Název projektu
        theme = sections.get("SONG_THEME", sections.get("PROJECT_THEME", self.project_dir.name.upper()))
        timeline_lines.append(f"{self.project_dir.name.upper()} — AUTOMATICKY GENEROVANÝ STŘIHOVÝ PLÁN")
        timeline_lines.append("")
        timeline_lines.append("=" * 78)

        current_section = None
        for t_start, t_end, clip, trim, desc in all_segments:
            # Zjistíme, zda jsme v nové hudební sekci
            sec_name = get_section_name(t_start)
            if sec_name != current_section:
                current_section = sec_name
                # Najdeme popis sekce
                sec_desc = ""
                sec_time = ""
                for sn, ss, se, sd in song_sections:
                    if sn == sec_name:
                        mm_s, ss_s = divmod(int(ss), 60)
                        mm_e, ss_e = divmod(int(se), 60)
                        sec_time = f"({mm_s:02d}:{ss_s:02d} - {mm_e:02d}:{ss_e:02d})"
                        sec_desc = sd
                        break
                timeline_lines.append("")
                timeline_lines.append(f"--- {sec_name.upper()} {sec_time} ---")
                timeline_lines.append("")

            # Efekty pro speciální sekce
            effect_tag = ""
            if "drop" in (current_section or "").lower():
                effect_tag = " [GLITCH]"

            timeline_lines.append(f"{t_start:06.2f} - {t_end:06.2f} | {clip} | 0.00{effect_tag} | {desc}")

        timeline_text = "\n".join(timeline_lines)

        # ── 8. Zápis do full_plan.txt ──
        # Odstraníme starou MUSIC_VIDEO_TIMELINE sekci, pokud existuje
        plan_content = self.full_plan.read_text(encoding="utf-8")

        # Najdeme a odstraníme starou sekci
        mvt_pattern = r'(### MUSIC_VIDEO_TIMELINE\s*\n)(.*?)(?=\n### [A-Z]|\Z)'
        if re.search(mvt_pattern, plan_content, re.DOTALL):
            plan_content = re.sub(mvt_pattern, '', plan_content, flags=re.DOTALL)

        # Přidáme novou sekci na konec
        plan_content = plan_content.rstrip() + "\n\n### MUSIC_VIDEO_TIMELINE\n" + timeline_text + "\n"

        self.full_plan.write_text(plan_content, encoding="utf-8")
        print(f"\n✅ Sekce MUSIC_VIDEO_TIMELINE byla vygenerována a zapsána do full_plan.txt")
        print(f"   Celkem {len(all_segments)} segmentů ({len(timeline_segments)} rap + {len(broll_segments)} B-roll)")
        print(f"   Nyní spusťte 'sync' pro načtení timeline a 'render' pro vykreslení.")

    def render_video(self, mode="draft", hd_mode="draft", use_fades=False, use_beat_sync=False):
        """Vyrenderuje finální hudební klip."""
        if mode == "final" and not self.validate_project(final=True, no_rap=False):
            print("❌ Final render zablokován: preflight validace projektu selhala.")
            return False
        lipsync_manifest_path = self.edit_dir / "word_phoneme_alignment.json"
        if lipsync_manifest_path.exists():
            lipsync_manifest = self._load_json(lipsync_manifest_path, {})
            rap_alignment_json = self._load_json(self.edit_dir / "rap_alignment.json", {})
            lipsync_ranges = []
            if isinstance(rap_alignment_json, dict):
                for clip_name, item in rap_alignment_json.items():
                    match = item.get("song_match", {}) if isinstance(item, dict) else {}
                    if match.get("song_start") is not None and match.get("song_end") is not None:
                        lipsync_ranges.append({
                            "clip": clip_name,
                            "start": match.get("song_start"),
                            "end": match.get("song_end"),
                        })
            lipsync_qa = validate_manifest_against_ranges(
                lipsync_manifest, lipsync_ranges, tolerance_ms=DEFAULT_WORD_TOLERANCE_MS
            ) if lipsync_ranges else {"ok": True, "errors": [], "warnings": ["rap_alignment.json neobsahuje segmenty"]}
            self._write_json(self.edit_dir / "word_phoneme_qa.json", lipsync_qa)
            if lipsync_qa.get("errors"):
                print("⚠️  Word-level lipsync drift:")
                for error in lipsync_qa["errors"][:10]:
                    print(f"   - {error}")
                if mode == "final":
                    print("❌ Final render zablokován: word-level lipsync drift překračuje toleranci.")
                    return False
        audio_path = self.find_audio()
        if not audio_path:
            print("❌ Audio soubor nebyl nalezen v INPUT/.")
            return

        audio_dur = probe_duration(audio_path)
        mode_label = "DRAFT" if mode == "draft" else "FINAL"
        res_label = {"draft": "640x360", "hd": "1280x720", "fullhd": "1920x1080"}.get(hd_mode, hd_mode)
        print(f"🎬 Inicializace střihu videoklipu (Režim: {mode_label}, Rozlišení: {res_label})")
        print(f"🎵 Audio: {audio_path.name} ({audio_dur:.2f}s)")

        if not self.timeline_file.exists():
            print(f"❌ Časová osa {self.timeline_file} neexistuje.")
            return

        if hd_mode == "fullhd":
            width, height, fps = 1920, 1080, 30
        elif hd_mode == "hd":
            width, height, fps = 1280, 720, 30
        else:
            width, height, fps = 640, 360, 24

        render_settings = self.load_settings()
        seed = ensure_seed(self.project_dir, render_settings.get("seed"))
        profile = profile_for(mode, hd_mode, fps)
        speed_min = float(render_settings.get("speed_min", 0.5) or 0.5)
        speed_max = float(render_settings.get("speed_max", 2.0) or 2.0)
        fps_override = render_settings.get("fps_override")
        if fps_override:
            try:
                fps = int(fps_override)
            except (TypeError, ValueError):
                pass
        profile = profile_for(mode, hd_mode, fps)

        beats = []
        beats_file = self.edit_dir / "beats.json"
        if use_beat_sync and beats_file.exists():
            try:
                with open(beats_file, "r") as f:
                    beat_data = json.load(f)
                beats = beat_data.get("beat_events") or enrich_beats(
                    beat_data.get("beats", []), bpm=beat_data.get("bpm")
                )
            except Exception:
                pass

        tmp_dir = self.export_dir / "_render_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True, parents=True)

        import re as _re
        raw_sections = []
        for line in self.timeline_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "|" not in line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 3:
                continue
            time_range = parts[0]
            asset_name = parts[1]
            note = " | ".join(parts[2:])
            try:
                start_raw, end_raw = [x.strip() for x in time_range.split("-", 1)]
                start = parse_timecode(start_raw)
                end = parse_timecode(end_raw)
                if start >= audio_dur:
                    continue
                raw_sections.append((start, min(end, audio_dur), asset_name, note))
            except Exception:
                continue

        def parse_speed_factor(note_text):
            m = _re.search(r'\[SPEED(?:\s+|=)([\d.]+)x?\]', note_text, _re.IGNORECASE)
            if m:
                return float(m.group(1))
            if '[SPEED]' in note_text.upper():
                return 1.4
            return None

        def parse_speed_applied(note_text):
            return "[SPEED_APPLIED]" in note_text.upper()

        def parse_trim_factor(note_text):
            m = _re.search(r'\[trim=([\d.]+)\]', note_text, _re.IGNORECASE)
            if m:
                return float(m.group(1))
            return 0.0

        segments = []
        for start, end, asset_name, note in raw_sections:
            duration = end - start
            speed = parse_speed_factor(note)
            trim = parse_trim_factor(note)
            speed_applied = parse_speed_applied(note)
            sync_anchor = nearest_sync_point(
                start, beats,
                prefer_downbeat=not asset_name.lower().startswith("rap"),
                tolerance_sec=0.20,
            ) if beats else None
            segment_motion = transition_plan({
                "note": note,
                "section": current_section,
                "energy": 0.75 if current_section in ("chorus", "drop") else 0.5,
                "duration": duration,
                "beat_is_downbeat": bool(sync_anchor and sync_anchor.get("is_downbeat")),
            })
            segments.append({
                "index": len(segments) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "asset": asset_name,
                "note": note,
                "speed": speed,
                "speed_applied": speed_applied,
                "trim": trim,
                "glitch": '[GLITCH]' in note.upper(),
                "whippan": '[WHIPPAN]' in note.upper(),
                "beat_anchor_ms": sync_anchor.get("time_ms") if sync_anchor else None,
                "beat_is_downbeat": bool(sync_anchor and sync_anchor.get("is_downbeat")),
                "beat_phrase_index": sync_anchor.get("phrase_index") if sync_anchor else None,
                "motion_style": segment_motion["style"],
                "motion_reason": segment_motion["reason"],
            })

        print(f"🎞️ Sestaveno {len(segments)} segmentů.")

        rendered_parts = []
        render_start_time = time.time()
        total_segments = len(segments)

        for i, seg in enumerate(segments, 1):
            asset_id = seg["asset"]
            prefix = asset_id.split("_", 1)[0]

            # Vypočet progresu a ETA
            elapsed = time.time() - render_start_time
            avg_per_seg = elapsed / (i - 1) if i > 1 else 0.0
            rem_segs = total_segments - i + 1
            eta_sec = int(avg_per_seg * rem_segs)
            mm, ss = divmod(eta_sec, 60)
            eta_str = f"{mm:02d}:{ss:02d}" if i > 1 else "--:--"

            bar_len = 20
            filled = int(bar_len * i / total_segments)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(f"  🎬 [{bar}] {i}/{total_segments} ({int(i/total_segments*100):2d}%) | {asset_id:<12} | ETA: {eta_str}", flush=True)

            if prefix == "rap":
                adjusted_path = self.project_dir / "gen_rap_adjusted" / f"{asset_id}.mp4"
                if adjusted_path.exists() and adjusted_path.stat().st_size > 500:
                    src_path = adjusted_path
                else:
                    src_path = self.gen_rap / f"{asset_id}.mp4"
            elif prefix == "vid":
                src_path = self.gen_vid / f"{asset_id}.mp4"
            elif prefix == "char":
                src_path = self.gen_char / f"{asset_id}.mp4"
            elif prefix == "pic":
                for ext in (".mp4", ".png", ".jpg"):
                    p = self.gen_pic / f"{asset_id}{ext}"
                    if p.exists():
                        src_path = p
                        break
                else:
                    src_path = self.gen_pic / f"{asset_id}.mp4"
            else:
                src_path = None

            if not src_path or not src_path.exists():
                err_msg = f"❌ Chybí soubor {asset_id}"
                print(err_msg)
                log_file = self.export_dir / "render_errors.log"
                import datetime
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"[{datetime.datetime.now().isoformat()}] Projekt: {self.project_dir.name} | Chyba: {err_msg}\n")
                return

            out_segment = tmp_dir / f"seg_{seg['index']:03d}.mp4"
            duration = seg["duration"]
            is_image = src_path.suffix.lower() in (".png", ".jpg", ".jpeg")

            speed_factor = seg.get("speed")

            filters = [
                f"scale={width}:{height}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
                "setsar=1",
                f"fps={fps}"
            ]

            if prefix == "rap":
                # Rap klipy jsou vždy už fyzicky přepočítané na cílovou délku
                # (before + adjusted_rap + after baked do souboru) — žádná
                # dodatečná úprava rychlosti při renderu se neaplikuje.
                pass
            elif speed_factor is not None:
                pts_factor = round(1.0 / speed_factor, 4)
                filters.append(f"setpts={pts_factor}*PTS")
            elif not is_image:
                src_dur = probe_duration(src_path)
                if src_dur > 0 and src_dur < duration - 0.05:
                    needed_speed = round(src_dur / duration, 4)
                    # OPRAVA: dřív se tu klip natáhl na přesnou délku slotu BEZ OHLEDU
                    # na speed_min/speed_max z nastavení — pokud align_vid_clips (volba V)
                    # už rychlost oříznula na limit a klipu tak zbyl deficit, render ho
                    # tady potichu dotahoval libovolně extrémním zpomalením (žádný clamp,
                    # žádné varování). Teď se použije STEJNÝ limit jako ve volbě V; pokud
                    # by ho natažení na přesnou délku porušilo, radši se klip nechá kratší
                    # (a řádek v timeline se pak při concatu jen na chvíli podrží poslední
                    # snímek/zkrátí) než aby vznikl neomezeně "gumový" záběr.
                    clamped_speed = max(speed_min, min(speed_max, needed_speed))
                    if clamped_speed != needed_speed:
                        print(f"  ⚠️  {asset_id}: dopočet délky v renderu by vyžadoval rychlost "
                              f"{needed_speed}x, oříznuto na limit {speed_min}-{speed_max} "
                              f"(nastav volbou 13, nebo klip prodluž/zkrať).")
                    pts_factor = round(1.0 / clamped_speed, 4)
                    filters.append(f"setpts={pts_factor}*PTS")

            if mode == "final":
                filters.extend([
                    "eq=contrast=1.22:saturation=1.25:brightness=-0.035",
                    "colorbalance=rs=0.12:gs=-0.08:bs=0.18",
                    "vignette=0.5"
                ])
            else:
                filters.append("eq=contrast=1.10:saturation=1.10")

            if is_image:
                filters.append(f"zoompan=z='zoom+0.0018':x=iw/2-(iw/zoom/2):y=ih/2-(ih/zoom/2):d=1:s={width}x{height}")

            filters.extend(motion_filters(seg.get("motion_style", "clean"), duration))
            vf_filter = ",".join(filters)

            cmd = video_segment_command(
                src_path, out_segment, duration, vf_filter,
                is_image=is_image, profile=profile,
            )

            if not run_ffmpeg(cmd, quiet=True):
                err_msg = f"❌ Selhalo renderování segmentu {asset_id}"
                print(err_msg)
                log_file = self.export_dir / "render_errors.log"
                import datetime
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"[{datetime.datetime.now().isoformat()}] Projekt: {self.project_dir.name} | Asset: {asset_id} | Příkaz: {' '.join(cmd)}\n")
                return

            rendered_parts.append(out_segment)

        # Concat + audio + fades
        print("🔗 Spojuji segmenty...")
        concat_list_file = tmp_dir / "concat_list.txt"
        concat_manifest(rendered_parts, concat_list_file)
        merged_video = tmp_dir / "merged_no_audio.mp4"
        run_cmd(concat_command(concat_list_file, merged_video), quiet=True)


        today_str = __import__('datetime').date.today().strftime("%Y-%m-%d")
        seq = 1
        while True:
            candidate = self.export_dir / f"final_{mode}_{hd_mode}_{today_str}_{seq:02d}.mp4"
            if not candidate.exists():
                break
            seq += 1
        final_video = candidate

        print("🎵 Připojuji audio...")
        run_cmd(mux_audio_command(merged_video, audio_path, final_video, profile=profile), quiet=True)

        if use_fades:
            fade_video = self.export_dir / f"final_{mode}_{hd_mode}_{today_str}_{seq:02d}_fades.mp4"
            dur = probe_duration(final_video)
            run_cmd(fade_command(final_video, fade_video, dur, profile=profile), quiet=True)
            final_video = fade_video

        loudness = run_loudness_audit(final_video)
        qa = ffprobe_media_qa(final_video, expected_duration=audio_dur)
        video_meta = qa.get("video", {}) if isinstance(qa, dict) else {}
        visual = audit_video(
            final_video, audio_dur,
            width=video_meta.get("width"),
            height=video_meta.get("height"),
            sample_count=12,
        )
        qa["loudness"] = loudness
        qa["visual"] = visual
        if not loudness.get("ok"):
            qa.setdefault("errors", []).extend(loudness.get("errors", []))
            qa["ok"] = False
        if not visual.get("ok"):
            qa.setdefault("errors", []).extend(visual.get("errors", []))
            qa["ok"] = False
        if visual.get("warnings"):
            qa.setdefault("warnings", []).extend(visual.get("warnings", []))
        qa_report = final_video.with_suffix(final_video.suffix + ".qa.json")
        self._write_json(qa_report, qa)
        registry_path = append_render_event(
            self.project_dir,
            output=final_video,
            mode=mode,
            resolution=hd_mode,
            duration=probe_duration(final_video),
            qa=qa,
            seed=seed,
        )
        qa_summary = write_qa_summary(self.project_dir)
        print(f"   Render registry: {registry_path}")
        print(f"   QA summary: {qa_summary}")
        if not qa.get("ok"):
            print("❌ Post-render QA selhala:")
            for error in qa.get("errors", []):
                print(f"   - {error}")
            if mode == "final":
                print(f"   Report: {qa_report}")
                return False
            print("⚠️ Draft výstup bude ponechán pro diagnostiku.")

        print(f"\n✅ Hotovo! Finální soubor: {final_video}")
        preview_report = write_preview_report(self.project_dir, qa=qa, seed=seed)
        contact_paths = sorted(
            list(self.gen_pic.glob("*.png")) + list(self.gen_pic.glob("*.jpg"))
        )[:64]
        contact_sheet = generate_contact_sheet(
            contact_paths, self.edit_dir / "contact_sheet.jpg"
        ) if contact_paths else None
        print(f"   QA report: {qa_report}")
        print(f"   Preview report: {preview_report}")
        if contact_sheet:
            print(f"   Kontaktní list: {contact_sheet}")
        return True

    def generate_preview_report(self):
        """Vytvoří souhrnný report bez nutnosti spouštět nový render."""
        seed = ensure_seed(self.project_dir, self.load_settings().get("seed"))
        qa_files = sorted(self.export_dir.glob("*.mp4.qa.json"), key=lambda path: path.stat().st_mtime, reverse=True) if self.export_dir.exists() else []
        qa = None
        if qa_files:
            try:
                qa = json.loads(qa_files[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                qa = None
        catalog = self._load_klipy_md()
        catalog_quality_path = write_catalog_quality_report(self.project_dir, catalog) if catalog else None
        report = write_preview_report(self.project_dir, qa=qa, seed=seed)
        if catalog_quality_path:
            try:
                preview_data = json.loads(report.read_text(encoding="utf-8"))
                preview_data["catalog_quality_report"] = str(catalog_quality_path.relative_to(self.project_dir))
                report.write_text(json.dumps(preview_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                pass
        contact_paths = sorted(list(self.gen_pic.glob("*.png")) + list(self.gen_pic.glob("*.jpg")))[:64]
        contact_sheet = generate_contact_sheet(contact_paths, self.edit_dir / "contact_sheet.jpg") if contact_paths else None
        print(f"✅ Preview report: {report}")
        if catalog_quality_path:
            print(f"✅ Katalog quality report: {catalog_quality_path}")
        if contact_sheet:
            print(f"✅ Kontaktní list: {contact_sheet}")
        return True

    def generate_qa_summary(self):
        """Regeneruje agregovaný QA summary z dostupných QA reportů a registry."""
        output = write_qa_summary(self.project_dir)
        print(f"✅ QA summary: {output}")
        return True

    def show_render_registry(self):
        """Vypíše poslední render registry události a uloží aktuální QA summary."""
        records = read_render_registry(self.project_dir)
        summary = write_qa_summary(self.project_dir)
        print(f"Render registry: {len(records)} událostí")
        for record in records[-10:]:
            print(f"- {record.get('timestamp_utc')} | {record.get('mode')} | {record.get('resolution')} | {record.get('status')} | QA={record.get('qa_ok')} | {record.get('output_relative')}")
        print(f"QA summary: {summary}")
        return True

    def generate_social_exports(self, source: Path | None = None, profiles: tuple[str, ...] = ("youtube", "vertical", "square")):
        """Vytvoří standardizované social exporty z posledního nebo zadaného masteru."""
        source = source or self._latest_render()
        if not source or not source.exists():
            print("❌ Nebyl nalezen zdrojový render pro social export.")
            return False
        social_dir = self.export_dir / "social"
        social_dir.mkdir(parents=True, exist_ok=True)
        for name in profiles:
            profile = social_profile_for(name)
            output = social_dir / f"{source.stem}_{profile.name}.mp4"
            command = social_export_command(source, output, profile)
            if not run_ffmpeg(command, quiet=True):
                print(f"❌ Social export selhal: {profile.name}")
                return False
            print(f"✅ Social export: {output}")
        return True

    def generate_thumbnail_candidates(self, source: Path | None = None, count: int = 8):
        """Vygeneruje časové thumbnail kandidáty z posledního renderu a zapíše manifest."""
        source = source or self._latest_render()
        if not source or not source.exists():
            print("❌ Nebyl nalezen zdrojový render pro thumbnails.")
            return False
        duration = probe_duration(source)
        if duration <= 0:
            return False
        thumb_dir = self.edit_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for index in range(max(1, count)):
            time_sec = duration * (index + 1) / (max(1, count) + 1)
            output = thumb_dir / f"candidate_{index + 1:02d}.jpg"
            if not run_ffmpeg(thumbnail_command(source, output, time_sec), quiet=True):
                continue
            candidates.append({"index": index + 1, "time_sec": round(time_sec, 3), "path": str(output), "sharpness": 0.5, "brightness": 0.5, "subject_score": 0.5, "black_ratio": 0.0})
        ranked = rank_thumbnail_candidates(candidates)
        manifest = self.edit_dir / "thumbnail_candidates.json"
        manifest.write_text(json.dumps({"source": str(source), "candidates": ranked}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"✅ Thumbnail kandidáti: {manifest}")
        return True

    def generate_ab_variants(self, base_seed: int | None = None):
        """Vytvoří reprodukovatelný manifest A/B variant pro následné renderování."""
        if base_seed is None:
            base_seed = int(self.load_settings().get("seed", 0) or 0)
        source = self._latest_render()
        output = self.edit_dir / "experiment_manifest.json"
        write_experiment_manifest(self.project_dir, output, base_seed, str(source) if source else None)
        print(f"✅ Experimentální manifest: {output}")
        return True

    def clean_exports(self, keep_last: int = 5):
        """Smaže staré render soubory z EXPORT/, ponechá posledních N verzí.
        Soubory final_stable_v*.mp4 jsou chráněny a nikdy se nemažou.
        """
        if not self.export_dir.exists():
            print("❌ EXPORT složka neexistuje.")
            return

        all_renders = sorted(
            [f for f in self.export_dir.glob("final_*.mp4") if "stable" not in f.name],
            key=lambda f: f.stat().st_mtime,
            reverse=True  # nejnovější první
        )
        stable_files = list(self.export_dir.glob("final_stable_v*.mp4"))
        to_delete = all_renders[keep_last:]

        print(f"\n📦 EXPORT složka: {self.export_dir}")
        print(f"   Render souborů celkem: {len(all_renders)}")
        print(f"   Stabilních verzí (chráněno): {len(stable_files)}")
        print(f"   Ponechat posledních: {keep_last}")

        if not to_delete:
            total_mb = sum(f.stat().st_size for f in all_renders) // 1024**2
            print(f"✅ Nic ke smazání ({total_mb} MB v {len(all_renders)} souborech).")
            return

        total_freed = sum(f.stat().st_size for f in to_delete)
        print(f"\n⚠️  Ke smazání ({len(to_delete)} souborů, seřazeno od nejstaršího):")
        for f in reversed(to_delete):
            size_mb = f.stat().st_size // 1024**2
            print(f"   🗑️  {f.name}  ({size_mb} MB)")

        confirm = input(f"\nOpravdu smazat {len(to_delete)} souborů? (a/N): ").strip().lower()
        if confirm != 'a':
            print("❌ Čištění zrušeno.")
            return

        for f in to_delete:
            f.unlink()
            print(f"   ✓ Smazán: {f.name}")

        freed_mb = total_freed // 1024**2
        print(f"\n✅ Uvolněno: {freed_mb} MB  ({len(to_delete)} souborů smazáno)")
        print(f"   Ponecháno: {len(all_renders) - len(to_delete)} render + {len(stable_files)} stabilní")

    def tag_stable(self):
        """Označí poslední render jako stabilní verzi (final_stable_vN.mp4).
        Stabilní verze nejsou mazány příkazem clean_exports().
        """
        if not self.export_dir.exists():
            print("❌ EXPORT složka neexistuje.")
            return

        all_renders = sorted(
            [f for f in self.export_dir.glob("final_*.mp4") if "stable" not in f.name],
            key=lambda f: f.stat().st_mtime
        )
        if not all_renders:
            print("❌ Žádný render soubor k označení v EXPORT/.")
            return

        last_render = all_renders[-1]
        size_mb = last_render.stat().st_size // 1024**2
        print(f"\n📁 Poslední render: {last_render.name}  ({size_mb} MB)")

        # Najdi příští číslo verze
        existing_versions = []
        for f in self.export_dir.glob("final_stable_v*.mp4"):
            m = re.search(r'final_stable_v(\d+)', f.name)
            if m:
                existing_versions.append(int(m.group(1)))
        version = max(existing_versions, default=0) + 1

        dest = self.export_dir / f"final_stable_v{version}.mp4"
        print(f"📌 Označit jako: {dest.name}")

        confirm = input("Potvrdit? (a/N): ").strip().lower()
        if confirm != 'a':
            print("❌ Tagování zrušeno.")
            return

        shutil.copy2(str(last_render), str(dest))
        size_dest = dest.stat().st_size // 1024**2
        print(f"✅ Vytvořena stabilní verze: {dest.name}  ({size_dest} MB)")

# ===== AUTOMATICKÁ DETEKCE PROJEKTU =====

def detect_project(project_arg) -> Path:
    """Zjistí složku projektu podle argumentu nebo aktuálního adresáře."""
    if project_arg:
        p = ROOT / project_arg
        if p.exists() and p.is_dir():
            return p
        p = Path(project_arg).resolve()
        if p.exists() and p.is_dir():
            return p
        print(f"❌ Složka projektu '{project_arg}' nebyla nalezena.")
        sys.exit(1)

    cwd = Path.cwd()
    if (cwd / "INPUT").exists() and (cwd / "Prompts").exists():
        return cwd

    subdirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name not in ("venv", "scripts", "EXPORT", "TEST", "__pycache__")])
    if not subdirs:
        print("❌ Nebyl nalezen žádný projekt v aktuálním adresáři a žádný nebyl zadán.")
        sys.exit(1)

    print("\nVyberte složku projektu:")
    for idx, d in enumerate(subdirs):
        print(f"{idx + 1} - {d.name}")

    while True:
        try:
            choice = input("Volba (nebo Q pro zrušení): ").strip().lower()
            if choice == 'q':
                sys.exit(0)
            val = int(choice)
            if 1 <= val <= len(subdirs):
                return subdirs[val - 1]
        except (ValueError, IndexError):
            pass
        print("Neplatná volba. Zadejte prosím číslo ze seznamu.")

# ===== INTERAKTIVNÍ MENU =====

def _list_project_dirs():
    """Vrátí seřazený seznam existujících projektových složek v kořenovém adresáři."""
    return sorted([
        d for d in ROOT.iterdir()
        if d.is_dir() and not d.name.startswith('.')
        and d.name not in ("venv", "scripts", "EXPORT", "TEST", "__pycache__")
    ])


def _select_project_dir(subdirs, allow_new: bool = True):
    """Zobrazí očíslovaný seznam projektových složek a nechá uživatele vybrat.
    Vrací Path k vybrané/nově vytvořené složce, nebo None při stornu (Enter/Q).
    Sdílená pomocná funkce pro úvodní výběr projektu i volbu 12 (Změnit aktivní
    projekt) — dřív byl tento výpis a smyčka duplikované na dvou místech."""
    print("\nVyberte složku projektu:")
    for idx, d in enumerate(subdirs):
        print(f"{idx + 1} - {d.name}")
    if allow_new:
        print("N - Vytvořit nový projekt")

    while True:
        choice = input("Volba (Enter nebo Q pro storno): ").strip().lower()
        if choice in ('', 'q'):
            return None
        if allow_new and choice == 'n':
            name = input("Název nového projektu: ").strip()
            if not name:
                continue
            new_dir = ROOT / name
            TemagenPipeline(new_dir).init_project()
            return new_dir
        try:
            val = int(choice)
            if 1 <= val <= len(subdirs):
                return subdirs[val - 1]
        except ValueError:
            pass
        hint = "číslo" + (", N" if allow_new else "") + " nebo Q"
        print(f"Neplatná volba. Vyberte {hint}.")


def interactive_menu():
    """Spustí interaktivní textové ovládací menu."""
    print("\n==============================================")
    print("🎬 TEMAGEN MUSIC VIDEO PIPELINE — OVLÁDACÍ MENU")
    print("==============================================")
    print("Plán střihu (Prompts/full_plan.txt) lze vytvořit 3 způsoby:")
    print("  1) 8a+8b zde — AI plán z lyrics.txt (rap zpracuješ kroky 5-7, nebo ho přeskočíš)")
    print("  2) 8c zde — starý algoritmus bez AI, jen doplní B-roll do děr v plánu")
    print("  3) samostatný klipy.py — plán jen z existujícího INPUT/klipy.md (bez AI časování)")
    print("Bez rapu (Character mód) = při krocích 9/10/11 použij volby C/R a kroky 5-6 přeskoč.")
    print("Po vytvoření plánu (jakýmkoliv způsobem) pokračuj odsud volbou 2 dál.")

    cwd = Path.cwd()
    project_dir = None
    if (cwd / "INPUT").exists() and (cwd / "Prompts").exists():
        project_dir = cwd
        print(f"👉 Automaticky detekován projekt v aktuální složce: {cwd.name}")

    if not project_dir:
        subdirs = _list_project_dirs()
        if not subdirs:
            print("⚠️ Nebyl nalezen žádný existující projekt v kořenovém adresáři.")
            name = input("Zadejte název pro vytvoření nového projektu (nebo Enter pro konec): ").strip()
            if not name:
                return
            project_dir = ROOT / name
            TemagenPipeline(project_dir).init_project()
        else:
            project_dir = _select_project_dir(subdirs, allow_new=True)
            if project_dir is None:
                return

    pipeline = TemagenPipeline(project_dir)

    menu_help = {
        "1": "[PROJEKT] Vytvoří adresářovou strukturu projektu (INPUT/, Prompts/, EDIT_PROJECT/...). Bezpečné spustit opakovaně, nic nepřepíše.",
        "12": "[PROJEKT] Přepne aktivní projekt na jinou složku (nebo založí novou).",
        "13": "[PROJEKT] Nastavení: Whisper/Groq transkripce, CPU/GPU, FPS, speed limity, AI provider pro 8a (8b vždy lokálně).",
        "4": "[1. PLÁN] Song analýza — Whisper/Groq transkripce písně + beaty → potřeba pro 8a/8b i pro přesné zarovnání rapu.",
        "8": "[1. PLÁN] Vytvoří Prompts/full_plan.txt: 8a AI scénář, 8b AI kompletní plán (s rapem, z klipy.md), 8c starý algoritmus bez AI.",
        "2": "[2. NAČTENÍ PLÁNU] Rozparsuje Prompts/full_plan.txt do jednotlivých promptů, timeline.txt a metadat.",
        "3": "[2. NAČTENÍ PLÁNU] Vytvoří prázdné placeholder soubory pro chybějící média podle promptů.",
        "5": "[3. RAP] Přepíše rap_xx.mp4 klipy → EDIT_PROJECT/rap_alignment.json. Vyžaduje klipy v gen_rap/.",
        "u": "[3. RAP] Po ruční úpravě 'transcript_raw'/'rap_start'/'rap_end' v rap_alignment.json znovu "
             "vyhledá lyrics_window podle lyrics.txt — BEZ nové Whisper transkripce.",
        "6": "[3. RAP] Spočítá speed korekce rap klipů podle song_alignment.json. Vyžaduje volbu 5.",
        "7": "[3. RAP] Přepočítá timeline.txt podle kotev rap_start ze speed_corrections.json. Vyžaduje volbu 6.",
        "l": "[3. RAP] Kompletní zkratka: přepis písně česky → song analýza → export WAV segmentů pro rap.",
        "i": "[3. RAP] Vloží hotové lip-sync segmenty (z gen_rap/) zpět do timeline.txt.",
        "t": "[3. RAP] Přepočítá rychlosti klipů podle timeline.txt beze změny časů/pořadí.",
        "v": "[4. B-ROLL] Zarovná vid_xx broll klipy na přesnou délku ze slotu v timeline.txt (se zálohou originálů).",
        "9": "[5. RENDER] Zkontroluje konzistenci projektu (chybějící média, časy, formát) — draft nebo final přísnost.",
        "10": "[5. RENDER] Vyrenderuje finální video S RAPEM podle timeline.txt (dotáže se na režim/rozlišení/efekty).",
        "r": "[5. RENDER] Jako 10, ale BEZ RAPU (Character mód).",
        "11": "[5. RENDER] Spustí VŠECHNY kroky od začátku po render S RAPEM na jedno tlačítko.",
        "c": "[5. RENDER] Jako 11, ale BEZ RAPU (Character mód).",
        "e": "[SPRÁVA] Vyčistí staré render soubory v EXPORT/ nebo označí aktuální jako stabilní verzi.",
        "j": "[SOCIAL] Vytvoří YouTube, vertikální a čtvercový export z posledního renderu.",
        "k": "[THUMBNAILS] Vytvoří a seřadí kandidáty titulních snímků.",
        "a": "[A/B] Vytvoří experimentální manifest s kontrolní a dvěma kreativními variantami.",
        "q": "[QA] Vytvoří agregovaný QA summary ze všech render reportů.",
        "g": "[OBSERVABILITY] Vypíše poslední události render registry.",
    }

    while True:
        settings = pipeline.load_settings()
        print("\n============================================================")
        print("            AI VIDEO PIPELINE (PRO) - HLAVNÍ MENU")
        print("============================================================")
        print(f"Aktivní projekt: {project_dir.name}")
        print(pipeline.status_summary_line(settings))
        print(pipeline.project_progress_checklist())
        print("------------------------------------------------------------")
        print("── PROJEKT ──────────────────────────────────────────────")
        print("1  - Založit nový projekt (vytvořit INPUT/, Prompts/, EDIT_PROJECT/...)")
        print("12 - Přepnout na jiný projekt")
        print("13 - Nastavení (Whisper, CPU/GPU, FPS, speed limity, AI provider)")
        print("── 1) VYTVOŘIT STŘIHOVÝ PLÁN (full_plan.txt) ───────────────")
        print("     Vyber JEDNU z cest níže (8a→8b s AI, nebo 8c bez AI).")
        print("     Plán postavený jen na INPUT/klipy.md dělá samostatný klipy.py.")
        print("4  - Analyzovat píseň (Whisper přepis + beaty) — udělej před 8a/8b")
        print("8a - Vygenerovat SCÉNÁŘ pomocí AI (Fáze A, z lyrics.txt)")
        print("8b - Vygenerovat CELÝ PLÁN pomocí AI (Fáze B, ze scénáře + klipy.md, S RAPEM)")
        print("8c - Vygenerovat plán BEZ AI (starý algoritmus, jen doplní B-roll)")
        print("── 2) NAČÍST PLÁN DO PROJEKTU ───────────────────────────────")
        print("2  - Rozparsovat full_plan.txt → timeline, prompty, metadata")
        print("3  - Vytvořit prázdné placeholdery pro chybějící média")
        print("── 3) RAP KLIPY (přeskoč, pokud je klip BEZ RAPU) ───────────")
        print("5  - Přepsat rap klipy (transkripce)")
        print("U  - Po ruční úpravě rap_alignment.json znovu vyhledat lyrics_window (bez re-transkripce)")
        print("6  - Doladit tempo rap klipů na píseň (speed)")
        print("7  - Zarovnat timeline podle skutečné pozice rapu v písni")
        print("L  - Kompletní zkratka přípravy lip-sync (přepis → analýza → export segmentů)")
        print("I  - Vložit hotové lip-sync segmenty zpět do timeline")
        print("T  - Přepočítat rychlosti klipů podle timeline (beze změny časů/pořadí)")
        print("── 4) B-ROLL KLIPY ──────────────────────────────────────────")
        print("V  - Doladit vid_xx klipy na přesnou délku slotu v timeline (se zálohou)")
        print("── 5) KONTROLA A RENDER ─────────────────────────────────────")
        print("9  - Zkontrolovat projekt (validace) před renderem")
        print("10 - Vyrenderovat video (S RAPEM)")
        print("R  - Vyrenderovat video (BEZ RAPU, Character mód)")
        print("11 - Spustit VŠECHNY kroky od začátku po render (S RAPEM)")
        print("C  - Spustit VŠECHNY kroky od začátku po render (BEZ RAPU, Character mód)")
        print("── SPRÁVA ────────────────────────────────────────────────")
        print("E  - Spravovat staré rendery v EXPORT/")
        print("J  - Exportovat social formáty (YouTube / vertical / square)")
        print("K  - Vygenerovat thumbnail kandidáty")
        print("A  - Vytvořit A/B experimentální manifest")
        print("Q  - Vytvořit automatický QA summary")
        print("G  - Zobrazit render registry")
        print("H  - Nápověda (co která volba dělá)")
        print("0  - Konec")
        print("============================================================")

        choice = input("\nVolba: ").strip().lower()
        if choice in ('q', '0'):
            print("👋 Ahoj!")
            break
        elif choice in ('h', '?'):
            print("\n📖 NÁPOVĚDA K VOLBÁM")
            print("------------------------------------------------------------")
            for key, desc in menu_help.items():
                print(f"{key.upper():>3} - {desc}")
            print("------------------------------------------------------------")
        elif choice == '1':
            pipeline.init_project()
        elif choice == '2':
            pipeline.parse_plan()
        elif choice == '3':
            pipeline.create_placeholders()
        elif choice == '4':
            pipeline.analyze_song()
        elif choice == '5':
            pipeline.transcribe_rap_clips()
        elif choice == 'u':
            pipeline.resync_rap_alignment_from_lyrics()
        elif choice == '6':
            pipeline.align_rap_clips()
        elif choice == '7':
            pipeline.update_timeline_from_alignment()
        elif choice == 'v':
            pipeline.align_vid_clips()
        elif choice == '8':
            print("\n🎬 VYGENEROVAT STŘIHOVÝ PLÁN")
            print("------------------------------------------------------------")
            print(f"    (AI provider pro 8a podle Nastavení: {pipeline._text_ai_provider(settings)}  |  8b vždy: local/{pipeline._ollama_plan_model(settings)})")
            print("8a - Fáze A: Vygenerovat scénář (lokální AI / Groq Cloud — nastaveno v 13, z lyrics.txt + mood.txt + postava.txt)")
            print("8b - Fáze B: Vygenerovat full_plan.txt (VŽDY lokální Ollama, ze scénáře + klipy.md)")
            print("8c - Starý algoritmický postup (bez AI scénáře, jen doplní díry B-rollem)")
            sub_choice = input("Volba (nebo Enter pro návrat): ").strip().lower()
            if sub_choice == '8a':
                pipeline.generate_scenario_ai()
            elif sub_choice == '8b':
                pipeline.generate_full_plan_ai()
            elif sub_choice == '8c':
                pipeline.generate_video_plan()
            else:
                print("Návrat do hlavního menu.")
        elif choice == '8a':
            pipeline.generate_scenario_ai()
        elif choice == '8b':
            pipeline.generate_full_plan_ai()
        elif choice == '8c':
            pipeline.generate_video_plan()
        elif choice == '9':
            m_choice = input("Režim validace (1 - draft, 2 - final): ").strip()
            mode_label = "final" if m_choice == "2" else "draft"
            print(f"   Použit režim: {mode_label}.")
            pipeline.validate_project(final=(mode_label == "final"))
        elif choice == 'l':
            if pipeline.transcribe_song_czech():
                if pipeline.analyze_song():
                    if not pipeline.export_lipsync_audio_segments():
                        print("⚠️  Export lip-sync WAV segmentů selhal — song analýza proběhla, ale poslední krok se nedokončil (viz chyba výše).")
                else:
                    print("⚠️  Song analýza po přepisu selhala — export WAV segmentů se nespustí (viz chyba výše).")
            else:
                print("⚠️  Přepis písně česky selhal — zbytek přípravy lip-sync se nespustí (viz chyba výše).")
        elif choice == 'i':
            if not pipeline.inject_lipsync_segments_into_timeline():
                print("⚠️  Vložení lip-sync segmentů do timeline se nezdařilo (viz chyba výše).")
        elif choice == '10':
            pipeline.run_render_flow(no_rap=False)
        elif choice == '11':
            pipeline.run_all_flow(no_rap=False)
        elif choice == 'c':
            pipeline.run_all_flow(no_rap=True)
        elif choice == 'r':
            pipeline.run_render_flow(no_rap=True)
        elif choice == 'j':
            pipeline.generate_social_exports()
        elif choice == 'k':
            pipeline.generate_thumbnail_candidates()
        elif choice == 'a':
            pipeline.generate_ab_variants()
        elif choice == 'q':
            pipeline.generate_qa_summary()
        elif choice == 'g':
            pipeline.show_render_registry()
        elif choice == 'e':
            export_dir = pipeline.export_dir
            if export_dir.exists():
                renders = [f for f in export_dir.glob("final_*.mp4") if "stable" not in f.name]
                stable = list(export_dir.glob("final_stable_v*.mp4"))
                total_mb = sum(f.stat().st_size for f in renders + stable) // 1024**2
                print(f"\n📦 EXPORT/: {len(renders)} běžných renderů, {len(stable)} stabilních verzí, celkem ~{total_mb} MB")
            else:
                print("\n📦 EXPORT/ zatím neexistuje (žádný render zatím neproběhl).")
            print("SPRÁVA EXPORT SLOŽKY")
            print("------------------------------------------------------------")
            print("1 - Vyčistit staré rendery (ponechat posledních N)")
            print("2 - Označit poslední render jako stabilní verzi")
            print("------------------------------------------------------------")
            e_choice = input("Volba (nebo Enter pro návrat): ").strip()
            if e_choice == '1':
                keep_str = input("Počet posledních verzí k ponechání [5]: ").strip()
                try:
                    keep = int(keep_str) if keep_str else 5
                    if keep < 0:
                        keep = 0
                except ValueError:
                    print("⚠️ Neplatná hodnota, použiji 5.")
                    keep = 5
                pipeline.clean_exports(keep_last=keep)
            elif e_choice == '2':
                pipeline.tag_stable()
            else:
                print("Návrat do hlavního menu.")
        elif choice == 't':
            pipeline.apply_speeds_from_timeline()
        elif choice == '12':
            subdirs = _list_project_dirs()
            selected = _select_project_dir(subdirs, allow_new=True)
            if selected is not None:
                project_dir = selected
                pipeline = TemagenPipeline(project_dir)
        elif choice == '13':
            pipeline.configure_settings()
        else:
            print(f"❌ Neplatná volba: '{choice}'. Napiš 'H' pro nápovědu k jednotlivým volbám.")

# ===== VSTUPNÍ BOD CLI =====

def main():
    # Pokud nejsou zadány žádné CLI argumenty, spustíme grafické menu
    if len(sys.argv) == 1:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n👋 Ukončeno uživatelem.")
        return

    command_aliases = MAIN_COMMAND_ALIASES

    parser = argparse.ArgumentParser(
        description="Temagen Music Video - sjednocená produkční pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser._positionals.title = "příkazy"
    parser._optionals.title = "volby"
    parser.add_argument("-h", "--help", "--napoveda", action="help", help="Zobrazit tuto nápovědu a skončit")
    parser.add_argument(
        "command",
        choices=sorted(command_aliases.keys()),
        help=(
            "Příkaz, který se má provést.\n"
            "Základní příkazy: init, parse, placeholders, analyze-song, transcribe-rap, resync-rap, align-rap, update-timeline, validate, prepare-lipsync, inject-lipsync, render, all.\n"
            "České aliasy: inicializuj, parsuj, zastupce, analyzuj-song, transkribuj-rap, resynchronizuj-rap, zarovnej-rap, prepocitej-timeline, validuj, priprav-lipsync, vloz-lipsync, renderuj, vse."
        ),
    )
    parser.add_argument("--project", "--projekt", "-p", default=None, help="Název nebo cesta ke složce projektu")

    # Argumenty pro render
    parser.add_argument("--mode", "--rezim", choices=["draft", "final"], default="draft", help="Režim renderu: draft nebo final")
    parser.add_argument("--res", "--rozliseni", choices=["draft", "hd", "fullhd"], default="draft", help="Rozlišení videa: draft, hd nebo fullhd")
    parser.add_argument("--fades", "--stmivacky", action="store_true", help="Přidat stmívačky na začátek a konec")
    parser.add_argument("--beat-sync", "--synchronizace-beatu", action="store_true", help="Synchronizovat střih a efekty s beaty")
    parser.add_argument("--no-rap", "--bez-rapu", action="store_true", help="Vypne procesy související s rap klipy (lipsync, zarovnání)")
    parser.add_argument("--force", "--vynutit", action="store_true", help="Přeskočí validaci a vynutí render i s nalezenými problémy (POUZE draft režim)")

    args = parser.parse_args()

    project_dir = detect_project(args.project)
    pipeline = TemagenPipeline(project_dir)

    command = command_aliases[args.command]

    if command == "init":
        pipeline.init_project()
    elif command == "parse":
        pipeline.parse_plan()
    elif command == "placeholders":
        pipeline.create_placeholders()
    elif command == "analyze":
        pipeline.analyze_audio()
    elif command == "analyze-song":
        pipeline.analyze_song()
    elif command == "plan":
        pipeline.generate_video_plan()
    elif command == "scenario":
        pipeline.generate_scenario_ai()
    elif command == "plan-ai":
        pipeline.generate_full_plan_ai()
    elif command == "sync":
        pipeline.sync_timeline()
    elif command == "transcribe-rap":
        pipeline.transcribe_rap_clips()
    elif command == "resync-rap":
        if not pipeline.resync_rap_alignment_from_lyrics():
            sys.exit(1)
    elif command == "align-rap":
        pipeline.align_rap_clips()
    elif command == "align-vid":
        pipeline.align_vid_clips()
    elif command == "update-timeline":
        pipeline.update_timeline_from_alignment()
    elif command == "apply-speeds-timeline":
        pipeline.apply_speeds_from_timeline()
    elif command == "validate":
        if not pipeline.validate_project(final=(args.mode == "final"), no_rap=args.no_rap):
            sys.exit(1)
    elif command == "prepare-lipsync":
        if pipeline.transcribe_song_czech():
            pipeline.analyze_song()
            pipeline.export_lipsync_audio_segments()
    elif command == "inject-lipsync":
        ok = pipeline.inject_lipsync_segments_into_timeline()
        sys.exit(0 if ok else 1)
    elif command == "all":
        pipeline.run_all(mode=args.mode, hd_mode=args.res, no_rap=args.no_rap, force=args.force)
    elif command == "render":
        is_final = args.mode == "final"
        validated = pipeline.validate_project(final=is_final, no_rap=args.no_rap)
        if not validated:
            if args.force and not is_final:
                print("⚠️  Validace selhala, ale --force je zapnuté (draft režim) → pokračuji v renderu i s problémy výše.")
            else:
                if args.force and is_final:
                    print("❌ --force nelze použít ve final režimu — lip-sync problémy musí být opravené (spusť prepare-lipsync / inject-lipsync).")
                sys.exit(1)
        pipeline.render_video(mode=args.mode, hd_mode=args.res, use_fades=args.fades, use_beat_sync=args.beat_sync)
    elif command == "settings":
        pipeline.configure_settings()
    elif command == "preview-report":
        ok = pipeline.generate_preview_report()
        sys.exit(0 if ok else 1)
    elif command == "social-export":
        ok = pipeline.generate_social_exports()
        sys.exit(0 if ok else 1)
    elif command == "thumbnails":
        ok = pipeline.generate_thumbnail_candidates()
        sys.exit(0 if ok else 1)
    elif command == "ab-variants":
        ok = pipeline.generate_ab_variants()
        sys.exit(0 if ok else 1)
    elif command == "qa-summary":
        ok = pipeline.generate_qa_summary()
        sys.exit(0 if ok else 1)
    elif command == "render-registry":
        ok = pipeline.show_render_registry()
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
