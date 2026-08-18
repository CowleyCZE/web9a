from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_STORK_ANCHOR = (
    "a constantly recurring anthropomorphic rapper character in the form of a stork, "
    "with a long orange beak, a black cap featuring an MLK motif, dark hooded streetwear, "
    "expressive body language, a recognizable silhouette, consistent proportions, "
    "and a music video-style character design"
)
DEFAULT_NEGATIVE = (
    "another character, a human face, a visible human mouth, another beak, a deformed beak, "
    "duplicate limbs, extra fingers, a distorted logo on a hoodie, illegible text, unstable identity, "
    "flickering, morphing, low resolution, watermark, subtitles"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip() if path.exists() else ""


def _clean_line(line: str) -> str:
    line = re.sub(r"^\s*(?:\[[^\]]+\]|\([^\)]+\))\s*", "", line)
    return re.sub(r"\s+", " ", line).strip(" -\t")


def load_inputs(project_dir: Path) -> dict[str, Any]:
    input_dir = project_dir / "INPUT"
    edit_dir = project_dir / "EDIT_PROJECT"
    lyrics = _read(input_dir / "lyrics.txt")
    mood = _read(input_dir / "mood.txt")
    character = _read(input_dir / "postava.txt") or _read(input_dir / "character.txt")
    brief = _read(input_dir / "generation_brief.txt") or _read(input_dir / "brief.txt")
    alignment = {}
    for name in ("song_alignment.json", "song_transcription.json"):
        candidate = edit_dir / name
        if candidate.exists():
            try:
                alignment = json.loads(candidate.read_text(encoding="utf-8"))
                break
            except (OSError, json.JSONDecodeError):
                pass
    return {"lyrics": lyrics, "mood": mood, "character": character, "brief": brief, "alignment": alignment}


def _words_for_lines(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    words = alignment.get("words", []) if isinstance(alignment, dict) else []
    return [word for word in words if isinstance(word, dict) and word.get("word") is not None]


def _line_time(line: str, words: list[dict[str, Any]], cursor: int) -> tuple[float | None, float | None, int]:
    tokens = [re.sub(r"[^a-záčďéěíňóřšťúůýž0-9]", "", token.lower()) for token in line.split()]
    tokens = [token for token in tokens if token]
    if not tokens or not words:
        return None, None, cursor
    normalized = [re.sub(r"[^a-záčďéěíňóřšťúůýž0-9]", "", str(w.get("word", "")).lower()) for w in words]
    for index in range(max(0, cursor), len(words)):
        if normalized[index] != tokens[0]:
            continue
        matched = 1
        while matched < len(tokens) and index + matched < len(words):
            if normalized[index + matched] != tokens[matched]:
                break
            matched += 1
        if matched >= max(1, min(3, len(tokens))):
            start = float(words[index].get("start", 0.0))
            end = float(words[index + matched - 1].get("end", start))
            return start, end, index + matched
    return None, None, cursor


def select_key_rap_passages(
    lyrics: str,
    alignment: dict[str, Any] | None = None,
    max_passages: int = 4,
    min_duration: float = 2.5,
    max_duration: float = 6.0,
) -> list[dict[str, Any]]:
    """Select a small number of short, text-anchored rap passages."""
    lines = [_clean_line(line) for line in lyrics.splitlines() if _clean_line(line)]
    if not lines:
        return []
    words = _words_for_lines(alignment or {})
    candidates = []
    cursor = 0
    for index, line in enumerate(lines):
        start, end, next_cursor = _line_time(line, words, cursor)
        if next_cursor != cursor:
            cursor = next_cursor
        token_count = len(line.split())
        # Prioritize concise, memorable lines, repeated/emphatic lines and lines
        # with narrative/action language. Avoid generating a lipsync clip for every bar.
        keyword_bonus = sum(1 for word in ("já", "my", "svět", "vítěz", "oheň", "signál", "vrchol", "změnit", "bořím", "nevzdávej") if word in line.lower())
        score = min(1.0, 0.25 + keyword_bonus * 0.10 + (0.15 if "!" in line else 0.0) + (0.10 if 5 <= token_count <= 18 else 0.0))
        if start is not None and end is not None:
            duration = max(min_duration, min(max_duration, end - start + 0.35))
        else:
            duration = max(min_duration, min(max_duration, 0.23 * token_count + 0.45))
        candidates.append({"line_index": index, "text": line, "start": start, "end": end, "duration": round(duration, 3), "score": round(score, 3)})
    ranked = sorted(candidates, key=lambda item: (-item["score"], item["line_index"]))
    chosen = []
    target = max(1, int(max_passages))
    for candidate in ranked:
        if any(abs(candidate["line_index"] - item["line_index"]) <= 1 for item in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= target:
            break
    # Pokud je text krátký nebo jsou všechny silné řádky sousední, doplň počet
    # nejlepšími dosud nepoužitými kandidáty. Limit rap klipů je horní hranice,
    # nikoli důvod vracet méně klíčových pasáží bez vysvětlení.
    if len(chosen) < target:
        for candidate in ranked:
            if candidate in chosen:
                continue
            chosen.append(candidate)
            if len(chosen) >= target:
                break
    return sorted(chosen, key=lambda item: item["line_index"])


def _escape_prompt_lyrics(text: str) -> str:
    """Keep the lyric payload readable inside the prompt's quoted lyric clause."""
    return re.sub(r"\s+", " ", text).strip().replace('"', "'")


def _build_rap_prompt(
    *,
    clip_id: str,
    duration: float,
    lyric_text: str,
    mood_text: str,
) -> str:
    """Build the canonical stork rap prompt with lyrics in the lipsync anchor."""
    safe_lyrics = _escape_prompt_lyrics(lyric_text)
    duration_text = f"{duration:.2f} seconds"
    return (
        f"{DEFAULT_STORK_ANCHOR}, wearing a dark olive functional jacket over a distinctive "
        f"hooded sweatshirt, a close medium shot, the beak clearly visible in a three-quarter view, "
        f"subtle and controlled articulation of the beak matching the Czech-rapped lyrics: "
        f'"{safe_lyrics}", a confident gaze, {mood_text}, '
        f"The stork rapper breaks free from the pressure and achieves self-confidence and victory., "
        f"cinematic practical lighting, steady camera, realistic movement, a single continuous shot, "
        f"duration {duration_text}"
    )


def _outfit_for_section(section: str, index: int) -> str:
    outfits = {
        "intro": "black technical hoodie with subtle reflective trim",
        "verse": "dark olive utility jacket over the signature hoodie",
        "chorus": "black and gold performance bomber jacket",
        "bridge": "charcoal long coat with metallic details",
        "outro": "the signature dark hooded streetwear, slightly weathered",
    }
    return outfits.get(section, outfits["verse"])


def build_generation_package(
    project_dir: Path,
    *,
    max_rap_passages: int = 4,
    mood: str | None = None,
    creative_brief: str | None = None,
) -> dict[str, Any]:
    inputs = load_inputs(project_dir)
    lyrics = inputs["lyrics"]
    if not lyrics:
        raise ValueError("Chybí INPUT/lyrics.txt — bez textu nelze bezpečně vybrat rap pasáže.")
    mood_text = (mood or inputs["mood"] or "cinematic Czech rap, determined, nocturnal, urban")
    brief = creative_brief or inputs["brief"] or "A stork rapper moves from pressure to self-belief and victory."
    rap_passages = select_key_rap_passages(lyrics, inputs["alignment"], max_passages=max_rap_passages)
    anchor = DEFAULT_STORK_ANCHOR
    prompts = []
    for index, passage in enumerate(rap_passages, 1):
        section = "chorus" if index == 2 else "verse"
        prompts.append({
            "clip_id": f"rap_{index:02d}",
            "type": "rap_lipsync",
            "duration_sec": passage["duration"],
            "text": passage["text"],
            "section": section,
            "character_anchor": anchor,
            "prompt": _build_rap_prompt(
                clip_id=f"rap_{index:02d}",
                duration=passage["duration"],
                lyric_text=passage["text"],
                mood_text=mood_text,
            ),
            "negative_prompt": DEFAULT_NEGATIVE,
            "lipsync_constraints": {
                "character_mode": "character_lipsync",
                "beak_visibility_required": True,
                "max_phoneme_drift_ms": 35,
                "pre_roll_ms": 100,
                "post_roll_ms": 120,
            },
        })
    broll_prompts = []
    scenes = [
        ("broll_01", "intro", "empty wet city under sodium lights, distant silhouette, slow push-in"),
        ("broll_02", "verse", "the stork rapper walking through an industrial corridor, reflections and smoke"),
        ("broll_03", "chorus", "the stork rapper on a rooftop above the city, wind moving the jacket, wide hero shot"),
        ("broll_04", "bridge", "close detail of boots crossing a puddle, reflected lights, rhythmic camera movement"),
        ("broll_05", "outro", "the stork rapper facing the sunrise from a rooftop, calm victorious silhouette"),
    ]
    for clip_id, section, visual in scenes:
        prompts.append({
            "clip_id": clip_id,
            "type": "broll",
            "duration_sec": 4.0,
            "section": section,
            "character_anchor": anchor,
            "prompt": f"{anchor}, {_outfit_for_section(section, 0)}, {visual}, {mood_text}, {brief}, cinematic music video, stable identity, duration 4 seconds",
            "negative_prompt": DEFAULT_NEGATIVE,
        })
    package = {
        "schema_version": 1,
        "character_type": "masked_bird_stork_rapper",
        "character_anchor": anchor,
        "negative_prompt": DEFAULT_NEGATIVE,
        "creative_brief": brief,
        "mood": mood_text,
        "rap_policy": {"max_passages": max_rap_passages, "selected_count": len(rap_passages), "selection": "key_lines_only", "max_duration_sec": 6.0},
        "rap_passages": rap_passages,
        "clips": prompts,
    }
    edit_dir = project_dir / "EDIT_PROJECT"
    prompts_dir = project_dir / "Prompts"
    edit_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (edit_dir / "generation_manifest.json").write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (prompts_dir / "character_profile.txt").write_text(anchor + "\n\nNegative prompt:\n" + DEFAULT_NEGATIVE + "\n", encoding="utf-8")
    (prompts_dir / "scenario.txt").write_text(
        f"TITLE: {project_dir.name}\n\nCREATIVE BRIEF\n{brief}\n\nMOOD\n{mood_text}\n\nSTORY ARC\n"
        "The masked stork rapper moves from isolation and pressure through confrontation into controlled confidence and a clear victorious final image.\n",
        encoding="utf-8",
    )
    markdown = [f"# Generation prompts — {project_dir.name}", "", f"**Character anchor:** {anchor}", "", f"**Mood:** {mood_text}", ""]
    for clip in prompts:
        markdown += [f"## {clip['clip_id']} — {clip['type']} ({clip['duration_sec']:.2f}s)", "", clip["prompt"], "", f"**Negative prompt:** {clip['negative_prompt']}", ""]
    (prompts_dir / "generation_prompts.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return package
