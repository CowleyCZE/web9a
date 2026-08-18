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
    # B-roll IDs intentionally use the convention consumed by pro_pipeline.py and
    # klipy.py: vid_XX.mp4 files live in the project's gen_vid/ directory.
    # Keep these prompts lyric-free; only rap_lipsync prompts carry rap text.
    scenes = [
        ("vid_01", "intro", 5.0, "a cold blue bedroom before sunrise, the stork rapper sitting on the edge of the bed while a phone illuminates the orange beak, slow locked-off push-in"),
        ("vid_02", "intro", 5.0, "extreme close-up of a phone screen glow reflected in the stork rapper's eye and orange beak, notifications remain abstract and unreadable, shallow depth of field"),
        ("vid_03", "intro", 4.0, "the stork rapper alone at a small kitchen table, untouched coffee, blue screen light, empty chair opposite, restrained static composition"),
        ("vid_04", "intro", 5.0, "overhead view of the stork rapper lying awake in a dark room, phone held above the black MLK cap, the room feels like a digital cage"),
        ("vid_05", "verse", 5.0, "the stork rapper walking through a narrow apartment corridor lined with cold phone reflections, shoulders tense, slow tracking shot"),
        ("vid_06", "verse", 4.0, "macro detail of a thumb endlessly scrolling on a phone while the stork rapper's orange beak appears blurred in the background, no readable interface text"),
        ("vid_07", "verse", 5.0, "the stork rapper framed behind translucent glass covered with abstract reflections, trapped urban mood, gradual lateral camera move"),
        ("vid_08", "verse", 5.0, "a crowded subway platform where every passenger is represented only by soft silhouettes and phone light, the stork rapper stands isolated in the center"),
        ("vid_09", "verse", 4.0, "the stork rapper passes a shop window showing fragmented reflections of the same silhouette, controlled handheld movement, cold green and blue palette"),
        ("vid_10", "verse", 5.0, "close shot of the stork rapper's hand hovering over the phone power button, hesitation and pressure, practical screen light on the dark hoodie"),
        ("vid_11", "bridge", 4.0, "the stork rapper places the phone face down on a concrete table, screen light disappears, a warm practical light begins to enter the frame"),
        ("vid_12", "bridge", 5.0, "the stork rapper unplugs a glowing charging cable and steps away from it, subtle dust in the air, symbolic release, steady camera"),
        ("vid_13", "bridge", 5.0, "the stork rapper opens a heavy industrial door from darkness into an orange-lit night street, silhouette transition, single smooth camera move"),
        ("vid_14", "chorus", 5.0, "the stork rapper enters a small late-night bar, warm amber practical lights, a few background silhouettes turn toward the character, welcoming atmosphere"),
        ("vid_15", "chorus", 4.0, "a group at a bar table gradually places their phones face down, hands and glasses in frame, warm light, no readable logos or text"),
        ("vid_16", "chorus", 5.0, "the stork rapper and a small group walk together through a neon side street, jackets moving in the wind, confident forward tracking shot"),
        ("vid_17", "chorus", 4.0, "low angle of synchronized footsteps crossing a rain-wet street, reflections of amber and cyan lights, rhythmic but realistic camera motion"),
        ("vid_18", "chorus", 5.0, "wide shot of the stork rapper and friends under an urban overpass, practical sodium lights, collective energy, slow circular camera move"),
        ("vid_19", "verse", 5.0, "the stork rapper alone again in the bedroom, the phone lights up on the table, the character reaches toward it despite knowing better"),
        ("vid_20", "verse", 4.0, "rapid but controlled montage-like shot of phone reflections passing over the stork rapper's hoodie and orange beak, abstract unreadable interface shapes"),
        ("vid_21", "verse", 5.0, "the stork rapper sits on a bus at night while city lights streak across the window, phone glow isolates the character from the moving world"),
        ("vid_22", "verse", 4.0, "a wall clock and phone on a dark table share the same cold light, the stork rapper's silhouette moves out of focus behind them, visual time pressure"),
        ("vid_23", "bridge", 5.0, "the stork rapper notices the phone reflection in a puddle, then steps through the reflection and breaks its symmetry, symbolic return to reality"),
        ("vid_24", "bridge", 5.0, "the stork rapper switches the phone to airplane mode and places it inside a jacket pocket, close detail, warm light growing stronger"),
        ("vid_25", "chorus", 5.0, "the stork rapper raises the head toward an open city skyline, wind moving the dark olive jacket, rooftop hero composition"),
        ("vid_26", "chorus", 4.0, "a small group leaves their phones in a row on a rooftop ledge and faces the city together, wide practical night lighting"),
        ("vid_27", "chorus", 5.0, "the stork rapper moves through an open rooftop gathering, expressive shoulders and arms, confident body language, smooth gimbal shot"),
        ("vid_28", "outro", 5.0, "the stork rapper stands at the edge of a rooftop before dawn, deep blue sky shifting toward gold, calm victorious silhouette"),
        ("vid_29", "outro", 5.0, "close detail of the stork rapper lowering the phone and looking directly toward the real world, orange beak and MLK cap clearly readable, warm sunrise light"),
        ("vid_30", "outro", 6.0, "final wide shot of the stork rapper and friends walking into warm morning light through an open city street, no screens, peaceful victorious ending"),
    ]
    for clip_id, section, duration, visual in scenes:
        prompts.append({
            "clip_id": clip_id,
            "type": "broll",
            "duration_sec": duration,
            "section": section,
            "character_anchor": anchor,
            "prompt": f"{anchor}, {_outfit_for_section(section, 0)}, {visual}, {mood_text}, {brief}, cinematic music video, stable identity, duration {duration:.2f} seconds",
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
