#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
klipy.py
========
Zjednodušený doprovodný nástroj k pro_pipeline.py (dále "skript 1"), zaměřený
čistě na práci s existujícím katalogem klipů INPUT/klipy.md — sestavení
scénáře a POŘADÍ klipů (bez AI výpočtu časování), plus průchod navazujícími
kroky pipeline (transkripce rapu, zarovnání, timeline, render).

DŮLEŽITÉ: klipy.py NEDUPLIKUJE kód pro_pipeline.py — importuje a znovupoužívá
třídu TemagenPipeline (projekty, nastavení, Ollama/Groq vrstvu, parsování
klipy.md, render flow atd.) přímo z pro_pipeline.py. Proto musí klipy.py ležet
VE STEJNÉ SLOŽCE jako pro_pipeline.py.

Mapování voleb na skript 1 (podle zadání):
  1 - Scénář z klipů (INPUT/klipy.md)            → mechanismus jako 8a
  2 - Vhodné POŘADÍ klipů (INPUT/klipy.md),       → mechanismus jako 8b,
      BEZ výpočtu časování (jen řazení za sebe)     ale bez timeline/timingu
  3 - Rozparsovat full_plan.txt                   → jako volba 2 skriptu 1
  4 - Transkribovat rap klipy                     → jako volba 5 skriptu 1
  4b - Po ruční úpravě rap_alignment.json znovu    → jako volba U skriptu 1
       vyhledat lyrics_window (bez re-transkripce)
  5 - Synchronizovat rap klipy (segmentace+speed)  → jako volba 6 skriptu 1
  6 - Přepočítat timeline podle rap_start kotev    → jako volba 7 skriptu 1
  7 - Zarovnat vid_xx klipy na délku z timeline    → jako volba V skriptu 1
  8 - Render videa                                 → jako volba 10 skriptu 1

Návaznost voleb 2 → 8 (celý řetězec kroků od pořadí klipů po hotový render):
  2 (poradi)          Seřadí ID klipů z klipy.md podle scénáře/textu (AI, bez
                       výpočtu času) a ULOŽÍ výsledek na DVOU místech:
                         a) Prompts/poradi_klipu.txt   – jen seznam ID, pro kontrolu
                         b) Prompts/full_plan.txt       – sekce ### MUSIC_VIDEO_TIMELINE
                            a ### SHOT_ORDER, jako ČISTĚ SEKVENČNÍ řazení klipů za
                            sebou podle jejich SKUTEČNÉ délky z klipy.md (žádný nový
                            AI odhad času). Ostatní sekce full_plan.txt (pokud tam
                            už jsou, např. z volby 1) zůstanou beze změny.
  3 (parse)           Zděděno z pro_pipeline.py: rozparsuje full_plan.txt (vč. sekcí
                       zapsaných volbou 2) do EDIT_PROJECT/timeline.txt,
                       EDIT_PROJECT/shot_order.txt a metadata.json.
  4 (transcribe-rap)  Přepíše rap klipy a spočítá jejich pozici v songu
                       (rap_alignment.json).
  4b (resync-rap)     Po ruční úpravě 'transcript_raw' a/nebo 'rap_start'/
                       'rap_end' přímo v rap_alignment.json znovu vyhledá
                       lyrics_window podle INPUT/lyrics.txt — BEZ nové Whisper
                       transkripce. Použij, když oprava odrapovaného textu
                       podle lyrics.txt vyšla špatně (např. uřízlo okrajové
                       slovo) nebo ses rozhodl(a) opravit přeslech ručně.
  5 (align-rap)       Upraví (speed-rampingem) rap klipy na přesnou délku
                       odpovídajícího úseku v songu.
  6 (update-timeline) Ukotví rap_xx klipy na jejich skutečnou pozici v songu a
                       mezery mezi nimi rovnoměrně přerozdělí mezi B-roll klipy
                       přímo v timeline.txt.
  7 (align-vid)       Zjistí (z takto přepočítané timeline.txt) cílovou délku
                       každého vid_xx klipu — tedy přesně velikost mezery, kterou
                       má v timeline vyplnit — a fyzicky ho na tuto délku
                       speed-rampingem upraví.
  8 (render)          Vyrenderuje video podle takto kompletně vyplněné timeline.txt.

Použití (interaktivní menu):
  python3 klipy.py

Použití (CLI):
  python3 klipy.py scenario         # 1
  python3 klipy.py poradi           # 2
  python3 klipy.py parse            # 3
  python3 klipy.py transcribe-rap   # 4
  python3 klipy.py resync-rap       # 4b
  python3 klipy.py align-rap        # 5
  python3 klipy.py update-timeline  # 6
  python3 klipy.py align-vid        # 7
  python3 klipy.py render           # 8

Volitelné:
  --project [Název]   # stejná logika detekce projektu jako v pro_pipeline.py
"""

import re
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.cli_registry import natural_command_key
from pipeline.commands import KLIPY_COMMAND_ALIASES
from pipeline.visual_quality import rank_candidates

try:
    from pro_pipeline import (
        TemagenPipeline,
        detect_project,
        truncate_for_prompt,
        clean_code_block_text,
        format_timecode,
        probe_duration,
    )
except ImportError as e:
    print("❌ Nelze najít/naimportovat pro_pipeline.py.")
    print("   klipy.py musí ležet VE STEJNÉ SLOŽCE jako pro_pipeline.py (znovupoužívá jeho třídu TemagenPipeline).")
    print(f"   Detail chyby: {e}")
    sys.exit(1)


# ===== VOLBA 2: PROMPT PRO ŘAZENÍ EXISTUJÍCÍCH KLIPŮ (BEZ ČASOVÁNÍ) =====
#
# Na rozdíl od Fáze B (8b, generate_full_plan_ai) v pro_pipeline.py, která staví
# kompletní full_plan.txt s časovou osou podle transkripce písně, tady chceme
# od AI POUZE pořadí ID klipů z klipy.md — žádné délky, žádné časy, žádnou
# timeline. Proto vlastní, mnohem jednodušší prompt a parsování odpovědi.

KLIPY_PORADI_SYSTEM_PROMPT = (
    "Jsi zkušený filmový střihač a hudební režisér. Dostaneš katalog existujících "
    "videoklipů/obrázků (s jejich obsahem, případně odrapovaným textem), text "
    "písně/scénář a orientační ROZPOČET na celkovou délku B-rollu (vid_/pic_/char_ "
    "klipů). Tvým úkolem je: "
    "(1) zařadit VŠECHNY rap_ klipy z katalogu přesně v pořadí, v jakém se jejich "
    "SKUTEČNÝ odrapovaný text zpívá/rapuje v písni — ty se NIKDY nevynechávají ani "
    "nekrátí výběrem, nesou unikátní text, který by jinak ve videu chyběl; "
    "(2) k nim VYBRAT a seřadit takové B-roll klipy (vid_/pic_/char_), jejichž "
    "obsah nejlépe sedí k textu/náladě dané pasáže, a to tak, aby se SOUČET JEJICH "
    "DÉLEK blížil dodanému rozpočtu na B-roll — není nutné (a spíš NEŽÁDOUCÍ) použít "
    "úplně všechny katalogové B-roll klipy, pokud by jejich součet rozpočet výrazně "
    "přesáhl (přebytečné, obsahově nejméně vhodné vynech); pokud je naopak "
    "B-roll klipů v katalogu málo (jejich součet je pod rozpočtem), použij je "
    "všechny. Tohle rozhoduje o tom, jak moc bude nutné klipy později zrychlovat/"
    "zpomalovat, aby přesně vyplnily mezery mezi rap pasážemi v songu — proto na "
    "rozpočtu záleží. "
    "NEPOČÍTÁŠ žádné časování, délky ani časovou osu — jen POŘADÍ A VÝBĚR. "
    "Nevymýšlíš nové klipy, nepřejmenováváš existující ID. "
    "Odpověz VÝHRADNĚ seznamem ID klipů, jedno ID na řádek, přesně v pořadí, "
    "v jakém se mají v klipu objevit. Žádný jiný text, žádné komentáře, žádné odrážky, "
    "žádné číslování řádků, žádné markdown formátování."
)

KLIPY_PORADI_PROMPT_TEMPLATE = """## SCÉNÁŘ / TEXT PÍSNĚ
{scenario_or_lyrics}

## DOSTUPNÉ EXISTUJÍCÍ KLIPY (INPUT/klipy.md) — VYBÍREJ A ŘAĎ Z TĚCHTO, NEVYMÝŠLEJ NOVÉ:
{existing_clips}

## ROZPOČET NA B-ROLL (viz systémová instrukce — proč na tom záleží)
{budget_info}

## ÚKOL
1. Zařaď VŠECHNY rap_ klipy z katalogu, ve správném pořadí podle textu písně — nesou
   unikátní odrapovaný text a NESMÍ chybět ani se nesmí vynechat kvůli rozpočtu.
2. K nim VYBER a seřaď takové B-roll klipy (vid_/pic_/char_), jejichž obsah nejlépe
   sedí k textu/náladě dané pasáže, tak aby jejich SOUČET DÉLEK odpovídal výše
   uvedenému rozpočtu na B-roll — NEPOUŽÍVEJ automaticky úplně všechny, pokud by ses
   tím dostal výrazně nad rozpočet; vyber jen obsahově nejvhodnější.
NEPOČÍTEJ délky ani přesné časy — jen pořadí a výběr.
Výstup: jen ID klipů, jedno na řádek, přesně v pořadí, v jakém se mají ve videu objevit —
žádný další text.
"""


class KlipyPipeline(TemagenPipeline):
    """Rozšiřuje TemagenPipeline (ze skriptu 1) o volbu specifickou pro
    klipy.py — řazení existujících klipů z INPUT/klipy.md bez AI výpočtu
    časování. Všechny ostatní volby volají přímo zděděné metody skriptu 1."""

    def generate_clip_order_ai(self) -> bool:
        """Volba 2: Vygeneruje vhodné POŘADÍ existujících klipů z INPUT/klipy.md
        podle scénáře (Prompts/scenario.txt, pokud existuje z volby 1) nebo
        přímo podle INPUT/lyrics.txt.

        Na rozdíl od Fáze B (8b) v pro_pipeline.py AI NEPOČÍTÁ žádné
        časování/timeline — pouze řadí existující klipy za sebe.

        Stejně jako Fáze B (8b) běží VŽDY přes lokální Ollamu, nezávisle na
        nastavení 'text_ai_provider' (to řídí jen volbu 1 / Fázi A).

        Výstup: Prompts/poradi_klipu.txt — jeden ID klipu na řádek.
        """
        settings = self.load_settings()
        if not self._ollama_ready(settings):
            print("❌ Ollama není dostupná/vypnutá (nastavení 'ollama_enabled') — tato volba potřebuje lokální Ollamu.")
            print(f"   Zkontroluj, že běží `ollama serve` a je stažený model (`ollama pull {self._ollama_plan_model(settings)}`).")
            return False

        klipy = self._load_klipy_md()
        if not klipy:
            print(f"❌ {self.input_dir / 'klipy.md'} nenalezen nebo prázdný — není co řadit.")
            return False
        print(f"📼 Načteno {len(klipy)} existujících klipů z klipy.md "
              f"({sum(1 for v in klipy.values() if v['group'] == 'RAP')} rap, "
              f"{sum(1 for v in klipy.values() if v['group'] == 'VID')} vid, "
              f"{sum(1 for v in klipy.values() if v['group'] == 'PIC')} pic, "
              f"{sum(1 for v in klipy.values() if v['group'] == 'CHAR')} char).")
        existing_clips_block = self._format_existing_clips_for_prompt(klipy)

        scenario_path = self.prompts_dir / "scenario.txt"
        scenario_text = self._load_text_file(scenario_path)
        if scenario_text:
            print(f"📝 Použit scénář z {scenario_path.relative_to(self.project_dir)} jako kontext pro řazení.")
            context_text = scenario_text
        else:
            lyrics = self._load_lyrics_text()
            if not lyrics:
                print(f"❌ Chybí {scenario_path.relative_to(self.project_dir)} i {self.input_dir / 'lyrics.txt'} "
                      "— bez alespoň jednoho z nich nelze klipy smysluplně seřadit.")
                return False
            print("ℹ️  Prompts/scenario.txt nenalezen — jako kontext pro řazení použit přímo INPUT/lyrics.txt.")
            context_text = lyrics

        # ── Orientační rozpočet na B-roll — BEZ TOHOTO by AI (stejně jako dřív) klidně
        # seřadilo/vybralo VŠECHNY katalogové B-roll klipy bez ohledu na to, kolik místa
        # jim ve skutečnosti v songu zbyde. Volba 6 (update_timeline_from_alignment)
        # ale rap_xx klipy později přesune na jejich SKUTEČNOU pozici v songu a mezery
        # mezi nimi rovnoměrně rozdělí mezi vybrané B-roll klipy — pokud je jich vybráno
        # o moc víc, než kolik se do mezer reálně vejde, musí je volba 7 (align_vid_clips)
        # drasticky komprimovat (rychlost nad limit 2.0x, viz reálně zachycená chyba:
        # cíl 0.52s pro 8s klip → rychlost 14x, oříznuto na 2x → klip pak neodpovídá
        # timeline). Spočítáme proto orientační rozpočet: délka písně mínus součet délek
        # VŠECH rap_ klipů z katalogu (ty se do pořadí zařazují vždy celé) = kolik sekund
        # zbývá na B-roll, a pošleme ho AI jako vodítko pro VÝBĚR (ne jen řazení).
        song_duration = None
        audio_path = self.find_audio()
        if audio_path:
            try:
                song_duration = probe_duration(audio_path)
            except Exception:
                song_duration = None

        rap_total = sum(
            float(d.get("duration_sec") or 0.0) for d in klipy.values() if d.get("group") == "RAP"
        )
        broll_budget = None
        if song_duration and song_duration > 0:
            broll_budget = max(0.0, song_duration - rap_total)
            budget_info = (
                f"Celková délka písně: ~{song_duration:.1f}s. Součet délek VŠECH rap_ klipů "
                f"z katalogu (ty se použijí vždy celé): ~{rap_total:.1f}s. Orientační rozpočet "
                f"na B-roll (zbytek písně mezi rap pasážemi): ~{broll_budget:.1f}s (klidně "
                "v rozmezí zhruba ±40 % — přesné mezery se dopočítají později podle skutečné "
                "pozice rap pasáží v songu, tohle je jen vodítko pro VÝBĚR množství B-rollu)."
            )
        else:
            budget_info = (
                "(audio soubor v INPUT/ nenalezen — přesný rozpočet nelze spočítat; vyber "
                "přiměřené množství B-rollu podle délky/struktury textu písně, radši méně "
                "kvalitně padnoucích klipů než všechny za sebou.)"
            )

        model_preview = self._ollama_plan_model(settings)
        print(f"🎞️  Řadím existující klipy pomocí lokální AI (Ollama, model: {model_preview})...")
        print("   (na CPU bez GPU to může chvíli trvat — model se musí nejdřív načíst do paměti)")
        if broll_budget is not None:
            print(f"🎯 Rozpočet na B-roll: ~{broll_budget:.1f}s (píseň ~{song_duration:.1f}s − rap klipy ~{rap_total:.1f}s).")

        prompt = KLIPY_PORADI_PROMPT_TEMPLATE.format(
            scenario_or_lyrics=truncate_for_prompt(context_text, 6000),
            existing_clips=existing_clips_block,
            budget_info=budget_info,
        )
        messages = [
            {"role": "system", "content": KLIPY_PORADI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # phase="plan" → stejně jako 8b vždy lokální Ollama (viz _generate_with_text_ai
        # ve skriptu 1), bez ohledu na nastavení 'text_ai_provider'.
        raw, model_used, error = self._generate_with_text_ai(
            messages, phase="plan", settings=settings, temperature=0.5, timeout=900,
            num_ctx=int(settings.get("ollama_plan_num_ctx", 8192) or 8192),
        )
        if not raw or not raw.strip():
            print(f"❌ Generování pořadí klipů selhalo: {error}")
            print(f"   Zkontroluj: `ollama pull {model_used}` a že server neběží na jinou náročnou úlohu zároveň.")
            return False

        raw = clean_code_block_text(raw.strip())

        # Z odpovědi vytáhneme jen tokeny odpovídající skutečným ID z klipy.md
        # (AI se občas přesně nedrží formátu i přes explicitní zadání) — pořadí
        # zachováno, duplicity a neznámá ID zahozeny.
        known_ids = set(klipy.keys())
        ordered, seen = [], set()
        for raw_line in raw.splitlines():
            stripped = raw_line.strip().strip("-*•\t ")
            if not stripped:
                continue
            token = stripped.split()[0].strip(",.;:")
            if token in known_ids and token not in seen:
                ordered.append(token)
                seen.add(token)

        if not ordered:
            print("❌ Odpověď AI neobsahovala žádné rozpoznatelné ID klipů z klipy.md — zkus to znovu, "
                  "případně zkontroluj obsah scénáře/lyrics.")
            return False

        # Deterministická pojistka nad AI výběrem: (1) doplní rap_ klipy, které AI
        # přesto vynechalo (nesmí chybět, nesou unikátní text), (2) ořízne případný
        # přebytek B-roll klipů nad rozpočet (viz komentář u výpočtu broll_budget výše),
        # aby volba 7 nemusela klipy komprimovat nad bezpečný limit rychlosti.
        ordered = self._select_and_trim_order(ordered, klipy, broll_budget, context_text)
        seen = set(ordered)

        missing = sorted(known_ids - seen)
        if missing:
            preview = ", ".join(missing[:15]) + (" ..." if len(missing) > 15 else "")
            print(f"ℹ️  {len(missing)} klip(ů) z klipy.md není ve finálním pořadí — buď je AI nezařadilo, "
                  f"nebo byly (u B-rollu záměrně) vypuštěny kvůli rozpočtu: {preview}")

        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.prompts_dir / "poradi_klipu.txt"
        header = (
            f"# POŘADÍ KLIPŮ — vygenerováno klipy.py (volba 2), model: {model_used}\n"
            f"# Pouze pořadí existujících klipů z INPUT/klipy.md — BEZ výpočtu časování/timeline.\n"
            f"# Zkontroluj/uprav ručně podle potřeby, než budeš pokračovat dalšími volbami.\n\n"
        )
        out_path.write_text(header + "\n".join(ordered) + "\n", encoding="utf-8")
        print(f"✅ Pořadí {len(ordered)} klipů uloženo do {out_path.relative_to(self.project_dir)}.")

        # Navazující krok: pořadí samo o sobě (poradi_klipu.txt) není nic, co by
        # uměla přečíst volba 3 (parse_plan, dědí se z pro_pipeline.py) — ta čte
        # výhradně Prompts/full_plan.txt. Aby šlo na volbu 2 rovnou navázat volbou
        # 3 (a pak 4-8), zapíšeme totéž pořadí i jako prosté SEKVENČNÍ řazení
        # (klip za klipem, podle SKUTEČNÉ délky z klipy.md — žádné nové AI
        # dopočítávání času) do sekcí ### MUSIC_VIDEO_TIMELINE a ### SHOT_ORDER
        # v full_plan.txt. Selhání tohoto kroku volbu 2 jako celek nesmí shodit —
        # poradi_klipu.txt už je v pořádku uložené.
        try:
            self.write_clip_order_to_full_plan(ordered, klipy)
        except Exception as e:
            print(f"⚠️  Pořadí se nepodařilo zapsat do full_plan.txt ({e}) — poradi_klipu.txt zůstává platné, "
                  "zápis do full_plan.txt zkus případně ručně.")
        return True

    @staticmethod
    def _nat_key(cid: str):
        """Přirozené řazení ID klipů (rap_02 < rap_11), aby doplňování
        chybějících rap_ klipů a hledání "následujícího" ID fungovalo podle
        číselného pořadí, ne lexikograficky (kde by "rap_11" bylo před
        "rap_2")."""
        m = re.search(r"(\d+)\s*$", cid or "")
        prefix = re.split(r"\d+\s*$", cid or "", maxsplit=1)[0]
        return (prefix, int(m.group(1)) if m else -1)

    @staticmethod
    def _text_overlap_score(a: str, b: str) -> int:
        """Jednoduché (bezmodelové) skóre obsahové podobnosti dvou textů —
        počet společných slov (délka > 2 znaky, aby se nepočítaly předložky/
        spojky). Používá se jen jako heuristika při DOPLŇOVÁNÍ klipů do
        řídkých bloků (viz _select_and_trim_order) — nejde o nic
        sofistikovaného, jen o to, aby doplněné klipy obsahově co nejlíp
        seděly k nejbližší rapované pasáži/scénáři."""
        words_a = {w for w in re.findall(r"\w+", (a or "").lower()) if len(w) > 2}
        words_b = {w for w in re.findall(r"\w+", (b or "").lower()) if len(w) > 2}
        if not words_a or not words_b:
            return 0
        return len(words_a & words_b)

    def _select_and_trim_order(self, ordered: list, klipy: dict, broll_budget, context_text: str = "") -> list:
        """Deterministická pojistka nad AI výběrem/pořadím z volby 2 (viz komentář
        u volání v generate_clip_order_ai). Řeší DVA problémy, které dřív reálně
        vedly k chybě ve volbě 7 (align-vid): rychlost vyšla 14x, oříznuta na
        limit 2.0x → klip neodpovídá timeline:

        1) VŠECHNY rap_ klipy z katalogu MUSÍ být v pořadí zastoupeny (nesou
           unikátní odrapovaný text) — pokud je AI přesto vynechalo, doplní se
           zpět na odpovídající místo podle číselného pořadí ID.

        2) B-roll (vid_/pic_/char_) se neposuzuje jen podle GLOBÁLNÍHO součtu
           délek vůči rozpočtu, ale PO BLOCÍCH — tedy přesně tak, jak s ním
           později pracuje update_timeline_from_alignment(): mezera mezi
           dvěma sousedními rap_ kotvami se rovnoměrně rozdělí mezi VŠECHNY
           klipy, které do ní volba 2 zařadila. Pokud je jich v jedné mezeře
           nacpáno moc (blok), vyjde na klip směšně málo místa a volba 7 ho
           musí drasticky zrychlit nad limit. Pokud je jich naopak v mezeře
           málo, vyjde na klip až moc místa (zpomalení pod limit).
           Proto se tu přebytek OŘEZÁVÁ a nedostatek DOPLŇUJE — každý blok
           zvlášť — z nepoužitých katalogových klipů, vybraných podle
           obsahové podobnosti s nejbližší rapovanou pasáží.

        Bez známého song_duration/broll_budget (broll_budget is None) se
        krok 2) přeskočí (není vůči čemu porovnávat) a vrátí se jen pořadí
        doplněné o chybějící rap_ klipy."""
        settings = self.load_settings()
        speed_min = float(settings.get("speed_min", 0.5) or 0.5)
        speed_max = float(settings.get("speed_max", 2.0) or 2.0)

        ordered = list(ordered)

        # ── 1) Doplnění chybějících rap_ klipů ────────────────────────────
        all_rap_ids = sorted(
            (cid for cid, d in klipy.items() if d.get("group") == "RAP"),
            key=self._nat_key,
        )
        present_rap = {cid for cid in ordered if klipy.get(cid, {}).get("group") == "RAP"}
        missing_rap = [cid for cid in all_rap_ids if cid not in present_rap]
        if missing_rap:
            print(f"⚠️  AI vynechalo {len(missing_rap)} rap_ klip(ů) i přes explicitní zadání "
                  f"— doplňuji zpět na místo podle pořadí ID: {', '.join(missing_rap)}")
            for mcid in missing_rap:
                mkey = self._nat_key(mcid)
                insert_at = len(ordered)
                for i, cid in enumerate(ordered):
                    if klipy.get(cid, {}).get("group") == "RAP" and self._nat_key(cid) > mkey:
                        insert_at = i
                        break
                ordered.insert(insert_at, mcid)

        if broll_budget is None:
            return ordered

        # ── 2) Rozdělení na bloky B-rollu mezi rap_ kotvami ───────────────
        segments = []  # [("rap", cid)] nebo [("block", [cid, ...])]
        i, n = 0, len(ordered)
        while i < n:
            cid = ordered[i]
            if klipy.get(cid, {}).get("group") == "RAP":
                segments.append(("rap", cid))
                i += 1
            else:
                block_ids = []
                while i < n and klipy.get(ordered[i], {}).get("group") != "RAP":
                    block_ids.append(ordered[i])
                    i += 1
                segments.append(("block", block_ids))

        num_blocks = sum(1 for typ, _ in segments if typ == "block")
        if num_blocks == 0:
            return ordered  # samé rap_ klipy, žádný B-roll k řešení
        average_gap = broll_budget / num_blocks

        used_ids = set()
        for typ, payload in segments:
            used_ids.update([payload] if typ == "rap" else payload)
        broll_pool = {
            cid: d for cid, d in klipy.items()
            if d.get("group") in ("VID", "PIC", "CHAR") and cid not in used_ids
        }

        def _dur(cid: str) -> float:
            return float(klipy.get(cid, {}).get("duration_sec") or 0.0)

        max_block_dur = average_gap * speed_max
        min_block_dur = average_gap * speed_min
        trimmed_total, added_total = 0, 0

        new_segments = []
        for idx, (typ, payload) in enumerate(segments):
            if typ == "rap":
                new_segments.append((typ, payload))
                continue

            ids = list(payload)
            dur = sum(_dur(c) for c in ids)

            # OŘEZÁNÍ přebytku — poslední klip v bloku = AI ho zařadila jako
            # poslední/nejméně prioritní, ořezáváme proto od konce.
            while dur > max_block_dur and len(ids) > 1:
                removed = ids.pop()
                dur -= _dur(removed)
                trimmed_total += 1

            # DOPLNĚNÍ nedostatku — z nepoužitých katalogových klipů, podle
            # obsahové podobnosti s nejbližší rapovanou pasáží (před blokem,
            # jinak za blokem; bez ní se použije celkový kontext/scénář).
            if dur < min_block_dur and broll_pool:
                context_ref = ""
                for j in range(idx - 1, -1, -1):
                    if segments[j][0] == "rap":
                        context_ref = klipy.get(segments[j][1], {}).get("text", "")
                        break
                if not context_ref:
                    for j in range(idx + 1, len(segments)):
                        if segments[j][0] == "rap":
                            context_ref = klipy.get(segments[j][1], {}).get("text", "")
                            break
                context_ref = context_ref or context_text

                recent_ids = ids[-3:]
                candidates = rank_candidates(
                    broll_pool.keys(), klipy, context=context_ref,
                    previous_id=ids[-1] if ids else None,
                    recent_ids=recent_ids,
                )
                for cand in candidates:
                    if dur >= min_block_dur:
                        break
                    ids.append(cand)
                    dur += _dur(cand)
                    del broll_pool[cand]
                    added_total += 1

            new_segments.append(("block", ids))

        if trimmed_total or added_total:
            print(f"🧩 Rozpočet B-rollu dopočítán PO BLOCÍCH (mezi rap_ kotvami, ~{average_gap:.1f}s/blok, "
                  f"{num_blocks} blok(ů)): ořezáno {trimmed_total} přebytečných klipů, doplněno {added_total} "
                  f"klipů z katalogu do řídkých mezer — cílem je, aby volba 7 (align-vid) nemusela klipy "
                  f"komprimovat/natahovat nad limit rychlosti ({speed_min}x–{speed_max}x).")

        result = []
        for typ, payload in new_segments:
            if typ == "rap":
                result.append(payload)
            else:
                result.extend(payload)
        return result

    def _build_naive_timeline_from_order(self, ordered: list, klipy: dict) -> tuple[list, list]:
        """Postaví ČISTĚ SEKVENČNÍ řazení klipů za sebe (další klip začíná přesně
        tam, kde končí předchozí) podle SKUTEČNÉ délky (`duration_sec` z INPUT/
        klipy.md) — žádné AI odhadování ani přepočet časování. Toto je jen
        prozatímní/orientační rozvržení, které navazující kroky dál zpřesní:
        volba 6 (update_timeline_from_alignment) rap_xx klipy přesune na jejich
        skutečnou pozici v songu a mezery mezi nimi přerozdělí mezi B-roll, a
        volba 7 (align_vid_clips) podle toho fyzicky přizpůsobí délku vid_xx
        souborů. Vrací dvojici (řádky pro MUSIC_VIDEO_TIMELINE, řádky pro
        SHOT_ORDER)."""
        timeline_lines, shot_order_lines = [], []
        current = 0.0
        for cid in ordered:
            data = klipy.get(cid, {})
            dur = float(data.get("duration_sec") or 0.0)
            duration_source = "measured"
            if dur <= 0.0:
                # Odhad je bezpečný pouze pro provizorní plán, ne pro final render.
                dur = 4.0 if cid.startswith("rap_") else 8.0
                duration_source = "estimated"
            start, end = current, current + dur

            if cid.startswith("rap_") and data.get("text") and "neobsahuje" not in data.get("text", "").lower():
                popis = truncate_for_prompt(data.get("text", ""), 100)
            else:
                popis = truncate_for_prompt(data.get("obsah", ""), 100)

            suffix = " [ODHAD DÉLKY]" if duration_source == "estimated" else ""
            timeline_lines.append(f"{format_timecode(start)} - {format_timecode(end)} | {cid} | {popis}{suffix}")
            shot_order_lines.append(f"{format_timecode(start)} - {format_timecode(end)} | {cid} |{suffix}")
            current = end
        return timeline_lines, shot_order_lines

    def write_clip_order_to_full_plan(self, ordered: list, klipy: dict) -> bool:
        """Zapíše výsledek volby 2 (pořadí existujících klipů z INPUT/klipy.md,
        BEZ AI výpočtu časování) jako sekvenční řazení do sekcí
        ### MUSIC_VIDEO_TIMELINE a ### SHOT_ORDER v Prompts/full_plan.txt.

        Nahradí POUZE tyto dvě sekce (pokud v full_plan.txt už existují) —
        ostatní sekce (např. VIDEO_PROMPTS z generate_scenario_ai / volby 1)
        ponechá beze změny. Pokud full_plan.txt ještě neexistuje nebo je
        prázdný, vytvoří ho jen s těmito dvěma sekcemi. Před přepsáním uloží
        zálohu do full_plan.txt.bak.

        Po tomto kroku lze rovnou pokračovat volbou 3 (parse_plan, zděděná z
        pro_pipeline.py), která obě sekce rozdistribuuje do
        EDIT_PROJECT/timeline.txt a EDIT_PROJECT/shot_order.txt pro navazující
        volby 4-8."""
        if not ordered:
            print("❌ Prázdné pořadí klipů — není co zapsat do full_plan.txt.")
            return False

        timeline_lines, shot_order_lines = self._build_naive_timeline_from_order(ordered, klipy)
        timeline_text = "\n".join(timeline_lines)
        shot_order_text = "\n".join(shot_order_lines)

        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        plan_content = self.full_plan.read_text(encoding="utf-8") if self.full_plan.exists() else ""

        if plan_content.strip():
            backup_path = self.prompts_dir / "full_plan.txt.bak"
            backup_path.write_text(plan_content, encoding="utf-8")
            print(f"💾 Původní full_plan.txt zazálohován do {backup_path.relative_to(self.project_dir)}.")

        def _replace_section(content: str, header: str, body: str) -> str:
            pattern = rf'(###\s*{header}\s*\n)(.*?)(?=\n###\s+[A-Z]|\Z)'
            block = f"### {header}\n{body}\n"
            if re.search(pattern, content, re.DOTALL):
                return re.sub(pattern, lambda m: block, content, count=1, flags=re.DOTALL)
            sep = "\n\n" if content.strip() else ""
            return content.rstrip() + sep + block

        plan_content = _replace_section(plan_content, "MUSIC_VIDEO_TIMELINE", timeline_text)
        plan_content = _replace_section(plan_content, "SHOT_ORDER", shot_order_text)

        self.full_plan.write_text(plan_content.strip() + "\n", encoding="utf-8")
        print(f"✅ Pořadí {len(ordered)} klipů zapsáno jako sekvenční řazení do sekcí "
              f"MUSIC_VIDEO_TIMELINE a SHOT_ORDER v {self.full_plan.relative_to(self.project_dir)} "
              "(ostatní sekce beze změny).")
        print("   👉 Pokračuj volbou 3 (Rozparsovat full_plan.txt) → pak 4 (transkripce rapu), "
              "5 (sync rap), 6 (přepočet timeline), 7 (zarovnání vid_xx), 8 (render).")
        return True


# ===== MENU / CLI MAPOVÁNÍ =====

MENU_ACTIONS = {
    "1": ("[1. PLÁN] Vytvořit scénář z existujících klipů (INPUT/klipy.md)  [jako 8a skriptu 1]",
          lambda p: p.generate_scenario_ai()),
    "2": ("[1. PLÁN] Seřadit existující klipy BEZ výpočtu časování (jen pořadí, zapíše i "
          "full_plan.txt)  [jako 8b skriptu 1, bez timingu]",
          lambda p: p.generate_clip_order_ai()),
    "3": ("[2. NAČTENÍ PLÁNU] Rozparsovat full_plan.txt → timeline, prompty  [jako volba 2 skriptu 1]",
          lambda p: p.parse_plan()),
    "4": ("[3. RAP] Přepsat rap klipy (transkripce)                        [jako volba 5 skriptu 1]",
          lambda p: p.transcribe_rap_clips()),
    "4b": ("[3. RAP] Po ruční úpravě rap_alignment.json znovu vyhledat lyrics_window "
           "(bez re-transkripce)                [jako volba U skriptu 1]",
           lambda p: p.resync_rap_alignment_from_lyrics()),
    "5": ("[3. RAP] Doladit tempo rap klipů na píseň (segmentace + speed)  [jako volba 6 skriptu 1]",
          lambda p: p.align_rap_clips()),
    "5b": ("[3. RAP] Aplikovat rychlosti rap klipů z timeline.txt (beze změny pořadí/časů)",
           lambda p: p.apply_speeds_from_timeline()),
    "6": ("[3. RAP] Zarovnat timeline podle skutečné pozice rapu v písni   [jako volba 7 skriptu 1]",
          lambda p: p.update_timeline_from_alignment()),
    "7": ("[4. B-ROLL] Doladit vid_xx klipy na přesnou délku slotu         [jako volba V skriptu 1]",
          lambda p: p.align_vid_clips()),
    "8": ("[5. RENDER] Vyrenderovat video                                  [jako volba 10 skriptu 1]",
          lambda p: p.run_render_flow(no_rap=False)),
}

CLI_COMMANDS = KLIPY_COMMAND_ALIASES


def _list_project_dirs():
    return sorted([
        d for d in ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name not in ("venv", "scripts", "EXPORT", "TEST", "__pycache__")
    ])


def _menu_sort_key(k: str):
    return natural_command_key(k)


def interactive_menu(pipeline: "KlipyPipeline"):
    while True:
        print("\n" + "=" * 66)
        print(f"KLIPY.PY — aktivní projekt: {pipeline.project_dir.name}")
        print("Toto je 3. způsob tvorby full_plan.txt: jen z existujícího INPUT/klipy.md,")
        print("bez AI výpočtu časování (na rozdíl od 8a+8b v pro_pipeline.py).")
        print("=" * 66)
        for key in sorted(MENU_ACTIONS.keys(), key=_menu_sort_key):
            print(f"{key} - {MENU_ACTIONS[key][0]}")
        print("P - Změnit aktivní projekt")
        print("0 - Konec")
        print("-" * 66)
        choice = input("Volba: ").strip().lower()

        if choice == "0":
            break
        elif choice == "p":
            subdirs = _list_project_dirs()
            if not subdirs:
                print("❌ Žádné projekty nenalezeny v kořenové složce.")
                continue
            for idx, d in enumerate(subdirs):
                print(f"{idx + 1} - {d.name}")
            sel = input("Volba (nebo Enter pro zrušení): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(subdirs):
                pipeline.__init__(subdirs[int(sel) - 1])
        elif choice in MENU_ACTIONS:
            try:
                MENU_ACTIONS[choice][1](pipeline)
            except KeyboardInterrupt:
                print("\n⚠️  Přerušeno uživatelem.")
        else:
            print(f"❌ Neplatná volba: '{choice}'.")


def main():
    if len(sys.argv) == 1:
        project_dir = detect_project(None)
        pipeline = KlipyPipeline(project_dir)
        try:
            interactive_menu(pipeline)
        except KeyboardInterrupt:
            print("\n👋 Ukončeno uživatelem.")
        return

    parser = argparse.ArgumentParser(
        description="klipy.py — doprovodný nástroj k pro_pipeline.py pro práci s katalogem INPUT/klipy.md",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", "--napoveda", action="help", help="Zobrazit tuto nápovědu a skončit")
    parser.add_argument(
        "command",
        choices=sorted(CLI_COMMANDS.keys()),
        help=(
            "scenario/scenar (1), poradi/order (2), parse/parsuj (3), "
            "transcribe-rap/transkribuj-rap (4), resync-rap/resynchronizuj-rap (4b), "
            "align-rap/zarovnej-rap (5), "
            "apply-speeds/aplikuj-rychlosti (5b), "
            "update-timeline/prepocitej-timeline (6), align-vid/zarovnej-vid (7), "
            "render/renderuj (8)"
        ),
    )
    parser.add_argument("--project", "--projekt", "-p", default=None, help="Název nebo cesta ke složce projektu")
    args = parser.parse_args()

    project_dir = detect_project(args.project)
    pipeline = KlipyPipeline(project_dir)
    key = CLI_COMMANDS[args.command]
    MENU_ACTIONS[key][1](pipeline)


if __name__ == "__main__":
    main()
