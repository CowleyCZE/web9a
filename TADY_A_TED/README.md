# Tady a Teď

Projektový balíček pro přípravu videoklipu k songu **Tady a Teď**.

## Obsah

| Složka | Obsah |
|---|---|
| `INPUT/` | Audio, oficiální lyrics a timestampovaná pracovní transkripce |
| `EDIT_PROJECT/` | Beat mapa, song alignment, word/phoneme alignment a generativní manifest |
| `Prompts/` | Scénář, character profile a prompty pro generování klipů |

## Stav

Audio preflight byl dokončen pro soubor `INPUT/Tady_a_Teď.mp3`. Délka songu je přibližně 164,9 s a tempo 89,1 BPM. Generativní balíček obsahuje čtyři klíčové rap lipsync pasáže a pět základních B-roll promptů.

Rap lipsync používá character-specific režim pro maskovaného čapího rapera se stabilním zobákem, černou čepicí a konzistentní siluetou. Pro finální generování klipů je nutné postupovat podle `Prompts/generation_prompts.md` a zachovat uvedené délky jednotlivých klipů.

## Důležitá poznámka

`INPUT/lyrics.txt` je oficiální text songu převzatý z kořenového souboru repozitáře. `transcription.json` je timestampovaná automatická transkripce použitá pro alignment; před finálním renderem je vhodné případné rozdíly mezi textem a STT ještě ručně zkontrolovat.
