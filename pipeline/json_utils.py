from __future__ import annotations

import json
import re


def _balanced_candidates(text: str) -> list[str]:
    candidates = []
    pairs = {"{": "}", "[": "]"}
    for start, char in enumerate(text):
        if char not in pairs:
            continue
        stack = []
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in pairs:
                stack.append(pairs[ch])
            elif ch in pairs.values():
                if not stack or ch != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start:i + 1])
                    break
    return sorted(set(candidates), key=len, reverse=True)


def _parse(candidate: str):
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return None


def parse_json_response(text: str, require_dict: bool = True):
    """Bezpečně získá JSON z odpovědi LLM s markdown/preambulí/trailing-comma fallbackem."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.replace("\ufeff", "").strip()
    cleaned = re.sub(r"^```(?:json|javascript|js)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    parsed = _parse(cleaned)
    if parsed is not None and (not require_dict or isinstance(parsed, dict)):
        return parsed
    for candidate in _balanced_candidates(cleaned):
        parsed = _parse(candidate)
        if parsed is not None and (not require_dict or isinstance(parsed, dict)):
            return parsed
    return None
