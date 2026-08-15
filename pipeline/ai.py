from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AIResult:
    text: str | None
    model: str
    error: str | None = None
    provider: str = "local"


def normalize_provider(value: Any, default: str = "local") -> str:
    provider = str(value or default).strip().lower()
    return provider if provider in {"local", "groq"} else default


def finite_positive_int(value: Any, default: int, maximum: int | None = None) -> int:
    try:
        result = int(value)
        if result <= 0:
            raise ValueError
    except (TypeError, ValueError):
        result = default
    if maximum is not None:
        result = min(result, maximum)
    return result


def finite_temperature(value: Any, default: float = 0.7) -> float:
    try:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError
        return min(result, 2.0)
    except (TypeError, ValueError):
        return default


def dispatch_text(
    messages,
    phase: str,
    settings: dict,
    local_model: str,
    groq_model: str,
    local_call: Callable[..., str | None],
    local_stream_call: Callable[..., str | None],
    groq_call: Callable[..., str | None],
    temperature: float = 0.7,
    timeout: int = 900,
    num_ctx: int = 8192,
    max_tokens: int | None = None,
) -> AIResult:
    provider = normalize_provider(settings.get("text_ai_provider")) if phase == "scenario" else "local"
    temperature = finite_temperature(temperature)
    timeout = finite_positive_int(timeout, 900, maximum=7200)
    num_ctx = finite_positive_int(num_ctx, 8192, maximum=262144)

    if provider == "groq":
        model = str(groq_model or "")
        output_limit = finite_positive_int(max_tokens or settings.get("groq_scenario_max_tokens"), 3000, maximum=32768)
        text = groq_call(messages, model=model, temperature=temperature, timeout=min(timeout, 600), max_tokens=output_limit)
        return AIResult(text, model, None if text else "Groq nevrátil textovou odpověď", provider)

    model = str(local_model or "")
    if phase == "plan":
        read_timeout = finite_positive_int(settings.get("ollama_stream_read_timeout_sec"), 600, maximum=7200)
        max_total = finite_positive_int(settings.get("ollama_stream_max_total_sec"), 7200, maximum=86400)
        text = local_stream_call(
            messages, model=model, temperature=temperature, num_ctx=num_ctx,
            read_timeout=read_timeout, max_total_seconds=max(timeout, max_total),
        )
    else:
        text = local_call(messages, model=model, format="text", temperature=temperature, timeout=timeout, num_ctx=num_ctx)
    return AIResult(text, model, None if text else "Lokální Ollama nevrátila textovou odpověď", provider)


__all__ = ["AIResult", "normalize_provider", "finite_positive_int", "finite_temperature", "dispatch_text"]
