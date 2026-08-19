from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .models import StepResult, StepStatus


def execute_step(name: str, action: Callable[[], Any]) -> StepResult:
    """Spustí legacy krok a převede výsledek na jednotný stavový model.

    Legacy metody mohou vracet None, bool, StepResult nebo dict s ``status``.
    ``None`` zůstává kvůli zpětné kompatibilitě úspěchem; explicitní False je
    ``failed``. Stav ``clamped``/``mismatch`` lze vrátit přes StepResult nebo
    slovník, aniž by se ztratil při orchestrace.
    """
    try:
        value = action()
    except Exception as exc:
        return StepResult.failure(f"{name}: {type(exc).__name__}: {exc}", status=StepStatus.FAILED)

    if isinstance(value, StepResult):
        return value

    if isinstance(value, dict) and "status" in value:
        status = StepStatus.normalize(value.get("status"))
        if status in {StepStatus.OK, StepStatus.CLAMPED}:
            return StepResult.success(value=value, status=status)
        return StepResult.failure(f"{name}: krok skončil ve stavu {status}", value=value, status=status)

    if value is False:
        return StepResult.failure(f"{name}: krok oznámil neúspěch", value=value, status=StepStatus.FAILED)

    return StepResult.success(value=value, status=StepStatus.OK)


def execute_sequence(steps: Iterable[tuple[str, Callable[[], Any]]]) -> StepResult:
    warnings: list[str] = []
    values: dict[str, Any] = {}
    for name, action in steps:
        result = execute_step(name, action)
        warnings.extend(result.warnings)
        values[name] = result.value
        if not result.ok:
            return StepResult.failure(*result.errors, value=values, warnings=warnings, status=result.status)
    return StepResult.success(value=values, warnings=warnings)


__all__ = ["execute_step", "execute_sequence"]
