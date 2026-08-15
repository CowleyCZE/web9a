from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .models import StepResult


def execute_step(name: str, action: Callable[[], Any]) -> StepResult:
    """Spustí legacy krok a převede jeho výsledek na jednotný StepResult.

    Legacy metody často vracejí None při úspěchu, True/False při validaci nebo
    mohou vyhodit výjimku. Pouze explicitní False znamená selhání; None je
    kvůli zpětné kompatibilitě považováno za úspěch.
    """
    try:
        value = action()
    except Exception as exc:
        return StepResult.failure(f"{name}: {type(exc).__name__}: {exc}")
    if value is False:
        return StepResult.failure(f"{name}: krok oznámil neúspěch", value=value)
    if isinstance(value, StepResult):
        return value
    return StepResult.success(value=value)


def execute_sequence(steps: Iterable[tuple[str, Callable[[], Any]]]) -> StepResult:
    warnings: list[str] = []
    values: dict[str, Any] = {}
    for name, action in steps:
        result = execute_step(name, action)
        warnings.extend(result.warnings)
        values[name] = result.value
        if not result.ok:
            return StepResult.failure(*result.errors, value=values, warnings=warnings)
    return StepResult.success(value=values, warnings=warnings)


__all__ = ["execute_step", "execute_sequence"]
