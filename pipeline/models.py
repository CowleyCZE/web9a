from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    ok: bool
    value: Any = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def success(cls, value: Any = None, warnings: list[str] | None = None) -> "StepResult":
        return cls(True, value=value, warnings=list(warnings or []))

    @classmethod
    def failure(cls, *errors: str, value: Any = None, warnings: list[str] | None = None) -> "StepResult":
        return cls(False, value=value, errors=list(errors), warnings=list(warnings or []))


@dataclass
class TimelineEntry:
    start: float
    end: float
    clip_id: str
    description: str = ""
    source: str = "unknown"
    duration_source: str = "measured"

    @property
    def duration(self) -> float:
        return self.end - self.start
