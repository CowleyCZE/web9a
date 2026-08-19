from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class StepStatus:
    """Jednotný stavový model pipeline.

    ``ok``      = krok dokončen bez významné odchylky.
    ``clamped`` = krok dokončen, ale hodnota byla omezena bezpečným limitem.
    ``mismatch`` = krok doběhl, ale výsledek nesplňuje očekávanou vazbu/slot.
    ``failed``  = krok se nepodařilo dokončit nebo je výsledek nepoužitelný.
    """

    OK = "ok"
    CLAMPED = "clamped"
    MISMATCH = "mismatch"
    FAILED = "failed"
    ALL = frozenset({OK, CLAMPED, MISMATCH, FAILED})

    @classmethod
    def normalize(cls, value: str | None, default: str = FAILED) -> str:
        value = str(value or "").strip().lower()
        return value if value in cls.ALL else default

    @classmethod
    def is_success(cls, value: str | None) -> bool:
        return cls.normalize(value) in {cls.OK, cls.CLAMPED}


@dataclass
class StepResult:
    ok: bool
    value: Any = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = StepStatus.OK

    def __post_init__(self) -> None:
        self.status = StepStatus.normalize(self.status, StepStatus.OK if self.ok else StepStatus.FAILED)
        if self.status in {StepStatus.FAILED, StepStatus.MISMATCH}:
            self.ok = False
        elif self.status in {StepStatus.OK, StepStatus.CLAMPED}:
            self.ok = True

    @classmethod
    def success(cls, value: Any = None, warnings: list[str] | None = None, status: str = StepStatus.OK) -> "StepResult":
        status = StepStatus.normalize(status, StepStatus.OK)
        if status not in {StepStatus.OK, StepStatus.CLAMPED}:
            raise ValueError(f"success() nepodporuje stav {status!r}")
        return cls(True, value=value, warnings=list(warnings or []), status=status)

    @classmethod
    def failure(
        cls,
        *errors: str,
        value: Any = None,
        warnings: list[str] | None = None,
        status: str = StepStatus.FAILED,
    ) -> "StepResult":
        status = StepStatus.normalize(status, StepStatus.FAILED)
        if status not in {StepStatus.MISMATCH, StepStatus.FAILED}:
            raise ValueError(f"failure() nepodporuje stav {status!r}")
        return cls(False, value=value, errors=list(errors), warnings=list(warnings or []), status=status)


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

    @property
    def start_ms(self) -> int:
        return int(round(self.start * 1000))

    @property
    def end_ms(self) -> int:
        return int(round(self.end * 1000))

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
