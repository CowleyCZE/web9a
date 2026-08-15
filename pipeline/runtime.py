from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|authorization|bearer)\s*[:=]\s*\S+")


@dataclass
class RuntimeResult:
    ok: bool
    value: Any = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def redact_secrets(message: str) -> str:
    return _SECRET_RE.sub(r"\1=<redacted>", str(message))


def project_logger(project_dir: Path, name: str = "temagen") -> logging.Logger:
    """Vrátí idempotentně nakonfigurovaný logger s konzolí i pipeline.log."""
    logger = logging.getLogger(f"{name}.{project_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    log_dir = project_dir / "EDIT_PROJECT"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    logger.exception("%s: %s", redact_secrets(message), redact_secrets(exc))
