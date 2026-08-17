from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_render_event(project_dir: Path, *, output: Path, mode: str, resolution: str,
                        duration: float | None = None, qa: dict[str, Any] | None = None,
                        seed: int | None = None, command: list[str] | None = None,
                        status: str = "completed", error: str | None = None) -> Path:
    registry = project_dir / "EDIT_PROJECT" / "render_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    qa = qa if isinstance(qa, dict) else {}
    record = {
        "schema_version": 1,
        "event": "render_completed",
        "timestamp_utc": utc_now(),
        "status": status,
        "output": str(output),
        "output_relative": str(output.relative_to(project_dir)) if output.is_relative_to(project_dir) else str(output),
        "mode": mode,
        "resolution": resolution,
        "duration_sec": round(float(duration), 3) if duration is not None else None,
        "size_bytes": output.stat().st_size if output.exists() else None,
        "sha256": sha256_file(output),
        "qa_ok": qa.get("ok"),
        "qa_errors": list(qa.get("errors", [])),
        "qa_warnings": list(qa.get("warnings", [])),
        "error": error,
        "qa_report": str(output.with_suffix(output.suffix + ".qa.json")),
        "seed": seed,
        "command": command,
    }
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return registry


def append_render_failure(project_dir: Path, *, mode: str, resolution: str, error: str, seed: int | None = None, output: Path | None = None) -> Path:
    return append_render_event(project_dir, output=output or (project_dir / "EXPORT" / "unknown.mp4"), mode=mode, resolution=resolution, seed=seed, status="failed", error=error, qa={"ok": False, "errors": [error], "warnings": []})


def read_render_registry(project_dir: Path) -> list[dict[str, Any]]:
    path = project_dir / "EDIT_PROJECT" / "render_registry.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except json.JSONDecodeError:
            continue
    return records


def build_qa_summary(project_dir: Path, registry: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = list(registry) if registry is not None else read_render_registry(project_dir)
    qa_files = sorted((project_dir / "EXPORT").glob("*.mp4.qa.json")) if (project_dir / "EXPORT").exists() else []
    reports: list[dict[str, Any]] = []
    for path in qa_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                reports.append({"path": str(path.relative_to(project_dir)), "ok": bool(data.get("ok")), "errors": list(data.get("errors", [])), "warnings": list(data.get("warnings", []))})
        except (OSError, json.JSONDecodeError):
            reports.append({"path": str(path.relative_to(project_dir)), "ok": False, "errors": ["Nečitelný QA report"], "warnings": []})
    source = reports or [{"ok": r.get("qa_ok"), "errors": r.get("qa_errors", []), "warnings": r.get("qa_warnings", []), "path": r.get("qa_report")} for r in records]
    passed = sum(1 for item in source if item.get("ok") is True)
    failed = sum(1 for item in source if item.get("ok") is False)
    warnings = sum(1 for item in source if item.get("warnings"))
    status = "UNKNOWN" if not source else ("FAIL" if failed else ("WARN" if warnings else "PASS"))
    return {"schema_version": 1, "generated_at_utc": utc_now(), "status": status, "counts": {"reports": len(source), "passed": passed, "failed": failed, "with_warnings": warnings}, "latest_render": records[-1] if records else None, "reports": source}


def write_qa_summary(project_dir: Path) -> Path:
    output = project_dir / "EDIT_PROJECT" / "qa_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_qa_summary(project_dir), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output

__all__ = ["utc_now", "sha256_file", "append_render_event", "append_render_failure", "read_render_registry", "build_qa_summary", "write_qa_summary"]
