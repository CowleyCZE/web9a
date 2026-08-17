from __future__ import annotations
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class Variant:
    name: str
    seed: int
    overrides: dict[str, Any]

DEFAULT_VARIANTS = (
    ("control", {}),
    ("faster_cuts", {"cut_density_multiplier": 1.15}),
    ("cleaner_motion", {"motion_intensity": 0.75}),
)

def stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

def build_variants(base_seed: int = 0, definitions: Iterable[tuple[str, dict[str, Any]]] = DEFAULT_VARIANTS) -> list[Variant]:
    return [Variant(name, stable_seed(base_seed, name), dict(overrides)) for name, overrides in definitions]

def apply_variant_overrides(config: dict[str, Any], variant: Variant | dict[str, Any]) -> dict[str, Any]:
    overrides = variant.overrides if isinstance(variant, Variant) else dict(variant.get("overrides", {}))
    result = dict(config)
    if "cut_density_multiplier" in overrides:
        result["cut_density_multiplier"] = max(0.5, min(2.0, float(overrides["cut_density_multiplier"])))
    if "motion_intensity" in overrides:
        result["motion_intensity"] = max(0.0, min(1.0, float(overrides["motion_intensity"])))
    return result

def build_variant_plans(base_plan: Iterable[dict[str, Any]], variants: Iterable[Variant]) -> dict[str, list[dict[str, Any]]]:
    plans = {}
    for variant in variants:
        multiplier = float(variant.overrides.get("cut_density_multiplier", 1.0))
        intensity = float(variant.overrides.get("motion_intensity", 1.0))
        rows = []
        for item in base_plan:
            row = dict(item)
            if "cut_density" in row:
                row["cut_density"] = round(max(0.0, min(1.0, float(row["cut_density"]) * multiplier)), 6)
            row["motion_intensity"] = round(max(0.0, min(1.0, intensity)), 6)
            row["variant"] = variant.name
            row["variant_seed"] = variant.seed
            rows.append(row)
        plans[variant.name] = rows
    return plans

def compare_variant_qa(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison = []
    for name, report in reports.items():
        comparison.append({"variant": name, "ok": bool(report.get("ok")), "error_count": len(report.get("errors", [])), "warning_count": len(report.get("warnings", []))})
    comparison.sort(key=lambda item: (not item["ok"], item["error_count"], item["warning_count"], item["variant"]))
    return {"schema_version": 1, "variants": comparison, "recommended": comparison[0]["variant"] if comparison else None}

def create_experiment_manifest(project: Path, base_seed: int = 0, source: str | None = None, base_plan: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    variants = build_variants(base_seed)
    manifest = {
        "schema_version": 2,
        "experiment_id": hashlib.sha256(f"{project.resolve()}:{base_seed}".encode()).hexdigest()[:16],
        "project": str(project),
        "source": source,
        "base_seed": int(base_seed),
        "variants": [asdict(v) for v in variants],
    }
    if base_plan is not None:
        manifest["variant_plans"] = build_variant_plans(list(base_plan), variants)
    return manifest

def write_experiment_manifest(project: Path, output: Path, base_seed: int = 0, source: str | None = None, base_plan: Iterable[dict[str, Any]] | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(create_experiment_manifest(project, base_seed, source, base_plan), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output

def variant_rng(variant: dict[str, Any]) -> random.Random:
    return random.Random(int(variant["seed"]))

__all__ = ["Variant", "DEFAULT_VARIANTS", "stable_seed", "build_variants", "apply_variant_overrides", "build_variant_plans", "compare_variant_qa", "create_experiment_manifest", "write_experiment_manifest", "variant_rng"]
