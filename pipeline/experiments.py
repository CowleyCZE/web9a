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

def create_experiment_manifest(project: Path, base_seed: int = 0, source: str | None = None) -> dict[str, Any]:
    variants = build_variants(base_seed)
    return {
        "schema_version": 1,
        "experiment_id": hashlib.sha256(f"{project.resolve()}:{base_seed}".encode()).hexdigest()[:16],
        "project": str(project),
        "source": source,
        "base_seed": int(base_seed),
        "variants": [asdict(v) for v in variants],
    }

def write_experiment_manifest(project: Path, output: Path, base_seed: int = 0, source: str | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(create_experiment_manifest(project, base_seed, source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output

def variant_rng(variant: dict[str, Any]) -> random.Random:
    return random.Random(int(variant["seed"]))

__all__ = ["Variant", "DEFAULT_VARIANTS", "stable_seed", "build_variants", "create_experiment_manifest", "write_experiment_manifest", "variant_rng"]
