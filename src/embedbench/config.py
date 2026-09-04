"""Load model rows from configs/models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_model_specs(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"no models found in {path}")
    return models


def select_specs(specs: list[dict[str, Any]], model_ids: list[str] | None) -> list[dict[str, Any]]:
    if not model_ids:
        return specs
    by_id = {spec["id"]: spec for spec in specs}
    missing = [model_id for model_id in model_ids if model_id not in by_id]
    if missing:
        known = ", ".join(sorted(by_id))
        raise ValueError(f"unknown model ids {missing}; known: {known}")
    return [by_id[model_id] for model_id in model_ids]
