"""Radiomics YAML presets for extraction (image-level rows)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

PYR_ROOT = Path(__file__).resolve().parents[2]

RADIOMICS_PRESET_PATHS: Dict[str, Path] = {
    "full": PYR_ROOT / "radiomics_2d.yaml",
    "original_only": PYR_ROOT / "radiomics_2d_original.yaml",
    "shape_only": PYR_ROOT / "radiomics_2d_shape_only.yaml",
}

FILTER_MODES = ("none", "standard")


def resolve_radiomics_params(
    feature_preset: str | None = None,
    params_path: str | Path | None = None,
) -> Path:
    if params_path is not None:
        path = Path(params_path)
        if not path.is_file():
            raise FileNotFoundError(f"Radiomics params not found: {path}")
        return path

    key = (feature_preset or "full").strip().lower()
    if key not in RADIOMICS_PRESET_PATHS:
        raise KeyError(f"Unknown feature_preset {key!r}; choose from {list(RADIOMICS_PRESET_PATHS)}")
    path = RADIOMICS_PRESET_PATHS[key]
    if not path.is_file():
        raise FileNotFoundError(f"Preset file missing for {key}: {path}")
    return path
