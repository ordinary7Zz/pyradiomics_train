"""Train-only image-level feature filtering for AutoGluon task CSVs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from binary_class.new_code.feature_presets import FILTER_MODES

META_COLUMNS = frozenset({"filename", "patient_id", "image_path", "mask_path"})
MANIFEST_FILENAME = "feature_manifest.json"


@dataclass
class FilterConfig:
    mode: str = "standard"
    max_features_ratio: int = 10
    max_features_cap: int = 200
    min_features: int = 5
    corr_threshold: float = 0.95
    missing_frac_max: float = 0.05
    variance_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.mode not in FILTER_MODES:
            raise ValueError(f"filter mode must be one of {FILTER_MODES}, got {self.mode!r}")


def feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == label_col or c in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _max_features_for_n(n_train: int, cfg: FilterConfig) -> int:
    if n_train <= 0:
        return cfg.min_features
    by_ratio = max(cfg.min_features, n_train // max(1, cfg.max_features_ratio))
    return min(cfg.max_features_cap, by_ratio)


def _drop_high_missing(X: pd.DataFrame, cfg: FilterConfig) -> List[str]:
    frac = X.isna().mean()
    keep = frac[frac <= cfg.missing_frac_max].index.tolist()
    return keep


def _drop_low_variance(X: pd.DataFrame, cols: Sequence[str], cfg: FilterConfig) -> List[str]:
    keep: List[str] = []
    for c in cols:
        s = X[c].std(skipna=True)
        if s is None or (isinstance(s, float) and (np.isnan(s) or s <= cfg.variance_epsilon)):
            continue
        keep.append(c)
    return keep


def _drop_correlated(X: pd.DataFrame, cols: Sequence[str], cfg: FilterConfig) -> List[str]:
    if len(cols) <= 1:
        return list(cols)
    sub = X[list(cols)].astype(float)
    corr = sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set[str] = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > cfg.corr_threshold].tolist()
        for partner in partners:
            if col in to_drop or partner in to_drop:
                continue
            to_drop.add(partner)
    return [c for c in cols if c not in to_drop]


def _univariate_strength(X: pd.DataFrame, y: pd.Series, cols: Sequence[str]) -> pd.Series:
    scores: Dict[str, float] = {}
    yv = y.astype(int).to_numpy()
    for c in cols:
        xv = X[c].astype(float).to_numpy()
        mask = np.isfinite(xv)
        if mask.sum() < 3 or len(np.unique(yv[mask])) < 2:
            scores[c] = 0.0
            continue
        try:
            r = np.corrcoef(xv[mask], yv[mask])[0, 1]
            scores[c] = float(abs(r)) if np.isfinite(r) else 0.0
        except Exception:
            scores[c] = 0.0
    return pd.Series(scores)


def fit_feature_columns(
    train_df: pd.DataFrame,
    label_col: str = "label",
    cfg: Optional[FilterConfig] = None,
) -> List[str]:
    cfg = cfg or FilterConfig()
    if cfg.mode == "none":
        return feature_columns(train_df, label_col)

    y = train_df[label_col]
    cols = feature_columns(train_df, label_col)
    if not cols:
        raise ValueError("No numeric feature columns found in training CSV")

    X = train_df[cols]
    cols = _drop_high_missing(X, cfg)
    X = X[cols]
    cols = _drop_low_variance(X, cols, cfg)
    X = X[cols]
    cols = _drop_correlated(X, cols, cfg)

    cap = _max_features_for_n(len(train_df), cfg)
    if len(cols) > cap:
        strength = _univariate_strength(X, y, cols)
        cols = strength.sort_values(ascending=False).head(cap).index.tolist()

    if len(cols) < cfg.min_features:
        raise ValueError(f"Too few features after filtering ({len(cols)} < {cfg.min_features})")
    return cols


def build_task_frame(
    df: pd.DataFrame,
    selected: Sequence[str],
    label_col: str = "label",
    keep_filename: bool = True,
) -> pd.DataFrame:
    parts: List[pd.Series] = []
    if keep_filename and "filename" in df.columns:
        parts.append(df["filename"].astype(str))
    for c in selected:
        if c not in df.columns:
            raise KeyError(f"Missing feature column {c!r}")
        parts.append(df[c])
    parts.append(df[label_col].astype(int))
    out = pd.concat(parts, axis=1)
    out.columns = (["filename"] if keep_filename and "filename" in df.columns else []) + list(selected) + [label_col]
    return out


def save_manifest(
    path: Path,
    *,
    feature_columns: Sequence[str],
    label_col: str,
    n_train: int,
    feature_preset: str,
    filter_mode: str,
    radiomics_params: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "version": 1,
        "feature_preset": feature_preset,
        "filter_mode": filter_mode,
        "label_column": label_col,
        "n_train": int(n_train),
        "n_features": len(feature_columns),
        "radiomics_params": radiomics_params,
        "feature_columns": list(feature_columns),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_manifest(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "feature_columns" not in data:
        raise ValueError(f"Invalid manifest (missing feature_columns): {path}")
    return data


def validate_against_manifest(df: pd.DataFrame, manifest_path: Path | str, label_col: str = "label") -> None:
    manifest = load_manifest(manifest_path)
    expected = list(manifest["feature_columns"])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing {len(missing)} manifest features (e.g. {missing[:5]}); "
            f"manifest={manifest_path}"
        )
    _ = build_task_frame(df, expected, label_col=label_col, keep_filename=False)


def filter_train_test_tasks(
    train_csv: Path | str,
    test_csv: Optional[Path | str],
    manifest_path: Path | str,
    *,
    label_col: str = "label",
    cfg: Optional[FilterConfig] = None,
    feature_preset: str = "full",
    radiomics_params: str = "",
    keep_filename: bool = False,
) -> Tuple[Path, Optional[Path]]:
    cfg = cfg or FilterConfig()
    train_path = Path(train_csv)
    manifest_out = Path(manifest_path)
    train_df = pd.read_csv(train_path)
    if label_col not in train_df.columns:
        raise ValueError(f"Missing {label_col} in {train_path}")

    selected = fit_feature_columns(train_df, label_col=label_col, cfg=cfg)
    save_manifest(
        manifest_out,
        feature_columns=selected,
        label_col=label_col,
        n_train=len(train_df),
        feature_preset=feature_preset,
        filter_mode=cfg.mode,
        radiomics_params=radiomics_params or "",
    )

    train_out = train_path
    train_filtered = build_task_frame(train_df, selected, label_col=label_col, keep_filename=keep_filename)
    train_filtered.to_csv(train_out, index=False)
    print(f"Wrote train task ({len(train_filtered)} rows, {len(selected)} features) -> {train_out}")

    test_out: Optional[Path] = None
    if test_csv is not None:
        test_path = Path(test_csv)
        test_df = pd.read_csv(test_path)
        test_filtered = build_task_frame(test_df, selected, label_col=label_col, keep_filename=keep_filename)
        test_filtered.to_csv(test_path, index=False)
        test_out = test_path
        print(f"Wrote test task ({len(test_filtered)} rows, {len(selected)} features) -> {test_path}")

    return train_out, test_out


def parse_filter_config(mode: str, **kwargs: Any) -> FilterConfig:
    return FilterConfig(mode=mode, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image-level train-only feature filter for task CSVs.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--label", type=str, default="label")
    p.add_argument("--filter_mode", type=str, default="standard", choices=list(FILTER_MODES))
    p.add_argument("--feature_preset", type=str, default="full")
    p.add_argument("--radiomics_params", type=str, default="")
    p.add_argument("--keep_filename", action="store_true", default=True)
    p.add_argument("--drop_filename", action="store_true", help="omit filename column from filtered CSVs")
    p.add_argument("--max_features_ratio", type=int, default=10)
    p.add_argument("--max_features_cap", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FilterConfig(
        mode=args.filter_mode,
        max_features_ratio=args.max_features_ratio,
        max_features_cap=args.max_features_cap,
    )
    filter_train_test_tasks(
        args.train_csv,
        args.test_csv,
        args.manifest,
        label_col=args.label,
        cfg=cfg,
        feature_preset=args.feature_preset,
        radiomics_params=args.radiomics_params,
        keep_filename=not args.drop_filename,
    )


if __name__ == "__main__":
    main()
