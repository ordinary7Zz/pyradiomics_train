import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd


META_COLUMNS = ["filename", "image_path", "mask_path"]
AGGREGATION_CHOICES = ["mean", "median", "max", "min"]


def _normalize_rel_path(path: str) -> str:
    return os.path.normpath(path.replace("\\", "/").lstrip("/\\"))


def _extract_patient_key(filename: str) -> str:
    normalized = _normalize_rel_path(str(filename))
    parts = normalized.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(
            f"filename must contain at least 3 path segments (year/patient/file), got: {filename}"
        )
    return f"{parts[0]}/{parts[1]}"


def _coerce_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.dropna().astype(int)
    values = values[values != -1]
    return values


def _resolve_feature_columns(df: pd.DataFrame, label_col: str, filename_col: str) -> List[str]:
    excluded = set(META_COLUMNS + [label_col, filename_col])
    feature_cols = [c for c in df.columns if c not in excluded]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns available for patient-level aggregation")
    return numeric_feature_cols


def _build_aggregations(feature_cols: List[str], aggregation: str) -> Dict[str, str]:
    return {col: aggregation for col in feature_cols}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate image-level binary task CSV into a patient-level task CSV."
    )
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--label", type=str, default="label")
    p.add_argument("--filename_col", type=str, default="filename")
    p.add_argument("--feature_agg", type=str, default="mean", choices=AGGREGATION_CHOICES)
    p.add_argument("--summary_csv", type=str, default=None)
    p.add_argument("--mapping_csv", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if args.filename_col not in df.columns:
        raise ValueError(f"Missing filename column: {args.filename_col}")
    if args.label not in df.columns:
        raise ValueError(f"Missing label column: {args.label}")

    df = df.copy()
    df[args.filename_col] = df[args.filename_col].map(_normalize_rel_path)
    df[args.label] = pd.to_numeric(df[args.label], errors="coerce")
    df = df.dropna(subset=[args.label]).copy()
    df[args.label] = df[args.label].astype(int)
    df = df[df[args.label] != -1].copy()
    if df.empty:
        raise ValueError("No valid labeled rows remain after filtering")

    df["_patient_key"] = df[args.filename_col].map(_extract_patient_key)

    label_counts = df.groupby("_patient_key")[args.label].nunique(dropna=True)
    conflicting = label_counts[label_counts > 1]
    if not conflicting.empty:
        examples = conflicting.index.tolist()[:10]
        raise ValueError(
            "Found patient groups with inconsistent labels. "
            f"Examples: {examples}"
        )

    feature_cols = _resolve_feature_columns(df, label_col=args.label, filename_col=args.filename_col)
    agg_map = _build_aggregations(feature_cols, args.feature_agg)

    grouped_features = df.groupby("_patient_key", sort=True).agg(agg_map)
    grouped_features = grouped_features.rename(
        columns={col: f"{col}_{args.feature_agg}" for col in feature_cols}
    )

    summary_df = (
        df.groupby("_patient_key", sort=True)
        .agg(
            **{
                args.label: (args.label, "first"),
                "image_count": (args.filename_col, "size"),
            }
        )
        .reset_index()
    )

    patient_df = summary_df.merge(
        grouped_features.reset_index(),
        on="_patient_key",
        how="inner",
    )
    if patient_df.empty:
        raise ValueError("Patient-level aggregation produced no rows")

    patient_df = patient_df.rename(columns={"_patient_key": "filename"})

    ordered_cols = ["filename", "image_count"]
    ordered_cols += [c for c in patient_df.columns if c not in {"filename", "image_count", args.label}]
    ordered_cols += [args.label]
    patient_df = patient_df[ordered_cols]

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    patient_df.to_csv(args.output_csv, index=False)

    if args.summary_csv:
        summary_dir = os.path.dirname(os.path.abspath(args.summary_csv))
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary_out = summary_df.rename(columns={"_patient_key": "filename"})
        summary_out.to_csv(args.summary_csv, index=False)

    if args.mapping_csv:
        mapping_dir = os.path.dirname(os.path.abspath(args.mapping_csv))
        if mapping_dir:
            os.makedirs(mapping_dir, exist_ok=True)
        mapping_df = df[["_patient_key", args.filename_col, args.label]].copy()
        mapping_df = mapping_df.rename(
            columns={"_patient_key": "patient_key", args.filename_col: "image_filename"}
        )
        mapping_df.to_csv(args.mapping_csv, index=False)

    n_patients = int(patient_df.shape[0])
    n_neg = int((patient_df[args.label] == 0).sum())
    n_pos = int((patient_df[args.label] == 1).sum())
    print(
        f"Saved patient-level CSV: rows={n_patients} neg={n_neg} pos={n_pos} -> {args.output_csv}"
    )


if __name__ == "__main__":
    main()
