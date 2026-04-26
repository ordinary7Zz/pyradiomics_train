import argparse
import datetime as _dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Quick analysis for radiomics CSV: missing/Inf, constants, outliers, and simple "
            "per-feature association with binary label."
        )
    )
    p.add_argument("--csv", type=str, required=True, help="Input radiomics features CSV")
    p.add_argument("--label", type=str, default="label", help="Label column name")
    p.add_argument("--out_dir", type=str, default="./csv_analysis", help="Output directory")
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional: only analyze first N rows (debug)",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="How many top-ranked features to print",
    )
    p.add_argument(
        "--z_thresh",
        type=float,
        default=5.0,
        help="Z-score threshold to count extreme values",
    )
    p.add_argument(
        "--txt_path",
        type=str,
        default=None,
        help="Optional: save console summary to this txt path (default: <out_dir>/summary.txt)",
    )
    return p.parse_args()


def _cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    # Cohen's d for two groups (0 vs 1)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    m0, m1 = float(np.mean(x0)), float(np.mean(x1))
    s0, s1 = float(np.std(x0, ddof=1)), float(np.std(x1, ddof=1))
    sp = np.sqrt(((len(x0) - 1) * s0 * s0 + (len(x1) - 1) * s1 * s1) / (len(x0) + len(x1) - 2))
    if sp == 0:
        return 0.0
    return (m1 - m0) / sp


def _outlier_frac_iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo = q1 - 3.0 * iqr
    hi = q3 + 3.0 * iqr
    return float(np.mean((x < lo) | (x > hi)))


def _outlier_frac_z(x: np.ndarray, z_thresh: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    if sigma == 0:
        return 0.0
    z = (x - mu) / sigma
    return float(np.mean(np.abs(z) > z_thresh))


def _try_auc(feature: np.ndarray, y: np.ndarray) -> float:
    """Compute single-feature ROC AUC (tries both directions). Returns NaN if unavailable."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return np.nan

    mask = np.isfinite(feature) & np.isfinite(y)
    f = feature[mask]
    yy = y[mask].astype(int)
    if len(np.unique(yy)) != 2:
        return np.nan
    if len(np.unique(f)) < 2:
        return 0.5

    # roc_auc_score expects higher score => positive; try both signs
    try:
        auc1 = float(roc_auc_score(yy, f))
        auc2 = float(roc_auc_score(yy, -f))
        return max(auc1, auc2)
    except Exception:
        return np.nan


def analyze_csv(csv_path: str, label_col: str, z_thresh: float, max_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop obvious non-feature columns if present
    drop_cols = [c for c in DROP_IF_PRESENT if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")

    if max_rows is not None:
        df = df.head(max_rows).copy()

    # Keep only labeled rows (assume -1 means unlabeled)
    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)

    y = df[label_col].to_numpy()

    # Feature candidates: numeric cols excluding label
    feat_cols: List[str] = [c for c in df.columns if c != label_col]

    # Coerce to numeric where possible, keep non-numeric flagged
    non_numeric = []
    numeric_df = pd.DataFrame(index=df.index)
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_df[c] = df[c]
        else:
            # try coercion
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                numeric_df[c] = coerced
            else:
                non_numeric.append(c)

    X = numeric_df

    rows: List[Dict[str, object]] = []
    for c in X.columns:
        s = X[c]
        arr = s.to_numpy(dtype=float, copy=False)

        missing_frac = float(np.mean(pd.isna(arr)))
        inf_frac = float(np.mean(np.isinf(arr)))
        finite = arr[np.isfinite(arr)]

        unique = int(pd.Series(finite).nunique()) if finite.size else 0
        std = float(np.std(finite, ddof=0)) if finite.size else np.nan

        out_iqr = _outlier_frac_iqr(arr)
        out_z = _outlier_frac_z(arr, z_thresh=z_thresh)

        # signal proxy
        # Cohen's d is a simple effect size (directional); AUC gives rank-based separation
        x0 = arr[y == 0]
        x1 = arr[y == 1]
        d = _cohen_d(x0, x1)
        auc = _try_auc(arr, y)

        rows.append(
            {
                "feature": c,
                "missing_frac": missing_frac,
                "inf_frac": inf_frac,
                "unique_finite": unique,
                "std_finite": std,
                "outlier_frac_iqr3": out_iqr,
                "outlier_frac_z": out_z,
                "cohen_d": d,
                "abs_cohen_d": float(abs(d)) if np.isfinite(d) else np.nan,
                "single_feature_auc": auc,
            }
        )

    report = pd.DataFrame(rows)

    # Add simple flags
    report["is_constant_or_empty"] = (report["unique_finite"] <= 1) | report["std_finite"].fillna(0).eq(0)
    report["has_missing"] = report["missing_frac"] > 0
    report["has_inf"] = report["inf_frac"] > 0
    report["many_outliers"] = (report["outlier_frac_z"].fillna(0) > 0.01) | (report["outlier_frac_iqr3"].fillna(0) > 0.01)

    # Keep a note about non-numeric columns
    if non_numeric:
        nn = pd.DataFrame({"feature": non_numeric})
        nn["note"] = "non-numeric (ignored)"
        # append as special rows for visibility
        report = pd.concat([report, nn], ignore_index=True)

    return report


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_path = args.txt_path or os.path.join(args.out_dir, "summary.txt")
    txt_lines: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        txt_lines.append(s)

    report = analyze_csv(
        csv_path=args.csv,
        label_col=args.label,
        z_thresh=args.z_thresh,
        max_rows=args.max_rows,
    )

    # Print quick dataset summary
    df = pd.read_csv(args.csv)
    if args.label in df.columns:
        labeled = df[df[args.label] != -1]
        try:
            y = labeled[args.label].astype(int)
            vc = y.value_counts().sort_index()
            emit("Label distribution (excluding -1):")
            emit(vc.to_string())
        except Exception:
            pass

    # Save full report
    report_path = os.path.join(args.out_dir, "feature_quality_report.csv")
    report.to_csv(report_path, index=False)

    # Flagged subsets
    flagged = report[
        (report.get("is_constant_or_empty", False) == True)
        | (report.get("has_inf", False) == True)
        | (report.get("many_outliers", False) == True)
        | (report.get("missing_frac", 0) > 0.05)
    ].copy()
    flagged_path = os.path.join(args.out_dir, "flagged_features.csv")
    flagged.to_csv(flagged_path, index=False)

    # Rank by single-feature AUC if available, else by abs_cohen_d
    rank = report.copy()
    if "single_feature_auc" in rank.columns and rank["single_feature_auc"].notna().any():
        rank = rank.dropna(subset=["single_feature_auc"]).sort_values(
            ["single_feature_auc", "abs_cohen_d"], ascending=[False, False]
        )
        rank_key = "single_feature_auc"
    else:
        rank = rank.dropna(subset=["abs_cohen_d"]).sort_values("abs_cohen_d", ascending=False)
        rank_key = "abs_cohen_d"

    top_path = os.path.join(args.out_dir, "top_features.csv")
    rank.head(args.top_k).to_csv(top_path, index=False)

    emit(f"Saved: {report_path}")
    emit(f"Saved: {flagged_path}")
    emit(f"Saved: {top_path}")
    emit(f"Ranking key: {rank_key}")

    if not rank.empty:
        emit("\nTop features:")
        cols_to_show = ["feature", rank_key, "missing_frac", "inf_frac", "outlier_frac_z", "outlier_frac_iqr3"]
        cols_to_show = [c for c in cols_to_show if c in rank.columns]
        emit(rank[cols_to_show].head(args.top_k).to_string(index=False))

    # Save the console-style summary into a txt file for downstream analysis.
    header = [
        f"csv: {args.csv}",
        f"label_col: {args.label}",
        f"generated_at: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    content = "\n".join(header + txt_lines) + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    emit(f"Saved summary txt: {txt_path}")


if __name__ == "__main__":
    main()
