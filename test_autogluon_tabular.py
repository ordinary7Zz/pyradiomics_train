import argparse
import os
from typing import Dict, List

import pandas as pd
from autogluon.tabular import TabularPredictor


DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an AutoGluon TabularPredictor on multiple test CSV datasets."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="AutoGluon training output directory (contains predictor.pkl)",
    )
    p.add_argument(
        "--test_csv",
        type=str,
        nargs="+",
        required=True,
        help="Test CSV path(s). Example: --test_csv a.csv b.csv c.csv",
    )
    p.add_argument(
        "--test_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional dataset name(s), 1-1 aligned with --test_csv order",
    )
    p.add_argument("--label", type=str, default="label", help="Label column name")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for computing sensitivity/specificity (default: 0.5)",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional: save per-dataset metrics table to this CSV path",
    )
    return p.parse_args()


def _prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    drop_cols = [c for c in DROP_IF_PRESENT if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def _default_name(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    return name


def _safe_auprc(y_true: pd.Series, y_score: pd.Series) -> float:
    """AUPRC (Average Precision). Returns NaN if sklearn is unavailable or invalid input."""
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return float("nan")

    yy = y_true.to_numpy()
    ss = y_score.to_numpy()
    if len(pd.unique(yy)) != 2:
        return float("nan")
    try:
        return float(average_precision_score(yy, ss))
    except Exception:
        return float("nan")


def _sens_spec(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Sensitivity=Recall for positive class; Specificity=True negative rate."""
    yy = y_true.to_numpy()
    pp = y_pred.to_numpy()
    if len(pd.unique(yy)) != 2:
        return {"sensitivity": float("nan"), "specificity": float("nan")}

    tp = int(((yy == 1) & (pp == 1)).sum())
    tn = int(((yy == 0) & (pp == 0)).sum())
    fp = int(((yy == 0) & (pp == 1)).sum())
    fn = int(((yy == 1) & (pp == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"sensitivity": float(sensitivity), "specificity": float(specificity)}


def main() -> None:
    args = parse_args()

    predictor = TabularPredictor.load(args.model_dir)

    test_csvs: List[str] = args.test_csv

    if args.test_names is not None and len(args.test_names) != len(test_csvs):
        raise ValueError(
            f"--test_names count ({len(args.test_names)}) must match --test_csv count ({len(test_csvs)})"
        )

    test_names = args.test_names or [_default_name(p) for p in test_csvs]

    rows: List[Dict[str, object]] = []

    for name, csv_path in zip(test_names, test_csvs):
        df = pd.read_csv(csv_path)
        df = _prepare_df(df, args.label)

        perf: Dict[str, float] = predictor.evaluate(
            df,
            silent=True,
        )

        # Extra metrics: AUPRC + sensitivity/specificity
        # Sensitivity is equivalent to Recall for the positive class.
        y_true = df[args.label]
        proba = predictor.predict_proba(df)
        if 1 in proba.columns:
            pos_proba = proba[1]
        else:
            # Fallback if positive class column is not labeled as '1'
            pos_proba = proba.iloc[:, -1]
        y_pred = (pos_proba >= float(args.threshold)).astype(int)

        perf["auprc"] = _safe_auprc(y_true, pos_proba)
        perf.update(_sens_spec(y_true, y_pred))

        row: Dict[str, object] = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(df.shape[0]),
        }
        row.update(perf)
        rows.append(row)

        print(f"[{name}] rows={row['n_rows']} perf={perf}")

    results = pd.DataFrame(rows)

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(args.model_dir, "test_results.csv")

    parent = os.path.dirname(out_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)

    results.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
