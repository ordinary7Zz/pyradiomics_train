import argparse
import os
from typing import Dict, List, Optional

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
        help="Probability threshold for computing binary metrics (default: 0.5)",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional: save per-dataset metrics table to this CSV path",
    )
    p.add_argument("--mask_source", type=str, default=None, help="Optional mask source tag to include in results")
    p.add_argument("--train_dataset", type=str, default=None, help="Optional train dataset tag to include in results")
    p.add_argument("--task_name", type=str, default=None, help="Optional task tag to include in results")
    p.add_argument("--feature_csv", type=str, default=None, help="Optional training feature CSV path to include in results")
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


def _get_positive_proba(proba: pd.DataFrame) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.astype(float)
    if 1 in proba.columns:
        return proba[1].astype(float)
    return proba.iloc[:, -1].astype(float)


def _safe_auroc(y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")

    yy = y_true.to_numpy()
    ss = y_score.to_numpy()
    if len(pd.unique(yy)) != 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, ss))
    except Exception:
        return float("nan")


def _safe_auprc(y_true: pd.Series, y_score: pd.Series) -> float:
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


def _specificity(y_true: pd.Series, y_pred: pd.Series) -> float:
    yy = y_true.to_numpy()
    pp = y_pred.to_numpy()
    if len(pd.unique(yy)) != 2:
        return float("nan")
    tn = int(((yy == 0) & (pp == 0)).sum())
    fp = int(((yy == 0) & (pp == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def _compute_binary_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float) -> Dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception:
        accuracy_score = f1_score = precision_score = recall_score = None

    y_pred = (y_score >= float(threshold)).astype(int)
    metrics: Dict[str, float] = {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "acc": float("nan"),
        "prec": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "specificity": _specificity(y_true, y_pred),
    }
    if accuracy_score is not None:
        try:
            metrics["acc"] = float(accuracy_score(y_true.to_numpy(), y_pred.to_numpy()))
            metrics["prec"] = float(precision_score(y_true.to_numpy(), y_pred.to_numpy(), zero_division=0))
            metrics["recall"] = float(recall_score(y_true.to_numpy(), y_pred.to_numpy(), zero_division=0))
            metrics["f1"] = float(f1_score(y_true.to_numpy(), y_pred.to_numpy(), zero_division=0))
        except Exception:
            pass
    return metrics


def _extract_auroc(perf: Dict[str, float], metrics: Dict[str, float]) -> float:
    for key in ("roc_auc", "auroc"):
        if key in perf and pd.notna(perf[key]):
            return float(perf[key])
    return float(metrics.get("auroc", float("nan")))


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

        y_true = df[args.label]
        pos_proba = _get_positive_proba(predictor.predict_proba(df))
        metrics = _compute_binary_metrics(y_true, pos_proba, threshold=float(args.threshold))

        row: Dict[str, object] = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(df.shape[0]),
            "mask_source": args.mask_source,
            "train_dataset": args.train_dataset,
            "task": args.task_name,
            "model_dir": args.model_dir,
            "feature_csv": args.feature_csv,
            "auroc": _extract_auroc(perf, metrics),
            "auprc": float(metrics.get("auprc", float("nan"))),
            "acc": float(metrics.get("acc", float("nan"))),
            "sensitivity": float(metrics.get("recall", float("nan"))),
            "specificity": float(metrics.get("specificity", float("nan"))),
        }
        row.update(perf)
        row["recall"] = float(metrics.get("recall", float("nan")))
        row["precision"] = float(metrics.get("prec", float("nan")))
        row["f1"] = float(metrics.get("f1", float("nan")))
        row["ece"] = float(metrics.get("ece", float("nan")))
        rows.append(row)

        print(
            f"[{name}] rows={row['n_rows']} auroc={row['auroc']:.6f} auprc={row['auprc']:.6f} "
            f"acc={row['acc']:.6f} sensitivity={row['sensitivity']:.6f} specificity={row['specificity']:.6f}"
        )

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
