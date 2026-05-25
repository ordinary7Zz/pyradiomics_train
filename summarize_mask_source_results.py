import argparse
import os
from typing import List, Optional

import pandas as pd


KEY_COLUMNS = [
    "train_dataset",
    "dataset",
    "task",
    "train_mask_source",
    "test_mask_source",
    "mask_source",
    "auroc",
    "auprc",
    "acc",
    "sensitivity",
    "specificity",
    "model_dir",
    "feature_csv",
    "csv",
]

MASK_SOURCE_ORDER = ["gt", "gt_mild_perturb", "gt_moderate_perturb", "pred"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge mask-source evaluation CSVs into one summary table.")
    p.add_argument("--results_csv", type=str, nargs="+", required=True, help="One or more test_results.csv files")
    p.add_argument(
        "--perturb_stats_csv",
        type=str,
        nargs="*",
        default=None,
        help="Optional perturbation stats CSV(s) produced by extract_radiomics_2d.py",
    )
    p.add_argument("--out_csv", type=str, required=True, help="Output summary CSV path")
    p.add_argument("--metric", type=str, default="auroc", help="Metric column to pivot into matrix output")
    p.add_argument("--matrix_out_csv", type=str, default=None, help="Optional output CSV path for train×test metric matrix")
    return p.parse_args()


def _read_existing_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _summarize_perturb_stats(paths: Optional[List[str]]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["mask_source", "mean_dice_vs_gt", "mean_hd95_vs_gt"])

    rows = []
    for path in paths:
        df = _read_existing_csv(path)
        if df.empty:
            continue
        rows.append(
            {
                "mask_source": df.get("mask_source", pd.Series([None])).iloc[0],
                "mean_dice_vs_gt": float(df["dice_vs_gt"].mean()) if "dice_vs_gt" in df.columns else float("nan"),
                "mean_hd95_vs_gt": float(df["hd95_vs_gt"].mean()) if "hd95_vs_gt" in df.columns else float("nan"),
                "perturb_stats_csv": path,
            }
        )
    return pd.DataFrame(rows)


def _write_matrix(summary: pd.DataFrame, metric: str, out_csv: str) -> None:
    if metric not in summary.columns:
        raise ValueError(f"Metric column not found for matrix output: {metric}")

    required_cols = {"train_mask_source", "test_mask_source"}
    missing_cols = [col for col in required_cols if col not in summary.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for matrix output: {', '.join(missing_cols)}")

    matrix = summary.pivot_table(
        index="train_mask_source",
        columns="test_mask_source",
        values=metric,
        aggfunc="first",
    )
    matrix = matrix.reindex(index=MASK_SOURCE_ORDER, columns=MASK_SOURCE_ORDER)
    matrix.index.name = "train_mask_source"

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    matrix.to_csv(out_csv)
    print(f"Saved matrix: {out_csv}")



def main() -> None:
    args = parse_args()

    result_frames = []
    for path in args.results_csv:
        df = _read_existing_csv(path)
        df["results_csv"] = path
        result_frames.append(df)

    if not result_frames:
        raise ValueError("No result CSVs were loaded")

    summary = pd.concat(result_frames, ignore_index=True)

    if "test_mask_source" not in summary.columns:
        summary["test_mask_source"] = summary.get("mask_source")
    if "mask_source" not in summary.columns:
        summary["mask_source"] = summary.get("test_mask_source")
    if "train_mask_source" not in summary.columns:
        summary["train_mask_source"] = None

    for col in KEY_COLUMNS:
        if col not in summary.columns:
            summary[col] = None

    perturb_summary = _summarize_perturb_stats(args.perturb_stats_csv)
    if not perturb_summary.empty:
        summary = summary.merge(perturb_summary, on="mask_source", how="left")
    else:
        summary["mean_dice_vs_gt"] = float("nan")
        summary["mean_hd95_vs_gt"] = float("nan")
        summary["perturb_stats_csv"] = None

    summary = summary[
        KEY_COLUMNS
        + [
            "mean_dice_vs_gt",
            "mean_hd95_vs_gt",
            "perturb_stats_csv",
            "results_csv",
            "n_rows",
        ]
        + [c for c in summary.columns if c not in set(KEY_COLUMNS + ["mean_dice_vs_gt", "mean_hd95_vs_gt", "perturb_stats_csv", "results_csv", "n_rows"])]
    ]

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

    if args.matrix_out_csv:
        _write_matrix(summary, metric=args.metric, out_csv=args.matrix_out_csv)


if __name__ == "__main__":
    main()
