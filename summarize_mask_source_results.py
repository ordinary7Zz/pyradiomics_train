import argparse
import os
from typing import List, Optional

import pandas as pd


KEY_COLUMNS = [
    "train_dataset",
    "dataset",
    "task",
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


if __name__ == "__main__":
    main()
