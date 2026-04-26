import argparse
import os
import shutil
import time

import pandas as pd
from autogluon.tabular import TabularPredictor

def parse_args():
    p = argparse.ArgumentParser(description="Train AutoGluon Tabular classifier on radiomics features.")
    p.add_argument("--train_csv", type=str, required=True, help="features csv from extract_radiomics_2d.py")
    p.add_argument(
        "--test_csv",
        type=str,
        nargs="+",
        default=None,
        help="optional test features csv(s). Example: --test_csv a.csv b.csv c.csv",
    )
    p.add_argument(
        "--test_names",
        type=str,
        nargs="+",
        default=None,
        help="optional dataset name(s), 1-1 aligned with --test_csv order",
    )
    p.add_argument("--label", type=str, default="label", help="label column name")
    p.add_argument("--save_dir", type=str, required=True, help="output directory for AutoGluon")

    p.add_argument("--presets", type=str, default="best_quality")
    p.add_argument("--time_limit", type=int, default=None, help="seconds")
    p.add_argument("--holdout_frac", type=float, default=0.2, help="used when test_csv is not provided")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_metric", type=str, default=None, help="e.g. roc_auc, accuracy, log_loss")

    return p.parse_args()


def _infer_problem_type(y: pd.Series) -> str:
    uniq = pd.unique(y.dropna())
    if len(uniq) <= 2:
        return "binary"
    return "multiclass"


def _prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    # Drop non-feature columns if present
    drop_cols = [c for c in ["image_path", "mask_path", "filename"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Remove unlabeled rows
    df = df[df[label_col] != -1].copy()

    # Ensure label is int for classification
    df[label_col] = df[label_col].astype(int)

    return df


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    train_df = _prepare_df(train_df, args.label)

    problem_type = _infer_problem_type(train_df[args.label])

    predictor = TabularPredictor(
        label=args.label,
        path=args.save_dir,
        problem_type=problem_type,
        eval_metric=args.eval_metric,
        log_to_file=True,
        log_file_path="auto"
    )

    fit_common_kwargs = dict(
        presets=args.presets,
        time_limit=args.time_limit,
        ag_args_fit={"random_seed": args.seed},
    )

    test_csvs = args.test_csv
    if test_csvs is not None:
        if not isinstance(test_csvs, list) or len(test_csvs) == 0:
            raise ValueError("--test_csv was provided but no CSV paths were parsed")

        test_names = args.test_names
        if test_names is not None and len(test_names) != len(test_csvs):
            raise ValueError(
                f"--test_names count ({len(test_names)}) must match --test_csv count ({len(test_csvs)})"
            )
        if test_names is None:
            test_names = [os.path.splitext(os.path.basename(p))[0] for p in test_csvs]

        predictor.fit(
            train_data=train_df,
            **fit_common_kwargs,
        )

        for name, csv_path in zip(test_names, test_csvs):
            test_df = pd.read_csv(csv_path)
            test_df = _prepare_df(test_df, args.label)
            perf = predictor.evaluate(test_df)
            print(f"Test performance [{name}]: {perf}")

    lb = predictor.leaderboard(silent=True)
    lb_path = os.path.join(args.save_dir, "leaderboard.csv")
    lb.to_csv(lb_path, index=False)
    print(f"Saved leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
