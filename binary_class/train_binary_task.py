import argparse
import os

import pandas as pd
from autogluon.tabular import TabularPredictor

from binary_class.binary_metrics import bootstrap_metric_cis, compute_binary_metrics, get_positive_proba


DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]


def _prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    drop_cols = [c for c in DROP_IF_PRESENT if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)

    label_values = sorted(df[label_col].unique().tolist())
    if any(v not in (0, 1) for v in label_values):
        raise ValueError(f"Expected binary labels 0/1 after filtering, got {label_values}")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one binary AutoGluon classifier for one task CSV.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="label")
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--test_csv", type=str, nargs="+", default=None)
    p.add_argument("--test_names", type=str, nargs="+", default=None)
    p.add_argument("--presets", type=str, default="best_quality")
    p.add_argument("--time_limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_metric", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ece_bins", type=int, default=10)
    p.add_argument("--ci_bootstrap_iters", type=int, default=1000)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--ci_seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    train_df = _prepare_df(train_df, args.label)

    predictor = TabularPredictor(
        label=args.label,
        path=args.save_dir,
        problem_type="binary",
        eval_metric=args.eval_metric,
        log_to_file=True,
        log_file_path="auto",
    )

    predictor.fit(
        train_data=train_df,
        presets=args.presets,
        time_limit=args.time_limit,
        ag_args_fit={"random_seed": args.seed},
    )

    lb = predictor.leaderboard(silent=True)
    lb_path = os.path.join(args.save_dir, "leaderboard.csv")
    lb.to_csv(lb_path, index=False)
    print(f"Saved leaderboard: {lb_path}")

    test_csvs = args.test_csv or []
    test_names = args.test_names
    if test_names is not None and len(test_names) != len(test_csvs):
        raise ValueError("--test_names count must match --test_csv count")
    if test_names is None:
        test_names = [os.path.splitext(os.path.basename(p))[0] for p in test_csvs]

    results = []
    ci_results = []
    for name, csv_path in zip(test_names, test_csvs):
        test_df = pd.read_csv(csv_path)
        test_df = _prepare_df(test_df, args.label)
        y_true = test_df[args.label].reset_index(drop=True)
        proba = predictor.predict_proba(test_df)
        y_score = get_positive_proba(proba).reset_index(drop=True)

        metrics = compute_binary_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=args.threshold,
            ece_bins=args.ece_bins,
        )
        ci_metrics = bootstrap_metric_cis(
            y_true=y_true,
            y_score=y_score,
            threshold=args.threshold,
            ece_bins=args.ece_bins,
            n_boot=args.ci_bootstrap_iters,
            ci_level=args.ci_level,
            seed=args.ci_seed,
        )

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        row = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(test_df.shape[0]),
            "n_neg": n_neg,
            "n_pos": n_pos,
        }
        row.update(metrics)
        results.append(row)

        ci_row = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(test_df.shape[0]),
            "n_neg": n_neg,
            "n_pos": n_pos,
            "threshold": float(args.threshold),
            "ci_level": float(args.ci_level),
            "ci_bootstrap_iters": int(args.ci_bootstrap_iters),
        }
        ci_row.update(ci_metrics)
        ci_results.append(ci_row)

        print(f"Test performance [{name}]: {metrics}")

    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.save_dir, "test_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved test results: {results_path}")

        ci_results_df = pd.DataFrame(ci_results)
        ci_results_path = os.path.join(args.save_dir, "test_results_ci.csv")
        ci_results_df.to_csv(ci_results_path, index=False)
        print(f"Saved test result CIs: {ci_results_path}")


if __name__ == "__main__":
    main()
