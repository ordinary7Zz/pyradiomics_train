import argparse
import os

import pandas as pd
from autogluon.tabular import TabularPredictor

from multiclass.multiclass_metrics import (
    TIRADS_CLASSES,
    bootstrap_multiclass_cis,
    compute_multiclass_metrics,
)


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
    invalid = set(label_values) - set(TIRADS_CLASSES)
    if invalid:
        raise ValueError(
            f"Unexpected labels {invalid}. Expected TIRADS {TIRADS_CLASSES}, got {label_values}"
        )

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one multiclass AutoGluon classifier for TIRADS.")
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
        problem_type="multiclass",
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

        # predict_proba 返回 (n_samples, n_classes) DataFrame，columns 为类别标签
        proba_df = predictor.predict_proba(test_df)
        y_pred = predictor.predict(test_df).reset_index(drop=True)

        metrics = compute_multiclass_metrics(
            y_true=y_true,
            y_score_df=proba_df,
            y_pred=y_pred,
            classes=TIRADS_CLASSES,
            ece_bins=args.ece_bins,
        )
        ci_metrics = bootstrap_multiclass_cis(
            y_true=y_true,
            y_score_df=proba_df,
            classes=TIRADS_CLASSES,
            ece_bins=args.ece_bins,
            n_boot=args.ci_bootstrap_iters,
            ci_level=args.ci_level,
            seed=args.ci_seed,
        )

        # 逐类统计
        class_counts = y_true.value_counts().sort_index()
        row = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(test_df.shape[0]),
        }
        for cls in TIRADS_CLASSES:
            row[f"n_TIRADS{cls}"] = int(class_counts.get(cls, 0))
        row.update(metrics)
        results.append(row)

        ci_row = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(test_df.shape[0]),
            "ci_level": float(args.ci_level),
            "ci_bootstrap_iters": int(args.ci_bootstrap_iters),
        }
        for cls in TIRADS_CLASSES:
            ci_row[f"n_TIRADS{cls}"] = int(class_counts.get(cls, 0))
        ci_row.update(ci_metrics)
        ci_results.append(ci_row)

        # 打印关键指标
        key_metrics = {k: metrics[k] for k in [
            "accuracy", "balanced_accuracy", "kappa", "qwk",
            "macro_f1", "weighted_f1", "macro_auroc", "ece",
        ] if k in metrics}
        print(f"Test performance [{name}]: {key_metrics}")

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
