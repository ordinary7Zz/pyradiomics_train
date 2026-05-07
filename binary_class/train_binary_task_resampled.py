import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from binary_class.binary_metrics import bootstrap_metric_cis, compute_binary_metrics, get_positive_proba


DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]
MODEL_SET_CHOICES = ["all", "tree_fast", "tree_full", "gbm_cat", "gbm_only"]


def _get_model_hyperparameters(model_set: str) -> Optional[Dict[str, Any]]:
    if model_set == "all":
        return None
    if model_set == "tree_fast":
        return {
            "GBM": {},
            "CAT": {},
        }
    if model_set == "tree_full":
        return {
            "GBM": {},
            "CAT": {},
            "XGB": {},
        }
    if model_set == "gbm_cat":
        return {
            "GBM": {},
            "CAT": {},
        }
    if model_set == "gbm_only":
        return {
            "GBM": {},
        }
    raise ValueError(f"Unsupported model_set: {model_set}")


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
    p = argparse.ArgumentParser(description="Train one binary AutoGluon classifier with optional resampling.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="label")
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--test_csv", type=str, nargs="+", default=None)
    p.add_argument("--test_names", type=str, nargs="+", default=None)
    p.add_argument("--presets", type=str, default="best_quality")
    p.add_argument("--model_set", type=str, default="all", choices=MODEL_SET_CHOICES)
    p.add_argument("--time_limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_metric", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ece_bins", type=int, default=10)
    p.add_argument("--ci_bootstrap_iters", type=int, default=1000)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--ci_seed", type=int, default=42)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument(
        "--resample_strategy",
        type=str,
        default="oversample",
        choices=["none", "oversample", "undersample"],
    )
    p.add_argument(
        "--resample_target",
        type=str,
        default="median",
        choices=["median", "max"],
    )
    p.add_argument("--save_resampled_csv", type=str, default=None)
    p.add_argument(
        "--target_class0_count",
        type=int,
        default=-1,
        help="target number of class-0 samples after resampling; use -1 to keep the original count",
    )
    p.add_argument(
        "--target_class1_count",
        type=int,
        default=-1,
        help="target number of class-1 samples after resampling; use -1 to keep the original count",
    )
    return p.parse_args()


def _class_counts(df: pd.DataFrame, label_col: str) -> pd.Series:
    counts = df[label_col].value_counts(dropna=False).sort_index()
    counts.index = counts.index.astype(int)
    return counts.astype(int)


def _print_class_counts(stage: str, counts: pd.Series) -> None:
    print(f"Class distribution [{stage}]:")
    for label_value, count in counts.items():
        print(f"  label={int(label_value)} count={int(count)}")


def _append_balance_rows(rows: List[Dict[str, object]], stage: str, counts: pd.Series) -> None:
    total = int(counts.sum())
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    pos_ratio = float(n_pos / total) if total > 0 else float("nan")
    rows.append(
        {
            "stage": stage,
            "n_rows": total,
            "n_neg": n_neg,
            "n_pos": n_pos,
            "pos_ratio": pos_ratio,
            "imbalance_ratio_neg_to_pos": float(n_neg / n_pos) if n_pos > 0 else float("inf"),
        }
    )


def _target_count_from_counts(counts: pd.Series, target_mode: str) -> int:
    values = counts.astype(int).tolist()
    if not values:
        raise ValueError("Cannot determine target count from empty class counts")
    if target_mode == "max":
        return int(max(values))
    if target_mode == "median":
        return int(pd.Series(values).median())
    raise ValueError(f"Unsupported resample target: {target_mode}")


def _sample_group(group: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    class_count = int(len(group))
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    return resample(
        group,
        replace=n_samples > class_count,
        n_samples=n_samples,
        random_state=seed,
    )


def _binary_resample_with_target_counts(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    target_class0_count: int,
    target_class1_count: int,
) -> pd.DataFrame:
    counts = _class_counts(df, label_col)
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    if n_neg == 0 or n_pos == 0:
        raise ValueError("Binary resampling requires both class 0 and class 1 to be present")

    neg_df = df[df[label_col] == 0]
    pos_df = df[df[label_col] == 1]

    target_neg = n_neg if target_class0_count == -1 else int(target_class0_count)
    target_pos = n_pos if target_class1_count == -1 else int(target_class1_count)

    if target_neg <= 0:
        raise ValueError(
            f"target_class0_count must be -1 or a positive integer, got {target_class0_count}"
        )
    if target_pos <= 0:
        raise ValueError(
            f"target_class1_count must be -1 or a positive integer, got {target_class1_count}"
        )

    sampled_neg = _sample_group(neg_df, n_samples=target_neg, seed=seed)
    sampled_pos = _sample_group(pos_df, n_samples=target_pos, seed=seed)

    resampled_df = pd.concat([sampled_neg, sampled_pos], axis=0, ignore_index=True)
    return resampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _resample_train_df(
    df: pd.DataFrame,
    label_col: str,
    strategy: str,
    target_mode: str,
    seed: int,
    target_class0_count: int = -1,
    target_class1_count: int = -1,
) -> pd.DataFrame:
    if target_class0_count != -1 or target_class1_count != -1:
        return _binary_resample_with_target_counts(
            df=df,
            label_col=label_col,
            seed=seed,
            target_class0_count=target_class0_count,
            target_class1_count=target_class1_count,
        )

    if strategy == "none":
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = _class_counts(df, label_col)
    target_count = _target_count_from_counts(counts, target_mode)
    sampled_parts: List[pd.DataFrame] = []

    for _, group in df.groupby(label_col, sort=True):
        class_count = int(len(group))
        if class_count == 0:
            continue

        if strategy == "oversample":
            n_samples = max(class_count, target_count)
        elif strategy == "undersample":
            n_samples = min(class_count, target_count)
        else:
            raise ValueError(f"Unsupported resample strategy: {strategy}")

        sampled_parts.append(_sample_group(group, n_samples=n_samples, seed=seed))

    if not sampled_parts:
        raise ValueError("Resampling produced no rows")

    resampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)
    return resampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _split_train_holdout(
    df: pd.DataFrame,
    label_col: str,
    holdout_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")

    y = df[label_col]
    class_counts = _class_counts(df, label_col)
    stratify = y if int(class_counts.min()) >= 2 else None

    if stratify is None:
        print("Warning: stratified holdout disabled because at least one class has fewer than 2 samples.")

    try:
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_frac,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        print(f"Warning: stratified holdout split failed ({exc}); falling back to unstratified split.")
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_frac,
            random_state=seed,
            stratify=None,
        )

    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def _save_class_balance_summary(save_dir: str, rows: List[Dict[str, object]]) -> str:
    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, "class_balance_summary.csv")
    summary_df.to_csv(out_path, index=False)
    return out_path


def _save_optional_csv(df: pd.DataFrame, csv_path: Optional[str]) -> Optional[str]:
    if not csv_path:
        return None
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def _evaluate_test_sets(
    predictor: TabularPredictor,
    test_csvs: List[str],
    test_names: List[str],
    label_col: str,
    threshold: float,
    ece_bins: int,
    ci_bootstrap_iters: int,
    ci_level: float,
    ci_seed: int,
    save_dir: str,
    balance_rows: List[Dict[str, object]],
) -> None:
    results = []
    ci_results = []

    for name, csv_path in zip(test_names, test_csvs):
        test_df = pd.read_csv(csv_path)
        test_df = _prepare_df(test_df, label_col)
        counts = _class_counts(test_df, label_col)
        _print_class_counts(f"test::{name}", counts)
        _append_balance_rows(balance_rows, f"test::{name}", counts)

        y_true = test_df[label_col].reset_index(drop=True)
        proba = predictor.predict_proba(test_df)
        y_score = get_positive_proba(proba).reset_index(drop=True)

        metrics = compute_binary_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            ece_bins=ece_bins,
        )
        ci_metrics = bootstrap_metric_cis(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            ece_bins=ece_bins,
            n_boot=ci_bootstrap_iters,
            ci_level=ci_level,
            seed=ci_seed,
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
            "threshold": float(threshold),
            "ci_level": float(ci_level),
            "ci_bootstrap_iters": int(ci_bootstrap_iters),
        }
        ci_row.update(ci_metrics)
        ci_results.append(ci_row)

        print(f"Test performance [{name}]: {metrics}")

    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(save_dir, "test_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved test results: {results_path}")

        ci_results_df = pd.DataFrame(ci_results)
        ci_results_path = os.path.join(save_dir, "test_results_ci.csv")
        ci_results_df.to_csv(ci_results_path, index=False)
        print(f"Saved test result CIs: {ci_results_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    train_df = _prepare_df(train_df, args.label)
    if train_df.empty:
        raise ValueError("Training data is empty after filtering")

    original_counts = _class_counts(train_df, args.label)
    if len(original_counts) < 2:
        raise ValueError("Training data must contain both binary classes after filtering")

    balance_rows: List[Dict[str, object]] = []
    _print_class_counts("train_original", original_counts)
    _append_balance_rows(balance_rows, "train_original", original_counts)

    if args.test_csv is not None:
        fit_input_df = train_df
        holdout_df = None
    else:
        fit_input_df, holdout_df = _split_train_holdout(
            train_df,
            label_col=args.label,
            holdout_frac=args.holdout_frac,
            seed=args.seed,
        )
        holdout_counts = _class_counts(holdout_df, args.label)
        _print_class_counts("holdout_original", holdout_counts)
        _append_balance_rows(balance_rows, "holdout_original", holdout_counts)

    fit_input_counts = _class_counts(fit_input_df, args.label)
    if len(fit_input_counts) < 2:
        raise ValueError("Training split must contain both binary classes after holdout split")
    _print_class_counts("fit_input_before_resample", fit_input_counts)
    _append_balance_rows(balance_rows, "fit_input_before_resample", fit_input_counts)

    if args.target_class0_count != -1 or args.target_class1_count != -1:
        print(
            "Using target-count balancing: "
            f"target_class0_count={args.target_class0_count}, "
            f"target_class1_count={args.target_class1_count}"
        )

    fit_train_df = _resample_train_df(
        fit_input_df,
        label_col=args.label,
        strategy=args.resample_strategy,
        target_mode=args.resample_target,
        seed=args.seed,
        target_class0_count=args.target_class0_count,
        target_class1_count=args.target_class1_count,
    )
    resampled_counts = _class_counts(fit_train_df, args.label)
    _print_class_counts("fit_train_after_resample", resampled_counts)
    _append_balance_rows(balance_rows, "fit_train_after_resample", resampled_counts)

    saved_resampled_path = _save_optional_csv(fit_train_df, args.save_resampled_csv)
    if saved_resampled_path is not None:
        print(f"Saved resampled training CSV: {saved_resampled_path}")

    predictor = TabularPredictor(
        label=args.label,
        path=args.save_dir,
        problem_type="binary",
        eval_metric=args.eval_metric,
        log_to_file=True,
        log_file_path="auto",
    )

    hyperparameters = _get_model_hyperparameters(args.model_set)
    print(
        "Training configuration: "
        f"presets={args.presets}, "
        f"model_set={args.model_set}, "
        f"time_limit={args.time_limit}"
    )

    fit_kwargs = dict(
        train_data=fit_train_df,
        presets=args.presets,
        time_limit=args.time_limit,
        ag_args_fit={"random_seed": args.seed},
    )
    if hyperparameters is not None:
        fit_kwargs["hyperparameters"] = hyperparameters
    if holdout_df is not None:
        fit_kwargs["tuning_data"] = holdout_df

    predictor.fit(**fit_kwargs)

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

    if test_csvs:
        _evaluate_test_sets(
            predictor=predictor,
            test_csvs=test_csvs,
            test_names=test_names,
            label_col=args.label,
            threshold=args.threshold,
            ece_bins=args.ece_bins,
            ci_bootstrap_iters=args.ci_bootstrap_iters,
            ci_level=args.ci_level,
            ci_seed=args.ci_seed,
            save_dir=args.save_dir,
            balance_rows=balance_rows,
        )

    balance_summary_path = _save_class_balance_summary(args.save_dir, balance_rows)
    print(f"Saved class balance summary: {balance_summary_path}")


if __name__ == "__main__":
    main()
