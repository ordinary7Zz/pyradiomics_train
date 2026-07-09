import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from multiclass.multiclass_metrics import (
    TIRADS_CLASSES,
    bootstrap_multiclass_cis,
    compute_multiclass_metrics,
)


DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]
MODEL_SET_CHOICES = ["all", "tree_fast", "tree_full", "gbm_cat", "gbm_only"]
RESAMPLE_STRATEGY_CHOICES = ["none", "oversample", "undersample"]
RESAMPLE_TARGET_CHOICES = ["median", "max"]


def _get_model_hyperparameters(model_set: str) -> Optional[Dict[str, Any]]:
    if model_set == "all":
        return None
    if model_set == "tree_fast":
        return {"GBM": {}, "CAT": {}}
    if model_set == "tree_full":
        return {"GBM": {}, "CAT": {}, "XGB": {}}
    if model_set == "gbm_cat":
        return {"GBM": {}, "CAT": {}}
    if model_set == "gbm_only":
        return {"GBM": {}}
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
    invalid = set(label_values) - set(TIRADS_CLASSES)
    if invalid:
        raise ValueError(
            f"Unexpected labels {invalid}. Expected TIRADS {TIRADS_CLASSES}, got {label_values}"
        )

    return df


def _class_counts(df: pd.DataFrame, label_col: str) -> pd.Series:
    counts = df[label_col].value_counts(dropna=False).sort_index()
    counts.index = counts.index.astype(int)
    return counts.astype(int)


def _print_class_counts(stage: str, counts: pd.Series) -> None:
    print(f"Class distribution [{stage}]:")
    for label_value, count in counts.items():
        print(f"  TIRADS {int(label_value)}: {int(count)}")


def _append_balance_rows(rows: List[Dict[str, object]], stage: str, counts: pd.Series) -> None:
    row: Dict[str, object] = {"stage": stage, "n_rows": int(counts.sum())}
    for cls in TIRADS_CLASSES:
        row[f"n_TIRADS{cls}"] = int(counts.get(cls, 0))
    rows.append(row)


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


def _multiclass_resample(
    df: pd.DataFrame,
    label_col: str,
    strategy: str,
    target_mode: str,
    seed: int,
) -> pd.DataFrame:
    """多分类重采样：逐类 over/under sample 到目标数量。"""
    if strategy == "none":
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = _class_counts(df, label_col)
    target_count = _target_count_from_counts(counts, target_mode)

    sampled_parts: List[pd.DataFrame] = []
    for cls in sorted(df[label_col].unique()):
        group = df[df[label_col] == cls]
        class_count = int(len(group))
        if class_count == 0:
            continue

        if strategy == "oversample":
            n_samples = max(class_count, target_count)
        elif strategy == "undersample":
            n_samples = min(class_count, target_count)
        else:
            raise ValueError(f"Unsupported resample strategy: {strategy}")

        sampled_parts.append(_sample_group(group, n_samples=n_samples, seed=seed + cls))

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


def _evaluate_test_sets(
    predictor: TabularPredictor,
    test_csvs: List[str],
    test_names: List[str],
    label_col: str,
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
        proba_df = predictor.predict_proba(test_df)
        y_pred = predictor.predict(test_df).reset_index(drop=True)

        metrics = compute_multiclass_metrics(
            y_true=y_true,
            y_score_df=proba_df,
            y_pred=y_pred,
            classes=TIRADS_CLASSES,
            ece_bins=ece_bins,
        )
        ci_metrics = bootstrap_multiclass_cis(
            y_true=y_true,
            y_score_df=proba_df,
            classes=TIRADS_CLASSES,
            ece_bins=ece_bins,
            n_boot=ci_bootstrap_iters,
            ci_level=ci_level,
            seed=ci_seed,
        )

        class_counts = y_true.value_counts().sort_index()
        row: Dict[str, object] = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(test_df.shape[0]),
        }
        for cls in TIRADS_CLASSES:
            row[f"n_TIRADS{cls}"] = int(class_counts.get(cls, 0))
        row.update(metrics)
        results.append(row)

        ci_row: Dict[str, object] = {
            "dataset": name,
            "csv": csv_path,
            "n_rows": int(test_df.shape[0]),
            "ci_level": float(ci_level),
            "ci_bootstrap_iters": int(ci_bootstrap_iters),
        }
        for cls in TIRADS_CLASSES:
            ci_row[f"n_TIRADS{cls}"] = int(class_counts.get(cls, 0))
        ci_row.update(ci_metrics)
        ci_results.append(ci_row)

        key_metrics = {k: metrics[k] for k in [
            "accuracy", "balanced_accuracy", "kappa", "qwk",
            "macro_f1", "weighted_f1", "macro_auroc", "ece",
        ] if k in metrics}
        print(f"Test performance [{name}]: {key_metrics}")

    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(save_dir, "test_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved test results: {results_path}")

        ci_results_df = pd.DataFrame(ci_results)
        ci_results_path = os.path.join(save_dir, "test_results_ci.csv")
        ci_results_df.to_csv(ci_results_path, index=False)
        print(f"Saved test result CIs: {ci_results_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one multiclass AutoGluon classifier with optional resampling."
    )
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
    p.add_argument("--ece_bins", type=int, default=10)
    p.add_argument("--ci_bootstrap_iters", type=int, default=1000)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--ci_seed", type=int, default=42)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument(
        "--resample_strategy",
        type=str,
        default="oversample",
        choices=RESAMPLE_STRATEGY_CHOICES,
    )
    p.add_argument(
        "--resample_target",
        type=str,
        default="median",
        choices=RESAMPLE_TARGET_CHOICES,
    )
    p.add_argument("--save_resampled_csv", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    train_df = _prepare_df(train_df, args.label)
    if train_df.empty:
        raise ValueError("Training data is empty after filtering")

    original_counts = _class_counts(train_df, args.label)
    if len(original_counts) < 2:
        raise ValueError("Training data must contain at least 2 classes after filtering")

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
    _print_class_counts("fit_input_before_resample", fit_input_counts)
    _append_balance_rows(balance_rows, "fit_input_before_resample", fit_input_counts)

    fit_train_df = _multiclass_resample(
        fit_input_df,
        label_col=args.label,
        strategy=args.resample_strategy,
        target_mode=args.resample_target,
        seed=args.seed,
    )
    resampled_counts = _class_counts(fit_train_df, args.label)
    _print_class_counts("fit_train_after_resample", resampled_counts)
    _append_balance_rows(balance_rows, "fit_train_after_resample", resampled_counts)

    if args.save_resampled_csv:
        parent = os.path.dirname(args.save_resampled_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fit_train_df.to_csv(args.save_resampled_csv, index=False)
        print(f"Saved resampled training CSV: {args.save_resampled_csv}")

    predictor = TabularPredictor(
        label=args.label,
        path=args.save_dir,
        problem_type="multiclass",
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
