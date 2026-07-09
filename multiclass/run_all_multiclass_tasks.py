import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional, Set


ROOT = os.path.dirname(os.path.dirname(__file__))
THIS_DIR = os.path.dirname(__file__)
EXTRACT_SCRIPT = os.path.join(THIS_DIR, "..", "binary_class", "extract_base_radiomics.py")
BUILD_SCRIPT = os.path.join(THIS_DIR, "build_multiclass_task_csv.py")


def _load_tasks(label_json_path: str) -> List[str]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("label_json must be a list of dicts")

    tasks: Set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            if key != "filename":
                tasks.add(str(key))
    return sorted(tasks)


def _resolve_tasks(
    train_label_json: str, test_label_json: str, requested_tasks: Optional[List[str]]
) -> List[str]:
    train_tasks = set(_load_tasks(train_label_json))
    test_tasks = set(_load_tasks(test_label_json))

    if requested_tasks:
        missing_in_train = [t for t in requested_tasks if t not in train_tasks]
        missing_in_test = [t for t in requested_tasks if t not in test_tasks]
        if missing_in_train:
            raise ValueError(f"Tasks not found in train_label_json: {missing_in_train}")
        if missing_in_test:
            raise ValueError(f"Tasks not found in test_label_json: {missing_in_test}")
        return requested_tasks

    shared_tasks = sorted(train_tasks & test_tasks)
    if not shared_tasks:
        raise ValueError("No shared task keys found between train_label_json and test_label_json")
    return shared_tasks


def _run(cmd: List[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _extract_base_features(
    image_dir: str,
    mask_dir: str,
    label_json: str,
    output_csv: str,
    params: str,
    mask_threshold: int,
    spacing_x: float,
    spacing_y: float,
    skip_fail: bool,
) -> None:
    cmd = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--image_dir", image_dir,
        "--mask_dir", mask_dir,
        "--label_json", label_json,
        "--output_csv", output_csv,
        "--params", params,
        "--mask_threshold", str(mask_threshold),
        "--spacing_x", str(spacing_x),
        "--spacing_y", str(spacing_y),
    ]
    if skip_fail:
        cmd.append("--skip_fail")
    _run(cmd)


def _build_task_csv(
    base_features_csv: str, label_json: str, task: str, output_csv: str
) -> None:
    _run([
        sys.executable,
        BUILD_SCRIPT,
        "--base_features_csv", base_features_csv,
        "--label_json", label_json,
        "--task", task,
        "--output_csv", output_csv,
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all multiclass tasks with train/test JSON inputs."
    )
    p.add_argument("--train_image_dir", type=str, required=True)
    p.add_argument("--train_mask_dir", type=str, required=True)
    p.add_argument("--train_label_json", type=str, required=True)
    p.add_argument("--test_image_dir", type=str, required=True)
    p.add_argument("--test_mask_dir", type=str, required=True)
    p.add_argument("--test_label_json", type=str, required=True)
    p.add_argument("--work_dir", type=str, default=os.path.join(THIS_DIR, "outputs"))
    p.add_argument("--params", type=str, default=os.path.join(ROOT, "radiomics_2d.yaml"))
    p.add_argument("--mask_threshold", type=int, default=0)
    p.add_argument("--spacing_x", type=float, default=1.0)
    p.add_argument("--spacing_y", type=float, default=1.0)
    p.add_argument("--skip_fail", action="store_true")
    p.add_argument("--presets", type=str, default="best_quality")
    p.add_argument("--time_limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_metric", type=str, default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--ece_bins", type=int, default=10)
    p.add_argument("--ci_bootstrap_iters", type=int, default=1000)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--ci_seed", type=int, default=42)
    # 训练模式选择
    p.add_argument(
        "--training_mode",
        type=str,
        default="resampled",
        choices=["basic", "resampled"],
        help="basic: train_multiclass_task, resampled: train_multiclass_task_resampled",
    )
    # Resampled 模式参数
    p.add_argument("--resample_strategy", type=str, default="oversample")
    p.add_argument("--resample_target", type=str, default="median")
    p.add_argument("--model_set", type=str, default="all")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = os.path.abspath(args.work_dir)
    base_features_dir = os.path.join(base_dir, "base_features")
    task_csv_dir = os.path.join(base_dir, "task_csvs")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(base_features_dir, exist_ok=True)
    os.makedirs(task_csv_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    train_base_features_csv = os.path.join(base_features_dir, "train_base_features.csv")
    test_base_features_csv = os.path.join(base_features_dir, "test_base_features.csv")

    _extract_base_features(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        label_json=args.train_label_json,
        output_csv=train_base_features_csv,
        params=args.params,
        mask_threshold=args.mask_threshold,
        spacing_x=args.spacing_x,
        spacing_y=args.spacing_y,
        skip_fail=args.skip_fail,
    )
    _extract_base_features(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        label_json=args.test_label_json,
        output_csv=test_base_features_csv,
        params=args.params,
        mask_threshold=args.mask_threshold,
        spacing_x=args.spacing_x,
        spacing_y=args.spacing_y,
        skip_fail=args.skip_fail,
    )

    tasks = _resolve_tasks(args.train_label_json, args.test_label_json, args.tasks)
    summary_rows = []

    for task in tasks:
        train_task_csv = os.path.join(task_csv_dir, f"train_{task}.csv")
        test_task_csv = os.path.join(task_csv_dir, f"test_{task}.csv")
        model_dir = os.path.join(models_dir, task)

        _build_task_csv(train_base_features_csv, args.train_label_json, task, train_task_csv)
        _build_task_csv(test_base_features_csv, args.test_label_json, task, test_task_csv)

        if args.training_mode == "resampled":
            train_module = "multiclass.train_multiclass_task_resampled"
        else:
            train_module = "multiclass.train_multiclass_task"

        train_cmd = [
            sys.executable,
            "-m",
            train_module,
            "--train_csv", train_task_csv,
            "--save_dir", model_dir,
            "--task_name", task,
            "--test_csv", test_task_csv,
            "--test_names", f"test_{task}",
            "--presets", args.presets,
            "--seed", str(args.seed),
            "--ece_bins", str(args.ece_bins),
            "--ci_bootstrap_iters", str(args.ci_bootstrap_iters),
            "--ci_level", str(args.ci_level),
            "--ci_seed", str(args.ci_seed),
        ]

        if args.time_limit is not None:
            train_cmd.extend(["--time_limit", str(args.time_limit)])
        if args.eval_metric is not None:
            train_cmd.extend(["--eval_metric", args.eval_metric])
        if args.training_mode == "resampled":
            train_cmd.extend([
                "--resample_strategy", args.resample_strategy,
                "--resample_target", args.resample_target,
                "--model_set", args.model_set,
            ])

        _run(train_cmd)

        summary_rows.append({
            "task": task,
            "train_task_csv": train_task_csv,
            "test_task_csv": test_task_csv,
            "model_dir": model_dir,
            "training_mode": args.training_mode,
        })

    import pandas as pd

    summary_path = os.path.join(reports_dir, "run_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
