import argparse
import json
import os
import subprocess
import sys
from typing import List


ROOT = os.path.dirname(os.path.dirname(__file__))
THIS_DIR = os.path.dirname(__file__)
EXTRACT_SCRIPT = os.path.join(THIS_DIR, "extract_base_radiomics.py")
BUILD_SCRIPT = os.path.join(THIS_DIR, "build_binary_task_csv.py")
TRAIN_SCRIPT = os.path.join(THIS_DIR, "train_binary_task.py")


def _load_tasks(label_json_path: str) -> List[str]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("label_json must be a list of dicts")

    tasks = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            if key != "filename":
                tasks.add(str(key))
    return sorted(tasks)


def _run(cmd: List[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all binary tasks from one label json.")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, required=True)
    p.add_argument("--label_json", type=str, required=True)
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

    base_features_csv = os.path.join(base_features_dir, "base_features.csv")
    extract_cmd = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--image_dir", args.image_dir,
        "--mask_dir", args.mask_dir,
        "--label_json", args.label_json,
        "--output_csv", base_features_csv,
        "--params", args.params,
        "--mask_threshold", str(args.mask_threshold),
        "--spacing_x", str(args.spacing_x),
        "--spacing_y", str(args.spacing_y),
    ]
    if args.skip_fail:
        extract_cmd.append("--skip_fail")
    _run(extract_cmd)

    tasks = args.tasks or _load_tasks(args.label_json)
    summary_rows = []

    for task in tasks:
        task_csv = os.path.join(task_csv_dir, f"{task}.csv")
        model_dir = os.path.join(models_dir, task)

        _run([
            sys.executable,
            BUILD_SCRIPT,
            "--base_features_csv", base_features_csv,
            "--label_json", args.label_json,
            "--task", task,
            "--output_csv", task_csv,
        ])

        train_cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--train_csv", task_csv,
            "--save_dir", model_dir,
            "--task_name", task,
            "--presets", args.presets,
            "--seed", str(args.seed),
        ]
        if args.time_limit is not None:
            train_cmd.extend(["--time_limit", str(args.time_limit)])
        if args.eval_metric is not None:
            train_cmd.extend(["--eval_metric", args.eval_metric])
        _run(train_cmd)

        summary_rows.append({
            "task": task,
            "task_csv": task_csv,
            "model_dir": model_dir,
        })

    import pandas as pd

    summary_path = os.path.join(reports_dir, "run_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
