#!/usr/bin/env python3
"""B9 AutoGluon patient-level 5-fold CV (orchestration only)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PYR_ROOT = Path(__file__).resolve().parents[2]
DINO_ROOT = PYR_ROOT.parent
if str(PYR_ROOT) not in sys.path:
    sys.path.insert(0, str(PYR_ROOT))

from binary_class.new_code.metrics_oof import aggregate_patient_mean, binary_metrics, cluster_bootstrap_ci
from binary_class.new_code.packed_io import build_sample_records, load_image_roots, write_resolved_samples
from binary_class.new_code.patient_cv import split_samples_by_patient_cv
from binary_class.new_code.task_specs import get_task

NEW_CODE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["lnm", "ftcptc"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--packed_root", type=str, default=None)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--cv_seed", type=int, default=42)
    p.add_argument("--max_folds", type=int, default=None)
    p.add_argument("--limit_samples", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--reuse_feature_cache", action="store_true", help="extract all pool once, slice by fold")
    p.add_argument("--time_limit", type=int, default=3600)
    p.add_argument("--model_set", type=str, default="tree_full")
    p.add_argument("--presets", type=str, default="best_quality")
    p.add_argument("--ag_seed", type=int, default=42)
    p.add_argument("--skip_fail_features", action="store_true")
    return p.parse_args()


def _run(cmd: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _export_fold_splits(out_dir: Path, folds_meta: List[Dict[str, Any]]) -> None:
    with open(out_dir / "fold_splits.json", "w", encoding="utf-8") as f:
        json.dump({"folds": folds_meta, "n_splits": len(folds_meta)}, f, indent=2)


def main() -> None:
    args = parse_args()
    spec = get_task(args.task)
    packed = Path(args.packed_root or spec.packed_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = packed / spec.train_manifest
    roots = load_image_roots(packed)
    records, stats = build_sample_records(spec, manifest, roots)
    if args.limit_samples and len(records) > args.limit_samples:
        records = records[: args.limit_samples]
    print(f"Using {len(records)} samples; resolve stats: {stats}")

    folds = split_samples_by_patient_cv(records, n_splits=args.n_splits, seed=args.cv_seed)
    n_run = args.max_folds if args.max_folds is not None else len(folds)

    ag_config = {
        "time_limit": args.time_limit,
        "model_set": args.model_set,
        "presets": args.presets,
        "resample_strategy": spec.resample_strategy,
        "target_class0_count": spec.target_class0_count,
        "target_class1_count": spec.target_class1_count,
        "seed": args.ag_seed,
        "eval_metric": "roc_auc",
    }
    with open(out_dir / "ag_train_config.json", "w", encoding="utf-8") as f:
        json.dump(ag_config, f, indent=2)

    cache_csv = out_dir / "cache" / "all_features.csv"
    all_resolved_path = out_dir / "cache" / "all_resolved.json"
    if args.reuse_feature_cache and not args.dry_run:
        all_rows = [s.to_resolved_row(spec.label_key) for s in records]
        write_resolved_samples(all_resolved_path, all_rows)
        if not cache_csv.is_file():
            _run(
                [
                    sys.executable,
                    str(NEW_CODE / "extract_features_resolved.py"),
                    "--resolved_json",
                    str(all_resolved_path),
                    "--output_csv",
                    str(cache_csv),
                ]
                + (["--skip_fail"] if args.skip_fail_features else []),
                PYR_ROOT,
            )

    folds_meta: List[Dict[str, Any]] = []
    all_oof: List[Dict[str, Any]] = []
    fold_summaries = []

    for fold_idx in range(n_run):
        train_samples, val_samples = folds[fold_idx]
        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_rows = [s.to_resolved_row(spec.label_key) for s in train_samples]
        val_rows = [s.to_resolved_row(spec.label_key) for s in val_samples]
        train_res = fold_dir / "train_resolved.json"
        val_res = fold_dir / "val_resolved.json"
        write_resolved_samples(train_res, train_rows)
        write_resolved_samples(val_res, val_rows)

        folds_meta.append(
            {
                "fold": fold_idx,
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "train_patient_ids": sorted({s.patient_id for s in train_samples}),
                "val_patient_ids": sorted({s.patient_id for s in val_samples}),
            }
        )

        train_feat = fold_dir / "train_features.csv"
        val_feat = fold_dir / "val_features.csv"
        train_task = fold_dir / "train_task.csv"
        val_task = fold_dir / "val_task.csv"
        ag_dir = fold_dir / "ag_model"

        if args.dry_run:
            continue

        if args.reuse_feature_cache and cache_csv.is_file():
            import pandas as pd

            feat = pd.read_csv(cache_csv)
            train_fn = {r["filename"] for r in train_rows}
            val_fn = {r["filename"] for r in val_rows}
            feat[feat["filename"].isin(train_fn)].to_csv(train_feat, index=False)
            feat[feat["filename"].isin(val_fn)].to_csv(val_feat, index=False)
        else:
            _run(
                [
                    sys.executable,
                    str(NEW_CODE / "extract_features_resolved.py"),
                    "--resolved_json",
                    str(train_res),
                    "--output_csv",
                    str(train_feat),
                ]
                + (["--skip_fail"] if args.skip_fail_features else []),
                PYR_ROOT,
            )
            _run(
                [
                    sys.executable,
                    str(NEW_CODE / "extract_features_resolved.py"),
                    "--resolved_json",
                    str(val_res),
                    "--output_csv",
                    str(val_feat),
                ]
                + (["--skip_fail"] if args.skip_fail_features else []),
                PYR_ROOT,
            )

        for res, feat, task in ((train_res, train_feat, train_task), (val_res, val_feat, val_task)):
            _run(
                [
                    sys.executable,
                    str(NEW_CODE / "build_task_csv.py"),
                    "--features_csv",
                    str(feat),
                    "--resolved_json",
                    str(res),
                    "--label_key",
                    spec.label_key,
                    "--output_csv",
                    str(task),
                ],
                PYR_ROOT,
            )

        if not args.skip_train:
            cmd = [
                sys.executable,
                "-m",
                "binary_class.train_binary_task_resampled",
                "--train_csv",
                str(train_task),
                "--test_csv",
                str(val_task),
                "--test_names",
                "val_fold",
                "--save_dir",
                str(ag_dir),
                "--label",
                "label",
                "--eval_metric",
                "roc_auc",
                "--model_set",
                args.model_set,
                "--presets",
                args.presets,
                "--time_limit",
                str(args.time_limit),
                "--resample_strategy",
                spec.resample_strategy,
                "--target_class0_count",
                str(spec.target_class0_count),
                "--target_class1_count",
                str(spec.target_class1_count),
                "--seed",
                str(args.ag_seed),
                "--ci_seed",
                str(args.cv_seed),
            ]
            _run(cmd, PYR_ROOT)

        val_probs = fold_dir / "val_probs.json"
        _run(
            [
                sys.executable,
                str(NEW_CODE / "infer_val_probs.py"),
                "--model_dir",
                str(ag_dir),
                "--val_task_csv",
                str(val_task),
                "--resolved_json",
                str(val_res),
                "--output_json",
                str(val_probs),
            ],
            PYR_ROOT,
        )

        with open(val_probs, "r", encoding="utf-8") as f:
            preds = json.load(f)
        for row in preds:
            row["fold"] = fold_idx
        all_oof.extend(preds)

        yt, ys, pids = aggregate_patient_mean(
            [r["patient_id"] for r in preds],
            [r["label"] for r in preds],
            [r["prob"] for r in preds],
        )
        fold_summaries.append(
            {
                "fold": fold_idx,
                "image_level": binary_metrics(
                    np.array([r["label"] for r in preds]),
                    np.array([r["prob"] for r in preds]),
                ),
                "patient_level": binary_metrics(yt, ys),
                "patient_ci": cluster_bootstrap_ci(yt, ys, pids, seed=args.cv_seed),
            }
        )

    _export_fold_splits(out_dir, folds_meta)

    if args.dry_run:
        print(f"Dry run done -> {out_dir}")
        return

    with open(out_dir / "oof_predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_oof, f, ensure_ascii=False, indent=2)

    if all_oof:
        yt, ys, pids = aggregate_patient_mean(
            [r["patient_id"] for r in all_oof],
            [r["label"] for r in all_oof],
            [r["prob"] for r in all_oof],
        )
        summary = {
            "task": spec.name,
            "label_key": spec.label_key,
            "n_splits": args.n_splits,
            "cv_seed": args.cv_seed,
            "resolve_stats": stats,
            "folds": fold_summaries,
            "oof_patient_level": binary_metrics(yt, ys),
            "oof_patient_ci": cluster_bootstrap_ci(yt, ys, pids, seed=args.cv_seed),
        }
        with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary["oof_patient_level"], indent=2))


if __name__ == "__main__":
    main()
