import argparse
import json
import os
from typing import Dict, List

import pandas as pd


META_COLUMNS = ["filename", "image_path", "mask_path"]


def _normalize_rel_path(path: str) -> str:
    return os.path.normpath(path.replace("\\", "/").lstrip("/\\"))


def _load_label_df(label_json_path: str) -> pd.DataFrame:
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("label_json must be a list of dicts")

    rows: List[Dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue
        row = dict(item)
        row["filename"] = _normalize_rel_path(str(filename))
        rows.append(row)

    if not rows:
        raise ValueError("No valid rows found in label_json")
    return pd.DataFrame(rows)


def _list_tasks(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "filename"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one binary task CSV from base radiomics features and a label json.")
    p.add_argument("--base_features_csv", type=str, required=True)
    p.add_argument("--label_json", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_df = pd.read_csv(args.base_features_csv)
    if "filename" not in base_df.columns:
        raise ValueError("base_features_csv must contain filename column")
    base_df["filename"] = base_df["filename"].map(_normalize_rel_path)

    label_df = _load_label_df(args.label_json)
    available_tasks = _list_tasks(label_df)
    if args.task not in available_tasks:
        raise ValueError(f"task='{args.task}' not found in label_json. Available tasks: {available_tasks}")

    task_df = label_df[["filename", args.task]].copy()
    task_df = task_df.rename(columns={args.task: "label"})
    task_df["label"] = pd.to_numeric(task_df["label"], errors="coerce")
    task_df = task_df.dropna(subset=["label"])
    task_df["label"] = task_df["label"].astype(int)
    task_df = task_df[task_df["label"] != -1].copy()

    merged = base_df.merge(task_df, on="filename", how="inner")
    if merged.empty:
        raise ValueError(f"No matched labeled rows found for task={args.task}")

    label_values = sorted(merged["label"].unique().tolist())
    if label_values != [0, 1] and label_values != [0] and label_values != [1]:
        raise ValueError(f"Task {args.task} is not binary after filtering. labels={label_values}")

    feature_cols = [c for c in merged.columns if c not in META_COLUMNS + ["label"]]
    ordered_cols = META_COLUMNS + feature_cols + ["label"]
    ordered_cols = [c for c in ordered_cols if c in merged.columns]
    merged = merged[ordered_cols]

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    pos = int((merged["label"] == 1).sum())
    neg = int((merged["label"] == 0).sum())
    print(f"Saved {args.task}: rows={len(merged)} neg={neg} pos={pos} -> {args.output_csv}")


if __name__ == "__main__":
    main()
