"""Merge feature CSV with resolved labels -> train_binary_task_resampled format (column `label`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Not radiomics; kept in feature CSV only for traceability — strip before task training.
NON_FEATURE_COLUMNS = ["patient_id", "image_path", "mask_path"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", type=str, required=True)
    p.add_argument("--resolved_json", type=str, required=True)
    p.add_argument("--label_key", type=str, default="label")
    p.add_argument("--output_csv", type=str, required=True)
    return p.parse_args()


def _norm_filename(s: str) -> str:
    import os

    return os.path.normpath(str(s).replace("\\", "/").lstrip("/\\"))


def main() -> None:
    args = parse_args()
    feat = pd.read_csv(args.features_csv)
    feat["filename"] = feat["filename"].map(_norm_filename)
    # Features from extract_features_resolved may already carry `label` / label_key.
    for col in (args.label_key, "label"):
        if col in feat.columns:
            feat = feat.drop(columns=[col])

    with open(args.resolved_json, "r", encoding="utf-8") as f:
        resolved = json.load(f)
    labels = []
    for row in resolved:
        labels.append(
            {
                "filename": _norm_filename(row["filename"]),
                "label": int(row.get(args.label_key, row.get("label"))),
            }
        )
    lab = pd.DataFrame(labels).drop_duplicates(subset=["filename"])

    merged = feat.merge(lab, on="filename", how="inner")
    if merged.empty:
        raise ValueError("No rows after merging features and labels")

    drop_meta = [c for c in NON_FEATURE_COLUMNS if c in merged.columns]
    if drop_meta:
        merged = merged.drop(columns=drop_meta)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {len(merged)} rows -> {out}")


if __name__ == "__main__":
    main()
