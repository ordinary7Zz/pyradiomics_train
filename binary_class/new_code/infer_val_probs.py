from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

PYR_ROOT = Path(__file__).resolve().parents[2]
if str(PYR_ROOT) not in sys.path:
    sys.path.insert(0, str(PYR_ROOT))

from binary_class.binary_metrics import get_positive_proba  # noqa: E402
from binary_class.train_binary_task_resampled import DROP_IF_PRESENT, _prepare_df  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--val_task_csv", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--label", type=str, default="label")
    p.add_argument("--resolved_json", type=str, required=True, help="for patient_id")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.resolved_json, "r", encoding="utf-8") as f:
        resolved = {str(r["filename"]): r for r in json.load(f)}

    raw = pd.read_csv(args.val_task_csv)
    filenames = raw["filename"].astype(str).tolist()
    labels = raw[args.label].astype(int).tolist()
    feature_df = _prepare_df(raw.copy(), args.label)

    predictor = TabularPredictor.load(args.model_dir)
    proba = predictor.predict_proba(feature_df)
    scores = get_positive_proba(proba).reset_index(drop=True)

    rows = []
    for i, fn in enumerate(filenames):
        meta = resolved.get(fn, {})
        rows.append(
            {
                "filename": fn,
                "patient_id": str(meta.get("patient_id", fn)),
                "label": int(labels[i]),
                "prob": float(scores.iloc[i]),
            }
        )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
