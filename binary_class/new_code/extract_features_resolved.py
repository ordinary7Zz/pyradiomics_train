"""Extract radiomics from resolved JSON (multi-root paths); reuses extract_base_radiomics internals."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PYR_ROOT = Path(__file__).resolve().parents[2]
if str(PYR_ROOT) not in sys.path:
    sys.path.insert(0, str(PYR_ROOT))

from binary_class.extract_base_radiomics import (  # noqa: E402
    _extract_one,
    _normalize_rel_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resolved_json", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument(
        "--params",
        type=str,
        default=str(PYR_ROOT / "radiomics_2d.yaml"),
    )
    p.add_argument("--mask_threshold", type=int, default=0)
    p.add_argument("--spacing_x", type=float, default=1.0)
    p.add_argument("--spacing_y", type=float, default=1.0)
    p.add_argument("--skip_fail", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from radiomics import featureextractor

    with open(args.resolved_json, "r", encoding="utf-8") as f:
        rows_in = json.load(f)
    if args.limit:
        rows_in = rows_in[: args.limit]

    extractor = featureextractor.RadiomicsFeatureExtractor(args.params)
    out_rows: List[Dict[str, object]] = []
    failures = 0
    spacing = (args.spacing_x, args.spacing_y)

    for idx, row in enumerate(rows_in):
        try:
            image_path = str(row["image_path"])
            mask_path = str(row["mask_path"])
            filename = _normalize_rel_path(str(row["filename"]))
            feats = _extract_one(
                extractor=extractor,
                image_path=image_path,
                mask_path=mask_path,
                filename=filename,
                mask_threshold=args.mask_threshold,
                spacing=spacing,
            )
            if "patient_id" in row:
                feats["patient_id"] = row["patient_id"]
            if "label" in row:
                feats["label"] = row["label"]
            out_rows.append(feats)
        except Exception as e:
            failures += 1
            msg = f"[{idx}] failed: {row.get('filename')} err={type(e).__name__}: {e}"
            if args.skip_fail:
                print(msg)
                continue
            raise RuntimeError(msg) from e

    df = pd.DataFrame(out_rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Extracted {len(df)} rows (failures={failures}) -> {out_path}")


if __name__ == "__main__":
    main()
