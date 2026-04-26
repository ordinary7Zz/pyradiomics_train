import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")


def _list_images(image_dir: str) -> List[str]:
    files = []
    for name in os.listdir(image_dir):
        if name.endswith(_IMAGE_EXTS):
            files.append(os.path.join(image_dir, name))
    return sorted(files)


def _read_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={arr.shape} ({path})")
    return arr


def _read_mask(path: str, threshold: int) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.asarray(m)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape} ({path})")
    return (arr > threshold).astype(np.uint8)


def _normalize_rel_path(path: str) -> str:
    return os.path.normpath(path.replace("\\", "/").lstrip("/\\"))


def _resolve_image_path(image_dir: str, label_filename: str) -> str:
    rel = _normalize_rel_path(label_filename)

    # Preferred: root + relative path from label json (supports nested directories)
    direct = os.path.join(image_dir, rel)
    if os.path.exists(direct):
        return direct

    stem, ext = os.path.splitext(rel)
    if ext:
        # Keep old behavior as fallback: only basename under image_dir
        base_fallback = os.path.join(image_dir, os.path.basename(rel))
        if os.path.exists(base_fallback):
            return base_fallback
    else:
        for e in _IMAGE_EXTS:
            cand = os.path.join(image_dir, stem + e)
            if os.path.exists(cand):
                return cand

    raise FileNotFoundError(
        f"Image not found for label filename '{label_filename}'. Tried '{direct}' and basename fallback in {image_dir}"
    )


def _resolve_mask_path(mask_dir: str, label_filename: str, mask_suffix: str) -> str:
    rel = _normalize_rel_path(label_filename)
    stem, ext = os.path.splitext(rel)
    stem_with_suffix = stem + (mask_suffix or "")

    # Preferred: root + relative path from label json (supports nested directories)
    if ext:
        direct = os.path.join(mask_dir, stem_with_suffix + ext)
        if os.path.exists(direct):
            return direct

    for e in _IMAGE_EXTS:
        cand = os.path.join(mask_dir, stem_with_suffix + e)
        if os.path.exists(cand):
            return cand

    # Keep old behavior as fallback: only basename under mask_dir
    stem_base = os.path.basename(stem_with_suffix)
    if ext:
        base_direct = os.path.join(mask_dir, stem_base + ext)
        if os.path.exists(base_direct):
            return base_direct
    for e in _IMAGE_EXTS:
        base_cand = os.path.join(mask_dir, stem_base + e)
        if os.path.exists(base_cand):
            return base_cand

    raise FileNotFoundError(
        f"Mask not found for label filename '{label_filename}'. Looked for nested and basename paths in {mask_dir}"
    )


def _load_label_rows(label_json_path: str) -> List[Dict[str, object]]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("label_json must be a list of dicts and include key: filename")

    rows: List[Dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue
        item_copy = dict(item)
        item_copy["filename"] = str(filename)
        rows.append(item_copy)

    return rows


def _to_sitk(image: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float]):
    import SimpleITK as sitk

    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    msk_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

    img_sitk.SetSpacing(spacing)
    msk_sitk.SetSpacing(spacing)

    return img_sitk, msk_sitk


def _extract_one(
    extractor,
    image_path: str,
    mask_path: str,
    label_value: int,
    output_filename: str,
    mask_threshold: int,
    spacing: Tuple[float, float],
) -> Dict[str, object]:
    image = _read_gray(image_path)
    mask = _read_mask(mask_path, threshold=mask_threshold)

    if image.shape != mask.shape:
        raise ValueError(f"Image/mask size mismatch: image={image.shape}, mask={mask.shape}")

    if int(mask.sum()) == 0:
        raise ValueError("Empty mask (no foreground pixels)")

    # PyRadiomics expects SimpleITK images
    img_sitk, msk_sitk = _to_sitk(image, mask, spacing=spacing)

    result = extractor.execute(img_sitk, msk_sitk, label=1)

    # drop diagnostics; keep numeric features
    feats = {k: v for k, v in result.items() if not str(k).startswith("diagnostics_")}
    # add label and filename so downstream CSV has image identity
    feats["label"] = int(label_value)
    feats["filename"] = output_filename
    return feats


@dataclass
class Args:
    image_dir: str
    mask_dir: str
    label_json: str
    task: str
    output_csv: str
    params: str
    mask_threshold: int
    mask_suffix: str
    spacing_x: float
    spacing_y: float
    skip_fail: bool
    keep_unlabeled: bool
    limit: Optional[int]


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="2D pyradiomics feature extraction using predicted segmentation masks as ROI."
    )
    p.add_argument("--image_dir", type=str, required=True, help="directory of 2D images (png/jpg)")
    p.add_argument("--mask_dir", type=str, required=True, help="directory of 2D masks (png/jpg)")
    p.add_argument("--label_json", type=str, required=True, help="json file containing labels")

    p.add_argument(
        "--task",
        type=str,
        default="malignancy",
        help="label field to use from label_json (e.g. malignancy/tirads/LNM_CN01/FTCPTC)",
    )
    p.add_argument("--output_csv", type=str, required=True, help="output features CSV")

    p.add_argument(
        "--params",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "radiomics_2d.yaml"),
        help="pyradiomics parameter YAML",
    )

    p.add_argument("--mask_threshold", type=int, default=0, help="mask > threshold is treated as foreground")
    p.add_argument("--mask_suffix", type=str, default="", help="optional suffix added to image basename to find mask")

    # no real spacing; keep consistent pseudo spacing
    p.add_argument("--spacing_x", type=float, default=1.0)
    p.add_argument("--spacing_y", type=float, default=1.0)

    p.add_argument("--skip_fail", action="store_true", help="skip failed cases instead of stopping")
    p.add_argument("--keep_unlabeled", action="store_true", help="keep label=-1 rows (default: drop)")
    p.add_argument("--limit", type=int, default=None, help="debug: only process first N images")

    a = p.parse_args()
    return Args(
        image_dir=a.image_dir,
        mask_dir=a.mask_dir,
        label_json=a.label_json,
        task=a.task,
        output_csv=a.output_csv,
        params=a.params,
        mask_threshold=a.mask_threshold,
        mask_suffix=a.mask_suffix,
        spacing_x=a.spacing_x,
        spacing_y=a.spacing_y,
        skip_fail=bool(a.skip_fail),
        keep_unlabeled=bool(a.keep_unlabeled),
        limit=a.limit,
    )


def main() -> None:
    args = parse_args()

    from radiomics import featureextractor

    label_rows = _load_label_rows(args.label_json)
    if args.limit is not None:
        label_rows = label_rows[: args.limit]
    if not label_rows:
        raise ValueError("No valid rows found in label_json (require dict items with key 'filename').")

    # Task key is now fully configurable; validate against keys present in label_json.
    available_tasks = sorted(
        {
            str(k)
            for row in label_rows
            for k in row.keys()
            if str(k) not in {"filename"}
        }
    )
    if args.task not in available_tasks:
        raise ValueError(
            f"task='{args.task}' not found in label_json. Available tasks: {available_tasks}"
        )

    extractor = featureextractor.RadiomicsFeatureExtractor(args.params)

    rows: List[Dict[str, object]] = []
    failures = 0
    skipped_unlabeled = 0

    for idx, label_info in enumerate(label_rows):
        raw_filename = str(label_info.get("filename", ""))
        fname = os.path.basename(raw_filename)

        label_value = int(label_info.get(args.task, -1))
        if label_value == -1:
            skipped_unlabeled += 1
            continue

        try:
            img_path = _resolve_image_path(args.image_dir, raw_filename)
            mask_path = _resolve_mask_path(args.mask_dir, raw_filename, mask_suffix=args.mask_suffix)
            feats = _extract_one(
                extractor,
                image_path=img_path,
                mask_path=mask_path,
                label_value=label_value,
                output_filename=_normalize_rel_path(raw_filename),
                mask_threshold=args.mask_threshold,
                spacing=(args.spacing_x, args.spacing_y),
            )
            rows.append(feats)
        except Exception as e:
            failures += 1
            msg = f"[{idx}] failed: {fname} err={type(e).__name__}: {e}"
            if args.skip_fail:
                print(msg)
                continue
            raise RuntimeError(msg) from e

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(
        "Done. "
        f"images_seen={len(label_rows)} extracted={len(df)} failures={failures} skipped_unlabeled={skipped_unlabeled} "
        f"saved={args.output_csv}"
    )


if __name__ == "__main__":
    main()
