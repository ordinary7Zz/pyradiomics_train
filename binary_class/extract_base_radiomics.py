import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")


def _normalize_rel_path(path: str) -> str:
    return os.path.normpath(path.replace("\\", "/").lstrip("/\\"))


def _strip_ext(path: str) -> str:
    stem, _ = os.path.splitext(path)
    return stem


def _resolve_with_flexible_ext(base_dir: str, rel_path: str) -> str:
    rel_path = _normalize_rel_path(rel_path)
    direct = os.path.join(base_dir, rel_path)
    if os.path.exists(direct):
        return direct

    rel_stem = _strip_ext(rel_path)
    for ext in _IMAGE_EXTS:
        candidate = os.path.join(base_dir, rel_stem + ext)
        if os.path.exists(candidate):
            return candidate

    base_name = os.path.basename(rel_path)
    base_direct = os.path.join(base_dir, base_name)
    if os.path.exists(base_direct):
        return base_direct

    base_stem = _strip_ext(base_name)
    for ext in _IMAGE_EXTS:
        candidate = os.path.join(base_dir, base_stem + ext)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"File not found under {base_dir} for rel_path={rel_path}")


def _read_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={arr.shape} ({path})")
    return arr


def _read_mask(path: str, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape} ({path})")
    return (arr > threshold).astype(np.uint8)


def _load_label_rows(label_json_path: str) -> List[Dict[str, object]]:
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
        row["filename"] = str(filename)
        rows.append(row)
    return rows


def _to_sitk(image: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float]):
    import SimpleITK as sitk

    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    msk_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    img_sitk.SetSpacing(spacing)
    msk_sitk.SetSpacing(spacing)
    return img_sitk, msk_sitk


def _extract_one(extractor, image_path: str, mask_path: str, filename: str, mask_threshold: int, spacing: Tuple[float, float]) -> Dict[str, object]:
    image = _read_gray(image_path)
    mask = _read_mask(mask_path, threshold=mask_threshold)

    if image.shape != mask.shape:
        raise ValueError(f"Image/mask size mismatch: image={image.shape}, mask={mask.shape}")
    if int(mask.sum()) == 0:
        raise ValueError("Empty mask (no foreground pixels)")

    img_sitk, msk_sitk = _to_sitk(image, mask, spacing=spacing)
    result = extractor.execute(img_sitk, msk_sitk, label=1)

    features = {k: v for k, v in result.items() if not str(k).startswith("diagnostics_")}
    features["filename"] = _normalize_rel_path(filename)
    features["image_path"] = image_path
    features["mask_path"] = mask_path
    return features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract base radiomics features without binding to a single task label.")
    p.add_argument("--image_dir", type=str, required=True, help="base directory for images")
    p.add_argument("--mask_dir", type=str, required=True, help="base directory for masks")
    p.add_argument("--label_json", type=str, required=True, help="json file containing filename and task keys")
    p.add_argument("--output_csv", type=str, required=True, help="output base features csv")
    p.add_argument("--params", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "radiomics_2d.yaml"))
    p.add_argument("--mask_threshold", type=int, default=0)
    p.add_argument("--spacing_x", type=float, default=1.0)
    p.add_argument("--spacing_y", type=float, default=1.0)
    p.add_argument("--skip_fail", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from radiomics import featureextractor

    label_rows = _load_label_rows(args.label_json)
    if args.limit is not None:
        label_rows = label_rows[: args.limit]
    if not label_rows:
        raise ValueError("No valid rows found in label_json")

    extractor = featureextractor.RadiomicsFeatureExtractor(args.params)
    rows: List[Dict[str, object]] = []
    failures = 0

    for idx, label_info in enumerate(label_rows):
        raw_filename = str(label_info["filename"])
        try:
            image_path = _resolve_with_flexible_ext(args.image_dir, raw_filename)
            mask_path = _resolve_with_flexible_ext(args.mask_dir, raw_filename)
            features = _extract_one(
                extractor=extractor,
                image_path=image_path,
                mask_path=mask_path,
                filename=raw_filename,
                mask_threshold=args.mask_threshold,
                spacing=(args.spacing_x, args.spacing_y),
            )
            rows.append(features)
        except Exception as e:
            failures += 1
            msg = f"[{idx}] failed: {raw_filename} err={type(e).__name__}: {e}"
            if args.skip_fail:
                print(msg)
                continue
            raise RuntimeError(msg) from e

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Done. images_seen={len(label_rows)} extracted={len(df)} failures={failures} saved={args.output_csv}")


if __name__ == "__main__":
    main()
