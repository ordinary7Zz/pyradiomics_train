import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")
_PERTURB_MASK_SOURCES = {"gt_mild_perturb", "gt_moderate_perturb"}
_MASK_SOURCE_CHOICES = ("gt", "gt_mild_perturb", "gt_moderate_perturb", "pred")


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

    direct = os.path.join(image_dir, rel)
    if os.path.exists(direct):
        return direct

    stem, ext = os.path.splitext(rel)
    if ext:
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

    if ext:
        direct = os.path.join(mask_dir, stem_with_suffix + ext)
        if os.path.exists(direct):
            return direct

    for e in _IMAGE_EXTS:
        cand = os.path.join(mask_dir, stem_with_suffix + e)
        if os.path.exists(cand):
            return cand

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


def _mask_to_sitk(mask: np.ndarray, spacing: Tuple[float, float]):
    import SimpleITK as sitk

    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_sitk.SetSpacing(spacing)
    return sitk.Cast(mask_sitk, sitk.sitkUInt8)


def _extract_features(
    extractor,
    image: np.ndarray,
    mask: np.ndarray,
    label_value: int,
    output_filename: str,
    spacing: Tuple[float, float],
) -> Dict[str, object]:
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask size mismatch: image={image.shape}, mask={mask.shape}")

    if int(mask.sum()) == 0:
        raise ValueError("Empty mask (no foreground pixels)")

    img_sitk, msk_sitk = _to_sitk(image, mask, spacing=spacing)
    result = extractor.execute(img_sitk, msk_sitk, label=1)

    feats = {k: v for k, v in result.items() if not str(k).startswith("diagnostics_")}
    feats["label"] = int(label_value)
    feats["filename"] = output_filename
    return feats


def _stable_case_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    denom = a_sum + b_sum
    if denom == 0:
        return float("nan")
    inter = int(np.logical_and(a, b).sum())
    return float((2.0 * inter) / denom)


def _hd95(mask_a: np.ndarray, mask_b: np.ndarray, spacing: Tuple[float, float]) -> float:
    import SimpleITK as sitk

    if int(mask_a.sum()) == 0 or int(mask_b.sum()) == 0:
        return float("nan")

    sitk_a = _mask_to_sitk(mask_a, spacing)
    sitk_b = _mask_to_sitk(mask_b, spacing)

    contour_a = sitk.LabelContour(sitk_a)
    contour_b = sitk.LabelContour(sitk_b)

    dist_to_a = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk_a, squaredDistance=False, useImageSpacing=True))
    dist_to_b = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk_b, squaredDistance=False, useImageSpacing=True))

    contour_a_arr = sitk.GetArrayViewFromImage(contour_a) > 0
    contour_b_arr = sitk.GetArrayViewFromImage(contour_b) > 0
    dist_to_a_arr = sitk.GetArrayViewFromImage(dist_to_a)
    dist_to_b_arr = sitk.GetArrayViewFromImage(dist_to_b)

    distances_ab = dist_to_b_arr[contour_a_arr]
    distances_ba = dist_to_a_arr[contour_b_arr]

    if distances_ab.size == 0 and distances_ba.size == 0:
        return 0.0 if np.array_equal(mask_a, mask_b) else float("nan")

    surface_distances = np.concatenate([
        distances_ab.astype(np.float64, copy=False),
        distances_ba.astype(np.float64, copy=False),
    ])
    return float(np.percentile(surface_distances, 95))


def _perturb_radius_candidates(level: str) -> List[int]:
    if level == "gt_mild_perturb":
        return [2, 3]
    if level == "gt_moderate_perturb":
        return [5, 6, 7]
    raise ValueError(f"Unsupported perturbation level: {level}")


def _apply_binary_morphology(
    mask: np.ndarray,
    operation: str,
    radius: int,
    spacing: Tuple[float, float],
) -> np.ndarray:
    import SimpleITK as sitk

    mask_sitk = _mask_to_sitk(mask, spacing)
    kernel_radius = [int(radius), int(radius)]

    if operation == "dilation":
        out = sitk.BinaryDilate(mask_sitk, kernel_radius, sitk.sitkBall, 0.0, 1.0, False)
    elif operation == "erosion":
        out = sitk.BinaryErode(mask_sitk, kernel_radius, sitk.sitkBall, 0.0, 1.0, False)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return (sitk.GetArrayFromImage(out) > 0).astype(np.uint8)


def _perturb_mask(
    gt_mask: np.ndarray,
    mask_source: str,
    filename_key: str,
    perturb_seed: int,
    spacing: Tuple[float, float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(_stable_case_seed(perturb_seed, f"{mask_source}:{filename_key}"))
    radii = _perturb_radius_candidates(mask_source)
    radii = list(rng.permutation(radii))
    operations = ["dilation", "erosion"] if float(rng.random()) < 0.5 else ["erosion", "dilation"]

    fallback_mask: Optional[np.ndarray] = None
    fallback_meta: Optional[Dict[str, object]] = None

    for operation in operations:
        for radius in radii:
            candidate = _apply_binary_morphology(gt_mask, operation=operation, radius=radius, spacing=spacing)
            if int(candidate.sum()) == 0:
                continue

            meta = {
                "operation": operation,
                "kernel_radius": int(radius),
            }
            if fallback_mask is None:
                fallback_mask = candidate
                fallback_meta = meta
            if not np.array_equal(candidate, gt_mask):
                dice = _dice(gt_mask, candidate)
                hd95 = _hd95(gt_mask, candidate, spacing=spacing)
                meta.update(
                    {
                        "dice_vs_gt": dice,
                        "hd95_vs_gt": hd95,
                        "gt_foreground_pixels": int(gt_mask.sum()),
                        "perturbed_foreground_pixels": int(candidate.sum()),
                    }
                )
                return candidate, meta

    if fallback_mask is not None and fallback_meta is not None:
        fallback_meta = dict(fallback_meta)
        fallback_meta.update(
            {
                "dice_vs_gt": _dice(gt_mask, fallback_mask),
                "hd95_vs_gt": _hd95(gt_mask, fallback_mask, spacing=spacing),
                "gt_foreground_pixels": int(gt_mask.sum()),
                "perturbed_foreground_pixels": int(fallback_mask.sum()),
            }
        )
        return fallback_mask, fallback_meta

    raise ValueError("Failed to generate a non-empty perturbed mask")


def _default_perturb_stats_csv(output_csv: str, mask_source: str) -> str:
    root, ext = os.path.splitext(output_csv)
    if not ext:
        return f"{output_csv}.{mask_source}.mask_quality.csv"
    return f"{root}.{mask_source}.mask_quality.csv"


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
    mask_source: str
    perturb_seed: int
    perturb_stats_csv: Optional[str]


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="2D pyradiomics feature extraction with configurable mask sources and GT-based perturbations."
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
    p.add_argument(
        "--mask_source",
        type=str,
        default="gt",
        choices=_MASK_SOURCE_CHOICES,
        help="mask source setting: gt / gt_mild_perturb / gt_moderate_perturb / pred",
    )
    p.add_argument("--perturb_seed", type=int, default=42, help="base seed for reproducible GT mask perturbation")
    p.add_argument(
        "--perturb_stats_csv",
        type=str,
        default=None,
        help="optional sidecar CSV for perturbation-vs-GT metrics; auto-derived from output_csv when omitted",
    )

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
        mask_source=a.mask_source,
        perturb_seed=int(a.perturb_seed),
        perturb_stats_csv=a.perturb_stats_csv,
    )


def main() -> None:
    args = parse_args()

    from radiomics import featureextractor

    label_rows = _load_label_rows(args.label_json)
    if args.limit is not None:
        label_rows = label_rows[: args.limit]
    if not label_rows:
        raise ValueError("No valid rows found in label_json (require dict items with key 'filename').")

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
    perturb_rows: List[Dict[str, object]] = []
    failures = 0
    skipped_unlabeled = 0
    spacing = (args.spacing_x, args.spacing_y)

    for idx, label_info in enumerate(label_rows):
        raw_filename = str(label_info.get("filename", ""))
        normalized_filename = _normalize_rel_path(raw_filename)
        fname = os.path.basename(raw_filename)

        label_value = int(label_info.get(args.task, -1))
        if label_value == -1:
            skipped_unlabeled += 1
            continue

        try:
            img_path = _resolve_image_path(args.image_dir, raw_filename)
            mask_path = _resolve_mask_path(args.mask_dir, raw_filename, mask_suffix=args.mask_suffix)

            image = _read_gray(img_path)
            base_mask = _read_mask(mask_path, threshold=args.mask_threshold)

            mask_for_features = base_mask
            perturb_meta: Optional[Dict[str, object]] = None
            if args.mask_source in _PERTURB_MASK_SOURCES:
                mask_for_features, perturb_meta = _perturb_mask(
                    base_mask,
                    mask_source=args.mask_source,
                    filename_key=normalized_filename,
                    perturb_seed=args.perturb_seed,
                    spacing=spacing,
                )

            feats = _extract_features(
                extractor,
                image=image,
                mask=mask_for_features,
                label_value=label_value,
                output_filename=normalized_filename,
                spacing=spacing,
            )
            rows.append(feats)

            if perturb_meta is not None:
                perturb_row: Dict[str, object] = {
                    "filename": normalized_filename,
                    "mask_source": args.mask_source,
                    "mask_path": mask_path,
                }
                perturb_row.update(perturb_meta)
                perturb_rows.append(perturb_row)
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

    perturb_stats_csv = args.perturb_stats_csv
    if perturb_rows:
        if perturb_stats_csv is None:
            perturb_stats_csv = _default_perturb_stats_csv(args.output_csv, args.mask_source)
        perturb_dir = os.path.dirname(os.path.abspath(perturb_stats_csv))
        if perturb_dir:
            os.makedirs(perturb_dir, exist_ok=True)
        pd.DataFrame(perturb_rows).to_csv(perturb_stats_csv, index=False)

    print(
        "Done. "
        f"mask_source={args.mask_source} images_seen={len(label_rows)} extracted={len(df)} failures={failures} skipped_unlabeled={skipped_unlabeled} "
        f"saved={args.output_csv}"
    )
    if perturb_rows:
        print(f"Saved perturbation stats: {perturb_stats_csv}")


if __name__ == "__main__":
    main()
