#!/usr/bin/env python3
"""
这个脚本用于根据一个 txt 文件中给出的目标文件名列表，
从已有的 image 目录、mask 目录和标签 json 文件中筛选出对应样本，
并把筛选结果整理成一个新的子集目录。

主要作用
--------
1. 读取 `--filename_list` 指定的 txt 文件，每行一个目标文件名。
2. 在 `--image_dir` 中查找对应原图文件，并复制到输出目录下的 `images/`。
3. 在 `--mask_dir` 中查找对应 mask 文件，并复制到输出目录下的 `masks/`。
4. 在 `--input_json` 指定的标签 json 中，按 `filename` 精确匹配筛出对应记录，
   写入新的 `labels.json`。
5. 额外生成一个 `summary.json`，用于记录筛选统计信息、缺失项和重复项。

匹配规则
--------
- image 和 mask：
  只按“文件名去掉最后一层后缀后的 stem”匹配，不要求后缀相同。
  例如 txt 中是 `002.jpg`，则可以匹配到：
  - image 中的 `002.png`
  - mask 中的 `002.bmp`

- json：
  按 `filename` 字段做完整字符串精确匹配。
  例如 txt 中是 `002.jpg`，那么 json 中必须也是 `002.jpg` 才算匹配；
  如果 json 中是 `002.png`，则不会命中。

输出结构
--------
执行后会在 `--output_dir` 下生成：
- `images/`：复制后的原图
- `masks/`：复制后的 mask
- `labels.json`：筛选后的标签记录
- `summary.json`：处理统计、缺失信息、歧义匹配信息

默认模式与严格模式
------------------
- 默认模式：
  如果某个样本缺失 image、mask 或 json 记录，脚本会给出 warning，
  但不会中断整个流程，而是继续处理其他样本。

- `--strict` 模式：
  一旦出现缺失项、歧义匹配或重复异常，脚本会立即报错退出。

使用示例
--------
示例 1：普通模式
python filter_copy_dataset.py \
  --filename_list ./BM_any_doctor_wrong_filename_list.txt \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/masks \
  --input_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/500_TestData_Malignancy_Cls.json \
  --output_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/171_doctor_wrong

示例 2：严格模式
python filter_copy_dataset.py \
  --filename_list /path/to/targets.txt \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --input_json /path/to/input_labels.json \
  --output_dir /path/to/output_subset \
  --strict

适用场景
--------
这个脚本适合在数据清洗、错误样本回收、子集数据导出、
或者为后续 radiomics / SHAP 分析准备小规模数据集时使用。
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter images/masks by filename list and export subset labels JSON."
    )
    parser.add_argument("--filename_list", required=True, help="Path to txt file of target filenames")
    parser.add_argument("--image_dir", required=True, help="Source image directory")
    parser.add_argument("--mask_dir", required=True, help="Source mask directory")
    parser.add_argument("--input_json", required=True, help="Source label json path")
    parser.add_argument("--output_dir", required=True, help="Output root directory")
    parser.add_argument("--output_json_name", default="labels.json", help="Name of exported label json")
    parser.add_argument("--strict", action="store_true", help="Fail immediately on missing or ambiguous matches")
    return parser.parse_args()


def stem_of(name: str) -> str:
    return Path(str(name).strip()).stem


def fail_or_warn(message: str, *, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    print(f"Warning: {message}", file=sys.stderr)


def load_requested_filenames(path: Path) -> tuple[list[str], list[str]]:
    ordered: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        filename = line.strip()
        if not filename:
            continue
        if filename in seen:
            duplicates.append(filename)
            continue
        seen.add(filename)
        ordered.append(filename)
    return ordered, duplicates


def load_json_rows(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input_json must be a list of objects")

    rows: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue
        filename = str(filename)
        if filename in rows:
            duplicates.append(filename)
            continue
        rows[filename] = dict(item)
    return rows, duplicates


def build_stem_index(base_dir: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in sorted(p for p in base_dir.rglob("*") if p.is_file()):
        grouped[stem_of(path.name)].append(path)

    resolved: dict[str, Path] = {}
    ambiguous: dict[str, list[str]] = {}
    for stem, candidates in grouped.items():
        if len(candidates) == 1:
            resolved[stem] = candidates[0]
        else:
            ambiguous[stem] = [str(candidate.name) for candidate in candidates]
    return resolved, ambiguous


def ensure_not_ambiguous(
    kind: str,
    requested_name: str,
    stem: str,
    ambiguous_map: dict[str, list[str]],
    *,
    strict: bool,
) -> bool:
    if stem not in ambiguous_map:
        return True
    fail_or_warn(
        f"Ambiguous {kind} match for '{requested_name}' (stem='{stem}'): {ambiguous_map[stem]}",
        strict=strict,
    )
    return False


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_subset(args: argparse.Namespace) -> int:
    filename_list_path = Path(args.filename_list)
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_images = output_dir / "images"
    output_masks = output_dir / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    requested, duplicate_requested = load_requested_filenames(filename_list_path)
    json_rows, duplicate_json = load_json_rows(input_json)
    image_index, ambiguous_images = build_stem_index(image_dir)
    mask_index, ambiguous_masks = build_stem_index(mask_dir)

    if duplicate_requested:
        fail_or_warn(f"Duplicate requested filenames: {duplicate_requested}", strict=args.strict)
    if duplicate_json:
        fail_or_warn(f"Duplicate json filenames: {duplicate_json}", strict=args.strict)

    selected_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_masks: list[str] = []
    missing_json_rows: list[str] = []

    for requested_name in requested:
        requested_stem = stem_of(requested_name)
        image_ok = ensure_not_ambiguous(
            "image", requested_name, requested_stem, ambiguous_images, strict=args.strict
        )
        mask_ok = ensure_not_ambiguous(
            "mask", requested_name, requested_stem, ambiguous_masks, strict=args.strict
        )
        if not image_ok or not mask_ok:
            continue

        image_path = image_index.get(requested_stem)
        mask_path = mask_index.get(requested_stem)
        row = json_rows.get(requested_name)

        if image_path is None:
            fail_or_warn(f"Missing image for '{requested_name}'", strict=args.strict)
            missing_images.append(requested_name)
        if mask_path is None:
            fail_or_warn(f"Missing mask for '{requested_name}'", strict=args.strict)
            missing_masks.append(requested_name)
        if row is None:
            fail_or_warn(f"Missing json row for '{requested_name}'", strict=args.strict)
            missing_json_rows.append(requested_name)
        if image_path is None or mask_path is None or row is None:
            continue

        shutil.copy2(image_path, output_images / image_path.name)
        shutil.copy2(mask_path, output_masks / mask_path.name)
        selected_rows.append(row)

    summary = {
        "requested_count": len(requested),
        "selected_count": len(selected_rows),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "missing_json_rows": missing_json_rows,
        "ambiguous_image_stems": ambiguous_images,
        "ambiguous_mask_stems": ambiguous_masks,
        "duplicate_requested_filenames": duplicate_requested,
        "duplicate_json_filenames": duplicate_json,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / args.output_json_name, selected_rows)
    write_json(output_dir / "summary.json", summary)
    print(f"requested_count={len(requested)} selected_count={len(selected_rows)}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return copy_subset(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
