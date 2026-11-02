from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .task_specs import AgTaskSpec


@dataclass
class SampleRecord:
    filename: str
    label: int
    patient_id: str
    image_path: str
    mask_path: str
    image_root_key: str
    raw: Dict[str, Any]

    def to_resolved_row(self, label_key: str) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "patient_id": self.patient_id,
            label_key: self.label,
            "label": self.label,
        }


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_roots(packed_root: Path, image_roots_json: Optional[Path] = None) -> Dict[str, str]:
    path = image_roots_json or (packed_root / "image_roots.json")
    data = load_json(path)
    roots: Dict[str, str] = {}
    for k, v in data.items():
        if k in ("pack_root", "zjh_image_source", "fangdai_source"):
            continue
        if isinstance(v, str) and v:
            roots[k] = v
    return roots


def _mask_from_image_rel(mask_root: Path, image_root: str, image_path: str) -> Optional[str]:
    try:
        rel = os.path.relpath(image_path, image_root)
    except ValueError:
        return None
    rel_no_ext, _ = os.path.splitext(rel)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = os.path.normpath(str(mask_root / (rel_no_ext + ext)))
        if os.path.isfile(candidate):
            return candidate
    candidate = os.path.normpath(str(mask_root / rel))
    if os.path.isfile(candidate):
        return candidate
    return None


def resolve_mask_path(
    spec: AgTaskSpec,
    image_root_key: str,
    image_path: str,
    image_roots: Dict[str, str],
) -> Optional[str]:
    if image_root_key in ("zjh",):
        zjh_root = image_roots.get("zjh")
        if zjh_root:
            found = _mask_from_image_rel(spec.zjh_mask_root, zjh_root, image_path)
            if found:
                return found
    if image_root_key in ("lymphus_c1", "lymphus_c2") and spec.lymphus_mask_root:
        candidate = spec.lymphus_mask_root / os.path.basename(image_path)
        if candidate.is_file():
            return str(candidate)
    if image_root_key in ("fangdai",) and spec.fangdai_mask_root:
        fangdai_root = image_roots.get("fangdai")
        if fangdai_root:
            found = _mask_from_image_rel(spec.fangdai_mask_root, fangdai_root, image_path)
            if found:
                return found
        candidate = spec.fangdai_mask_root / os.path.basename(image_path)
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_image_path(image_roots: Dict[str, str], image_root_key: str, filename: str) -> Optional[str]:
    root = image_roots.get(image_root_key)
    if not root:
        return None
    normalized = str(filename).replace("\\", os.sep).replace("/", os.sep)
    candidate = os.path.normpath(os.path.join(root, normalized))
    if os.path.isfile(candidate):
        return candidate
    base = os.path.basename(normalized)
    for dirpath, _, files in os.walk(root):
        if base in files:
            return os.path.join(dirpath, base)
    return None


def _zjh_year_name_cluster(filename: str) -> Optional[str]:
    """ZJH packed paths: ``{year}/{name}/...`` or ``images/zjh/{year}/{name}/...``."""
    rel = str(filename).strip().replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 5 and parts[0] == "images" and parts[1] == "zjh":
        return f"{parts[2]}/{parts[3]}"
    if len(parts) >= 3 and len(parts[0]) == 4 and parts[0].isdigit():
        return f"{parts[0]}/{parts[1]}"
    return None


def _is_lymphus_dataset(record: Dict[str, Any], root_key: str, filename: str) -> bool:
    if root_key in ("lymphus_c1", "lymphus_c2"):
        return True
    source = str(record.get("source") or "")
    if source.startswith("lymphus"):
        return True
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    return "_center1" in stem or "_center2" in stem


def _fangdai_case_cluster(filename: str) -> Optional[str]:
    """FangDai paths: group by case token ``B_<id>`` when present."""
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] == "B" and parts[1].isdigit():
        return f"fangdai::B_{parts[1]}"
    return f"fangdai::{base}"


def patient_id_from_record(record: Dict[str, Any], filename_key: str = "filename") -> str:
    """CV cluster key by **dataset**, not task: ZJH ``year/name``; LymphUs one image; FangDai case id."""
    filename = str(record.get(filename_key, "")).strip().replace("\\", "/")
    root_key = str(record.get("image_root_key") or "")

    if _is_lymphus_dataset(record, root_key, filename):
        source = str(record.get("source") or root_key or "lymphus")
        prefix = source if source.startswith("lymphus") else "lymphus"
        return f"{prefix}::{os.path.basename(filename)}"

    if root_key == "zjh" or _zjh_year_name_cluster(filename):
        cluster = _zjh_year_name_cluster(filename)
        if cluster:
            return cluster

    if root_key == "fangdai" or "/fangdai/" in filename.lower():
        return _fangdai_case_cluster(filename)

    for key in ("patient_id", "case_id", "patient", "case"):
        if record.get(key) is not None:
            return str(record[key]).strip()
    rel = filename.lstrip("/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1:
        stem = parts[0].rsplit(".", 1)[0]
        return stem.split("_")[0] if "_" in stem else stem
    return rel or "unknown"


def build_sample_records(
    spec: AgTaskSpec,
    manifest_path: Path,
    image_roots: Dict[str, str],
    label_key: Optional[str] = None,
) -> Tuple[List[SampleRecord], Dict[str, int]]:
    label_key = label_key or spec.label_key
    items = load_json(manifest_path)
    if not isinstance(items, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")

    records: List[SampleRecord] = []
    stats = {"skipped_label": 0, "skipped_image": 0, "skipped_mask": 0}

    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get(label_key, -1)
        try:
            label = int(label)
        except (TypeError, ValueError):
            label = -1
        if label not in (0, 1):
            stats["skipped_label"] += 1
            continue

        filename = str(item.get("filename", ""))
        root_key = str(item.get("image_root_key", "zjh"))
        image_path = resolve_image_path(image_roots, root_key, filename)
        if not image_path:
            stats["skipped_image"] += 1
            continue
        mask_path = resolve_mask_path(spec, root_key, image_path, image_roots)
        if not mask_path:
            stats["skipped_mask"] += 1
            continue

        records.append(
            SampleRecord(
                filename=filename,
                label=int(label),
                patient_id=patient_id_from_record(item),
                image_path=image_path,
                mask_path=mask_path,
                image_root_key=root_key,
                raw=item,
            )
        )

    return records, stats


def build_sample_records_from_label_json(
    spec: AgTaskSpec,
    label_json_path: Path,
    image_root_key: str,
    image_root: Path,
    label_key: Optional[str] = None,
) -> Tuple[List[SampleRecord], Dict[str, int]]:
    """Resolve FangDai (or single-root) samples from a flat label JSON list."""
    label_key = label_key or spec.label_key
    items = load_json(label_json_path)
    if not isinstance(items, list):
        raise ValueError(f"Label JSON must be a list: {label_json_path}")

    image_roots = {image_root_key: str(image_root)}
    records: List[SampleRecord] = []
    stats = {"skipped_label": 0, "skipped_image": 0, "skipped_mask": 0}

    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get(label_key, -1)
        try:
            label = int(label)
        except (TypeError, ValueError):
            label = -1
        if label not in (0, 1):
            stats["skipped_label"] += 1
            continue

        filename = str(item.get("filename", ""))
        manifest_row = {**item, "filename": filename, "image_root_key": image_root_key}
        image_path = resolve_image_path(image_roots, image_root_key, filename)
        if not image_path:
            stats["skipped_image"] += 1
            continue
        mask_path = resolve_mask_path(spec, image_root_key, image_path, image_roots)
        if not mask_path:
            stats["skipped_mask"] += 1
            continue

        records.append(
            SampleRecord(
                filename=filename,
                label=int(label),
                patient_id=patient_id_from_record(manifest_row),
                image_path=image_path,
                mask_path=mask_path,
                image_root_key=image_root_key,
                raw=manifest_row,
            )
        )

    return records, stats


def write_resolved_samples(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
