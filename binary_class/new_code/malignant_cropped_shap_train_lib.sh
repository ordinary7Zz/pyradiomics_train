#!/usr/bin/env bash
# Malignant_ultrasound_images_cropped + pred mask: label JSON -> radiomics -> AutoGluon
set -euo pipefail

: "${PYR_ROOT:=/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train}"
: "${PYTHON:=/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
: "${MALIGNANT_IMAGE_ROOT:=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped}"
: "${MALIGNANT_TRAIN_LABELS:=${MALIGNANT_IMAGE_ROOT}/train_labels_filtered_by_csv.json}"
: "${MALIGNANT_TEST_LABELS:=${MALIGNANT_IMAGE_ROOT}/test_labels_filtered_by_csv.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=zjh3227_shap_train_lib.sh
source "${SCRIPT_DIR}/zjh3227_shap_train_lib.sh"

malignant_export_resolved() {
  local task="$1"
  local label_json="$2"
  local out_json="$3"
  "${PYTHON}" - <<PY
import sys
from pathlib import Path

pyr = Path("${PYR_ROOT}")
sys.path.insert(0, str(pyr))

from binary_class.new_code.packed_io import (
    build_sample_records_from_label_json,
    write_resolved_samples,
)
from binary_class.new_code.task_specs import get_task

spec = get_task("${task}")
records, stats = build_sample_records_from_label_json(
    spec,
    Path("${label_json}"),
    image_root_key="zjh",
    image_root=Path("${MALIGNANT_IMAGE_ROOT}"),
)
rows = [s.to_resolved_row(spec.label_key) for s in records]
out = Path("${out_json}")
write_resolved_samples(out, rows)
print(f"Wrote {len(rows)} rows -> {out}")
print("resolve_stats:", stats)
PY
}
