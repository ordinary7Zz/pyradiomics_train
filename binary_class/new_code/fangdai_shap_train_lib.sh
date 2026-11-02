#!/usr/bin/env bash
# FangDai cropped + pred mask: label JSON -> radiomics -> AutoGluon (SHAP / 做法1)
set -euo pipefail

: "${PYR_ROOT:=/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train}"
: "${PYTHON:=/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
: "${FANGDAI_IMAGE_ROOT:=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped}"
: "${FANGDAI_TRAIN_LABELS:=${FANGDAI_IMAGE_ROOT}/FangDai_train_labels_fake.json}"
: "${FANGDAI_TEST_LABELS:=${FANGDAI_IMAGE_ROOT}/FangDai_test_labels.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=zjh3227_shap_train_lib.sh
source "${SCRIPT_DIR}/zjh3227_shap_train_lib.sh"

NEW_CODE="${PYR_ROOT}/binary_class/new_code"
TASK="ftcptc"

fangdai_export_resolved() {
  local label_json="$1"
  local out_json="$2"
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

spec = get_task("${TASK}")
records, stats = build_sample_records_from_label_json(
    spec,
    Path("${label_json}"),
    image_root_key="fangdai",
    image_root=Path("${FANGDAI_IMAGE_ROOT}"),
)
rows = [s.to_resolved_row(spec.label_key) for s in records]
out = Path("${out_json}")
write_resolved_samples(out, rows)
print(f"Wrote {len(rows)} rows -> {out}")
print("resolve_stats:", stats)
PY
}
