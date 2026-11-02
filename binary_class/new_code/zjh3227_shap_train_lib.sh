#!/usr/bin/env bash
# Shared helpers: ZJH 3227 train + 801 test -> radiomics -> AutoGluon (SHAP / 做法1)
set -euo pipefail

: "${PYR_ROOT:=/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train}"
: "${PYTHON:=/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
: "${TRAIN_MANIFEST_REL:=manifests/zjh_dev_all.json}"
: "${TEST_MANIFEST_REL:=manifests/zjh_test.json}"

NEW_CODE="${PYR_ROOT}/binary_class/new_code"

zjh3227_export_resolved() {
  local task="$1"
  local packed_root="$2"
  local manifest_rel="$3"
  local out_json="$4"
  "${PYTHON}" - <<PY
import sys
from pathlib import Path

pyr = Path("${PYR_ROOT}")
sys.path.insert(0, str(pyr))

from binary_class.new_code.packed_io import build_sample_records, load_image_roots, write_resolved_samples
from binary_class.new_code.task_specs import get_task

spec = get_task("${task}")
packed = Path("${packed_root}")
manifest = packed / "${manifest_rel}"
out = Path("${out_json}")
roots = load_image_roots(packed)
records, stats = build_sample_records(spec, manifest, roots)
rows = [s.to_resolved_row(spec.label_key) for s in records]
write_resolved_samples(out, rows)
print(f"Wrote {len(rows)} rows -> {out}")
print("resolve_stats:", stats)
PY
}

zjh3227_extract_features() {
  local resolved_json="$1"
  local output_csv="$2"
  local extra=()
  if [[ "${SKIP_FAIL_FEATURES:-0}" == "1" ]]; then
    extra+=(--skip_fail)
  fi
  "${PYTHON}" "${NEW_CODE}/extract_features_resolved.py" \
    --resolved_json "${resolved_json}" \
    --output_csv "${output_csv}" \
    "${extra[@]}"
}

zjh3227_build_task_csv() {
  local task="$1"
  local features_csv="$2"
  local resolved_json="$3"
  local output_csv="$4"
  local label_key
  if [[ "${task}" == "lnm" ]]; then
    label_key="LNM_CN01"
  else
    label_key="FTCPTC"
  fi
  "${PYTHON}" "${NEW_CODE}/build_task_csv.py" \
    --features_csv "${features_csv}" \
    --resolved_json "${resolved_json}" \
    --label_key "${label_key}" \
    --output_csv "${output_csv}"
}

zjh3227_train_autogluon() {
  local train_csv="$1"
  local test_csv="$2"
  local save_dir="$3"
  local task_name="$4"
  : "${TIME_LIMIT:=3600}"
  : "${MODEL_SET:=tree_full}"
  : "${PRESETS:=best_quality}"
  : "${AG_SEED:=42}"
  : "${CI_SEED:=42}"
  : "${RESAMPLE_STRATEGY:=none}"

  mkdir -p "${save_dir}"
  "${PYTHON}" -m binary_class.train_binary_task_resampled \
    --train_csv "${train_csv}" \
    --test_csv "${test_csv}" \
    --test_names ZJH_test_801 \
    --save_dir "${save_dir}" \
    --task_name "${task_name}" \
    --label label \
    --eval_metric roc_auc \
    --model_set "${MODEL_SET}" \
    --presets "${PRESETS}" \
    --time_limit "${TIME_LIMIT}" \
    --resample_strategy "${RESAMPLE_STRATEGY}" \
    --target_class0_count -1 \
    --target_class1_count -1 \
    --threshold 0.5 \
    --ece_bins 10 \
    --ci_bootstrap_iters 1000 \
    --ci_level 0.95 \
    --ci_seed "${CI_SEED}" \
    --seed "${AG_SEED}"
}
