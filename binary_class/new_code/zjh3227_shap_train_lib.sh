#!/usr/bin/env bash
# Shared helpers: ZJH 3227 train + 801 test -> radiomics -> AutoGluon (SHAP / 做法1)
set -euo pipefail

: "${PYR_ROOT:=/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train}"
: "${PYTHON:=/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
: "${TRAIN_MANIFEST_REL:=manifests/zjh_dev_all.json}"
: "${TEST_MANIFEST_REL:=manifests/zjh_test.json}"
: "${FEATURE_PRESET:=full}"
: "${FEATURE_FILTER:=standard}"
: "${RADIOMICS_PARAMS:=}"

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
  if [[ -n "${RADIOMICS_PARAMS}" ]]; then
    extra+=(--params "${RADIOMICS_PARAMS}")
  fi
  extra+=(--feature_preset "${FEATURE_PRESET}")
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
  zjh3227_assert_task_csv "${output_csv}"
}

zjh3227_assert_task_csv() {
  local csv_path="$1"
  "${PYTHON}" - <<PY
import pandas as pd
from pathlib import Path
p = Path("${csv_path}")
cols = pd.read_csv(p, nrows=0).columns.tolist()
forbidden = {"patient_id", "image_path", "mask_path"}
bad = forbidden & set(cols)
if bad:
    raise SystemExit(f"task CSV must not contain {bad}: {p}")
if "label" not in cols:
    raise SystemExit(f"missing label column: {p}")
print(f"OK task CSV ({len(cols)} cols, no patient_id): {p}")
PY
}

zjh3227_apply_feature_filter() {
  local work_dir="$1"
  local train_task="$2"
  local test_task="$3"
  local manifest="${work_dir}/feature_manifest.json"
  : "${FEATURE_FILTER:=standard}"
  : "${FEATURE_PRESET:=full}"
  local params_arg=()
  if [[ -n "${RADIOMICS_PARAMS}" ]]; then
    params_arg=(--radiomics_params "${RADIOMICS_PARAMS}")
  else
    local resolved_params
    resolved_params="$("${PYTHON}" - <<PY
from binary_class.new_code.feature_presets import resolve_radiomics_params
print(resolve_radiomics_params("${FEATURE_PRESET}", None))
PY
)"
    params_arg=(--radiomics_params "${resolved_params}")
  fi
  "${PYTHON}" -m binary_class.new_code.feature_pipeline \
    --train_csv "${train_task}" \
    --test_csv "${test_task}" \
    --manifest "${manifest}" \
    --filter_mode "${FEATURE_FILTER}" \
    --feature_preset "${FEATURE_PRESET}" \
    --drop_filename \
    "${params_arg[@]}"
  echo "Feature manifest: ${manifest}"
}

zjh3227_maybe_extract_features() {
  local resolved_json="$1"
  local output_csv="$2"
  if [[ "${REUSE_FEATURES:-0}" == "1" && -s "${output_csv}" ]]; then
    echo "REUSE_FEATURES=1 — skip extract -> ${output_csv}"
    return 0
  fi
  zjh3227_extract_features "${resolved_json}" "${output_csv}"
}

zjh3227_train_autogluon() {
  local train_csv="$1"
  local test_csv="$2"
  local save_dir="$3"
  local task_name="$4"
  local manifest="${5:-}"
  : "${TIME_LIMIT:=3600}"
  : "${MODEL_SET:=tree_full}"
  : "${PRESETS:=best_quality}"
  : "${AG_SEED:=42}"
  : "${CI_SEED:=42}"
  : "${RESAMPLE_STRATEGY:=none}"
  : "${TEST_NAMES:=ZJH_test_801}"
  local extra_manifest=()
  if [[ -n "${manifest}" && -f "${manifest}" ]]; then
    extra_manifest=(--feature_manifest "${manifest}")
  fi

  mkdir -p "${save_dir}"
  "${PYTHON}" -m binary_class.train_binary_task_resampled \
    --train_csv "${train_csv}" \
    --test_csv "${test_csv}" \
    --test_names "${TEST_NAMES}" \
    "${extra_manifest[@]}" \
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
