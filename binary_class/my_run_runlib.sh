#!/usr/bin/env bash
# Shared paths for my_run-style pipelines: legacy base_features are read-only.
set -euo pipefail

# SHAP 图/CSV 放在 binary_class/shap_analysis_outputs/<run 目录名>/，与 outputs/runs/<run> 对应；
# 勿复用 legacy 数据集目录名（如 FTCPTC_FangDai），避免覆盖旧 SHAP。
my_run_shap_output_dir() {
  local pyr_root="$1"
  local work_dir="$2"
  echo "${pyr_root}/binary_class/shap_analysis_outputs/$(basename "${work_dir}")"
}

# Resolve TRAIN_FEAT / TEST_FEAT for reading (work dir first, then legacy).
my_run_resolve_feature_csvs() {
  local work_feat_dir="$1"
  local legacy_feat_dir="$2"
  local train_name="${3:-train_base_features.csv}"
  local test_name="${4:-test_base_features.csv}"

  TRAIN_FEAT=""
  TEST_FEAT=""
  FEAT_READ_SOURCE=""

  if [[ -s "${work_feat_dir}/${train_name}" && -s "${work_feat_dir}/${test_name}" ]]; then
    TRAIN_FEAT="${work_feat_dir}/${train_name}"
    TEST_FEAT="${work_feat_dir}/${test_name}"
    FEAT_READ_SOURCE="work:${work_feat_dir}"
  elif [[ -s "${legacy_feat_dir}/${train_name}" && -s "${legacy_feat_dir}/${test_name}" ]]; then
    TRAIN_FEAT="${legacy_feat_dir}/${train_name}"
    TEST_FEAT="${legacy_feat_dir}/${test_name}"
    FEAT_READ_SOURCE="legacy:${legacy_feat_dir}"
  elif [[ -s "${work_feat_dir}/${train_name}" ]]; then
    TRAIN_FEAT="${work_feat_dir}/${train_name}"
    FEAT_READ_SOURCE="work_partial:${work_feat_dir}"
    if [[ -s "${legacy_feat_dir}/${test_name}" ]]; then
      TEST_FEAT="${legacy_feat_dir}/${test_name}"
    fi
  elif [[ -s "${legacy_feat_dir}/${train_name}" ]]; then
    TRAIN_FEAT="${legacy_feat_dir}/${train_name}"
    FEAT_READ_SOURCE="legacy_partial:${legacy_feat_dir}"
    if [[ -s "${work_feat_dir}/${test_name}" ]]; then
      TEST_FEAT="${work_feat_dir}/${test_name}"
    elif [[ -s "${legacy_feat_dir}/${test_name}" ]]; then
      TEST_FEAT="${legacy_feat_dir}/${test_name}"
    fi
  fi
}

# Pick directory for new extracts (never under legacy). Avoid overwriting existing work CSVs.
my_run_pick_extract_out_dir() {
  local work_feat_dir="$1"
  if [[ -n "${FEAT_WRITE_DIR:-}" ]]; then
    echo "${FEAT_WRITE_DIR}"
    return 0
  fi
  if [[ ! -s "${work_feat_dir}/train_base_features.csv" ]]; then
    echo "${work_feat_dir}"
  else
    echo "${work_feat_dir}/run_$(date +%Y%m%d_%H%M%S)"
  fi
}

my_run_run_extract() {
  local python_bin="$1"
  local image_dir="$2"
  local mask_dir="$3"
  local train_label_json="$4"
  local test_label_json="$5"
  local extract_out="$6"

  mkdir -p "${extract_out}"
  local out_train="${extract_out}/train_base_features.csv"
  local out_test="${extract_out}/test_base_features.csv"

  if [[ -s "${out_train}" || -s "${out_test}" ]]; then
    echo "ERROR: refuse to overwrite existing features in ${extract_out}" >&2
    echo "  Set FEAT_WRITE_DIR to a new empty directory, or remove those files." >&2
    exit 1
  fi

  echo "==> extract base radiomics -> ${extract_out} (legacy paths are never written)"
  echo "==> [extract] train"
  "${python_bin}" binary_class/extract_base_radiomics.py \
    --image_dir "${image_dir}" \
    --mask_dir "${mask_dir}" \
    --label_json "${train_label_json}" \
    --output_csv "${out_train}" \
    --skip_fail

  echo "==> [extract] test"
  "${python_bin}" binary_class/extract_base_radiomics.py \
    --image_dir "${image_dir}" \
    --mask_dir "${mask_dir}" \
    --label_json "${test_label_json}" \
    --output_csv "${out_test}" \
    --skip_fail

  TRAIN_FEAT="${out_train}"
  TEST_FEAT="${out_test}"
  FEAT_READ_SOURCE="extract:${extract_out}"
}

# Train-only image-level feature filter (same as new_code/zjh3227_shap_train_lib.sh).
# Writes feature_manifest.json; overwrites train/test task CSVs in place (runs/ only).
my_run_apply_feature_filter() {
  local python_bin="$1"
  local work_dir="$2"
  local train_task="$3"
  local test_task="$4"
  local manifest="${work_dir}/feature_manifest.json"
  : "${FEATURE_FILTER:=standard}"
  : "${FEATURE_PRESET:=full}"
  if [[ "${FEATURE_FILTER}" == "none" ]]; then
    echo "FEATURE_FILTER=none — skip train-only feature filter"
    FEATURE_MANIFEST=""
    return 0
  fi
  local params_arg=()
  if [[ -n "${RADIOMICS_PARAMS:-}" ]]; then
    params_arg=(--radiomics_params "${RADIOMICS_PARAMS}")
  else
    local resolved_params
    resolved_params="$("${python_bin}" - <<PY
from binary_class.new_code.feature_presets import resolve_radiomics_params
print(resolve_radiomics_params("${FEATURE_PRESET}", None))
PY
)"
    params_arg=(--radiomics_params "${resolved_params}")
  fi
  "${python_bin}" -m binary_class.new_code.feature_pipeline \
    --train_csv "${train_task}" \
    --test_csv "${test_task}" \
    --manifest "${manifest}" \
    --filter_mode "${FEATURE_FILTER}" \
    --feature_preset "${FEATURE_PRESET}" \
    --drop_filename \
    "${params_arg[@]}"
  FEATURE_MANIFEST="${manifest}"
  echo "Feature manifest: ${FEATURE_MANIFEST}"
}
