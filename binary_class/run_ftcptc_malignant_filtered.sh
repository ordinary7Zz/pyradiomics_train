#!/usr/bin/env bash
# 整理后的 FTC/PTC · Malignant cropped + filtered JSON（与 binary_class/my_run.md 该节一致）
# Legacy 特征：outputs/base_features/FTCPTC_FangDai（只读，永不写入）
# 本脚本训练产物：outputs/runs/ftcptc_malignant_filtered/{task_csvs,ag_model[,feature_manifest.json]}
# 默认 FEATURE_FILTER=none（全 radiomics 维训练+SHAP）；53 维筛选：FEATURE_FILTER=standard bash ...
# Legacy 特征只读：outputs/base_features/FTCPTC_FangDai
# SHAP 输出：binary_class/shap_analysis_outputs/ftcptc_malignant_filtered（与 WORK_DIR 同名，不覆盖 legacy FTCPTC_FangDai）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=my_run_runlib.sh
source "${SCRIPT_DIR}/my_run_runlib.sh"

PYR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PYR_ROOT}"
export PYTHONPATH="${PYR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${PYTHON:=/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
: "${TIME_LIMIT:=3600}"
: "${MODEL_SET:=tree_full}"
: "${TEST_NAME:=FTCPTC_test}"
: "${SKIP_EXTRACT:=1}"
: "${FEATURE_FILTER:=none}"
: "${FEATURE_PRESET:=full}"

IMAGE_ROOT="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped"
MASK_ROOT="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions"

TRAIN_LABEL_JSON="${TRAIN_LABEL_JSON:-/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/train_labels_filtered_by_csv.json}"
TEST_LABEL_JSON="${TEST_LABEL_JSON:-/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/test_labels_filtered_by_csv.json}"

LEGACY_FEAT_DIR="${PYR_ROOT}/binary_class/outputs/base_features/FTCPTC_FangDai"
WORK_DIR="${WORK_DIR:-${PYR_ROOT}/binary_class/outputs/runs/ftcptc_malignant_filtered}"
FEAT_DIR="${FEAT_DIR:-${WORK_DIR}/features}"
TASK_DIR="${WORK_DIR}/task_csvs"
MODEL_DIR="${WORK_DIR}/ag_model"
SHAP_OUTPUT_DIR="${SHAP_OUTPUT_DIR:-$(my_run_shap_output_dir "${PYR_ROOT}" "${WORK_DIR}")}"

TRAIN_TASK="${TASK_DIR}/train_FTCPTC.csv"
TEST_TASK="${TASK_DIR}/test_FTCPTC.csv"

mkdir -p "${TASK_DIR}" "${MODEL_DIR}"

echo "==> FTC/PTC · Malignant filtered (my_run) | PYTHON=${PYTHON}"
echo "    SKIP_EXTRACT=${SKIP_EXTRACT} (default 1: do not re-extract)"
echo "    legacy feats (read-only): ${LEGACY_FEAT_DIR}"
echo "    work dir:                 ${WORK_DIR}"
echo "    FEATURE_FILTER=${FEATURE_FILTER} (none=~900维; standard=train-only 筛选~53维)"
echo "    train labels: ${TRAIN_LABEL_JSON}"
echo "    test labels:  ${TEST_LABEL_JSON}"

if [[ "${SKIP_EXTRACT}" == "1" ]]; then
  my_run_resolve_feature_csvs "${FEAT_DIR}" "${LEGACY_FEAT_DIR}"
  if [[ -z "${TRAIN_FEAT}" || -z "${TEST_FEAT}" ]]; then
    echo "ERROR: no feature CSVs found. Work: ${FEAT_DIR}  Legacy: ${LEGACY_FEAT_DIR}" >&2
    echo "  Run with SKIP_EXTRACT=0 to extract into ${FEAT_DIR} (or FEAT_WRITE_DIR)." >&2
    exit 1
  fi
  echo "    using features: ${FEAT_READ_SOURCE}"
  echo "    train feat: ${TRAIN_FEAT}"
  echo "    test feat:  ${TEST_FEAT}"
else
  EXTRACT_OUT="$(my_run_pick_extract_out_dir "${FEAT_DIR}")"
  echo "    extract output dir: ${EXTRACT_OUT}"
  my_run_run_extract "${PYTHON}" "${IMAGE_ROOT}" "${MASK_ROOT}" \
    "${TRAIN_LABEL_JSON}" "${TEST_LABEL_JSON}" "${EXTRACT_OUT}"
  echo "    using features: ${FEAT_READ_SOURCE}"
fi

echo "==> [2/5] build task CSV (FTCPTC) -> ${TASK_DIR}"
"${PYTHON}" binary_class/build_binary_task_csv.py \
  --base_features_csv "${TRAIN_FEAT}" \
  --label_json "${TRAIN_LABEL_JSON}" \
  --task FTCPTC \
  --output_csv "${TRAIN_TASK}"

"${PYTHON}" binary_class/build_binary_task_csv.py \
  --base_features_csv "${TEST_FEAT}" \
  --label_json "${TEST_LABEL_JSON}" \
  --task FTCPTC \
  --output_csv "${TEST_TASK}"

echo "==> [3/5] train-only feature filter (FEATURE_FILTER=${FEATURE_FILTER}, preset=${FEATURE_PRESET})"
my_run_apply_feature_filter "${PYTHON}" "${WORK_DIR}" "${TRAIN_TASK}" "${TEST_TASK}"

echo "==> [4/5] AutoGluon (model_set=${MODEL_SET}, time_limit=${TIME_LIMIT}s, no resample) -> ${MODEL_DIR}"
AG_MANIFEST=()
if [[ -n "${FEATURE_MANIFEST:-}" && -f "${FEATURE_MANIFEST}" ]]; then
  AG_MANIFEST=(--feature_manifest "${FEATURE_MANIFEST}")
fi
"${PYTHON}" -m binary_class.train_binary_task_resampled \
  --train_csv "${TRAIN_TASK}" \
  --test_csv "${TEST_TASK}" \
  --test_names "${TEST_NAME}" \
  "${AG_MANIFEST[@]}" \
  --save_dir "${MODEL_DIR}" \
  --eval_metric roc_auc \
  --model_set "${MODEL_SET}" \
  --time_limit "${TIME_LIMIT}" \
  --resample_strategy none \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42

: "${RUN_SHAP:=1}"
if [[ "${RUN_SHAP}" == "1" && "${SKIP_SHAP:-0}" != "1" ]]; then
  echo "==> [5/5] SHAP -> ${SHAP_OUTPUT_DIR}"
  export PYTHON
  export SHAP_MODEL_DIR="${MODEL_DIR}"
  export SHAP_TRAIN_CSV="${TRAIN_TASK}"
  export SHAP_FILENAMES_CSV="${TRAIN_FEAT}"
  export SHAP_OUTPUT_DIR="${SHAP_OUTPUT_DIR}"
  echo "    SHAP model:   ${SHAP_MODEL_DIR}"
  echo "    SHAP train:   ${SHAP_TRAIN_CSV}"
  echo "    SHAP output:  ${SHAP_OUTPUT_DIR}"
  bash "${PYR_ROOT}/shap_analyze/shap_analyze_autogluon_fixed_FTCPTC_FangDai.sh"
else
  echo "SKIP_SHAP=1 or RUN_SHAP=0 — skip SHAP"
fi

echo ""
echo "Done."
echo "  legacy feats (unchanged): ${LEGACY_FEAT_DIR}"
echo "  features used:          ${TRAIN_FEAT}"
echo "  feature manifest:       ${FEATURE_MANIFEST:- (none)}"
echo "  leaderboard:            ${MODEL_DIR}/leaderboard.csv"
echo "  test metrics:           ${MODEL_DIR}/test_results.csv"
echo "  SHAP out:               ${SHAP_OUTPUT_DIR}"
