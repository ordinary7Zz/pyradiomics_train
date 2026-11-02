#!/usr/bin/env bash
# 做法1 · LNM：Malignant cropped train/test JSON + pred mask -> AutoGluon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=malignant_cropped_shap_train_lib.sh
source "${SCRIPT_DIR}/malignant_cropped_shap_train_lib.sh"

TASK="lnm"
WORK_DIR="${WORK_DIR:-${PYR_ROOT}/binary_class/new_code/runs/shap_full_lnm_zjh3227}"
SKIP_FAIL_FEATURES="${SKIP_FAIL_FEATURES:-1}"
REUSE_FEATURES="${REUSE_FEATURES:-0}"
TEST_NAME="${TEST_NAME:-Malignant_test_filtered}"

TRAIN_RES="${WORK_DIR}/train_resolved.json"
TEST_RES="${WORK_DIR}/test_resolved.json"
TRAIN_FEAT="${WORK_DIR}/train_features.csv"
TEST_FEAT="${WORK_DIR}/test_features.csv"
TRAIN_TASK="${WORK_DIR}/train_task.csv"
TEST_TASK="${WORK_DIR}/test_task.csv"
AG_DIR="${WORK_DIR}/ag_model"

mkdir -p "${WORK_DIR}"
cd "${PYR_ROOT}"

echo "==> LNM full-train | PYTHON=${PYTHON}"
echo "    image:  ${MALIGNANT_IMAGE_ROOT}"
echo "    train:  ${MALIGNANT_TRAIN_LABELS}"
echo "    test:   ${MALIGNANT_TEST_LABELS}"
echo "    work:   ${WORK_DIR}"

echo "==> [1/4] label JSON -> resolved"
malignant_export_resolved "${TASK}" "${MALIGNANT_TRAIN_LABELS}" "${TRAIN_RES}"
malignant_export_resolved "${TASK}" "${MALIGNANT_TEST_LABELS}" "${TEST_RES}"

echo "==> [2/4] radiomics (pred mask)"
zjh3227_maybe_extract_features "${TRAIN_RES}" "${TRAIN_FEAT}"
zjh3227_maybe_extract_features "${TEST_RES}" "${TEST_FEAT}"

echo "==> [3/4] labels -> task CSV (strip patient_id / paths)"
zjh3227_build_task_csv "${TASK}" "${TRAIN_FEAT}" "${TRAIN_RES}" "${TRAIN_TASK}"
zjh3227_build_task_csv "${TASK}" "${TEST_FEAT}" "${TEST_RES}" "${TEST_TASK}"

echo "==> [3b/4] image-level feature filter (train-only fit)"
zjh3227_apply_feature_filter "${WORK_DIR}" "${TRAIN_TASK}" "${TEST_TASK}"

echo "==> [4/4] AutoGluon (tree_full, time_limit=${TIME_LIMIT:-3600}s)"
rm -rf "${AG_DIR}"
TEST_NAMES="${TEST_NAME}" zjh3227_train_autogluon "${TRAIN_TASK}" "${TEST_TASK}" "${AG_DIR}" "LNM_CN01" "${WORK_DIR}/feature_manifest.json"

cat <<EOF

Done.
  model:  ${AG_DIR}
  train:  ${TRAIN_TASK}
  test:   ${AG_DIR}/test_results.csv

Next SHAP:
  cd ${PYR_ROOT}
  bash shap_analyze/shap_analyze_autogluon_fixed_LNM_zjh3227.sh
EOF
