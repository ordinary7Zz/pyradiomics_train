#!/usr/bin/env bash
# 做法1 · FangDai FTC/PTC：FangDai_train_labels.json + test_labels -> AutoGluon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=fangdai_shap_train_lib.sh
source "${SCRIPT_DIR}/fangdai_shap_train_lib.sh"

FANGDAI_TRAIN_LABELS="${FANGDAI_IMAGE_ROOT}/FangDai_train_labels.json"

WORK_DIR="${WORK_DIR:-${PYR_ROOT}/binary_class/new_code/runs/shap_full_ftcptc_fangdai_fake}"
SKIP_FAIL_FEATURES="${SKIP_FAIL_FEATURES:-1}"
REUSE_FEATURES="${REUSE_FEATURES:-0}"
TEST_NAME="${TEST_NAME:-FangDai_test}"

TRAIN_RES="${WORK_DIR}/train_resolved.json"
TEST_RES="${WORK_DIR}/test_resolved.json"
TRAIN_FEAT="${WORK_DIR}/train_features.csv"
TEST_FEAT="${WORK_DIR}/test_features.csv"
TRAIN_TASK="${WORK_DIR}/train_task.csv"
TEST_TASK="${WORK_DIR}/test_task.csv"
AG_DIR="${WORK_DIR}/ag_model"

mkdir -p "${WORK_DIR}"
cd "${PYR_ROOT}"

echo "==> FangDai FTC/PTC full-train | PYTHON=${PYTHON}"
echo "    image:  ${FANGDAI_IMAGE_ROOT}"
echo "    train:  ${FANGDAI_TRAIN_LABELS}"
echo "    test:   ${FANGDAI_TEST_LABELS}"
echo "    work:   ${WORK_DIR}"

echo "==> [1/4] label JSON -> resolved"
fangdai_export_resolved "${FANGDAI_TRAIN_LABELS}" "${TRAIN_RES}"
fangdai_export_resolved "${FANGDAI_TEST_LABELS}" "${TEST_RES}"

echo "==> [2/4] radiomics (FangDai pred mask)"
zjh3227_maybe_extract_features "${TRAIN_RES}" "${TRAIN_FEAT}"
zjh3227_maybe_extract_features "${TEST_RES}" "${TEST_FEAT}"

echo "==> [3/4] labels -> task CSV (strip patient_id / paths)"
zjh3227_build_task_csv "${TASK}" "${TRAIN_FEAT}" "${TRAIN_RES}" "${TRAIN_TASK}"
zjh3227_build_task_csv "${TASK}" "${TEST_FEAT}" "${TEST_RES}" "${TEST_TASK}"

echo "==> [3b/4] image-level feature filter (train-only fit)"
zjh3227_apply_feature_filter "${WORK_DIR}" "${TRAIN_TASK}" "${TEST_TASK}"

echo "==> [4/4] AutoGluon (tree_full, time_limit=${TIME_LIMIT:-3600}s)"
rm -rf "${AG_DIR}"
TEST_NAMES="${TEST_NAME}" zjh3227_train_autogluon "${TRAIN_TASK}" "${TEST_TASK}" "${AG_DIR}" "FTCPTC" "${WORK_DIR}/feature_manifest.json"

cat <<EOF

Done.
  model:  ${AG_DIR}
  train:  ${TRAIN_TASK}
  test:   ${AG_DIR}/test_results.csv

Next SHAP:
  cd ${PYR_ROOT}
  bash shap_analyze/shap_analyze_autogluon_fixed_FTCPTC_fangdai_fake.sh
EOF
