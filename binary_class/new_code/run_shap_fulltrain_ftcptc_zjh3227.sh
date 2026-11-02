#!/usr/bin/env bash
# 做法1 · FTC/PTC：ZJH 3227 全量训 AutoGluon（供 SHAP），ZJH 801 作 test 评估
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=zjh3227_shap_train_lib.sh
source "${SCRIPT_DIR}/zjh3227_shap_train_lib.sh"

TASK="ftcptc"
PACKED_ROOT="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FTCPTC_ZJH8K_FangDai_packed"
WORK_DIR="${WORK_DIR:-${PYR_ROOT}/binary_class/new_code/runs/shap_full_ftcptc_zjh3227}"
SKIP_FAIL_FEATURES="${SKIP_FAIL_FEATURES:-1}"

TRAIN_RES="${WORK_DIR}/train_resolved.json"
TEST_RES="${WORK_DIR}/test_resolved.json"
TRAIN_FEAT="${WORK_DIR}/train_features.csv"
TEST_FEAT="${WORK_DIR}/test_features.csv"
TRAIN_TASK="${WORK_DIR}/train_task.csv"
TEST_TASK="${WORK_DIR}/test_task.csv"
AG_DIR="${WORK_DIR}/ag_model"

mkdir -p "${WORK_DIR}"
cd "${PYR_ROOT}"

echo "==> FTC/PTC SHAP full-train (3227 train / 801 test)"
echo "    packed:  ${PACKED_ROOT}"
echo "    work:    ${WORK_DIR}"

echo "==> [1/4] manifest -> resolved JSON"
zjh3227_export_resolved "${TASK}" "${PACKED_ROOT}" "${TRAIN_MANIFEST_REL}" "${TRAIN_RES}"
zjh3227_export_resolved "${TASK}" "${PACKED_ROOT}" "${TEST_MANIFEST_REL}" "${TEST_RES}"

echo "==> [2/4] radiomics (pred mask)"
zjh3227_extract_features "${TRAIN_RES}" "${TRAIN_FEAT}"
zjh3227_extract_features "${TEST_RES}" "${TEST_FEAT}"

echo "==> [3/4] merge labels -> task CSV"
zjh3227_build_task_csv "${TASK}" "${TRAIN_FEAT}" "${TRAIN_RES}" "${TRAIN_TASK}"
zjh3227_build_task_csv "${TASK}" "${TEST_FEAT}" "${TEST_RES}" "${TEST_TASK}"

echo "==> [4/4] AutoGluon fit on full 3227"
zjh3227_train_autogluon "${TRAIN_TASK}" "${TEST_TASK}" "${AG_DIR}" "FTCPTC"

cat <<EOF

Done.
  SHAP --model_dir:  ${AG_DIR}
  SHAP --train_csv:  ${TRAIN_TASK}
  Test metrics:      ${AG_DIR}/test_results.csv

Example SHAP:
  cd ${PYR_ROOT}
  python shap_analyze/shap_analyze_autogluon_fixed.py \\
    --model_dir ${AG_DIR} \\
    --train_csv ${TRAIN_TASK} \\
    --label label \\
    --output_dir binary_class/shap_analysis_outputs/shap_full_ftcptc_zjh3227 \\
    --task_name "FTC vs PTC" \\
    --positive_class_name "FTC" \\
    --negative_class_name "PTC" \\
    --output_space "raw score" \\
    --main_models LightGBMXT_BAG_L1 \\
    --plot_beeswarm_for LightGBMXT_BAG_L1 \\
    --background_samples 500 \\
    --explain_samples 2000 \\
    --skip_neural_net \\
    --plot_waterfall \\
    --top_features 10 \\
    --waterfall_samples 10

(若 leaderboard 榜首不是 LightGBMXT_BAG_L1，请改 --main_models / --plot_beeswarm_for)
EOF
