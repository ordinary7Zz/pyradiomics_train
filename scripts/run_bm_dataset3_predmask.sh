#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BM_dataset3_predmask 完整训练流水线
# 阶段 1: 提取 Radiomics 基础特征
# 阶段 2: 构建二分类任务 CSV
# 阶段 3: 训练 + 评估
# ============================================================

# ---- 路径配置 ----
DATA_BASE="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test"

IMAGE_DIR_TRAIN="${DATA_BASE}/Superimposed_multitask/dataset_3/train/images"
IMAGE_DIR_TEST="${DATA_BASE}/Superimposed_multitask/dataset_3/test/images"

MASK_DIR_TRAIN="${DATA_BASE}/Superimposed_multitask/dataset_3/train/masks"
MASK_DIR_TEST="${DATA_BASE}/Superimposed_multitask/dataset_3/test/masks"

LABEL_JSON_TRAIN="${DATA_BASE}/Superimposed_multitask/dataset_3/train/dataset_3_train_label.json"
LABEL_JSON_TEST="${DATA_BASE}/Superimposed_multitask/dataset_3/test/dataset_3_test_label.json"

# ---- 输出路径 ----
OUT_BASE="binary_class/BM/outputs"
BASE_FEAT_DIR="${OUT_BASE}/base_features/BM_dataset3_gt"
TASK_CSV_DIR="${OUT_BASE}/task_csvs/BM_dataset3_gt"
MODEL_DIR="${OUT_BASE}/models/BM_dataset3_gt/BM"

# ---- 任务与标签 ----
TASK="malignancy"

# ---- 训练参数 ----
MODEL_SET="all"
TIME_LIMIT=7200
RESAMPLE_STRATEGY="none"
EVAL_METRIC="roc_auc"
THRESHOLD="0.5"
ECE_BINS=10
CI_BOOTSTRAP_ITERS=1000
CI_LEVEL="0.95"
CI_SEED=42
SEED=42

# ============================================================
# 阶段 1: 提取 Radiomics 基础特征
# ============================================================
echo "===== [1/5] 提取训练集基础特征 ====="
python binary_class/extract_base_radiomics.py \
  --image_dir "${IMAGE_DIR_TRAIN}" \
  --mask_dir "${MASK_DIR_TRAIN}" \
  --label_json "${LABEL_JSON_TRAIN}" \
  --output_csv "${BASE_FEAT_DIR}/train_base_features.csv" \
  --skip_fail

echo "===== [2/5] 提取测试集基础特征 ====="
python binary_class/extract_base_radiomics.py \
  --image_dir "${IMAGE_DIR_TEST}" \
  --mask_dir "${MASK_DIR_TEST}" \
  --label_json "${LABEL_JSON_TEST}" \
  --output_csv "${BASE_FEAT_DIR}/test_base_features.csv" \
  --skip_fail

# ============================================================
# 阶段 2: 构建二分类任务 CSV
# ============================================================
echo "===== [3/5] 构建训练集任务 CSV ====="
python binary_class/build_binary_task_csv.py \
  --base_features_csv "${BASE_FEAT_DIR}/train_base_features.csv" \
  --label_json "${LABEL_JSON_TRAIN}" \
  --task "${TASK}" \
  --output_csv "${TASK_CSV_DIR}/train_BM.csv"

echo "===== [4/5] 构建测试集任务 CSV ====="
python binary_class/build_binary_task_csv.py \
  --base_features_csv "${BASE_FEAT_DIR}/test_base_features.csv" \
  --label_json "${LABEL_JSON_TEST}" \
  --task "${TASK}" \
  --output_csv "${TASK_CSV_DIR}/test_BM.csv"

# ============================================================
# 阶段 3: 训练 + 评估
# ============================================================
echo "===== [5/5] 训练二分类模型 ====="
python -m binary_class.train_binary_task_resampled \
  --train_csv "${TASK_CSV_DIR}/train_BM.csv" \
  --test_csv "${TASK_CSV_DIR}/test_BM.csv" \
  --test_names BM_test \
  --save_dir "${MODEL_DIR}" \
  --eval_metric "${EVAL_METRIC}" \
  --model_set "${MODEL_SET}" \
  --time_limit "${TIME_LIMIT}" \
  --resample_strategy "${RESAMPLE_STRATEGY}" \
  --threshold "${THRESHOLD}" \
  --ece_bins "${ECE_BINS}" \
  --ci_bootstrap_iters "${CI_BOOTSTRAP_ITERS}" \
  --ci_level "${CI_LEVEL}" \
  --ci_seed "${CI_SEED}" \
  --seed "${SEED}"

echo ""
echo "===== 流水线完成 ====="
echo "特征文件: ${BASE_FEAT_DIR}/"
echo "任务 CSV: ${TASK_CSV_DIR}/"
echo "模型结果: ${MODEL_DIR}/"
echo "  - leaderboard.csv"
echo "  - test_results.csv"
echo "  - test_results_ci.csv"
