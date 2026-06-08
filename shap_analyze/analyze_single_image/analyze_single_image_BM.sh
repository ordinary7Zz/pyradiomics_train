#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/BM_dataset3_predmask/BM"
TRAIN_CSV="./binary_class/outputs/task_csvs/BM_dataset3_predmask/test_BM.csv"
TARGET_FILENAME="TN3K_test_0040.jpg"
OUTPUT_DIR="./binary_class/single_image/BM"

python shap_analyze/analyze_single_image/analyze_single_image.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filename "$TARGET_FILENAME" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --task_name "Benign vs Malignant" \
  --positive_class_name "malignant" \
  --negative_class_name "benign" \
  --output_space "raw score" \
  --main_models LightGBM_BAG_L1 \
  --background_samples 500 \
  --top_features 5 \
  --skip_neural_net
