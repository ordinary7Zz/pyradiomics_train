#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/BM_dataset3/BM"
TRAIN_CSV="./binary_class/outputs/task_csvs/BM_dataset3/test_BM.csv"
TARGET_FILENAME="ThyroidXL_test_00003189_31AD2E2C_1.png"
OUTPUT_DIR="./binary_class/single_image"

python shap_analyze/analyze_single_image/analyze_single_image.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filename "$TARGET_FILENAME" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --main_models LightGBM_BAG_L1 \
  --background_samples 100 \
  --top_features 5 \
  --skip_neural_net
