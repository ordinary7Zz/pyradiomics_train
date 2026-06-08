#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/FTCPTC_FangDai/FTCPTC"
TRAIN_CSV="./binary_class/outputs/task_csvs/FTCPTC_FangDai/train_FTCPTC.csv"
TARGET_FILENAME="FangDai/PTC/A_b17171017081122.png"
OUTPUT_DIR="./binary_class/single_image/FTCPTC"

python shap_analyze/analyze_single_image/analyze_single_image.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filename "$TARGET_FILENAME" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --task_name "FTC vs PTC" \
  --positive_class_name "FTC" \
  --negative_class_name "PTC" \
  --output_space "raw score" \
  --main_models LightGBM_BAG_L1 \
  --background_samples 500 \
  --top_features 5 \
  --skip_neural_net
