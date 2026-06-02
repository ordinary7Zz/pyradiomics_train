#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/LymphUs_fake_predmask"
TRAIN_CSV="./binary_class/outputs/task_csvs/LymphUs_fake_predmask/train_LNM_CN01.csv"
TARGET_FILENAME="22_Benign_center1.png"
OUTPUT_DIR="./binary_class/single_image/LNM_CN01"

python shap_analyze/analyze_single_image/analyze_single_image.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filename "$TARGET_FILENAME" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --main_models LightGBM_BAG_L1 \
  --background_samples 500 \
  --top_features 5 \
  --skip_neural_net
