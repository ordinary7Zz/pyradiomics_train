#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/BM_dataset3/BM"
TRAIN_CSV="./binary_class/outputs/task_csvs/BM_dataset3/train_BM.csv"
TARGET_FILENAME="ThyroidXL_train_00003284_1F60E8DF_3.png"
OUTPUT_DIR="./binary_class/shap_analysis_outputs/single_image"

python shap_analyze/analyze_single_image/analyze_single_image.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filename "$TARGET_FILENAME" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --main_models LightGBM_BAG_L1 \
  --background_samples 100 \
  --top_features 10 \
  --skip_neural_net
