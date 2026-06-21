#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/BM_dataset3_predmask/BM"
TRAIN_CSV="./binary_class/outputs/task_csvs/BM_500_predmask/train_BM.csv"
FILENAME_LIST="./BM_any_doctor_wrong_filename_list.txt"
OUTPUT_DIR="./binary_class/single_image/BM_171_doctor_wrong"
FEATURE_LABEL_LANG="${FEATURE_LABEL_LANG:-cn}"

python shap_analyze/analyze_single_image/analyze_single_image.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filename_list "$FILENAME_LIST" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --task_name "Benign vs Malignant" \
  --positive_class_name "malignant" \
  --negative_class_name "benign" \
  --output_space "raw score" \
  --feature_label_lang "$FEATURE_LABEL_LANG" \
  --main_models LightGBM_BAG_L1 \
  --background_samples 500 \
  --top_features 10 \
  --skip_neural_net
