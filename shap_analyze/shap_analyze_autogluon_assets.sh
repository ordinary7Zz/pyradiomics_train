#!/usr/bin/env bash
DATASET_NAME="dataset_4"

MODEL_DIR="./autogluon_model/gtmask/${DATASET_NAME}/autogluon_model_20260107_233246"
TRAIN_CSV="./csv_data/train_with_gtmask/${DATASET_NAME}_radiomics_2d_features.csv"
OUTPUT_DIR="./shap_analysis"

python shap_analyze/shap_analyze_autogluon_assets.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --label "label" \
  --background_samples 100 \
  --explain_samples 500 \
  --threshold 0.5 \
  --kernel_nsamples 100