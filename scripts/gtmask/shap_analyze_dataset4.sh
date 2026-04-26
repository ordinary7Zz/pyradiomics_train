#!/usr/bin/env bash
DATASET_NAME="dataset_4"

MODEL_DIR="./autogluon_model/gtmask/${DATASET_NAME}/autogluon_model_20260107_233246"
TRAIN_CSV="./csv_data/train_with_gtmask/${DATASET_NAME}_radiomics_2d_features.csv"
OUTPUT_DIR="./shap_analysis_LightGBMXT_BAG_L1"

python shap_analyze_autogluon.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --main_models LightGBMXT_BAG_L1
  --background_samples 100 \
  --explain_samples 500 \
  --skip_neural_net

