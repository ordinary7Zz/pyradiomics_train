#!/usr/bin/env bash
DATASET_NAME="dataset_3"

MODEL_DIR="./autogluon_model/predmask/${DATASET_NAME}/autogluon_model_20260111_192856"
TRAIN_CSV="./csv_data/train_with_predmask/${DATASET_NAME}_radiomics_2d_features.csv"
OUTPUT_DIR="${MODEL_DIR}/shap_analysis"

python shap_analyze_autogluon.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --background_samples 100 \
  --explain_samples 500 \
  --skip_neural_net

