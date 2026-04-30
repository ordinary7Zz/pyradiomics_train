#!/usr/bin/env bash
MODEL_DIR="./binary_class/outputs/models/LNM_CN01"
TRAIN_CSV="./binary_class/outputs/base_features/train_base_features.csv"
OUTPUT_DIR="./binary_class/shap_analysis_outputs"

python shap_analyze_autogluon_fixed.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --main_models LightGBMXT_BAG_L1 \
  --plot_beeswarm_for LightGBMXT_BAG_L1 \
  --background_samples 100 \
  --explain_samples 2000 \
  --skip_neural_net \
  --plot_waterfall \
  --top_features 5 \
  --waterfall_samples 10

