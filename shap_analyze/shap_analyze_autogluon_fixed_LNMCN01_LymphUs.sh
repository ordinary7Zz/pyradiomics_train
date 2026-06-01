#!/usr/bin/env bash
MODEL_DIR="./binary_class/outputs/models/LymphUs_fake_predmask"
TRAIN_CSV="./binary_class/outputs/task_csvs/LymphUs_fake_predmask/train_LNM_CN01.csv"
OUTPUT_DIR="./binary_class/shap_analysis_outputs/LymphUs_fake_predmask/LNM_CN01"

python shap_analyze/shap_analyze_autogluon_fixed.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --main_models LightGBM_BAG_L1 \
  --plot_beeswarm_for LightGBM_BAG_L1 \
  --background_samples 100 \
  --explain_samples 2000 \
  --skip_neural_net \
  --plot_waterfall \
  --top_features 10 \
  --waterfall_samples 10