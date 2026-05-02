#!/usr/bin/env bash
MODEL_DIR="./autogluon_model/gtmask/dataset_3/autogluon_model_20260106_215755"
TRAIN_CSV="./csv_data/train_with_gtmask/dataset_3_radiomics_2d_features.csv"
OUTPUT_DIR="./shap_analysis_outputs/Malignancy_Cls"

python shap_analyze/shap_analyze_autogluon_fixed.py \
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