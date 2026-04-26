#!/usr/bin/env bash
DATASET_NAME="dataset_4"

MODEL_DIR="./autogluon_model/gtmask/${DATASET_NAME}/autogluon_model_20260107_233246"
TRAIN_CSV="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/outputs/csv_data/TN3K_radiomics_2d_with_dice.csv"
OUTPUT_DIR="./shap_analysis_LightGBMXT_BAG_L1"

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
  --waterfall_samples 10 \
  --sample_filename "TN3K_test_0306.jpg" "TN3K_test_0332.jpg" "TN3K_test_0586.jpg" "TN3K_test_0231.jpg" "TN3K_test_0606.jpg" "TN3K_test_0360.jpg"

