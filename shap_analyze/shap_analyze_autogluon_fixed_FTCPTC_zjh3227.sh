#!/usr/bin/env bash
PYTHON="${PYTHON:-/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
MODEL_DIR="./binary_class/new_code/runs/shap_full_ftcptc_zjh3227/ag_model"
TRAIN_CSV="./binary_class/new_code/runs/shap_full_ftcptc_zjh3227/train_task.csv"
FILENAMES_CSV="./binary_class/new_code/runs/shap_full_ftcptc_zjh3227/train_features.csv"
OUTPUT_DIR="./binary_class/shap_analysis_outputs/shap_full_ftcptc_zjh3227"

"$PYTHON" shap_analyze/shap_analyze_autogluon_fixed.py \
  --model_dir "$MODEL_DIR" \
  --train_csv "$TRAIN_CSV" \
  --filenames_csv "$FILENAMES_CSV" \
  --label "label" \
  --output_dir "$OUTPUT_DIR" \
  --task_name "FTC vs PTC" \
  --positive_class_name "FTC" \
  --negative_class_name "PTC" \
  --output_space "raw score" \
  --main_models CatBoost_BAG_L1 LightGBM_BAG_L1 \
  --plot_beeswarm_for CatBoost_BAG_L1 LightGBM_BAG_L1 \
  --background_samples 500 \
  --explain_samples 532 \
  --skip_neural_net \
  --plot_waterfall \
  --top_features 10 \
  --waterfall_samples 10
