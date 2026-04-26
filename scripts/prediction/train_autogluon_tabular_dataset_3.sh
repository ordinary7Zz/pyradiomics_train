#!/usr/bin/env bash
DATASET_NAME="dataset_3"

TRAIN_CSV="./csv_data/new_train_with_predmask/${DATASET_NAME}_radiomics_2d_features.csv"
TEST_NAMES=(
  "TN3K"
  "ThyroidXL"
  "TN5K"
)
TEST_CSVS=(
  "./csv_data/test_with_gtmask/TN3K_radiomics_2d_features.csv"
  "./csv_data/test_with_gtmask/ThyroidXL_radiomics_2d_features.csv"
  "./csv_data/test_with_gtmask/TN5K_radiomics_2d_features.csv"
)
SAVE_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train/autogluon_model/predmask/${DATASET_NAME}/autogluon_model_$(date +%Y%m%d_%H%M%S)"

python train_autogluon_tabular.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "${TEST_CSVS[@]}" \
  --test_names "${TEST_NAMES[@]}" \
  --save_dir "$SAVE_DIR" \
  --eval_metric "roc_auc" \
  --presets "best_quality" \
  --time_limit 600 \
  --seed 42