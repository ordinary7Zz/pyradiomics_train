#!/usr/bin/env bash
DATASET_NAME="dataset_3"

MODEL_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train/autogluon_model/gtmask/dataset_3/autogluon_model_20260106_215755"
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
OUT_CSV="./test_logs/${DATASET_NAME}/autogluon_model_$(date +%Y%m%d_%H%M%S)"

python test_autogluon_tabular.py \
  --model_dir "$MODEL_DIR" \
  --test_csv "${TEST_CSVS[@]}" \
  --test_names "${TEST_NAMES[@]}" \
  --out_csv "$OUT_CSV"