#!/usr/bin/env bash
DATASET_NAME="dataset_3"

MODEL_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train/autogluon_model/predmask/dataset_3/autogluon_model_20260111_192856"
TEST_NAMES=(
  "TN3K"
  "ThyroidXL"
  "TN5K"
)
TEST_CSVS=(
  "./csv_data/new_test_with_predmask/TN3K_radiomics_2d_features.csv"
  "./csv_data/new_test_with_predmask/ThyroidXL_radiomics_2d_features.csv"
  "./csv_data/new_test_with_predmask/TN5K_radiomics_2d_features.csv"
)
OUT_CSV="./test_logs/predmask/${DATASET_NAME}/autogluon_model_$(date +%Y%m%d_%H%M%S)"

python test_autogluon_tabular.py \
  --model_dir "$MODEL_DIR" \
  --test_csv "${TEST_CSVS[@]}" \
  --test_names "${TEST_NAMES[@]}" \
  --out_csv "$OUT_CSV"