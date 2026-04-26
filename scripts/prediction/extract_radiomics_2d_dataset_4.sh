#!/usr/bin/env bash
set -euo pipefail

Train_DATASET="dataset_4"

IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/${Train_DATASET}/train/images"
MASK_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/pred_masks/${Train_DATASET}"
LABEL_JSON="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/${Train_DATASET}/train/${Train_DATASET}_train_label.json"
PARAMS_YAML="./radiomics_2d.yaml"
OUT_CSV="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train/csv_data/new_train_with_predmask/${Train_DATASET}_radiomics_2d_features.csv"

TASK=malignancy          # malignancy 或 tirads
MASK_THRESHOLD=0
MASK_SUFFIX=""
SPACING_X=1.0
SPACING_Y=1.0

python extract_radiomics_2d.py \
  --image_dir "$IMAGE_DIR" \
  --mask_dir "$MASK_DIR" \
  --label_json "$LABEL_JSON" \
  --task "$TASK" \
  --params "$PARAMS_YAML" \
  --output_csv "$OUT_CSV" \
  --mask_threshold "$MASK_THRESHOLD" \
  --mask_suffix "$MASK_SUFFIX" \
  --spacing_x "$SPACING_X" \
  --spacing_y "$SPACING_Y" \
  --skip_fail
