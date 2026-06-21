#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="./binary_class/outputs/models/BM_dataset3_predmask/BM"
TRAIN_CSV="./binary_class/outputs/task_csvs/BM_500_predmask/train_BM.csv"
OUTPUT_DIR="./binary_class/single_image/BM_any_doctor_wrong"

run_single_case() {
  local target_filename="$1"

  python shap_analyze/analyze_single_image/analyze_single_image.py \
    --model_dir "$MODEL_DIR" \
    --train_csv "$TRAIN_CSV" \
    --filename "$target_filename" \
    --label "label" \
    --output_dir "$OUTPUT_DIR" \
    --task_name "Benign vs Malignant" \
    --positive_class_name "malignant" \
    --negative_class_name "benign" \
    --output_space "raw score" \
    --main_models LightGBM_BAG_L1 \
    --background_samples 500 \
    --top_features 10 \
    --skip_neural_net
}

run_group() {
  local group_name="$1"
  shift

  echo "Running group: ${group_name}"
  for target_filename in "$@"; do
    echo "  -> ${target_filename}"
    run_single_case "$target_filename"
  done
}

# Benign good mask
benign_good_masks=(
  "ThyroidXL_train_00002730_C9690598_1.png"
)

# Benign bad mask
benign_bad_masks=(
  "TN3K_test_0040.jpg"
)

# Malignant good mask
malignant_good_masks=(
  "TN5K_test_003323.jpg"
)

# Malignant bad mask
malignant_bad_masks=(
  "ThyroidXL_test_00001873_6923593C_2.png"
  "ThyroidXL_test_00001978_DC398883_0.png"
  "ThyroidXL_test_00001378_BA5E9CC4_0.png"
  "ThyroidXL_test_00003932_89B4CFAD_1.png"
  "TN3K_test_0586.jpg"
  "ThyroidXL_test_00002838_EDBD208B_2.png"
  "ThyroidXL_test_00002838_1E638EAB_1.png"
  "ThyroidXL_test_00002755_F79615B3_0.png"
  "ThyroidXL_test_00002838_A9C56A4B_0.png"
)

run_group "Benign good mask" "${benign_good_masks[@]}"
run_group "Benign bad mask" "${benign_bad_masks[@]}"
run_group "Malignant good mask" "${malignant_good_masks[@]}"
run_group "Malignant bad mask" "${malignant_bad_masks[@]}"
