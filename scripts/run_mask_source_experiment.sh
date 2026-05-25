#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/run_mask_source_experiment.sh extract_train gt dataset_1
#   bash scripts/run_mask_source_experiment.sh extract_train gt_mild_perturb dataset_1
#   bash scripts/run_mask_source_experiment.sh extract_test pred TN3K
#   bash scripts/run_mask_source_experiment.sh train gt dataset_1
#   bash scripts/run_mask_source_experiment.sh test gt dataset_1 /path/to/model_dir
#   bash scripts/run_mask_source_experiment.sh extract_all dataset_1

ACTION="${1:-}"
MASK_SOURCE="${2:-}"
DATASET_NAME="${3:-}"
MODEL_DIR="${4:-}"

if [[ -z "$ACTION" ]]; then
  echo "Missing ACTION"
  exit 1
fi

ROOT_DATA="/mnt/wangbd8/workspace/DataSets/ThyroidAgent"
ROOT_OUT="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train"
TASK="malignancy"
MASK_THRESHOLD=0
MASK_SUFFIX=""
SPACING_X=1.0
SPACING_Y=1.0
PERTURB_SEED=42
PARAMS_YAML="./radiomics_2d.yaml"

TRAIN_TEST_NAMES=("TN3K" "ThyroidXL" "TN5K")

is_train_dataset() {
  [[ "$1" == dataset_1 || "$1" == dataset_2 || "$1" == dataset_3 || "$1" == dataset_4 ]]
}

mask_feature_dir() {
  local mask_source="$1"
  printf "%s/csv_data/%s" "$ROOT_OUT" "$mask_source"
}

mask_model_dir() {
  local mask_source="$1"
  local dataset_name="$2"
  printf "%s/autogluon_model/%s/%s/autogluon_model_%s" "$ROOT_OUT" "$mask_source" "$dataset_name" "$(date +%Y%m%d_%H%M%S)"
}

mask_test_log_csv() {
  local mask_source="$1"
  local dataset_name="$2"
  printf "%s/test_logs/%s/%s/test_results_%s.csv" "$ROOT_OUT" "$mask_source" "$dataset_name" "$(date +%Y%m%d_%H%M%S)"
}

resolve_train_paths() {
  local mask_source="$1"
  local dataset_name="$2"

  IMAGE_DIR="$ROOT_DATA/train_val_test/Superimposed_multitask/${dataset_name}/train/images"
  LABEL_JSON="$ROOT_DATA/train_val_test/Superimposed_multitask/${dataset_name}/train/${dataset_name}_train_label.json"
  if [[ "$mask_source" == "pred" ]]; then
    MASK_DIR="$ROOT_DATA/train_val_test/Superimposed_multitask/pred_masks/${dataset_name}"
  else
    MASK_DIR="$ROOT_DATA/Superimposed_multitask/${dataset_name}/masks"
  fi
  OUT_CSV="$(mask_feature_dir "$mask_source")/${dataset_name}_radiomics_2d_features.csv"
}

resolve_test_paths() {
  local mask_source="$1"
  local dataset_name="$2"

  IMAGE_DIR="$ROOT_DATA/train_val_test/${dataset_name}/test/images"
  LABEL_JSON="$ROOT_DATA/train_val_test/${dataset_name}/test/${dataset_name}_test_label.json"
  if [[ "$mask_source" == "pred" ]]; then
    MASK_DIR="$ROOT_DATA/train_val_test/pred_test_masks/${dataset_name}"
  else
    MASK_DIR="$ROOT_DATA/Superimposed_multitask/test_${dataset_name}/masks"
  fi
  OUT_CSV="$(mask_feature_dir "$mask_source")/${dataset_name}_radiomics_2d_features.csv"
}

run_extract() {
  local split="$1"
  local mask_source="$2"
  local dataset_name="$3"

  if [[ "$split" == "train" ]]; then
    resolve_train_paths "$mask_source" "$dataset_name"
  else
    resolve_test_paths "$mask_source" "$dataset_name"
  fi

  python extract_radiomics_2d.py \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --label_json "$LABEL_JSON" \
    --task "$TASK" \
    --params "$PARAMS_YAML" \
    --output_csv "$OUT_CSV" \
    --mask_threshold "$MASK_THRESHOLD" \
    --mask_suffix "$MASK_SUFFIX" \
    --mask_source "$mask_source" \
    --perturb_seed "$PERTURB_SEED" \
    --spacing_x "$SPACING_X" \
    --spacing_y "$SPACING_Y" \
    --skip_fail
}

run_train() {
  local mask_source="$1"
  local dataset_name="$2"

  local train_csv="$(mask_feature_dir "$mask_source")/${dataset_name}_radiomics_2d_features.csv"
  local save_dir
  save_dir="$(mask_model_dir "$mask_source" "$dataset_name")"

  local test_csvs=()
  local test_name
  for test_name in "${TRAIN_TEST_NAMES[@]}"; do
    test_csvs+=("$(mask_feature_dir "$mask_source")/${test_name}_radiomics_2d_features.csv")
  done

  python train_autogluon_tabular.py \
    --train_csv "$train_csv" \
    --test_csv "${test_csvs[@]}" \
    --test_names "${TRAIN_TEST_NAMES[@]}" \
    --save_dir "$save_dir" \
    --eval_metric "roc_auc" \
    --presets "best_quality" \
    --time_limit 600 \
    --seed 42
}

run_test() {
  local mask_source="$1"
  local dataset_name="$2"
  local model_dir="$3"

  if [[ -z "$model_dir" ]]; then
    echo "MODEL_DIR is required for test action"
    exit 1
  fi

  local out_csv
  out_csv="$(mask_test_log_csv "$mask_source" "$dataset_name")"
  local test_csvs=()
  local test_name
  for test_name in "${TRAIN_TEST_NAMES[@]}"; do
    test_csvs+=("$(mask_feature_dir "$mask_source")/${test_name}_radiomics_2d_features.csv")
  done

  python test_autogluon_tabular.py \
    --model_dir "$model_dir" \
    --test_csv "${test_csvs[@]}" \
    --test_names "${TRAIN_TEST_NAMES[@]}" \
    --mask_source "$mask_source" \
    --train_dataset "$dataset_name" \
    --task_name "$TASK" \
    --feature_csv "$(mask_feature_dir "$mask_source")/${dataset_name}_radiomics_2d_features.csv" \
    --out_csv "$out_csv"
}

run_extract_all() {
  local dataset_name="$1"
  local mask_source
  for mask_source in gt gt_mild_perturb gt_moderate_perturb pred; do
    run_extract train "$mask_source" "$dataset_name"
  done
  local test_name
  for test_name in "${TRAIN_TEST_NAMES[@]}"; do
    for mask_source in gt gt_mild_perturb gt_moderate_perturb pred; do
      run_extract test "$mask_source" "$test_name"
    done
  done
}

case "$ACTION" in
  extract_train)
    is_train_dataset "$DATASET_NAME" || { echo "extract_train expects dataset_1..dataset_4"; exit 1; }
    run_extract train "$MASK_SOURCE" "$DATASET_NAME"
    ;;
  extract_test)
    run_extract test "$MASK_SOURCE" "$DATASET_NAME"
    ;;
  train)
    is_train_dataset "$DATASET_NAME" || { echo "train expects dataset_1..dataset_4"; exit 1; }
    run_train "$MASK_SOURCE" "$DATASET_NAME"
    ;;
  test)
    is_train_dataset "$DATASET_NAME" || { echo "test expects dataset_1..dataset_4"; exit 1; }
    run_test "$MASK_SOURCE" "$DATASET_NAME" "$MODEL_DIR"
    ;;
  extract_all)
    is_train_dataset "$MASK_SOURCE" || { echo "extract_all expects dataset name as second arg"; exit 1; }
    run_extract_all "$MASK_SOURCE"
    ;;
  *)
    echo "Unsupported ACTION: $ACTION"
    exit 1
    ;;
esac
