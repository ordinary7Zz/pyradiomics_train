#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:-}"
if [[ -z "$DATASET_NAME" ]]; then
  echo "Usage: bash scripts/run_mask_source_cross_experiments.sh <DATASET_NAME>"
  echo "Example: bash scripts/run_mask_source_cross_experiments.sh TN5K"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LABEL_COL="${LABEL_COL:-label}"
TASK_NAME="${TASK_NAME:-malignancy}"
MODEL_SET="${MODEL_SET:-tree_full}"
EVAL_METRIC="${EVAL_METRIC:-roc_auc}"
PRESETS="${PRESETS:-best_quality}"
TIME_LIMIT="${TIME_LIMIT:-600}"
SEED="${SEED:-42}"
FEATURE_ROOT="${FEATURE_ROOT:-$REPO_ROOT/csv_data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/cross_mask_runs/$DATASET_NAME/$RUN_ID}"
TRAIN_SUFFIX="${TRAIN_SUFFIX:-_train.csv}"
TEST_SUFFIX="${TEST_SUFFIX:-_test.csv}"

MASK_SOURCES=(gt gt_mild_perturb gt_moderate_perturb pred)
RESULT_CSVS=()
declare -A MODEL_DIRS

echo "Dataset: $DATASET_NAME"
echo "Feature root: $FEATURE_ROOT"
echo "Run root: $RUN_ROOT"
echo "Seed: $SEED"

mkdir -p "$RUN_ROOT/models" "$RUN_ROOT/results" "$RUN_ROOT/summary" "$RUN_ROOT/logs"

feature_csv_path() {
  local mask_source="$1"
  local suffix="$2"
  printf '%s/%s/%s%s' "$FEATURE_ROOT" "$mask_source" "$DATASET_NAME" "$suffix"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path"
    exit 1
  fi
}

for mask_source in "${MASK_SOURCES[@]}"; do
  require_file "$(feature_csv_path "$mask_source" "$TRAIN_SUFFIX")"
  require_file "$(feature_csv_path "$mask_source" "$TEST_SUFFIX")"
done

for train_mask in "${MASK_SOURCES[@]}"; do
  train_csv="$(feature_csv_path "$train_mask" "$TRAIN_SUFFIX")"
  matched_test_csv="$(feature_csv_path "$train_mask" "$TEST_SUFFIX")"
  model_dir="$RUN_ROOT/models/train_${train_mask}"
  log_path="$RUN_ROOT/logs/train_${train_mask}.log"

  echo "=== Training model: train_mask_source=$train_mask ==="
  python "$REPO_ROOT/train_autogluon_tabular.py" \
    --train_csv "$train_csv" \
    --test_csv "$matched_test_csv" \
    --test_names "$DATASET_NAME" \
    --label "$LABEL_COL" \
    --save_dir "$model_dir" \
    --model_set "$MODEL_SET" \
    --eval_metric "$EVAL_METRIC" \
    --presets "$PRESETS" \
    --time_limit "$TIME_LIMIT" \
    --seed "$SEED" | tee "$log_path"

  MODEL_DIRS["$train_mask"]="$model_dir"
done

for train_mask in "${MASK_SOURCES[@]}"; do
  model_dir="${MODEL_DIRS[$train_mask]}"
  train_feature_csv="$(feature_csv_path "$train_mask" "$TRAIN_SUFFIX")"

  for test_mask in "${MASK_SOURCES[@]}"; do
    test_csv="$(feature_csv_path "$test_mask" "$TEST_SUFFIX")"
    out_csv="$RUN_ROOT/results/train_${train_mask}__test_${test_mask}.csv"
    log_path="$RUN_ROOT/logs/test_train_${train_mask}__test_${test_mask}.log"

    echo "=== Evaluating model: train_mask_source=$train_mask test_mask_source=$test_mask ==="
    python "$REPO_ROOT/test_autogluon_tabular.py" \
      --model_dir "$model_dir" \
      --test_csv "$test_csv" \
      --test_names "$DATASET_NAME" \
      --label "$LABEL_COL" \
      --train_mask_source "$train_mask" \
      --test_mask_source "$test_mask" \
      --train_dataset "$DATASET_NAME" \
      --task_name "$TASK_NAME" \
      --feature_csv "$train_feature_csv" \
      --out_csv "$out_csv" | tee "$log_path"

    RESULT_CSVS+=("$out_csv")
  done
done

python "$REPO_ROOT/summarize_mask_source_results.py" \
  --results_csv "${RESULT_CSVS[@]}" \
  --out_csv "$RUN_ROOT/summary/all_results_long.csv" \
  --metric auroc \
  --matrix_out_csv "$RUN_ROOT/summary/auroc_4x4.csv"

for metric in auprc acc sensitivity specificity; do
  python "$REPO_ROOT/summarize_mask_source_results.py" \
    --results_csv "${RESULT_CSVS[@]}" \
    --out_csv "$RUN_ROOT/summary/all_results_long.csv" \
    --metric "$metric" \
    --matrix_out_csv "$RUN_ROOT/summary/${metric}_4x4.csv" >/dev/null
done

echo "Saved 16 result CSVs under: $RUN_ROOT/results"
echo "Saved long summary: $RUN_ROOT/summary/all_results_long.csv"
echo "Saved metric matrices under: $RUN_ROOT/summary"
