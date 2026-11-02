#!/usr/bin/env bash
# Legacy FangDai 模型 + 全特征 task 的 SHAP（默认写入 shap_analysis_outputs/FTCPTC_FangDai）。
# run_ftcptc_malignant_filtered.sh 会 export SHAP_OUTPUT_DIR=.../ftcptc_malignant_filtered，勿覆盖本目录。
PYTHON="${PYTHON:-/mnt/wangbd8/workspace/envs/thyroid_infer/bin/python}"
MODEL_DIR="${SHAP_MODEL_DIR:-./binary_class/outputs/models/FTCPTC_FangDai/FTCPTC}"
TRAIN_CSV="${SHAP_TRAIN_CSV:-./binary_class/outputs/task_csvs/FTCPTC_FangDai/train_FTCPTC.csv}"
FILENAMES_CSV="${SHAP_FILENAMES_CSV:-./binary_class/outputs/base_features/FTCPTC_FangDai/train_base_features.csv}"
OUTPUT_DIR="${SHAP_OUTPUT_DIR:-./binary_class/shap_analysis_outputs/FTCPTC_FangDai}"

EXTRA_MAIN=()
EXTRA_BEESWARM=()
if [[ -n "${SHAP_MAIN_MODELS:-}" ]]; then
  read -ra EXTRA_MAIN <<< "${SHAP_MAIN_MODELS}"
fi
if [[ -n "${SHAP_PLOT_BEESWARM_FOR:-}" ]]; then
  read -ra EXTRA_BEESWARM <<< "${SHAP_PLOT_BEESWARM_FOR}"
fi

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
  ${EXTRA_MAIN:+--main_models "${EXTRA_MAIN[@]}"} \
  ${EXTRA_BEESWARM:+--plot_beeswarm_for "${EXTRA_BEESWARM[@]}"} \
  --background_samples 500 \
  --explain_samples 2000 \
  --skip_neural_net \
  --plot_waterfall \
  --top_features 10 \
  --waterfall_samples 10
