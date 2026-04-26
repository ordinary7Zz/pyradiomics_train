#!/usr/bin/env bash

python plot_beeswarm_batch.py \
  --summary_txt ./shap_analysis_LightGBMXT_BAG_L1/shap_analysis_summary.txt \
  --shap_dir ./autogluon_model/gtmask/dataset_4/autogluon_model_20260107_233246/shap_analysis \
  --max_display 10 \
  --out_dir ./beeswarm_plots