#!/usr/bin/env bash
# B9 AutoGluon only — LNM → FTC/PTC（DINO 见 Malignant_Cls_Model/b9/run_b9_*_dino_then_ag.sh）
set -euo pipefail

B9_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/Malignant_Cls_Model/b9"
export SKIP_DINO=1

echo "==> LNM (AutoGluon only)"
"${B9_DIR}/run_b9_lnm_dino_then_ag.sh"

echo "==> FTC/PTC (AutoGluon only)"
"${B9_DIR}/run_b9_ftcptc_dino_then_ag.sh"
