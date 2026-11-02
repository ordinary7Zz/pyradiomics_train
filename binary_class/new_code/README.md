# B9 · AutoGluon 患者级五折（`binary_class/new_code`）

> **原则**：不修改 `extract_base_radiomics.py` / `build_binary_task_csv.py` / `train_binary_task_resampled.py`；本目录只做 **编排、packed 多根路径特征、OOF 汇总**。  
> **与 DINO 对齐**：同一 packed manifest、`patient_id`、**`cv_seed=42`、`n_splits=5`**（逻辑与 `Malignant_Cls_Model/b9/patient_cv.py` 一致）。

## 推荐架构

```text
Phase A  全 CV 池解析（packed_io）→ 患者折分（patient_cv）→ 写 fold_k/{train,val}_resolved.json
Phase B  每折（或全池一次）提 radiomics → fold_k/{train,val}_features.csv
Phase C  合并 label → fold_k/{train,val}_task.csv（列 label=0/1）
Phase D  子进程：python -m binary_class.train_binary_task_resampled
         --train_csv train_task.csv --test_csv val_task.csv   # val 仅作 OOF 评估，不参与 fit 重采样
Phase E  TabularPredictor 读 val 概率 → 患者 mean → 五折拼 oof_predictions.json / cv_summary.json
```

### 为何分「resolved + 新提取脚本」

现有 `extract_base_radiomics.py` 假定 **单一** `--image_dir` + `--mask_dir`。  
LNM 的 CV 池含 **ZJH + LymphUs C1** 两套根目录，因此在 **`extract_features_resolved.py`** 中按 `image_path`/`mask_path` 逐行提取（内部复用 `extract_base_radiomics._extract_one`，**不改原文件**）。

### 特征可否只算一次

- **推荐（快）**：对 **全 4753/4756** 张图提一次 `cache/all_features.parquet`，各折用 `filename` **inner join** 切 train/val。  
- **严格（慢）**：每折仅用训练折图 fit 的 mask 提特征（mask 来自 DINO 折内 seg 时再用此模式）。

当前默认：**固定 pred mask + 全池特征缓存**（与历史 `my_run.md` 一致）；`methods_snapshot.json` 记录。

### 与 DINO `b9` 折对齐

任选其一：

1. **独立重算**（本目录 `patient_cv`，应与 DINO 一致若 manifest/seed 相同）。  
2. **强制对齐**：`--fold_assignments` 指向 DINO 产物  
   `Malignant_Cls_Model/b9/runs/lnm_cv/fold_splits.json`（由 `export_fold_splits.py` 生成，见下）。

## 目录

| 文件 | 作用 |
|------|------|
| `task_specs.py` | `lnm` / `ftcptc` 的 packed 路径、label_key、AG 超参默认 |
| `packed_io.py` | manifest → `SampleRecord`（与 DINO b9 同思路） |
| `patient_cv.py` | 患者级 StratifiedKFold |
| `export_fold_splits.py` | 写 `fold_splits.json` + 每折 `*_resolved.json` |
| `extract_features_resolved.py` | 从 resolved JSON 批量提特征 CSV |
| `build_task_csv.py` | features ⋈ label_key → `label` 列（等价 build_binary_task_csv） |
| `infer_val_probs.py` | 加载 AG 目录，对 val_task.csv 出 prob |
| `aggregate_oof.py` | 患者聚合 + bootstrap CI |
| `run_patient_cv.py` | 主入口 |
| `run_b9_ag_lnm_then_ftcptc.sh` | 顺序跑 LNM + FTC（可选） |

## 用法

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train

# 干跑：只导出折分与空配置
python -m binary_class.new_code.run_patient_cv \
  --task lnm --output_dir binary_class/new_code/runs/lnm_ag_cv --dry_run

# 满跑（特征缓存 + 5 折 AG）
python -m binary_class.new_code.run_patient_cv \
  --task lnm \
  --output_dir binary_class/new_code/runs/lnm_ag_cv \
  --cv_seed 42 --n_splits 5 \
  --time_limit 3600 --model_set tree_full
```

## 交付物

```text
runs/lnm_ag_cv/
  cache/all_features.csv          # 可选全池缓存
  fold_splits.json
  fold_00/
    train_resolved.json
    val_resolved.json
    train_task.csv / val_task.csv
    ag_model/                     # TabularPredictor
    val_probs.json
  oof_predictions.json
  cv_summary.json
  ag_train_config.json            # 记录 resample / model_set / seed
```

## 后续（全 ThyroidXAgent）

DINO OOF + AG OOF + 固定路由规则在 `ThyroidCascadeAgent` 合并；本目录 **只负责 AG 五折**。
