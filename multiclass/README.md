# multiclass — TIRADS 五分类训练

本目录包含基于 AutoGluon 的 **TIRADS 五分类**（标签 1-5）训练代码，与 `binary_class/` 目录平行独立，不修改其任何文件。

## 文件说明

| 文件 | 用途 |
|------|------|
| `multiclass_metrics.py` | 多分类评估指标：逐类 Precision/Recall/F1/Specificity/AUROC/ECE、Macro/Weighted 聚合、Cohen's Kappa、Quadratic Weighted Kappa (QWK)、bootstrap 置信区间 |
| `build_multiclass_task_csv.py` | 从 base_features.csv + label_json 中提取 `tirads` 列，构建五分类训练/测试 CSV |
| `train_multiclass_task.py` | 基础版 AutoGluon 多分类训练（不做重采样） |
| `train_multiclass_task_resampled.py` | 带类别重采样的 AutoGluon 多分类训练（支持 oversample/undersample） |
| `build_patient_task_csv.py` | 将图像级 task CSV 聚合成病人级 CSV（均值聚合特征，校验同病人标签一致） |
| `run_all_multiclass_tasks.py` | 完整流水线：提取特征 → 构表 → 训练 → 评估 |

## 标签规范

- **TIRADS 标签范围**：1, 2, 3, 4, 5
- **无效标签（负样本）**：-1，会被自动过滤
- **label_json 中的列名**：`"tirads"`

## 评估指标

输出的 `test_results.csv` 包含以下指标：

### 聚合指标
| 指标 | 说明 |
|------|------|
| `accuracy` | 总体准确率 |
| `balanced_accuracy` | 平衡准确率（逐类 recall 的均值） |
| `kappa` | Cohen's Kappa |
| `qwk` | **Quadratic Weighted Kappa**（适合 TIRADS 有序分类，跨级错误惩罚更重） |
| `macro_f1` / `weighted_f1` | 宏平均/加权平均 F1 |
| `macro_auroc` | 宏平均 AUROC（OvR） |
| `ece` | Confidence-ECE（最大预测概率校准误差） |
| `macro_ece` / `weighted_ece` | 逐类 ECE 的宏平均/加权平均 |

### 逐类指标
每类 (class_1 ~ class_5) 均有：`precision`, `recall`, `specificity`, `f1`, `support`
每类 (class_1 ~ class_5) 均有：`auroc`, `ece`

## 使用示例

### 1. 提取基础特征（复用 binary_class 的 extract_base_radiomics.py）

```bash
python binary_class/extract_base_radiomics.py \
  --image_dir /path/to/train/images \
  --mask_dir /path/to/train/masks \
  --label_json /path/to/train_labels.json \
  --output_csv multiclass/outputs/base_features/train_base_features.csv \
  --skip_fail
```

### 2. 构建五分类 CSV

```bash
python multiclass/build_multiclass_task_csv.py \
  --base_features_csv multiclass/outputs/base_features/train_base_features.csv \
  --label_json /path/to/train_labels.json \
  --task tirads \
  --output_csv multiclass/outputs/task_csvs/train_tirads.csv
```

### 3. 训练（基础版）

```bash
python -m multiclass.train_multiclass_task \
  --train_csv multiclass/outputs/task_csvs/train_tirads.csv \
  --save_dir multiclass/outputs/models/tirads \
  --eval_metric balanced_accuracy \
  --time_limit 3600
```

### 4. 训练（重采样版，推荐）

```bash
python -m multiclass.train_multiclass_task_resampled \
  --train_csv multiclass/outputs/task_csvs/train_tirads.csv \
  --save_dir multiclass/outputs/models/tirads \
  --eval_metric balanced_accuracy \
  --model_set tree_full \
  --time_limit 3600 \
  --resample_strategy oversample \
  --resample_target max \
  --seed 42
```

### 5. 带测试集的训练

```bash
python -m multiclass.train_multiclass_task_resampled \
  --train_csv multiclass/outputs/task_csvs/train_tirads.csv \
  --test_csv multiclass/outputs/task_csvs/test_tirads.csv \
  --test_names tirads_test \
  --save_dir multiclass/outputs/models/tirads \
  --eval_metric balanced_accuracy \
  --model_set tree_full \
  --time_limit 3600 \
  --resample_strategy oversample \
  --resample_target max \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42
```

### 6. 完整流水线（一键运行）

```bash
python multiclass/run_all_multiclass_tasks.py \
  --train_image_dir /path/to/train/images \
  --train_mask_dir /path/to/train/masks \
  --train_label_json /path/to/train_labels.json \
  --test_image_dir /path/to/test/images \
  --test_mask_dir /path/to/test/masks \
  --test_label_json /path/to/test_labels.json \
  --work_dir multiclass/outputs \
  --tasks tirads \
  --training_mode resampled \
  --resample_strategy oversample \
  --resample_target max \
  --model_set tree_full \
  --time_limit 3600 \
  --eval_metric balanced_accuracy \
  --skip_fail
```

## 病人级训练

如果标签是病人级，可先聚合：

```bash
python multiclass/build_patient_task_csv.py \
  --input_csv multiclass/outputs/task_csvs/train_tirads.csv \
  --output_csv multiclass/outputs/task_csvs/patient_train_tirads.csv \
  --feature_agg mean
```

再用同样的训练脚本进行病人级训练。

## 输出文件

```
multiclass/outputs/
├── base_features/
│   ├── train_base_features.csv
│   └── test_base_features.csv
├── task_csvs/
│   ├── train_tirads.csv
│   └── test_tirads.csv
├── models/
│   └── tirads/
│       ├── leaderboard.csv
│       ├── test_results.csv
│       ├── test_results_ci.csv
│       └── class_balance_summary.csv
└── reports/
    └── run_summary.csv
```

## 指标关注建议

对于 TIRADS 有序五分类，建议重点关注的指标优先级：

1. **QWK (Quadratic Weighted Kappa)** — 最契合有序分类场景
2. **Weighted F1 / Macro F1** — 综合考虑类别不平衡
3. **Accuracy / Balanced Accuracy** — 基础参考
4. **Kappa** — 去除随机一致性的准确率
5. **逐类 Recall** — 检查模型在哪几个 TIRADS 级别表现较差
6. **ECE** — 校准度检查
