# binary_class 病人级训练与测试说明

## 适用场景

当标签是病人级，但当前样本还是图像级时，建议先把图像级 task CSV 聚合成病人级 CSV，再复用现有训练脚本完成训练与评估。

## 核心脚本

### `build_patient_task_csv.py`

作用：将图像级 task CSV 聚合成病人级 task CSV。

主要行为：

- 默认从 `filename` 中提取“年份/病人目录名”作为病人键
- 会校验同一病人组内 `label` 是否一致
- 默认对数值特征做 `mean` 聚合
- 新增 `image_count` 列，表示该病人的图像数量
- 输出结果可直接用于 `train_binary_task.py` 或 `train_binary_task_resampled.py`

## 推荐流程

### 第 1 步：构建训练集病人级 CSV

```bash
python binary_class/build_patient_task_csv.py \
  --input_csv binary_class/outputs/task_csvs/train_FTCPTC.csv \
  --output_csv binary_class/outputs/task_csvs/train_FTCPTC_patient.csv \
  --summary_csv binary_class/outputs/task_csvs/train_FTCPTC_patient_summary.csv \
  --mapping_csv binary_class/outputs/task_csvs/train_FTCPTC_patient_mapping.csv
```

### 第 2 步：构建测试集病人级 CSV

```bash
python binary_class/build_patient_task_csv.py \
  --input_csv binary_class/outputs/task_csvs/test_FTCPTC.csv \
  --output_csv binary_class/outputs/task_csvs/test_FTCPTC_patient.csv \
  --summary_csv binary_class/outputs/task_csvs/test_FTCPTC_patient_summary.csv \
  --mapping_csv binary_class/outputs/task_csvs/test_FTCPTC_patient_mapping.csv
```

### 第 3 步：做病人级训练与测试评估

```bash
python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/train_FTCPTC_patient.csv \
  --test_csv binary_class/outputs/task_csvs/test_FTCPTC_patient.csv \
  --test_names FTCPTC_test_patient \
  --save_dir binary_class/outputs/models/FTCPTC_patient_tree_fast \
  --eval_metric roc_auc \
  --model_set tree_fast \
  --time_limit 1800 \
  --target_class0_count -1 \
  --target_class1_count -1 \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42
```

## 主要输出

### 构表阶段

- `train_FTCPTC_patient.csv`：训练集病人级特征表
- `test_FTCPTC_patient.csv`：测试集病人级特征表
- `*_patient_summary.csv`：病人级聚合摘要
- `*_patient_mapping.csv`：图像到病人的映射关系

### 训练与测试阶段

- `binary_class/outputs/models/<task>/leaderboard.csv`
- `binary_class/outputs/models/<task>/test_results.csv`
- `binary_class/outputs/models/<task>/test_results_ci.csv`
- `binary_class/outputs/models/<task>/class_balance_summary.csv`（若使用重采样训练）

## 使用建议

- 如果病人级标签来自多个图像，请先确认同一病人内标签定义一致
- 聚合前先检查 `filename` 是否能稳定反映病人目录结构
- 默认均值聚合适合快速实验；如果后续需要更复杂的聚合策略，再单独扩展脚本
- 评估时仍建议重点看 `AUPRC`、`Recall`、`F1`、`Specificity`，不要只看 `AUROC`
