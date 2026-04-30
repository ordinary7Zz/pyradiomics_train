# SHAP分析运行指南

SHAP（SHapley Additive exPlanations）分析用于解释AutoGluon模型的预测决策。本目录提供了SHAP分析的完整运行流程和相关脚本。

## 📊 什么是SHAP分析？

SHAP分析计算每个特征对模型预测的贡献值（Shapley值），用于：
- 理解模型为什么做出某个预测
- 识别哪些特征最重要
- 发现特征与预测的关系

## 🔄 SHAP分析的必要性

**重点：绘制beeswarm图、waterfall图等可视化结果之前，必须先完成SHAP分析。**

```
模型训练完成
    ↓
SHAP分析（生成*_shap_values.csv）
    ↓
绘制可视化图表（beeswarm、waterfall等）
```

## 📋 输入数据

| 数据 | 说明 |
|------|------|
| **AutoGluon模型目录** | 包含 `predictor.pkl` 和 `logs/predictor_log.txt` |
| **训练CSV文件** | 用于提取背景样本和解释样本 |

## 🚀 快速开始

### 最小示例
```bash
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv
```

### 带绘图的完整示例
```bash
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv \
  --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3 \
  --plot_waterfall \
  --top_features 20
```

## 📝 常用参数说明

### 必需参数
- `--model_dir`：AutoGluon模型保存目录
- `--train_csv`：训练数据CSV路径

### 可选参数（SHAP计算）
- `--label`：标签列名，默认为 `label`
- `--output_dir`：输出目录，默认为 `<model_dir>/shap_analysis`
- `--background_samples`：背景样本数，默认100（用于加速SHAP计算）
- `--explain_samples`：解释样本数，默认500（或所有样本）
- `--skip_neural_net`：跳过神经网络模型加速计算
- `--main_models`：指定要分析的模型，默认自动检测

### 可选参数（绘图）
- `--plot_beeswarm_for`：为指定模型生成beeswarm图
  ```bash
  --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3
  ```
- `--plot_waterfall`：生成waterfall图
- `--waterfall_samples`：每类样本绘制数量，默认3
- `--sample_filename`：为特定文件名的样本生成waterfall图
  ```bash
  --sample_filename sample1.png sample2.png
  ```
- `--top_features`：waterfall/beeswarm显示的特征数，默认5

## 📂 输出文件结构

```
<output_dir>/  (default: <model_dir>/shap_analysis)
├── <ModelName>_shap_values.csv        # 样本级SHAP值（重要！）
├── <ModelName>_feature_values.csv     # 对应的特征值
├── <ModelName>_predictions.csv        # 模型预测结果
├── shap_analysis_summary.txt          # 分析摘要（包含训练CSV路径、模型权重等）
├── beeswarm/                          # beeswarm图输出
│   ├── <ModelName>_beeswarm_top20.png
│   ├── <ModelName>_beeswarm_top20.svg
│   ├── <ModelName>_beeswarm_top20.pdf
│   └── ...
└── waterfall/                         # waterfall图输出
    ├── <ModelName>_correct_sample_1.png
    ├── <ModelName>_incorrect_sample_1.png
    ├── waterfall_sample_images.csv
    └── ...
```

## 🔑 关键输出文件说明

### `*_shap_values.csv`
- **用途**：绘图脚本（beeswarm、waterfall等）的核心输入
- **内容**：每行是一个样本，每列是一个特征的SHAP值
- **大小**：(样本数 × 特征数)

### `shap_analysis_summary.txt`
- **用途**：记录分析元数据，供绘图脚本使用
- **内容**：
  - 训练CSV路径
  - 各模型的ensemble权重
  - 各模型的top特征排序

## 📊 分析步骤详解

### 步骤1：准备数据
```
读取训练CSV → 移除无效样本 → 分割背景样本和解释样本
```

### 步骤2：加载模型
```
加载AutoGluon predictor → 识别ensemble中的主模型
```

### 步骤3：计算SHAP值
```
对每个主模型计算SHAP值
├── 树模型（LightGBM、XGBoost等）：使用TreeExplainer（快）
└── 其他模型：使用KernelExplainer（慢但通用）
```

### 步骤4：保存结果
```
保存*_shap_values.csv → 保存summary.txt → （可选）生成可视化图表
```

## 💡 性能优化建议

| 场景 | 建议 |
|------|------|
| 特征很多（>500） | 降低 `--background_samples` 到50，或增加 `--explain_samples` |
| 模型很复杂 | 使用 `--skip_neural_net` 跳过NN模型 |
| 只关心某些模型 | 使用 `--main_models` 仅分析需要的模型 |
| 第一次运行 | 不用 `--plot_*` 参数，先生成SHAP值，再单独绘图 |

## 🔗 与绘图脚本的关系

SHAP分析生成的 `*_shap_values.csv` 是后续绘图的必要输入：

| 绘图脚本 | 需要的SHAP文件 | 说明 |
|---------|---------------|------|
| `plots/plot_beeswarm_batch.py` | `*_shap_values.csv` | 批量生成beeswarm图 |
| `plots/plot_beeswarm_from_outputs.py` | `*_shap_values.csv` | 单个模型beeswarm图 |
| `plots/plot_global_bar_from_txt.py` | `shap_analysis_summary.txt` | 全局特征重要性柱状图 |
| `plots/generate_shap_plots.py` | 模型 + 特征 + SHAP值 | 多种补充图表 |

详见 [plots/README.md](../plots/README.md)。

## ❓ 常见问题

### Q: SHAP分析需要多久？
A: 取决于样本数、特征数和背景样本数。通常：
- 500个样本 + 200个特征 + 100背景样本：5-10分钟
- 树模型比NN模型快10倍

### Q: 可以跳过某些模型吗？
A: 可以用 `--skip_neural_net` 跳过所有NN模型，或用 `--main_models` 指定特定模型。

### Q: SHAP值可以复用吗？
A: 可以。一旦生成了 `*_shap_values.csv`，就可以用不同参数重复运行绘图脚本。

### Q: 如何只为某个样本绘制waterfall图？
A: 使用 `--sample_filename` 参数，例如：
```bash
--sample_filename thyroid_001.png thyroid_002.png
```

## 🔍 调试技巧

### 检查SHAP计算是否成功
```bash
# 查看是否生成了*_shap_values.csv
ls <output_dir>/*_shap_values.csv

# 检查summary文件是否包含training CSV路径
cat <output_dir>/shap_analysis_summary.txt
```

### 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| `predictor.pkl not found` | 模型目录不正确 | 检查 `--model_dir` 路径 |
| `label column not found` | 标签列名错误 | 检查 `--label` 参数 |
| 内存不足 | 样本数太多 | 降低 `--explain_samples` |
| SHAP计算超时 | 特征太多 | 增加 `--background_samples` 或用特征筛选 |

## 📚 相关文件

- [根目录README](../README.md)：完整工作流程
- [plots/README.md](../plots/README.md)：绘图脚本使用说明
- [shap_analyze_autogluon_fixed.py](../shap_analyze_autogluon_fixed.py)：主要SHAP分析脚本
