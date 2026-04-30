# 绘图脚本说明

这个目录集中放置当前项目中的绘图相关脚本和共享绘图辅助代码。

## ⚠️ 重要：SHAP分析是绘图的前置步骤

**所有绘图脚本的运行都依赖SHAP分析的结果。** 绘图前必须先完成SHAP分析！

### 完整工作流程
```
1. 特征提取（extract_radiomics_2d.py）
   → radiomics_features.csv

2. 模型训练（train_autogluon_tabular.py）
   → 训练好的AutoGluon模型

3. ⭐ SHAP分析（shap_analyze_autogluon_fixed.py）
   → *_shap_values.csv + shap_analysis_summary.txt

4. 绘制图表（本目录的各绘图脚本）
   → beeswarm、waterfall、global bar等可视化
```

### SHAP分析快速命令
```bash
# 进入shap_analyze目录查看详细说明
cd ../shap_analyze

# 运行SHAP分析（基础）
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv

# 运行SHAP分析并同时生成图表（推荐）
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv \
  --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3 \
  --plot_waterfall
```

**更详细的说明请参阅：[shap_analyze/README.md](../shap_analyze/README.md)**

---

## 目录概览
- `plotting_utils.py`：共享绘图辅助函数
- `plot_beeswarm_batch.py`：批量生成 beeswarm 图
- `plot_beeswarm_from_outputs.py`：基于单个 `*_shap_values.csv` 生成 beeswarm 图
- `plot_global_bar_from_txt.py`：从 `shap_analysis_summary.txt` 生成 global bar 图
- `generate_shap_plots.py`：生成补充图，包括 summary bar、waterfall、force、FP/FN 对比、TP/TN 对比

## 各脚本与SHAP分析的依赖关系

### 强制需要SHAP分析的脚本
| 脚本 | 依赖的SHAP输出 | 说明 |
|------|---|---|
| `plot_beeswarm_batch.py` | `*_shap_values.csv` | 必须先完成SHAP分析 |
| `plot_beeswarm_from_outputs.py` | `*_shap_values.csv` | 必须先完成SHAP分析 |
| `plot_global_bar_from_txt.py` | `shap_analysis_summary.txt` | 需要SHAP分析生成的summary文件 |
| `generate_shap_plots.py` | 完整SHAP分析结果 | 需要模型+特征+SHAP值 |

### SHAP分析内置的绘图功能
除了上述独立脚本，SHAP分析脚本本身也提供绘图功能：
- `shap_analyze_autogluon_fixed.py --plot_beeswarm_for XXX`：直接生成beeswarm图
- `shap_analyze_autogluon_fixed.py --plot_waterfall`：直接生成waterfall图

**推荐做法**：大多数情况下直接在SHAP分析时使用 `--plot_beeswarm_for` 和 `--plot_waterfall` 参数，一次性完成分析和绘图。

## 推荐使用顺序

### 方案A：SHAP分析时一次性生成所有图表（推荐）
这是最高效的方式，一次性完成SHAP计算和绘图：

```bash
# 第1步：运行SHAP分析并同时绘制所有需要的图表
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv \
  --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3 \
  --plot_waterfall \
  --waterfall_samples 3 \
  --top_features 20

# 完成！所有图表已生成，可在 <model_dir>/shap_analysis/ 中查看
```

### 方案B：SHAP分析与绘图分开（灵活，用于后期调整）
如果需要调整参数或添加新的图表：

```bash
# 第1步：先完成SHAP分析（生成*_shap_values.csv）
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv

# 第2步：然后按需要运行本目录的各绘图脚本
python plots/plot_global_bar_from_txt.py \
  --summary_txt ./shap_analysis_LightGBMXT_BAG_L1/shap_analysis_summary.txt \
  --topk 20

python plots/plot_beeswarm_from_outputs.py \
  --summary_txt ./shap_analysis_LightGBMXT_BAG_L1/shap_analysis_summary.txt \
  --shap_values_csv ./autogluon_model/shap_analysis/LightGBM_BAG_L1_shap_values.csv \
  --max_display 20

python plots/plot_beeswarm_batch.py \
  --summary_txt ./shap_analysis_LightGBMXT_BAG_L1/shap_analysis_summary.txt \
  --shap_dir ./autogluon_model/shap_analysis
```

### 流程对比
| 方案 | 何时使用 | 优点 | 缺点 |
|------|---------|------|------|
| **A（一体化）** | 首次分析、常规应用 | 快速、一次性、参数一致 | 修改参数需重新计算SHAP |
| **B（分开）** | 多次调整图表参数 | 灵活、可复用SHAP结果 | 步骤多、需要两次运行 |

---

## 输入文件说明

所有下列文件都由SHAP分析脚本（[shap_analyze_autogluon_fixed.py](../shap_analyze_autogluon_fixed.py)）生成，不需要手动准备：

| 文件 | 用途 | 生成方式 |
|------|------|--------|
| `shap_analysis_summary.txt` | 记录训练CSV路径、ensemble权重、各模型top特征 | SHAP分析自动生成 |
| `*_shap_values.csv` | 样本级SHAP值表，是beeswarm、waterfall等图表的核心数据 | SHAP分析自动生成 |
| `*_feature_values.csv` | 对应的原始特征值 | SHAP分析自动生成 |
| `*_predictions.csv` | 模型预测结果（预测标签、概率等） | SHAP分析自动生成 |

**重点**：确保SHAP分析已完成，才能使用这些文件进行绘图。

## 1. plot_beeswarm_batch.py

**依赖**：✅ 需要完成SHAP分析，生成 `*_shap_values.csv`

作用：批量读取一个目录下的 `*_shap_values.csv`，为每个文件生成 beeswarm 图。

最小示例：
```bash
python plots/plot_beeswarm_batch.py \
  --summary_txt ./shap_analysis_LightGBMXT_BAG_L1/shap_analysis_summary.txt \
  --shap_dir ./autogluon_model/gtmask/dataset_4/autogluon_model_20260107_233246/shap_analysis
```

常用参数：
- `--summary_txt`：`shap_analysis_summary.txt` 路径
- `--shap_dir`：存放 `*_shap_values.csv` 的目录
- `--max_display`：图中显示的 top 特征数
- `--out_dir`：输出目录
- `--pattern`：待处理文件匹配模式，默认 `*_shap_values.csv`

输出：
- `*_beeswarm_topK.png`
- `*_beeswarm_topK.svg`
- `*_beeswarm_topK.pdf`
- `*_feature_name_map.csv`

## 2. plot_beeswarm_from_outputs.py

**依赖**：✅ 需要完成SHAP分析，生成 `*_shap_values.csv`

作用：针对单个 `*_shap_values.csv` 生成一张 beeswarm 图。

最小示例：
```bash
python plots/plot_beeswarm_from_outputs.py \
  --summary_txt ./shap_analysis_LightGBMXT_BAG_L1/shap_analysis_summary.txt \
  --shap_values_csv ./autogluon_model/shap_analysis/LightGBM_BAG_L1_shap_values.csv \
  --out_png ./beeswarm.png
```

常用参数：
- `--summary_txt`：`shap_analysis_summary.txt` 路径
- `--shap_values_csv`：单个 SHAP 值 CSV 路径
- `--max_display`：图中显示的 top 特征数
- `--out_png`：输出 png 路径

输出：
- 单个 beeswarm png 文件

## 3. plot_global_bar_from_txt.py

**依赖**：✅ 需要完成SHAP分析，生成 `shap_analysis_summary.txt`

作用：从 `shap_analysis_summary.txt` 中读取各模型 top feature 和 ensemble weight，生成加权 global bar 图。

最小示例：
```bash
python plots/plot_global_bar_from_txt.py \
  --summary_txt ./autogluon_model/shap_analysis/shap_analysis_summary.txt \
  --topk 20
```

常用参数：
- `--summary_txt`：summary 文件路径
- `--topk`：显示前多少个特征
- `--out_dir`：输出目录
- `--prefix`：输出文件名前缀

输出：
- `global_bar_weighted_topK.png`
- `global_bar_weighted_topK.pdf`

## 4. generate_shap_plots.py

**依赖**：✅ 需要完成SHAP分析，生成SHAP值矩阵和相关数据

作用：生成补充型图像，适合做单独分析或早期结果整理。

包含的图：
- SHAP summary bar 图
- waterfall 图
- force 图
- FP vs FN 特征重要性对比图
- TP vs TN 特征重要性对比图

说明：
- 这个脚本依赖 `shap_analysis_path` 目录结构
- 当前入口仍是脚本内直接指定路径，不是统一 CLI 风格

## 5. SHAP分析脚本说明

[shap_analyze_autogluon_fixed.py](../shap_analyze_autogluon_fixed.py) 是完整SHAP分析的核心脚本，必须首先运行。

详细说明和参数配置见：[shap_analyze/README.md](../shap_analyze/README.md)

### SHAP分析的双重作用

#### 作用1：生成绘图数据（核心）
生成以下输出文件，供上述绘图脚本（1-4）使用：
- `*_shap_values.csv`：SHAP值矩阵
- `shap_analysis_summary.txt`：分析摘要和模型权重
- `*_feature_values.csv`：特征值矩阵

#### 作用2：内置绘图功能（可选）
分析脚本本身也支持直接生成图表，参数包括：
- `--plot_beeswarm_for <model_names>`：为指定模型生成 beeswarm 图
- `--plot_waterfall`：生成 waterfall 图
- `--waterfall_samples <N>`：控制每类样本绘制数量（默认3）
- `--sample_filename <filenames>`：为特定样本生成 waterfall 图
- `--top_features <K>`：控制 beeswarm/waterfall 显示的特征数（默认5）

### 完整示例
```bash
# SHAP分析 + 同步绘图（推荐一次性方案）
python ../shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/radiomics_features.csv \
  --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3 \
  --plot_waterfall \
  --top_features 20 \
  --waterfall_samples 3
```

## 输出文件说明
常见输出包括：
- `png`：常用位图
- `svg`：矢量图  
- `pdf`：论文或汇报可直接使用
- `csv`：特征名映射表、waterfall 样本映射表、SHAP值矩阵等
