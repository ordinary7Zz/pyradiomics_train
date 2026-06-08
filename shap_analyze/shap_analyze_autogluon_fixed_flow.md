# `shap_analyze_autogluon_fixed.py` 的 SHAP 分析全流程说明

本文基于 `shap_analyze/shap_analyze_autogluon_fixed.py` 及其依赖函数整理，说明这套 AutoGluon SHAP 分析脚本从输入到输出的完整执行链路。

## 1. 这个脚本的职责

这个脚本的目标是：

1. 加载 AutoGluon `TabularPredictor` 模型。
2. 读取训练 CSV，构造 SHAP 的背景样本和解释样本。
3. 自动识别要分析的主模型。
4. 针对每个主模型计算 SHAP 值。
5. 保存样本级 SHAP 结果和特征重要性结果。
6. 可选生成 beeswarm、waterfall 和紧凑版局部条形图。
7. 汇总 ensemble 权重并生成加权 ensemble 的 SHAP 结果。
8. 写出分析摘要，供后续绘图或排查使用。

---

## 2. 输入参数

### 必需参数

- `--model_dir`
  - AutoGluon 模型目录，里面通常包含 `predictor.pkl` 和 `logs/predictor_log.txt`。
- `--train_csv`
  - 训练数据 CSV，用来构建背景样本和解释样本。

### 常用可选参数

- `--label`
  - 标签列名，默认 `label`。
- `--output_dir`
  - 输出目录，默认是 `<model_dir>/shap_analysis`。
- `--background_samples`
  - 背景样本数，默认 100。
- `--explain_samples`
  - 解释样本数；不传时，默认最多解释 500 条样本。
- `--skip_neural_net`
  - 跳过神经网络模型。
- `--main_models`
  - 手工指定要分析的主模型列表。
- `--plot_beeswarm_for`
  - 指定哪些模型额外画 beeswarm 图。
- `--plot_waterfall`
  - 是否生成 waterfall 图和紧凑局部条形图。
- `--sample_filename`
  - 指定某些样本文件名，额外生成对应 waterfall 图。
- `--task_name` / `--positive_class_name` / `--negative_class_name` / `--output_space`
  - 主要用于图例、坐标轴和论文风格输出。

---

## 3. 全流程

### 第 1 步：加载模型和训练数据

脚本启动后会先：

1. 通过 `TabularPredictor.load(args.model_dir)` 加载 AutoGluon predictor。
2. 读取 `train_csv`。
3. 调用 `prepare_df(raw_df, args.label)` 做基础清洗：
   - 删除 `image_path`、`mask_path`、`filename` 这些列（如果存在）。
   - 丢弃 `label == -1` 的样本。
   - 将标签列转成 `int`。

这一步的作用是把训练数据整理成可以直接喂给 SHAP 的特征表。

---

### 第 2 步：提取样本 ID

脚本会尝试从训练 CSV 中找到样本标识列：

- 优先用 `filename`
- 如果没有，则用 `image_path`

如果使用的是 `image_path`，会取 basename，方便后续 waterfall 图按文件名匹配样本。

这个 ID 主要用于：

- 记录 waterfall 图对应的是哪一张图
- 支持 `--sample_filename` 精确选图

---

### 第 3 步：确定要分析的主模型

主模型选择逻辑是：

1. 如果传了 `--main_models`，就直接使用用户指定列表。
2. 否则先读取 `logs/predictor_log.txt`，尝试提取 ensemble weights。
3. 如果日志里能拿到 ensemble 权重，就把这些权重对应的模型作为主模型。
4. 如果日志里拿不到，再尝试从 leaderboard 和 ensemble 模型信息里推断。
5. 最后仍然失败时，回退到 leaderboard 前几个非 ensemble 模型。

所以这里的目标不是分析所有模型，而是优先分析 ensemble 里真正起作用的主子模型。

---

### 第 4 步：构造背景样本和解释样本

这是 SHAP 计算里最关键的采样步骤。

#### 4.1 背景样本 `X_background`

- 先固定随机种子 `42`。
- 如果训练集大于 `--background_samples`，就不放回随机抽样。
- 否则直接使用全部训练样本。

背景样本的作用是给 SHAP 提供参考分布。

#### 4.2 解释样本 `X_explain`

如果用户传了 `--explain_samples`：

- 最多解释这么多样本。
- 若样本数少于全量训练集，则用 `train_test_split(..., stratify=y_train, random_state=42)` 做分层抽样。

如果没有传：

- 训练集不大于 500，就全量解释。
- 训练集大于 500，就默认最多解释 500 条，并使用分层抽样。

同时脚本也会同步保存对应的 `y_explain` 和样本 ID，后面用于 waterfall 图。

---

### 第 5 步：为每个主模型计算 SHAP

脚本对 `main_models` 逐个循环处理，每个模型都会走 `compute_shap_for_model(...)`。

#### 5.1 加载模型

`compute_shap_for_model` 会先通过 `load_autogluon_model` 加载单个模型：

- 优先尝试 `predictor._trainer.load_model(model_name)`
- 如果不行，再尝试通过 `model_info` / `load_pkl` 兜底加载

如果模型无法加载，会跳过该模型。

#### 5.2 选择 SHAP 计算方式

脚本会根据模型类型选择不同策略：

- **BAG 树模型**：尝试提取 bag 里的 tree model，再用 `TreeExplainer`
- **普通树模型**：为了稳妥，直接用 `KernelExplainer`
- **其他模型**：统一使用 `KernelExplainer`

这里的判断来自：

- `is_bag_model(model_name)`
- `is_tree_model(model_name)`

#### 5.3 BAG 树模型的特殊处理

对于 `_BAG_` 树模型，脚本会：

1. 调用 `get_tree_model_from_bag(model, model_name, predictor)` 提取底层树模型。
2. 调用 predictor 的特征处理流程，把 `X_background` 和 `X_explain` 先做预处理。
3. 检查预处理后的特征数是否和树模型 `num_feature()` 对得上。
4. 如果对不上，就退回 `KernelExplainer`。
5. 如果对得上，就用 `shap.TreeExplainer(tree_model)` 计算 SHAP。

这个分支的目的是：**尽量用更快的 TreeExplainer，但一旦预处理或特征维度不匹配，就自动回退到通用方案。**

#### 5.4 KernelExplainer 的处理方式

`KernelExplainer` 分支会包装一个 `model_wrapper(X)`：

- 把数组转回 DataFrame，保证列名和训练特征一致。
- 调用 `model.predict_proba(X_df)`。
- 统一抽取正类概率（label=1）。

然后：

- `shap.KernelExplainer(model_wrapper, X_background.values)`
- `explainer.shap_values(X_explain.values, nsamples=100)`

因为 KernelExplainer 比较慢，所以这里只抽 100 个 nsamples。

#### 5.5 输出 SHAP 值

每个模型都会返回：

- `shap_values`：`numpy.ndarray`
- `shap_df`：`DataFrame`

其中 `shap_df` 的行索引对齐 `X_explain.index`，列名就是特征名。

---

### 第 6 步：保存单模型结果

每个模型都会保存两类结果：

1. `*_shap_values.csv`
   - 样本级 SHAP 值
   - 是后续绘图的核心输入
2. `*_feature_importance.csv`
   - 对 `abs(SHAP)` 求均值后排序
   - 用来表示全局特征重要性

脚本还会把每个模型的：

- top 20 特征
- mean absolute SHAP 总和

记录到内存里的 `shap_summary`，最后统一写进摘要文件。

---

### 第 7 步：可选生成 beeswarm 图

如果传了 `--plot_beeswarm_for` 且当前模型名在列表中，脚本会：

1. 检查 `shap` 和 `matplotlib` 是否安装。
2. 调用 `save_beeswarm_plot(...)`。
3. 输出到 `output_dir/beeswarm/`。

这里会同时生成：

- PNG 图
- SVG 图
- 可选 textless SVG

beeswarm 图展示的是某个模型里最重要的若干特征在全体样本上的 SHAP 分布。

---

### 第 8 步：按 ensemble 权重聚合 SHAP

脚本会再次读取 `logs/predictor_log.txt`，提取 ensemble weights。

如果：

- 成功拿到了权重
- 且前面对应模型的 SHAP 结果都存在

就会做加权聚合：

1. 按权重对每个模型的 `shap_values` 乘权重。
2. 对所有模型按维度求和。
3. 保存为：
   - `WeightedEnsemble_L3_shap_values.csv`
   - `WeightedEnsemble_L3_feature_importance.csv`

如果某个模型的 SHAP shape 和其他模型不一致，会跳过该模型，避免错误聚合。

---

### 第 9 步：写分析摘要

脚本会输出 `shap_analysis_summary.txt`，里面包含：

- 模型目录
- 训练 CSV 路径
- background 样本数
- explained 样本数
- 分析了多少个主模型
- ensemble weights（如果有）
- 每个模型的 top 10 特征
- 每个模型的 mean absolute SHAP 总和

这个文件适合作为后续绘图脚本的元信息来源。

---

### 第 10 步：可选生成 waterfall 图和紧凑局部条形图

如果传了 `--plot_waterfall`，脚本会调用 `plot_waterfall_samples(...)`。

这个函数会：

1. 对 `X_explain` 计算预测标签 `predict()` 和预测概率 `predict_proba()`。
2. 按真实标签和预测结果，把样本分成：
   - best
   - medium
   - worst
   - correct
3. 默认尽量在正类和负类之间做平衡抽样。
4. 如果传了 `--sample_filename`，还会按文件名额外找指定样本。
5. 对每个选中的样本：
   - 取该样本 SHAP 值绝对值最大的前 `n_top_features`
   - 画 waterfall 图
   - 画紧凑版局部 SHAP 条形图
6. 保存到：
   - `output_dir/waterfall/`
   - `output_dir/compact_shap_bar/`

如果能拿到样本 ID，还会额外写出：

- `waterfall_sample_images.csv`

用于记录每张图对应的样本名称、真实标签、预测标签和概率。

---

## 4. 输出目录结构

默认输出到：`<model_dir>/shap_analysis`

典型结构如下：

```text
shap_analysis/
├── <ModelName>_shap_values.csv
├── <ModelName>_feature_importance.csv
├── WeightedEnsemble_L3_shap_values.csv
├── WeightedEnsemble_L3_feature_importance.csv
├── shap_analysis_summary.txt
├── beeswarm/
│   ├── <ModelName>_beeswarm.png
│   ├── <ModelName>_beeswarm.svg
│   └── ...
├── waterfall/
│   ├── <ModelName>_waterfall_best_1.png
│   ├── <ModelName>_waterfall_correct_1.png
│   ├── <ModelName>_waterfall_sample_1.png
│   └── ...
└── compact_shap_bar/
    ├── <ModelName>_compact_shap_bar_best_1.png
    ├── <ModelName>_compact_shap_bar_correct_1.png
    └── ...
```

---

## 5. 这个流程的几个关键点

### 5.1 SHAP 分析是后续绘图的前置步骤

先生成 `*_shap_values.csv`，再做 beeswarm / waterfall / bar 图，流程是分层的。

### 5.2 树模型优先尝试 TreeExplainer，但会自动回退

对于 BAG 树模型，脚本会尽量走 TreeExplainer；只要底层树模型抽不出来、预处理对不上，都会自动退回 KernelExplainer。

### 5.3 结果不仅有单模型，也有 ensemble 汇总

这是这个脚本和普通 SHAP 脚本最大的区别之一：

- 它不仅保存单模型 SHAP
- 还会尝试按照 ensemble 权重合成一个加权 ensemble SHAP

### 5.4 可视化不是强制步骤

如果只想先拿 SHAP 数值，不传 `--plot_beeswarm_for` 和 `--plot_waterfall` 也可以。

---

## 6. 一个最小运行示例

```bash
python shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/train.csv
```

如果要额外生成图：

```bash
python shap_analyze_autogluon_fixed.py \
  --model_dir /path/to/autogluon_model \
  --train_csv /path/to/train.csv \
  --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3 \
  --plot_waterfall \
  --top_features 20
```

---

## 7. 总结

这套脚本的核心流程可以概括为：

**加载 AutoGluon 模型 → 清洗训练数据 → 构造背景/解释样本 → 识别主模型 → 逐模型计算 SHAP → 保存 CSV 和重要性 → 可选画图 → 汇总 ensemble → 输出摘要。**

如果后面要做 beeswarm、waterfall 或论文风格的 SHAP 图，这个脚本就是第一步。