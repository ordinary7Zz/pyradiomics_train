# TN5K 4×4 交叉实验总结

## 1. 实验目的

本实验用于分析 **分割掩码误差对影像组学分类的影响**，重点回答以下问题：

1. 训练阶段和测试阶段的掩码来源变化，是否会影响分类性能；
2. 真实预测掩码 `pred` 对分类的影响，是否强于基于 GT 的人工扰动；
3. `gt_mild_perturb` 与 `gt_moderate_perturb` 能否近似真实预测掩码误差；
4. 训练/测试 mask 来源匹配时，是否能够缓解性能下降。

## 2. 实验设置

### 2.1 四种 mask_source

本实验比较 4 种 ROI 来源：

- `gt`：直接使用人工标注 GT mask；
- `gt_mild_perturb`：在 GT mask 基础上做轻度形态学扰动；
- `gt_moderate_perturb`：在 GT mask 基础上做中度形态学扰动；
- `pred`：使用分割模型输出的预测 mask。

### 2.2 mild / moderate 的设置方式

`gt_mild_perturb` 和 `gt_moderate_perturb` 都不是读取独立文件，而是：

1. 先读取 GT mask；
2. 在内存中进行**可复现的形态学扰动**；
3. 再基于扰动后的 mask 提取 radiomics 特征。

当前 README 中的约定是：

- `gt_mild_perturb`：较小扰动，形态学半径约 **2–3**；
- `gt_moderate_perturb`：较大扰动，形态学半径约 **5–7**；
- 扰动随机种子：`perturb_seed=42`。

因此：

- `mild` / `moderate` 是**基于 GT 的人工误差模拟**；
- `pred` 是**真实分割模型产生的误差**；
- 三者不能简单看成同一种误差的强弱版本，`pred` 可能包含更复杂的结构性偏差。

### 2.3 特征与数据输入

本实验使用 TN5K 的显式 train/test split：

- `csv_data/gt/TN5K_train.csv`
- `csv_data/gt/TN5K_test.csv`
- `csv_data/gt_mild_perturb/TN5K_train.csv`
- `csv_data/gt_mild_perturb/TN5K_test.csv`
- `csv_data/gt_moderate_perturb/TN5K_train.csv`
- `csv_data/gt_moderate_perturb/TN5K_test.csv`
- `csv_data/pred/TN5K_train.csv`
- `csv_data/pred/TN5K_test.csv`

四种 CSV 的列结构保持一致，只是 mask 来源不同。

### 2.4 交叉实验如何进行

使用脚本：

```bash
bash scripts/run_mask_source_cross_experiments.sh TN5K
```

脚本执行逻辑如下：

1. 分别用 4 种 `train_mask_source` 训练 4 个 AutoGluon 分类模型：
   - `train_gt`
   - `train_gt_mild_perturb`
   - `train_gt_moderate_perturb`
   - `train_pred`
2. 每个训练好的模型，再分别在 4 种 `test_mask_source` 上评估；
3. 最终得到 **4 × 4 = 16 组 train×test 组合**。

### 2.5 训练参数

交叉实验脚本中的默认训练参数为：

- `MODEL_SET=tree_full`
- `EVAL_METRIC=roc_auc`
- `PRESETS=best_quality`
- `TIME_LIMIT=600`
- `SEED=42`
- `LABEL_COL=label`
- `TASK_NAME=malignancy`

### 2.6 评估指标

本次交叉实验统一汇报以下指标：

- `auroc`
- `auprc`
- `acc`
- `sensitivity`
- `specificity`

其中：

- `auroc` 反映整体排序能力；
- `auprc` 更关注正类识别质量；
- `sensitivity` 和 `specificity` 用于观察阈值 0.5 下的正负类取舍。

## 3. 16 组交叉实验结果

### 3.1 完整结果表

| train_mask_source | test_mask_source | AUROC | AUPRC | ACC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| gt | gt | 0.860841 | 0.923376 | 0.852 | 0.943912 | 0.602230 |
| gt | gt_mild_perturb | 0.843383 | 0.914471 | 0.827 | 0.926129 | 0.557621 |
| gt | gt_moderate_perturb | 0.821546 | 0.909737 | 0.794 | 0.898769 | 0.509294 |
| gt | pred | 0.793688 | 0.892803 | 0.782 | 0.969904 | 0.271375 |
| gt_mild_perturb | gt | 0.853437 | 0.917739 | 0.841 | 0.939808 | 0.572491 |
| gt_mild_perturb | gt_mild_perturb | 0.845316 | 0.914402 | 0.827 | 0.920657 | 0.572491 |
| gt_mild_perturb | gt_moderate_perturb | 0.848377 | 0.924427 | 0.813 | 0.901505 | 0.572491 |
| gt_mild_perturb | pred | 0.798178 | 0.897375 | 0.794 | 0.975376 | 0.301115 |
| gt_moderate_perturb | gt | 0.850920 | 0.915942 | 0.812 | 0.967168 | 0.390335 |
| gt_moderate_perturb | gt_mild_perturb | 0.847584 | 0.912331 | 0.827 | 0.956224 | 0.475836 |
| gt_moderate_perturb | gt_moderate_perturb | 0.867158 | 0.926103 | 0.846 | 0.950752 | 0.561338 |
| gt_moderate_perturb | pred | 0.811467 | 0.912516 | 0.794 | 0.948016 | 0.375465 |
| pred | gt | 0.835592 | 0.912080 | 0.811 | 0.871409 | 0.646840 |
| pred | gt_mild_perturb | 0.832093 | 0.913350 | 0.802 | 0.853625 | 0.661710 |
| pred | gt_moderate_perturb | 0.825935 | 0.912758 | 0.781 | 0.829001 | 0.650558 |
| pred | pred | 0.853895 | 0.924884 | 0.823 | 0.949384 | 0.479554 |

### 3.2 AUROC 4×4 矩阵

| train \ test | gt | mild | moderate | pred |
|---|---:|---:|---:|---:|
| gt | 0.860841 | 0.843383 | 0.821546 | 0.793688 |
| mild | 0.853437 | 0.845316 | 0.848377 | 0.798178 |
| moderate | 0.850920 | 0.847584 | 0.867158 | 0.811467 |
| pred | 0.835592 | 0.832093 | 0.825935 | 0.853895 |

### 3.3 AUPRC 4×4 矩阵

| train \ test | gt | mild | moderate | pred |
|---|---:|---:|---:|---:|
| gt | 0.923376 | 0.914471 | 0.909737 | 0.892803 |
| mild | 0.917739 | 0.914402 | 0.924427 | 0.897375 |
| moderate | 0.915942 | 0.912331 | 0.926103 | 0.912516 |
| pred | 0.912080 | 0.913350 | 0.912758 | 0.924884 |

## 4. 结果解读

### 4.1 测试阶段的 mask 误差影响最明显

固定 `train=gt` 时：

- `gt -> gt`: 0.860841
- `gt -> mild`: 0.843383
- `gt -> moderate`: 0.821546
- `gt -> pred`: 0.793688

说明随着测试掩码从 GT 偏离到 mild、moderate、pred，分类性能持续下降。

这表明：

> **测试阶段的分割误差会直接削弱 radiomics 特征的可用性。**

### 4.2 训练阶段的 mask 误差也有影响，但小于测试阶段

固定 `test=gt` 时：

- `gt -> gt`: 0.860841
- `mild -> gt`: 0.853437
- `moderate -> gt`: 0.850920
- `pred -> gt`: 0.835592

可以看到训练掩码变差也会降性能，但下降幅度小于固定 GT 训练、让测试 mask 逐步变差的情形。

这说明：

> **测试阶段的掩码误差，比训练阶段的掩码误差更伤模型。**

### 4.3 `pred` 对分类的影响最大

固定 `train=gt` 时，`test=pred` 的 AUROC 最低（0.793688），并且 specificity 只有 **0.271375**，远低于：

- `gt -> gt_moderate_perturb` 的 0.509294
- `gt -> gt` 的 0.602230

这说明：

1. `pred` 对下游 radiomics 分类的影响强于当前人工 mild / moderate 扰动；
2. `pred` 不只是让排序能力下降，还会使模型在阈值 0.5 下明显偏向正类，假阳性增加。

因此：

> **当前 `pred` 带来的问题不仅是噪声更大，还包含明显的分布偏移与阈值失衡。**

### 4.4 训练/测试 mask 来源匹配可以明显缓解性能下降

比较：

- `gt -> pred`: 0.793688
- `pred -> pred`: 0.853895

提升约 **0.0602**。

再比较：

- `gt -> gt_moderate_perturb`: 0.821546
- `gt_moderate_perturb -> gt_moderate_perturb`: 0.867158

提升约 **0.0456**。

说明：

> **当训练和测试使用相同类型的 mask 特征时，模型能更好适应该分布。**

这意味着性能下降不只是来自信息损失，也来自 **train/test 特征分布错配（domain shift）**。

### 4.5 moderate 比 mild 更接近 pred，但仍不能完全替代 pred

固定 `test=pred` 时：

- `gt -> pred`: 0.793688
- `mild -> pred`: 0.798178
- `moderate -> pred`: 0.811467
- `pred -> pred`: 0.853895

在非 `pred` 训练模型中，`moderate -> pred` 最好，说明：

> **当前人工扰动方案里，moderate 比 mild 更接近真实预测掩码误差。**

但它仍明显低于 `pred -> pred`，因此：

> **moderate 只能部分模拟 pred，尚不能完全等价替代真实预测误差。**

## 5. 当前实验支持的主要结论

1. **分割掩码误差会显著影响影像组学分类性能。**
2. **测试阶段的分割误差影响大于训练阶段。**
3. **真实预测掩码 `pred` 对分类的负面影响大于当前 mild / moderate 人工扰动。**
4. **`pred` 的影响不能仅用“扰动幅度更大”来解释，还涉及明显的特征分布偏移。**
5. **训练/测试 mask 来源匹配可以显著缓解性能下降。**
6. **在当前人工扰动设计中，`gt_moderate_perturb` 是最接近 `pred` 的代理，但仍不完全等价。**

## 6. 对后续实验的启发

### 6.1 当前结果不能直接说明 `pred` 的分割性能很差

本实验衡量的是 **下游分类性能**，不能直接等价为分割性能本身。

当前结果只能说明：

> **`pred mask` 对 radiomics 分类的干扰最大。**

如果要判断 `pred` 的分割性能是否差，还需要直接计算：

- Dice
- IoU
- HD95
- 边界距离
- 面积 / 体积差

### 6.2 是否需要增大 mild / moderate 的扰动强度

当前结果表明：

- mild / moderate 已经形成了明显的强度梯度；
- 但 `pred` 造成的影响仍强于 `moderate`；
- 因此当前人工扰动可能**幅度偏小**，同时也可能**误差形态不够接近真实预测误差**。

因此更合理的下一步不是直接替换现有实验，而是：

1. 保留当前 mild / moderate 结果；
2. 先比较 `GT vs pred`、`GT vs mild`、`GT vs moderate` 的 Dice / HD95 分布；
3. 如果 `pred` 明显比 `moderate` 更差，则新增更强一档扰动（例如 `severe perturbation`）；
4. 再评估更强扰动是否能更接近 `pred` 的 cross-test 表现。

## 7. 建议后续补充分析

1. **多随机种子重复实验**：验证上述趋势是否稳定；
2. **bootstrap 置信区间**：为 AUROC / AUPRC / specificity 提供统计区间；
3. **特征稳定性分析**：比较 GT、moderate、pred 下 radiomics 特征的变化；
4. **mask 质量分析**：直接计算 GT vs pred 的 Dice / HD95，并与 mild / moderate 对齐；
5. **更强扰动实验**：新增 `severe perturbation`，验证是否能更贴近 `pred`。
