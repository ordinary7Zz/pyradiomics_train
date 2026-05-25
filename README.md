# pyradiomics_train: radiomics + AutoGluon (2D)

目标流程：
- 输入：2D 图像 + 分割预测 mask + 图像级标签 JSON
- `pyradiomics`：用 mask 作为 ROI 提取 radiomics 特征（每张图一行）
- `autogluon.tabular`：用特征表训练分类模型

## 1) 安装依赖

在仓库根目录执行：

```bash
pip install -r pyradiomics_train/requirements.txt
```

## 2) 提取 radiomics 特征

`extract_radiomics_2d.py` 现在支持 4 种 mask 来源：
- `gt`
- `gt_mild_perturb`
- `gt_moderate_perturb`
- `pred`

其中：
- `gt` / `pred` 直接读取 `--mask_dir` 下的 mask
- `gt_mild_perturb` / `gt_moderate_perturb` 会先读取 `--mask_dir` 下的 **GT mask**，再在内存中做可复现的形态学扰动后提取 radiomics 特征

### 输入
- `--image_dir`: 图像目录（png/jpg 等）
- `--mask_dir`: mask 目录（png/jpg 等），默认按 **同名文件** 匹配（可用 `--mask_suffix`）
- `--label_json`: 标签 JSON，结构与当前工程一致：list，每个元素包含 `filename/malignancy/tirads`
- `--task`: 任务名，例如 `malignancy` 或 `tirads`
- `--mask_source`: `gt | gt_mild_perturb | gt_moderate_perturb | pred`
- `--perturb_seed`: GT 扰动的随机种子，默认 `42`

### 输出
- `--output_csv`: radiomics 特征表
- 当 `--mask_source` 为 `gt_mild_perturb` 或 `gt_moderate_perturb` 时，还会额外输出一个 sidecar CSV，记录：
  - `filename`
  - `mask_source`
  - `operation`
  - `kernel_radius`
  - `dice_vs_gt`
  - `hd95_vs_gt`
  - 前景像素统计

### 例子（GT mask）

```bash
python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/TN5K_train_label.json \
  --task malignancy \
  --mask_source gt \
  --perturb_seed 42 \
  --output_csv csv_data/gt/TN5K_train.csv \
  --skip_fail

python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json \
  --task malignancy \
  --mask_source gt \
  --perturb_seed 42 \
  --output_csv csv_data/gt/TN5K_test.csv \
  --skip_fail
```

### 例子（GT mild perturbation）

```bash
python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/TN5K_train_label.json \
  --task malignancy \
  --mask_source gt_mild_perturb \
  --perturb_seed 42 \
  --output_csv csv_data/gt_mild_perturb/TN5K_train.csv \
  --skip_fail

python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json \
  --task malignancy \
  --mask_source gt_mild_perturb \
  --perturb_seed 42 \
  --output_csv csv_data/gt_mild_perturb/TN5K_test.csv \
  --skip_fail
```

### 例子（GT moderate perturbation）

```bash
python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/TN5K_train_label.json \
  --task malignancy \
  --mask_source gt_moderate_perturb \
  --perturb_seed 42 \
  --output_csv csv_data/gt_moderate_perturb/TN5K_train.csv \
  --skip_fail

python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json \
  --task malignancy \
  --mask_source gt_moderate_perturb \
  --perturb_seed 42 \
  --output_csv csv_data/gt_moderate_perturb/TN5K_test.csv \
  --skip_fail
```

### 例子（predicted mask）

```bash
python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/train/TN5K \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/TN5K_train_label.json \
  --task malignancy \
  --mask_source pred \
  --perturb_seed 42 \
  --output_csv csv_data/pred/TN5K_train.csv \
  --skip_fail

python extract_radiomics_2d.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/test/TN5K \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json \
  --task malignancy \
  --mask_source pred \
  --perturb_seed 42 \
  --output_csv csv_data/pred/TN5K_test.csv \
  --skip_fail
```

### 统一实验脚本

项目中新增了统一脚本：`scripts/run_mask_source_experiment.sh`。

示例：

```bash
bash pyradiomics_train/scripts/run_mask_source_experiment.sh extract_train gt dataset_1
bash pyradiomics_train/scripts/run_mask_source_experiment.sh extract_train gt_mild_perturb dataset_1
bash pyradiomics_train/scripts/run_mask_source_experiment.sh extract_train gt_moderate_perturb dataset_1
bash pyradiomics_train/scripts/run_mask_source_experiment.sh extract_train pred dataset_1
```

一次提取某个训练集及 3 个测试集上的全部 4 种 mask 设置：

```bash
bash pyradiomics_train/scripts/run_mask_source_experiment.sh extract_all dataset_1
```

输出目录约定：
- radiomics 特征：`.../csv_data/<mask_source>/<dataset>_radiomics_2d_features.csv`
- perturbation 统计：`.../csv_data/<mask_source>/<dataset>_radiomics_2d_features.<mask_source>.mask_quality.csv`

说明：
- 你当前数据没有真实 spacing，因此脚本使用伪 spacing `(1.0, 1.0)`（可通过 `--spacing_x/--spacing_y` 固定一致值）。
- YAML 配置里禁用了重采样（`resampledPixelSpacing: null`），避免引入虚假的物理尺度。
- `gt_mild_perturb` 使用较小形态学扰动（半径 2–3），`gt_moderate_perturb` 使用较大形态学扰动（半径 5–7）。
- 两种 perturbation 都基于 GT mask 生成，不依赖 predicted mask。

## 3) AutoGluon 训练

对于 AutoGluon 来说，4 种 `mask_source` 的输入没有本质区别：
- `gt`
- `gt_mild_perturb`
- `gt_moderate_perturb`
- `pred`

它们都只是不同来源的 radiomics 特征 CSV。只要 CSV 列结构一致，训练和测试命令本身不需要改，只需要替换输入 CSV 路径即可。

### 用单个 CSV（内部自动划分 holdout，推荐先用 `--model_set tree_full` 加快训练）

```bash
python train_autogluon_tabular.py \
  --train_csv csv_data/gt_mild_perturb/TN5K_train.csv \
  --label label \
  --save_dir /path/to/ag_malignancy \
  --model_set tree_full
```

### 用显式 train/test CSV（推荐先用 `--model_set tree_full`）

```bash
python train_autogluon_tabular.py \
  --train_csv csv_data/gt_mild_perturb/TN5K_train.csv \
  --test_csv csv_data/gt_mild_perturb/TN5K_test.csv \
  --label label \
  --save_dir autogluon_model/gt_mild_perturb \
  --model_set tree_full
```

### 例子：

```bash
python train_autogluon_tabular.py \
  --train_csv csv_data/gt/TN5K_train.csv \
  --test_csv csv_data/gt/TN5K_test.csv \
  --label label \
  --save_dir autogluon_model/gt/TN5K \
  --model_set tree_full \
  --eval_metric roc_auc \
  --presets best_quality \
  --time_limit 600 \
  --seed 42
```
Test performance [TN5K_test]: {'roc_auc': 0.8608414404060233, 'accuracy': 0.852, 'balanced_accuracy': 0.7730714659858929, 'mcc': 0.6020935382637363, 'f1': 0.9031413612565445, 'precision': 0.8657465495608532, 'recall': 0.9439124487004104}
Saved leaderboard: autogluon_model/gt/TN5K/leaderboard.csv

```bash
python train_autogluon_tabular.py \
  --train_csv csv_data/gt_mild_perturb/TN5K_train.csv \
  --test_csv csv_data/gt_mild_perturb/TN5K_test.csv \
  --label label \
  --save_dir autogluon_model/gt_mild_perturb/TN5K \
  --model_set tree_full \
  --eval_metric roc_auc \
  --presets best_quality \
  --time_limit 600 \
  --seed 42
```
Test performance [TN5K_test]: {'roc_auc': 0.8453155274386058, 'accuracy': 0.827, 'balanced_accuracy': 0.7465736705333124, 'mcc': 0.5350328961105683, 'f1': 0.8861092824226465, 'precision': 0.8540609137055838, 'recall': 0.920656634746922}
Saved leaderboard: autogluon_model/gt_mild_perturb/TN5K/leaderboard.csv

```bash
python train_autogluon_tabular.py \
  --train_csv csv_data/gt_moderate_perturb/TN5K_train.csv \
  --test_csv csv_data/gt_moderate_perturb/TN5K_test.csv \
  --label label \
  --save_dir autogluon_model/gt_moderate_perturb/TN5K \
  --model_set tree_full \
  --eval_metric roc_auc \
  --presets best_quality \
  --time_limit 600 \
  --seed 42
```
Test performance [TN5K_test]: {'roc_auc': 0.8671575831854311, 'accuracy': 0.846, 'balanced_accuracy': 0.7560453419718367, 'mcc': 0.5823923042322434, 'f1': 0.9002590673575129, 'precision': 0.8548585485854858, 'recall': 0.9507523939808481}
Saved leaderboard: autogluon_model/gt_moderate_perturb/TN5K/leaderboard.csv

```bash
python train_autogluon_tabular.py \
  --train_csv csv_data/pred/TN5K_train.csv \
  --test_csv csv_data/pred/TN5K_test.csv \
  --label label \
  --save_dir autogluon_model/pred/TN5K \
  --model_set tree_full \
  --eval_metric roc_auc \
  --presets best_quality \
  --time_limit 600 \
  --seed 42
```
Test performance [TN5K_test]: {'roc_auc': 0.8538947004409094, 'accuracy': 0.823, 'balanced_accuracy': 0.7144691541352428, 'mcc': 0.5112020582368589, 'f1': 0.8869009584664537, 'precision': 0.8321342925659473, 'recall': 0.9493844049247606}
Saved leaderboard: autogluon_model/pred/TN5K/leaderboard.csv

输出：
- `--save_dir`：AutoGluon 模型目录
- `leaderboard.csv`：各模型效果对比

## 4) 在测试集上验证性能

训练完成后，可以使用 `test_autogluon_tabular.py` 在一个或多个测试集上评估性能。

当前脚本会输出至少以下指标：
- `auroc`
- `auprc`
- `acc`
- `sensitivity`
- `specificity`

### 通用命令

```bash
python test_autogluon_tabular.py \
  --model_dir /path/to/model_dir \
  --test_csv /path/to/test_a.csv /path/to/test_b.csv \
  --test_names test_a test_b \
  --out_csv /path/to/test_results.csv
```

### 例子：

```bash
python test_autogluon_tabular.py \
  --model_dir autogluon_model/gt/TN5K \
  --test_csv csv_data/gt/TN5K_test.csv \
  --test_names TN5K \
  --mask_source gt \
  --train_dataset TN5K \
  --task_name malignancy \
  --feature_csv csv_data/gt/TN5K_train.csv \
  --out_csv test_logs/gt/TN5K_results.csv
```

```bash
python test_autogluon_tabular.py \
  --model_dir autogluon_model/gt_mild_perturb/TN5K \
  --test_csv csv_data/gt_mild_perturb/TN5K_test.csv \
  --test_names TN5K \
  --mask_source gt_mild_perturb \
  --train_dataset TN5K \
  --task_name malignancy \
  --feature_csv csv_data/gt_mild_perturb/TN5K_train.csv \
  --out_csv test_logs/gt_mild_perturb/TN5K_results.csv
```

```bash
python test_autogluon_tabular.py \
  --model_dir autogluon_model/gt_moderate_perturb/TN5K \
  --test_csv csv_data/gt_moderate_perturb/TN5K_test.csv \
  --test_names TN5K \
  --mask_source gt_moderate_perturb \
  --train_dataset TN5K \
  --task_name malignancy \
  --feature_csv csv_data/gt_moderate_perturb/TN5K_train.csv \
  --out_csv test_logs/gt_moderate_perturb/TN5K_results.csv
```


```bash
python test_autogluon_tabular.py \
  --model_dir autogluon_model/pred/TN5K \
  --test_csv csv_data/pred/TN5K_test.csv \
  --test_names TN5K \
  --mask_source pred \
  --train_dataset TN5K \
  --task_name malignancy \
  --feature_csv csv_data/pred/TN5K_train.csv \
  --out_csv test_logs/pred/TN5K_results.csv
```



### 16 组 train×test 交叉实验

如果你已经按上面的 split 约定准备好了 4 种 mask_source 的特征 CSV：
- `csv_data/gt/<DATASET>_train.csv`
- `csv_data/gt/<DATASET>_test.csv`
- `csv_data/gt_mild_perturb/<DATASET>_train.csv`
- `csv_data/gt_mild_perturb/<DATASET>_test.csv`
- `csv_data/gt_moderate_perturb/<DATASET>_train.csv`
- `csv_data/gt_moderate_perturb/<DATASET>_test.csv`
- `csv_data/pred/<DATASET>_train.csv`
- `csv_data/pred/<DATASET>_test.csv`

可以直接运行：

```bash
bash pyradiomics_train/scripts/run_mask_source_cross_experiments.sh TN5K
```

脚本会固定 4 个 `train_mask_source`：
- `gt`
- `gt_mild_perturb`
- `gt_moderate_perturb`
- `pred`

并对 4 个 `test_mask_source` 全部做交叉评估，共输出 16 组结果：
- `gt -> gt`
- `gt -> gt_mild_perturb`
- `gt -> gt_moderate_perturb`
- `gt -> pred`
- `...`
- `pred -> pred`

默认参数与当前 README 示例保持一致：
- `LABEL_COL=label`
- `TASK_NAME=malignancy`
- `MODEL_SET=tree_full`
- `EVAL_METRIC=roc_auc`
- `PRESETS=best_quality`
- `TIME_LIMIT=600`
- `SEED=42`

这些参数都可以通过环境变量覆盖，例如：

```bash
TIME_LIMIT=1200 MODEL_SET=all bash pyradiomics_train/scripts/run_mask_source_cross_experiments.sh TN5K
```

输出目录默认位于：
- `cross_mask_runs/<DATASET>/<RUN_ID>/models/`：4 个训练模型
- `cross_mask_runs/<DATASET>/<RUN_ID>/results/`：16 个单独结果 CSV
- `cross_mask_runs/<DATASET>/<RUN_ID>/summary/all_results_long.csv`：长表汇总
- `cross_mask_runs/<DATASET>/<RUN_ID>/summary/auroc_4x4.csv`：AUROC 交叉矩阵
- `cross_mask_runs/<DATASET>/<RUN_ID>/summary/auprc_4x4.csv`
- `cross_mask_runs/<DATASET>/<RUN_ID>/summary/acc_4x4.csv`
- `cross_mask_runs/<DATASET>/<RUN_ID>/summary/sensitivity_4x4.csv`
- `cross_mask_runs/<DATASET>/<RUN_ID>/summary/specificity_4x4.csv`

交叉实验的结果 CSV 会额外记录：
- `train_mask_source`
- `test_mask_source`

这样可以直接区分模型训练时和测试时使用的 mask 来源。

### 四种 mask_source 的使用方式

对于另外三种设置，只需要替换对应的特征 CSV 与模型目录：
- `gt` → `pyradiomics_train/csv_data/gt/...`
- `gt_mild_perturb` → `pyradiomics_train/csv_data/gt_mild_perturb/...`
- `gt_moderate_perturb` → `pyradiomics_train/csv_data/gt_moderate_perturb/...`
- `pred` → `pyradiomics_train/csv_data/pred/...`

### 使用统一脚本训练 / 测试

也可以使用统一脚本：

```bash
bash pyradiomics_train/scripts/run_mask_source_experiment.sh train gt_mild_perturb dataset_3
bash pyradiomics_train/scripts/run_mask_source_experiment.sh test gt_mild_perturb dataset_3 /path/to/model_dir
```

### 测试结果输出

测试结果 CSV 中每一行对应一个测试集，通常包含：
- `dataset`
- `csv`
- `n_rows`
- `auroc`
- `auprc`
- `acc`
- `sensitivity`
- `specificity`
- `model_dir`
- `feature_csv`

## 5) 绘图脚本

绘图相关脚本已整理到 [plots/](plots/) 目录，使用说明见 [plots/README.md](plots/README.md)。
