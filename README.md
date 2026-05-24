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
python pyradiomics_train/extract_radiomics_2d.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/gt_masks \
  --label_json /path/to/labels.json \
  --task malignancy \
  --mask_source gt \
  --output_csv /path/to/gt_radiomics_malignancy.csv \
  --skip_fail
```

### 例子（GT mild perturbation）

```bash
python pyradiomics_train/extract_radiomics_2d.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/gt_masks \
  --label_json /path/to/labels.json \
  --task malignancy \
  --mask_source gt_mild_perturb \
  --perturb_seed 42 \
  --output_csv /path/to/gt_mild_perturb_radiomics_malignancy.csv \
  --skip_fail
```

### 例子（GT moderate perturbation）

```bash
python pyradiomics_train/extract_radiomics_2d.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/gt_masks \
  --label_json /path/to/labels.json \
  --task malignancy \
  --mask_source gt_moderate_perturb \
  --perturb_seed 42 \
  --output_csv /path/to/gt_moderate_perturb_radiomics_malignancy.csv \
  --skip_fail
```

### 例子（predicted mask）

```bash
python pyradiomics_train/extract_radiomics_2d.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/pred_masks \
  --label_json /path/to/labels.json \
  --task malignancy \
  --mask_source pred \
  --output_csv /path/to/pred_radiomics_malignancy.csv \
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

### 用单个 CSV（内部自动划分 holdout）

```bash
python pyradiomics_train/train_autogluon_tabular.py \
  --train_csv /path/to/radiomics_malignancy.csv \
  --label label \
  --save_dir /path/to/ag_malignancy
```

### 用显式 train/test CSV

```bash
python pyradiomics_train/train_autogluon_tabular.py \
  --train_csv /path/to/train.csv \
  --test_csv /path/to/test.csv \
  --label label \
  --save_dir /path/to/ag_run
```

输出：
- `--save_dir`：AutoGluon 模型目录
- `leaderboard.csv`：各模型效果对比

## 4) 绘图脚本

绘图相关脚本已整理到 [plots/](plots/) 目录，使用说明见 [plots/README.md](plots/README.md)。
