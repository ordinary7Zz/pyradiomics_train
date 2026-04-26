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

### 输入
- `--image_dir`: 图像目录（png/jpg 等）
- `--mask_dir`: mask 目录（png/jpg 等），默认按 **同名文件** 匹配（可用 `--mask_suffix`）
- `--label_json`: 标签 JSON，结构与当前工程一致：list，每个元素包含 `filename/malignancy/tirads`

### 输出
- `--output_csv`: radiomics 特征表

### 例子（二分类良恶性）

```bash
python pyradiomics_train/extract_radiomics_2d.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --label_json /path/to/labels.json \
  --task malignancy \
  --output_csv /path/to/radiomics_malignancy.csv \
  --skip_fail
```

### 例子（TIRADS 五分类）

```bash
python pyradiomics_train/extract_radiomics_2d.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --label_json /path/to/labels.json \
  --task tirads \
  --output_csv /path/to/radiomics_tirads.csv \
  --skip_fail
```

说明：
- 你当前数据没有真实 spacing，因此脚本使用伪 spacing `(1.0, 1.0)`（可通过 `--spacing_x/--spacing_y` 固定一致值）。
- YAML 配置里禁用了重采样（`resampledPixelSpacing: null`），避免引入虚假的物理尺度。

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
