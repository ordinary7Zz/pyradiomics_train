binary_class 目录说明

文件：
1. extract_base_radiomics.py
   - 从 image_dir 和 mask_dir 中按 label_json 的 filename 提取基础 radiomics 特征
   - 不绑定具体任务标签
   - 允许 image 和 mask 扩展名不一致，只要相对路径主体一致即可自动匹配

2. build_binary_task_csv.py
   - 从 base_features.csv 和 label_json 中生成某一个二分类任务的训练表
   - 会过滤 label=-1 的样本

3. train_binary_task.py
   - 对单个任务 CSV 训练一个 AutoGluon 二分类模型
   - 这是基础训练版本，不做类别重采样

4. train_binary_task_resampled.py
   - 对单个任务 CSV 训练一个带重采样能力的 AutoGluon 二分类模型
   - 支持训练集重采样，以缓解类别不平衡问题
   - 支持通过 `--time_limit` 控制总训练时长
   - 支持通过 `--model_set` 选择训练哪些模型族
   - 支持两种使用方式：
     a) 通用重采样：oversample / undersample
     b) 目标数量重采样：直接指定 0 类和 1 类各自的目标数量
   - `--target_class0_count` / `--target_class1_count` 输入 -1 表示保持该类原始数量不变
   - 若不提供 test_csv，会先切分 holdout，再只对训练部分做重采样
   - 测试集 / holdout 保持原始分布，不参与重采样

5. build_patient_task_csv.py
   - 将图像级 task CSV 聚合成病人级 task CSV
   - 默认从 filename 中提取“年份/病人目录名”作为病人键
   - 会校验同一病人组内 label 是否一致
   - 默认对数值特征做 mean 聚合，并新增 image_count 列
   - 输出结果可直接复用 train_binary_task.py 或 train_binary_task_resampled.py 训练

6. run_all_binary_tasks.py
   - 自动串联：提训练/测试基础特征 -> 逐任务构表 -> 逐任务训练 -> 在测试集上评估
   - 当前默认调用的是 train_binary_task.py
   - 测试输出指标：AUROC, AUPRC, Acc, Prec, Recall, F1, Specificity, ECE
   - 同时输出上述指标的 bootstrap 置信区间

输出位置：
- 训练基础特征：binary_class/outputs/base_features/train_base_features.csv
- 测试基础特征：binary_class/outputs/base_features/test_base_features.csv
- 各任务训练 CSV：binary_class/outputs/task_csvs/train_<task>.csv
- 各任务测试 CSV：binary_class/outputs/task_csvs/test_<task>.csv
- 各任务模型与测试结果：binary_class/outputs/models/<task>/
  - leaderboard.csv
  - test_results.csv
  - test_results_ci.csv
  - class_balance_summary.csv   （仅 train_binary_task_resampled.py 输出）
- 总汇总：binary_class/outputs/reports/run_summary.csv

说明：
- --resample_strategy / --resample_target：通用重采样模式
- --target_class0_count / --target_class1_count：分别指定训练时 0 类和 1 类的目标数量
- 某一类设为 -1 表示保持该类原始数量不变
- --time_limit：控制 AutoGluon 总训练时长，单位为秒
- --model_set：控制训练哪些模型族，可选值为 all / tree_fast / tree_full / gbm_cat / gbm_only
- 当传入 --target_class0_count 或 --target_class1_count 且其值不为 -1 时，会优先使用目标数量重采样
- 如需保存重采样后的训练表，可附加：
  --save_resampled_csv binary_class/outputs/task_csvs/FTCPTC_resampled_train.csv

批量所有任务：
python binary_class/run_all_binary_tasks.py \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --train_mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --train_label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --test_mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --test_label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json \
  --work_dir binary_class/outputs \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --skip_fail

补充说明：
- LNM_CN01 这类轻中度不平衡任务，通常可先尝试通用重采样：
  --resample_strategy oversample --resample_target median
- FTCPTC 这类明显不平衡任务，更推荐目标数量重采样，例如先限制 0 类数量，再视情况补足 1 类数量
- 若希望只调整其中一类数量，可将另一类设为 -1 保持原始数量
- 若希望更快试验，可优先使用 `--model_set tree_fast --time_limit 1800`
- 若希望兼顾效果与速度，可优先使用 `--model_set tree_full --time_limit 3600`
- 若标签是病人级、样本却是图像级，建议先构建病人级 CSV，再复用现有训练脚本做病人级训练与评估
- 建议重点关注 test_results.csv 中的 AUPRC、Recall、F1、Specificity，而不只看 AUROC

## LymphUs数据集 - predmask：
python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/Lymph_Node_Metastasis_fake/Lymph_images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/LymphUs_train_labels.json \
  --output_csv binary_class/outputs/base_features/LymphUs_fake_predmask/train_base_features.csv \
  --skip_fail

python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/Lymph_Node_Metastasis_fake/Lymph_images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/LymphUs_test_labels.json \
  --output_csv binary_class/outputs/base_features/LymphUs_fake_predmask/test_base_features.csv \
  --skip_fail

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/LymphUs_fake_predmask/train_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/LymphUs_train_labels.json \
  --task LNM_CN01 \
  --output_csv binary_class/outputs/task_csvs/LymphUs_fake_predmask/train_LNM_CN01.csv

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/LymphUs_fake_predmask/test_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/LymphUs_test_labels.json \
  --task LNM_CN01 \
  --output_csv binary_class/outputs/task_csvs/LymphUs_fake_predmask/test_LNM_CN01.csv

python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/LymphUs_fake_predmask/train_LNM_CN01.csv \
  --test_csv binary_class/outputs/task_csvs/LymphUs_fake_predmask/test_LNM_CN01.csv \
  --test_names LymphUs_test \
  --save_dir binary_class/outputs/models/LymphUs_fake_predmask \
  --eval_metric roc_auc \
  --model_set tree_full \
  --time_limit 3600 \
  --resample_strategy none \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42

## FangDai数据集：
python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped_predictions \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped/FangDai_train_labels.json \
  --output_csv binary_class/outputs/base_features/FangDai/train_base_features.csv \
  --skip_fail

python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped_predictions \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped/FangDai_test_labels.json \
  --output_csv binary_class/outputs/base_features/FangDai/test_base_features.csv \
  --skip_fail

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/FangDai/train_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped/FangDai_train_labels.json \
  --task FTCPTC \
  --output_csv binary_class/outputs/task_csvs/FangDai/train_FTCPTC.csv

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/FangDai/test_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped/FangDai_test_labels.json \
  --task FTCPTC \
  --output_csv binary_class/outputs/task_csvs/FangDai/test_FTCPTC.csv

python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/FangDai/train_FTCPTC.csv \
  --test_csv binary_class/outputs/task_csvs/FangDai/test_FTCPTC.csv \
  --test_names FTCPTC_test \
  --save_dir binary_class/outputs/models/FangDai/FTCPTC \
  --eval_metric roc_auc \
  --model_set tree_full \
  --time_limit 3600 \
  --resample_strategy none \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42

## 整理后的FTCPTC数据集：
python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --label_json /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/train_labels_filtered_by_csv.json \
  --output_csv binary_class/outputs/base_features/FTCPTC_FangDai/train_base_features.csv \
  --skip_fail

python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --label_json /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/test_labels_filtered_by_csv.json \
  --output_csv binary_class/outputs/base_features/FTCPTC_FangDai/test_base_features.csv \
  --skip_fail

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/FTCPTC_FangDai/train_base_features.csv \
  --label_json /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/train_labels_filtered_by_csv.json \
  --task FTCPTC \
  --output_csv binary_class/outputs/task_csvs/FTCPTC_FangDai/train_FTCPTC.csv

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/FTCPTC_FangDai/test_base_features.csv \
  --label_json /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/my_json/test_labels_filtered_by_csv.json \
  --task FTCPTC \
  --output_csv binary_class/outputs/task_csvs/FTCPTC_FangDai/test_FTCPTC.csv

python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/FTCPTC_FangDai/train_FTCPTC.csv \
  --test_csv binary_class/outputs/task_csvs/FTCPTC_FangDai/test_FTCPTC.csv \
  --test_names FTCPTC_test \
  --save_dir binary_class/outputs/models/FTCPTC_FangDai/FTCPTC \
  --eval_metric roc_auc \
  --model_set tree_full \
  --time_limit 3600 \
  --resample_strategy none \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42

## 良恶性数据集 - predmask
python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/train/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/cls_dataset_3/dataset_3_train \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/train/dataset_3_train_label.json \
  --output_csv binary_class/outputs/base_features/BM_dataset3_predmask/train_base_features.csv \
  --skip_fail

python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/cls_dataset_3/dataset_3_test \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/dataset_3_test_label.json \
  --output_csv binary_class/outputs/base_features/BM_dataset3_predmask/test_base_features.csv \
  --skip_fail

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/BM_dataset3_predmask/train_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/train/dataset_3_train_label.json \
  --task malignancy \
  --output_csv binary_class/outputs/task_csvs/BM_dataset3_predmask/train_BM.csv

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/BM_dataset3_predmask/test_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/dataset_3_test_label.json \
  --task malignancy \
  --output_csv binary_class/outputs/task_csvs/BM_dataset3_predmask/test_BM.csv

python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/BM_dataset3_predmask/train_BM.csv \
  --test_csv binary_class/outputs/task_csvs/BM_dataset3_predmask/test_BM.csv \
  --test_names BM_test \
  --save_dir binary_class/outputs/models/BM_dataset3_predmask/BM \
  --eval_metric roc_auc \
  --model_set tree_full \
  --time_limit 3600 \
  --resample_strategy none \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42

## 500BM数据集获得csv
python binary_class/extract_base_radiomics.py \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/images \
  --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/masks \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/500_TestData_Malignancy_Cls.json \
  --output_csv binary_class/outputs/base_features/BM_500_predmask/train_base_features.csv \
  --skip_fail

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/BM_500_predmask/train_base_features.csv \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/500_TestData_Malignancy_Cls.json \
  --task malignancy \
  --output_csv binary_class/outputs/task_csvs/BM_500_predmask/train_BM.csv