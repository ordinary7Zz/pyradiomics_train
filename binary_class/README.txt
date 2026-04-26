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

4. run_all_binary_tasks.py
   - 自动串联：提训练/测试基础特征 -> 逐任务构表 -> 逐任务训练 -> 在测试集上评估

输出位置：
- 训练基础特征：binary_class/outputs/base_features/train_base_features.csv
- 测试基础特征：binary_class/outputs/base_features/test_base_features.csv
- 各任务训练 CSV：binary_class/outputs/task_csvs/train_<task>.csv
- 各任务测试 CSV：binary_class/outputs/task_csvs/test_<task>.csv
- 各任务模型与测试结果：binary_class/outputs/models/<task>/
- 总汇总：binary_class/outputs/reports/run_summary.csv

推荐运行方式：

单任务：
python binary_class/extract_base_radiomics.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --label_json /path/to/test_labels.json \
  --output_csv binary_class/outputs/base_features/base_features.csv \
  --skip_fail

python binary_class/build_binary_task_csv.py \
  --base_features_csv binary_class/outputs/base_features/base_features.csv \
  --label_json /path/to/test_labels.json \
  --task LNM_CN01 \
  --output_csv binary_class/outputs/task_csvs/LNM_CN01.csv

python binary_class/train_binary_task.py \
  --train_csv binary_class/outputs/task_csvs/LNM_CN01.csv \
  --save_dir binary_class/outputs/models/LNM_CN01

批量所有任务：
python binary_class/run_all_binary_tasks.py \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --train_mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --train_label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --test_mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --test_label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json \
  --work_dir binary_class/outputs \
  --skip_fail
