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
   - 支持两种使用方式：
     a) 通用重采样：oversample / undersample
     b) 比例控制重采样：限制 0 类数量，并指定目标负正比
   - 若不提供 test_csv，会先切分 holdout，再只对训练部分做重采样
   - 测试集 / holdout 保持原始分布，不参与重采样

5. run_all_binary_tasks.py
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

推荐运行方式：

单任务基础版：
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

python -m binary_class.train_binary_task \
  --train_csv binary_class/outputs/task_csvs/LNM_CN01.csv \
  --save_dir binary_class/outputs/models/LNM_CN01

单任务重采样版（推荐用于类别不平衡任务）：
python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/LNM_CN01.csv \
  --save_dir binary_class/outputs/models/LNM_CN01_resampled \
  --eval_metric roc_auc \
  --resample_strategy oversample \
  --resample_target median \
  --seed 42

若任务类别极不平衡，推荐使用“比例控制重采样”：
python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/FTCPTC.csv \
  --save_dir binary_class/outputs/models/FTCPTC_resampled \
  --eval_metric roc_auc \
  --max_neg_count 4000 \
  --target_neg_pos_ratio 4 \
  --seed 42

说明：
- --resample_strategy / --resample_target：通用重采样模式
- --max_neg_count：限制训练时 0 类最多保留多少样本
- --target_neg_pos_ratio：指定重采样后的目标负正比，例如 4 表示约 4:1
- 当传入 --max_neg_count 或 --target_neg_pos_ratio 时，会优先使用比例控制重采样
- 如需保存重采样后的训练表，可附加：
  --save_resampled_csv binary_class/outputs/task_csvs/FTCPTC_resampled_train.csv

带测试集评估的重采样训练示例：
python -m binary_class.train_binary_task_resampled \
  --train_csv binary_class/outputs/task_csvs/train_FTCPTC.csv \
  --test_csv binary_class/outputs/task_csvs/test_FTCPTC.csv \
  --test_names FTCPTC_test \
  --save_dir binary_class/outputs/models/FTCPTC_resampled \
  --eval_metric roc_auc \
  --max_neg_count 4000 \
  --target_neg_pos_ratio 4 \
  --threshold 0.5 \
  --ece_bins 10 \
  --ci_bootstrap_iters 1000 \
  --ci_level 0.95 \
  --ci_seed 42 \
  --seed 42

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
- FTCPTC 这类明显不平衡任务，更推荐比例控制重采样，而不是单纯把 1 类重复很多次
- 建议重点关注 test_results.csv 中的 AUPRC、Recall、F1、Specificity，而不只看 AUROC
