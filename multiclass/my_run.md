# TIRADS 五分类训练运行记录

## 训练命令参考

```bash
# 完整流水线
python multiclass/run_all_multiclass_tasks.py \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train/images \
  --train_mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train/masks \
  --train_label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train/Cine-Clip_train_label.json \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images \
  --test_mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/masks \
  --test_label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/Cine-Clip_test_label.json \
  --work_dir multiclass/outputs \
  --tasks tirads \
  --training_mode resampled \
  --resample_strategy oversample \
  --resample_target max \
  --model_set tree_full \
  --time_limit 3600 \
  --eval_metric balanced_accuracy \
  --skip_fail
```

## 参数组合建议

| 场景 | presets | model_set | time_limit | resample_strategy |
|------|---------|-----------|------------|-------------------|
| 快速实验 | medium_quality | tree_fast | 1800 | oversample + median |
| 标准训练 | best_quality | tree_full | 3600 | oversample + max |
| 极致效果 | best_quality | all | 7200+ | oversample + max |
| 无重采样 | best_quality | tree_full | 3600 | none |

## eval_metric 参考

- `balanced_accuracy` — 最常用，对不平衡数据友好
- `f1_weighted` — 关注加权 F1
- `log_loss` — 关注概率校准
- 默认（不指定）— AutoGluon 自动选择
