#!/usr/bin/env python
"""
快速查看AutoGluon模型目录中的关键诊断信息。

主要功能：
- 读取并打印 leaderboard.csv，展示模型排名与分数。
- 从 predictor_log.txt 中提取 ensemble 权重，展示各子模型贡献。
- 给出后续SHAP分析的推荐命令模板。

用法：
    python shap_analyze/inspect_autogluon_models.py <model_dir>

例如：
    python shap_analyze/inspect_autogluon_models.py ./autogluon_model_20260430_120000/
"""

import argparse
import os
import re
import sys
from typing import Optional, Dict

def parse_args():
    p = argparse.ArgumentParser(description="Inspect AutoGluon model directory and show leaderboard")
    p.add_argument("model_dir", type=str, help="AutoGluon model directory")
    return p.parse_args()


def print_separator(title: str = "", width: int = 60):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print(f"{'=' * width}\n")


def read_leaderboard(model_dir: str) -> Optional[str]:
    """读取leaderboard.csv内容"""
    leaderboard_path = os.path.join(model_dir, "leaderboard.csv")
    if os.path.exists(leaderboard_path):
        try:
            import pandas as pd
            df = pd.read_csv(leaderboard_path)
            return df
        except Exception as e:
            print(f"Error reading leaderboard: {e}")
            return None
    return None


def extract_ensemble_weights(log_path: str) -> Optional[Dict[str, float]]:
    """从predictor_log.txt提取ensemble权重"""
    if not os.path.exists(log_path):
        return None
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找 "Ensemble Weights:" 部分
        pattern = r"Ensemble Weights:\s*\{([^}]+)\}"
        match = re.search(pattern, content)
        if not match:
            return None

        weights_str = match.group(1)
        weights = {}
        for item in weights_str.split(","):
            item = item.strip()
            if not item:
                continue
            # 格式: 'ModelName': 0.467
            model_match = re.search(r"'([^']+)':\s*([\d.]+)", item)
            if model_match:
                model_name = model_match.group(1)
                weight = float(model_match.group(2))
                weights[model_name] = weight

        return weights if weights else None
    except Exception as e:
        print(f"Warning: Failed to parse ensemble weights: {e}")
        return None


def main():
    args = parse_args()
    model_dir = args.model_dir

    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    print_separator("AutoGluon Model Inspection")
    print(f"Model directory: {model_dir}\n")

    # 1. 显示leaderboard
    print_separator("TOP MODELS (Leaderboard)")
    leaderboard_df = read_leaderboard(model_dir)
    
    if leaderboard_df is not None:
        print(f"\nTotal models: {len(leaderboard_df)}")
        print(f"\nTop 10 models by score:")
        print("-" * 80)
        
        # 显示前10个模型
        display_cols = [col for col in leaderboard_df.columns 
                       if col in ['model', 'score_test', 'score_val', 'score_train', 
                                 'pred_time_test', 'pred_time_val', 'pred_time_train']]
        
        for idx, (i, row) in enumerate(leaderboard_df.head(10).iterrows(), 1):
            print(f"\n{idx}. {row['model']}")
            for col in display_cols:
                if col != 'model' and col in row:
                    val = row[col]
                    if isinstance(val, float):
                        print(f"   {col}: {val:.6f}")
                    else:
                        print(f"   {col}: {val}")
    else:
        print("❌ Could not read leaderboard.csv")

    # 2. 显示ensemble权重
    print_separator("ENSEMBLE WEIGHTS")
    log_path = os.path.join(model_dir, "logs", "predictor_log.txt")
    weights = extract_ensemble_weights(log_path)

    if weights:
        print(f"\nEnsemble contains {len(weights)} models:")
        print("-" * 80)
        
        # 按权重排序
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        total_weight = sum(w for _, w in sorted_weights)
        
        for rank, (model_name, weight) in enumerate(sorted_weights, 1):
            pct = weight / total_weight * 100 if total_weight > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{rank}. {model_name:30s} | Weight: {weight:.4f} | {pct:5.1f}% {bar}")
    else:
        print("❌ Could not find ensemble weights in predictor_log.txt")
        print("   This may be a single model or the log file is not accessible.")

    # 3. 摘要信息
    print_separator("SUMMARY")
    
    if leaderboard_df is not None:
        best_model = leaderboard_df.iloc[0]
        print(f"Best model: {best_model['model']}")
        if 'score_test' in best_model:
            print(f"Test score: {best_model['score_test']:.6f}")
        elif 'score_val' in best_model:
            print(f"Val score: {best_model['score_val']:.6f}")

    if weights:
        print(f"\nTop 3 models in ensemble:")
        for rank, (model_name, weight) in enumerate(sorted_weights[:3], 1):
            print(f"  {rank}. {model_name} (weight: {weight:.4f})")
    
    print(f"\n💡 要为特定模型生成SHAP分析，使用:")
    if weights:
        top_models = " ".join([name for name, _ in sorted_weights[:3]])
        print(f"   python shap_analyze_autogluon_fixed.py \\")
        print(f"     --model_dir {model_dir} \\")
        print(f"     --train_csv <train.csv> \\")
        print(f"     --plot_beeswarm_for {top_models}")
    else:
        print(f"   python shap_analyze_autogluon_fixed.py \\")
        print(f"     --model_dir {model_dir} \\")
        print(f"     --train_csv <train.csv> \\")
        print(f"     --plot_beeswarm_for <model_name>")

    print()


if __name__ == "__main__":
    main()
