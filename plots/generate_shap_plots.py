import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np


def load_shap_data(shap_analysis_path):
    """加载 SHAP 分析结果文件"""
    shap_values_path = os.path.join(shap_analysis_path, "models")
    
    shap_data = {}
    
    # 加载 expected_values.json（每个模型的 expected_value 和 output_space）
    with open(os.path.join(shap_analysis_path, "expected_values.json"), "r") as f:
        shap_data["expected_values"] = json.load(f)
    
    # 遍历模型文件夹，加载 SHAP 值和 case_table
    for model_name in os.listdir(shap_values_path):
        model_dir = os.path.join(shap_values_path, model_name)
        
        if model_name.endswith("_case_table.csv"):
            case_table = pd.read_csv(model_dir)
            shap_data[model_name] = {
                "case_table": case_table,
                "shap_values": pd.read_csv(os.path.join(model_dir.replace("case_table", "shap_values"))),
            }
        
    return shap_data


def select_fp_fn_tp_tn(shap_data, model_name, threshold=0.5):
    """选择 FP, FN, TP, TN 样本"""
    case_table = shap_data[model_name]["case_table"]
    
    # 根据阈值确定预测标签
    case_table["y_pred"] = case_table["p_pos"].apply(lambda x: 1 if x >= threshold else 0)
    
    # 选择 FP, FN, TP, TN
    fp = case_table[(case_table["y_true"] == 0) & (case_table["y_pred"] == 1)]
    fn = case_table[(case_table["y_true"] == 1) & (case_table["y_pred"] == 0)]
    tp = case_table[(case_table["y_true"] == 1) & (case_table["y_pred"] == 1)]
    tn = case_table[(case_table["y_true"] == 0) & (case_table["y_pred"] == 0)]
    
    return fp, fn, tp, tn


def generate_beeswarm_plot(shap_values, X_explain, save_path):
    """生成 Beeswarm 图"""
    shap.initjs()
    beeswarm_fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_explain, plot_type="bar")
    beeswarm_fig.savefig(os.path.join(save_path, "beeswarm_plot.png"))
    plt.close()

    print(f"Beeswarm plot saved to {save_path}")


def generate_waterfall_force_plot(shap_values, X_explain, sample_idx, save_path):
    """生成 Waterfall 图和 Force 图"""
    shap_values_sample = shap_values[sample_idx]
    X_sample = X_explain.iloc[sample_idx]
    
    # 生成 Waterfall 图
    waterfall_fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values_sample)
    waterfall_fig.savefig(os.path.join(save_path, "waterfall_plot.png"))
    
    # 生成 Force 图
    force_fig = plt.figure(figsize=(10, 6))
    shap.force_plot(shap_values_sample, matplotlib=True)
    force_fig.savefig(os.path.join(save_path, "force_plot.png"))
    
    print(f"Waterfall and Force plots saved to {save_path}")


def plot_fp_fn_comparison(shap_data, shap_analysis_path, model_name):
    """绘制 FP 和 FN 样本的 SHAP 特征贡献对比"""
    fp, fn, _, _ = select_fp_fn_tp_tn(shap_data, model_name)
    
    # 获取 FP 和 FN 样本的 SHAP 值
    fp_shap_values = shap_data[model_name]["shap_values"].iloc[fp.index]
    fn_shap_values = shap_data[model_name]["shap_values"].iloc[fn.index]
    
    # 绘制 FP 和 FN 样本特征重要性对比
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.bar(fp_shap_values.columns, np.mean(np.abs(fp_shap_values.values), axis=0), label="FP")
    ax.bar(fn_shap_values.columns, np.mean(np.abs(fn_shap_values.values), axis=0), label="FN")
    ax.set_title(f"FP vs FN Feature Importance Comparison ({model_name})")
    ax.legend()
    
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_analysis_path, f"{model_name}_fp_fn_comparison.png"))
    plt.close()
    
    print(f"FP vs FN comparison saved to {shap_analysis_path}")


def generate_feature_importance_comparison(shap_data, shap_analysis_path, model_name):
    """生成 TP, TN, FP, FN 特征重要性对比图"""
    fp, fn, tp, tn = select_fp_fn_tp_tn(shap_data, model_name)
    
    # 获取每类样本的 SHAP 值
    tp_shap_values = shap_data[model_name]["shap_values"].iloc[tp.index]
    tn_shap_values = shap_data[model_name]["shap_values"].iloc[tn.index]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制每类样本的平均特征重要性
    ax.bar(tp_shap_values.columns, np.mean(np.abs(tp_shap_values.values), axis=0), label="TP")
    ax.bar(tn_shap_values.columns, np.mean(np.abs(tn_shap_values.values), axis=0), label="TN")
    
    ax.set_title(f"TP vs TN Feature Importance Comparison ({model_name})")
    ax.legend()
    
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_analysis_path, f"{model_name}_tp_tn_comparison.png"))
    plt.close()
    
    print(f"TP vs TN comparison saved to {shap_analysis_path}")


def main(shap_analysis_path):
    """主流程，生成 SHAP 图与报告"""
    shap_data = load_shap_data(shap_analysis_path)
    
    for model_name in shap_data:
        print(f"Generating plots for {model_name}...")
        
        # 创建保存路径
        model_save_path = os.path.join(shap_analysis_path, model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        # 生成 Beeswarm 图
        generate_beeswarm_plot(shap_data[model_name]["shap_values"], pd.read_csv(os.path.join(shap_analysis_path, "assets", "X_explain.csv")), model_save_path)
        
        # 随机选择一个 FP 或 FN 样本生成 Waterfall 和 Force 图
        fp, fn, tp, tn = select_fp_fn_tp_tn(shap_data, model_name)
        sample_idx = fp.index[0] if len(fp) > 0 else fn.index[0]  # 默认选择第一个 FP 或 FN 样本
        
        # 生成单样本 Waterfall/Force 图
        generate_waterfall_force_plot(shap_data[model_name]["shap_values"], pd.read_csv(os.path.join(shap_analysis_path, "assets", "X_explain.csv")), sample_idx, model_save_path)
        
        # 绘制 FP vs FN 比较图
        plot_fp_fn_comparison(shap_data, shap_analysis_path, model_name)
        
        # 绘制 TP vs TN 比较图
        generate_feature_importance_comparison(shap_data, shap_analysis_path, model_name)
        
    print("All plots generated successfully!")


if __name__ == "__main__":
    shap_analysis_path = "/path/to/shap_analysis_assets"  # 请替换为实际路径
    main(shap_analysis_path)