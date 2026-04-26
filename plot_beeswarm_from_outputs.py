import argparse
import os
import re

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]


def parse_training_csv_from_summary(summary_txt: str) -> str:
    with open(summary_txt, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("Training CSV:"):
                return line.split("Training CSV:", 1)[1].strip()
    raise ValueError("Cannot find 'Training CSV:' in summary txt.")


def prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    drop_cols = [c for c in DROP_IF_PRESENT if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 与你的 SHAP 脚本一致：过滤 label==-1，label 转 int
    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def reconstruct_X_explain(train_csv: str, label_col: str, background_samples: int, explain_samples: int):
    df = pd.read_csv(train_csv)
    df = prepare_df(df, label_col)

    X_train = df.drop(columns=[label_col]).copy()

    # 与 SHAP 脚本一致：同一个 RNG 流程
    np.random.seed(42)

    # 先抽 background（即使后面不直接用，也要“消耗 RNG”，否则 explain_idx 会不同）
    if len(X_train) > background_samples:
        _ = np.random.choice(len(X_train), size=background_samples, replace=False)

    # 再抽 explain
    if explain_samples is not None:
        n_explain = min(explain_samples, len(X_train))
        if n_explain < len(X_train):
            explain_idx = np.random.choice(len(X_train), size=n_explain, replace=False)
            X_explain = X_train.iloc[explain_idx].copy()
        else:
            X_explain = X_train.copy()
    else:
        max_explain = 500
        if len(X_train) <= max_explain:
            X_explain = X_train.copy()
        else:
            explain_idx = np.random.choice(len(X_train), size=max_explain, replace=False)
            X_explain = X_train.iloc[explain_idx].copy()

    return X_explain


def load_shap_values(shap_values_csv: str) -> pd.DataFrame:
    sv = pd.read_csv(shap_values_csv)
    # 有时会多出 index 列
    if sv.columns[0].lower().startswith("unnamed"):
        sv = sv.drop(columns=[sv.columns[0]])
    return sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_txt", required=True, help="shap_analysis_summary.txt path")
    ap.add_argument("--shap_values_csv", required=True, help="*_shap_values.csv path (sample-level SHAP)")
    ap.add_argument("--label", default="label", help="label column name in training csv")
    ap.add_argument("--background_samples", type=int, default=100, help="must match SHAP run (default 100)")
    ap.add_argument("--explain_samples", type=int, default=500, help="must match SHAP run (default 500)")
    ap.add_argument("--max_display", type=int, default=20, help="Top-K features to show")
    ap.add_argument("--out_png", default="beeswarm.png", help="output png path")
    args = ap.parse_args()

    train_csv = parse_training_csv_from_summary(args.summary_txt)
    X_explain = reconstruct_X_explain(
        train_csv=train_csv,
        label_col=args.label,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples,
    )

    shap_df = load_shap_values(args.shap_values_csv)

    # 确保列对齐：SHAP 列顺序为准
    missing = [c for c in shap_df.columns if c not in X_explain.columns]
    if missing:
        raise ValueError(f"Some SHAP columns are missing in X_explain: {missing[:10]} ...")

    X_explain = X_explain.loc[:, shap_df.columns]

    # 画 beeswarm（summary plot）
    plt.figure()
    shap.summary_plot(
        shap_df.values,
        X_explain,
        plot_type="dot",
        max_display=args.max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Saved beeswarm to: {args.out_png}")


if __name__ == "__main__":
    main()
