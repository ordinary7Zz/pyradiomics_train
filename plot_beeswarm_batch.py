import os
import re
import argparse
import glob

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]

# 1) 对你最常见/最重要的特征做“人工高质量翻译”（缩短版，参考 shap_analyze_autogluon_fixed.py）
MANUAL_MAP = {
    # ---- Shape2D ----
    "original_shape2D_Sphericity": "SHAPE: Sphericity",
    "original_shape2D_Elongation": "SHAPE: Elongation",
    "original_shape2D_Perimeter": "SHAPE: Perimeter",
    "original_shape2D_PerimeterSurfaceRatio": "SHAPE: Perim/Area",
    "original_shape2D_MajorAxisLength": "SHAPE: Major axis",
    "original_shape2D_MeshSurface": "SHAPE: Area",
    "original_shape2D_MinorAxisLength": "SHAPE: Minor axis",

    # ---- First-order intensity ----
    "original_firstorder_Skewness": "INT: Skewness",
    "original_firstorder_Minimum": "INT: Minimum",
    "original_firstorder_Median": "INT: Median",
    "original_firstorder_Energy": "INT: Energy",
    "original_firstorder_RobustMeanAbsoluteDeviation": "INT: Robust MAD",
    "original_firstorder_Kurtosis": "INT: Kurtosis",
    "original_firstorder_10Percentile": "INT: 10th pctl",

    # ---- GLRLM (Run-length texture) ----
    "original_glrlm_LongRunHighGrayLevelEmphasis": "RLM: LRHGLE",
    "original_glrlm_ShortRunHighGrayLevelEmphasis": "RLM: SRHGLE",
    "original_glrlm_LongRunLowGrayLevelEmphasis": "RLM: LRLGLE",
    "original_glrlm_ShortRunLowGrayLevelEmphasis": "RLM: SRLGLE",
    "original_glrlm_ShortRunEmphasis": "RLM: SRE",
    "original_glrlm_LongRunEmphasis": "RLM: LRE",
    "original_glrlm_RunVariance": "RLM: RunVar",
    "original_glrlm_RunEntropy": "RLM: RunEnt",
    "original_glrlm_GrayLevelNonUniformity": "RLM: GLNU",
    "original_glrlm_RunLengthNonUniformity": "RLM: RLNU",

    # ---- GLSZM (Size-zone texture) ----
    "original_glszm_SmallAreaHighGrayLevelEmphasis": "SZM: SAHGLE",
    "original_glszm_LargeAreaHighGrayLevelEmphasis": "SZM: LAHGLE",
    "original_glszm_SmallAreaLowGrayLevelEmphasis": "SZM: SALGLE",
    "original_glszm_LargeAreaLowGrayLevelEmphasis": "SZM: LALGLE",
    "original_glszm_ZoneVariance": "SZM: ZoneVar",
    "original_glszm_ZoneEntropy": "SZM: ZoneEnt",

    # ---- GLCM ----
    "original_glcm_Correlation": "GLCM: Corr",
    "original_glcm_Contrast": "GLCM: Contrast",
    "original_glcm_Energy": "GLCM: Energy",
    "original_glcm_Homogeneity": "GLCM: Homog",

    # ---- GLDM (Dependence texture) ----
    "original_gldm_DependenceNonUniformity": "DM: DepNU",
    "original_gldm_DependenceVariance": "DM: DepVar",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis": "DM: LDHGLE",
    "original_gldm_LargeDependenceEmphasis": "DM: LDE",

    # ---- NGTDM ----
    "original_ngtdm_Contrast": "NGTDM: Contrast",
    "original_ngtdm_Coarseness": "NGTDM: Coarse",
    "original_ngtdm_Complexity": "NGTDM: Complex",
}

# 2) 通用规则：把剩下的也变“更论文友好”
GROUP_PREFIX = {
    "shape2D": "SHAPE",
    "shape": "SHAPE",
    "firstorder": "INT",
    "glcm": "TEX(GLCM)",
    "glrlm": "TEX(RLM)",
    "glszm": "TEX(SZM)",
    "gldm": "TEX(DM)",
    "ngtdm": "TEX(NGTDM)",
}

# 常见术语微调（让词更像论文）
TOKEN_REWRITE = {
    "GrayLevel": "intensity",
    "High": "high",
    "Low": "low",
    "LongRun": "Long-run",
    "ShortRun": "Short-run",
    "SmallArea": "Small-area",
    "LargeArea": "Large-area",
    "NonUniformity": "non-uniformity",
    "Variance": "variance",
    "Emphasis": "emphasis",
    "Dependence": "dependence",
    "Run": "run",
    "Zone": "zone",
}

def _split_camel(s: str) -> str:
    # CamelCase -> space separated
    # "PerimeterSurfaceRatio" -> "Perimeter Surface Ratio"
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).strip()

def _rewrite_tokens(phrase: str) -> str:
    # 先按空格拆，再对常见 token 做替换
    tokens = phrase.split()
    out = []
    for t in tokens:
        t2 = TOKEN_REWRITE.get(t, t)
        out.append(t2)
    # 让后半部分整体更像英文短语（小写为主）
    # 但保留开头首字母（例如 "Major axis length"）
    res = " ".join(out)
    return res


def _truncate_display_name(name: str, max_len: int = 28) -> str:
    """缩短过长特征名（参考 shap_analyze_autogluon_fixed.py）"""
    if len(name) <= max_len:
        return name
    # 常见长短语缩写
    abbrev = {
        "Long run high Gray Level emphasis": "LRHGLE",
        "Long run high intensity emphasis": "LRHGLE",
        "Short run high Gray Level emphasis": "SRHGLE",
        "Short run high intensity emphasis": "SRHGLE",
        "Long run low Gray Level emphasis": "LRLGLE",
        "Short run low Gray Level emphasis": "SRLGLE",
        "Small area high Gray Level emphasis": "SAHGLE",
        "Small area high intensity emphasis": "SAHGLE",
        "Large area high Gray Level emphasis": "LAHGLE",
        "Gray Level emphasis": "GLE",
        "intensity emphasis": "GLE",
        "Gray Level": "GL",
        "non-uniformity": "NU",
        "emphasis": "emp",
        "Dependence non-uniformity": "DepNU",
        "Dependence variance": "DepVar",
    }
    result = name
    for long_phrase, short in abbrev.items():
        result = result.replace(long_phrase, short)
    return result[:max_len] if len(result) > max_len else result


def paper_friendly_name(col: str) -> str:
    # 优先人工映射
    if col in MANUAL_MAP:
        return MANUAL_MAP[col]

    # 去掉原始前缀
    s = re.sub(r"^original_", "", col)

    # 拆 group + name: e.g., "glrlm_LongRunEmphasis"
    m = re.match(r"^([A-Za-z0-9]+)_(.+)$", s)
    if m:
        group, name = m.group(1), m.group(2)
    else:
        group, name = "", s

    prefix = GROUP_PREFIX.get(group, group.upper() if group else "")

    # 名字转成更可读短语
    name_spaced = _split_camel(name)          # Camel -> words
    name_rw = _rewrite_tokens(name_spaced)    # token rewrite

    # 一些小美化：把多余双空格去掉
    name_rw = re.sub(r"\s+", " ", name_rw).strip()

    result = f"{prefix}: {name_rw}" if prefix else name_rw
    return _truncate_display_name(result)


def short_feature_name(col: str) -> str:
    # 去掉 radiomics 常见前缀
    col = re.sub(r"^original_", "", col)

    # 拆成 group + name，例如 shape2D_Sphericity
    m = re.match(r"^([A-Za-z0-9]+)_(.+)$", col)
    if m:
        group, name = m.group(1), m.group(2)
    else:
        group, name = "", col

    # 组名美化：匹配你 summary 中出现的类别
    group_map = {
        "shape2D": "Shape2D",
        "shape": "Shape",
        "firstorder": "FirstOrder",
        "glcm": "GLCM",
        "glrlm": "GLRLM",
        "glszm": "GLSZM",
        "gldm": "GLDM",
        "ngtdm": "NGTDM",
    }
    group_disp = group_map.get(group, group.upper() if group else "")

    return f"{group_disp}: {name}" if group_disp else name


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

    # 与你 SHAP 脚本一致：过滤 label == -1，label 转 int
    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)
    return df

def reconstruct_X_explain(train_csv: str, label_col: str, background_samples: int, explain_samples: int):
    df = pd.read_csv(train_csv)
    df = prepare_df(df, label_col)
    X_train = df.drop(columns=[label_col]).copy()

    # 关键：与 SHAP 计算时一致的 RNG 流程
    np.random.seed(42)

    # 先抽 background（即使后面不用，也要消耗 RNG，保证 explain_idx 一致）
    if len(X_train) > background_samples:
        _ = np.random.choice(len(X_train), size=background_samples, replace=False)

    # 再抽 explain
    n_explain = min(explain_samples, len(X_train))
    if n_explain < len(X_train):
        explain_idx = np.random.choice(len(X_train), size=n_explain, replace=False)
        X_explain = X_train.iloc[explain_idx].copy()
    else:
        X_explain = X_train.copy()

    return X_explain

def load_shap_values(shap_values_csv: str) -> pd.DataFrame:
    sv = pd.read_csv(shap_values_csv)
    if len(sv.columns) > 0 and sv.columns[0].lower().startswith("unnamed"):
        sv = sv.drop(columns=[sv.columns[0]])
    return sv

def plot_beeswarm(shap_df: pd.DataFrame, X_explain: pd.DataFrame, out_png: str, max_display: int):
    missing = [c for c in shap_df.columns if c not in X_explain.columns]
    if missing:
        raise ValueError(f"X_explain missing columns: {missing[:10]} ...")

    X_use = X_explain.loc[:, shap_df.columns].copy()

    # ✅ 关键：改显示名（只影响图，不影响对齐）
    rename_map = {c: paper_friendly_name(c) for c in shap_df.columns}
    X_use.rename(columns=rename_map, inplace=True)

    # 基础文件名（不含扩展名）
    base, _ = os.path.splitext(out_png)

    # ✅ 保存映射表（论文附录可用）
    map_path = f"{base}_feature_name_map.csv"
    (pd.Series(rename_map, name="paper_name")
       .rename_axis("raw_name")
       .to_csv(map_path))

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_df.values,
        X_use,
        plot_type="dot",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()

    # 同一张图导出为 png/svg/pdf 三种格式
    for ext in ("png", "svg", "pdf"):
        out_path = f"{base}.{ext}"
        plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_txt", required=True, help="shap_analysis_summary.txt path")
    ap.add_argument("--shap_dir", required=True, help="directory containing *_shap_values.csv")
    ap.add_argument("--label", default="label", help="label column in training csv")
    ap.add_argument("--background_samples", type=int, default=100)
    ap.add_argument("--explain_samples", type=int, default=500)
    ap.add_argument("--max_display", type=int, default=20)
    ap.add_argument("--out_dir", default=None, help="output dir (default: <shap_dir>/plots)")
    ap.add_argument("--pattern", default="*_shap_values.csv", help="glob pattern for shap values files")
    args = ap.parse_args()

    train_csv = parse_training_csv_from_summary(args.summary_txt)
    X_explain = reconstruct_X_explain(
        train_csv=train_csv,
        label_col=args.label,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples,
    )

    out_dir = args.out_dir or os.path.join(args.shap_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.shap_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.shap_dir, args.pattern)}")

    for f in files:
        name = os.path.basename(f).replace("_shap_values.csv", "")
        out_png = os.path.join(out_dir, f"{name}_beeswarm_top{args.max_display}.png")

        shap_df = load_shap_values(f)
        # 空文件/列异常跳过
        if shap_df.shape[1] == 0 or shap_df.shape[0] == 0:
            print(f"[SKIP] empty shap file: {f}")
            continue

        try:
            plot_beeswarm(shap_df, X_explain, out_png, args.max_display)
            base, _ = os.path.splitext(out_png)
            outputs = ", ".join(f"{base}.{ext}" for ext in ("png", "svg", "pdf"))
            print(f"[OK] {name} -> {outputs}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

if __name__ == "__main__":
    main()
