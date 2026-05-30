from __future__ import annotations

import os
import re
from typing import Callable, Optional, Sequence

np = None
pd = None
plt = None
shap = None

DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]

MANUAL_MAP = {
    "original_shape2D_Sphericity": "SHAPE: Sphericity",
    "original_shape2D_Elongation": "SHAPE: Elongation",
    "original_shape2D_Perimeter": "SHAPE: Perimeter",
    "original_shape2D_PerimeterSurfaceRatio": "SHAPE: Perim/Area",
    "original_shape2D_MajorAxisLength": "SHAPE: Major axis",
    "original_shape2D_MeshSurface": "SHAPE: Area",
    "original_shape2D_MinorAxisLength": "SHAPE: Minor axis",
    "original_firstorder_Skewness": "INT: Skewness",
    "original_firstorder_Minimum": "INT: Minimum",
    "original_firstorder_Median": "INT: Median",
    "original_firstorder_Energy": "INT: Energy",
    "original_firstorder_RobustMeanAbsoluteDeviation": "INT: Robust MAD",
    "original_firstorder_Kurtosis": "INT: Kurtosis",
    "original_firstorder_10Percentile": "INT: 10th pctl",
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
    "original_glszm_SmallAreaHighGrayLevelEmphasis": "SZM: SAHGLE",
    "original_glszm_LargeAreaHighGrayLevelEmphasis": "SZM: LAHGLE",
    "original_glszm_SmallAreaLowGrayLevelEmphasis": "SZM: SALGLE",
    "original_glszm_LargeAreaLowGrayLevelEmphasis": "SZM: LALGLE",
    "original_glszm_ZoneVariance": "SZM: ZoneVar",
    "original_glszm_ZoneEntropy": "SZM: ZoneEnt",
    "original_glcm_Correlation": "GLCM: Corr",
    "original_glcm_Contrast": "GLCM: Contrast",
    "original_glcm_Energy": "GLCM: Energy",
    "original_glcm_Homogeneity": "GLCM: Homog",
    "original_gldm_DependenceNonUniformity": "DM: DepNU",
    "original_gldm_DependenceVariance": "DM: DepVar",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis": "DM: LDHGLE",
    "original_gldm_LargeDependenceEmphasis": "DM: LDE",
    "original_ngtdm_Contrast": "NGTDM: Contrast",
    "original_ngtdm_Coarseness": "NGTDM: Coarse",
    "original_ngtdm_Complexity": "NGTDM: Complex",
}

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

TOKEN_REWRITE = {}


def _require_numpy():
    global np
    if np is None:
        import numpy as _np
        np = _np
    return np


def _require_pandas():
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd
    return pd


def _require_matplotlib_pyplot():
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt


def _require_shap():
    global shap
    if shap is None:
        import shap as _shap
        shap = _shap
    return shap


def _split_camel(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip()


def _rewrite_tokens(phrase: str) -> str:
    return " ".join(TOKEN_REWRITE.get(token, token) for token in phrase.split())


def _abbreviate_tail_words(phrase: str) -> str:
    words = phrase.split()
    if len(words) < 3:
        return phrase

    tail_initials = "".join(word[0].upper() for word in words[2:] if word)
    if not tail_initials:
        return " ".join(words[:2])

    return " ".join([words[0], words[1], tail_initials])


def paper_friendly_name(col: str) -> str:
    if col in MANUAL_MAP:
        return MANUAL_MAP[col]

    stripped = re.sub(r"^original_", "", col)
    match = re.match(r"^([A-Za-z0-9]+)_(.+)$", stripped)
    if match:
        group, name = match.group(1), match.group(2)
    else:
        group, name = "", stripped

    prefix = GROUP_PREFIX.get(group, group.upper() if group else "")
    display_name = _rewrite_tokens(_split_camel(name))
    display_name = re.sub(r"\s+", " ", display_name).strip()
    display_name = _abbreviate_tail_words(display_name)
    result = f"{prefix}: {display_name}" if prefix else display_name
    return result


def short_feature_name(col: str) -> str:
    stripped = re.sub(r"^original_", "", col)
    match = re.match(r"^([A-Za-z0-9]+)_(.+)$", stripped)
    if not match:
        return stripped

    group, name = match.group(1), match.group(2)
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
    group_display = group_map.get(group, group.upper() if group else "")
    return f"{group_display}: {name}" if group_display else name


def parse_training_csv_from_summary(summary_txt: str) -> str:
    with open(summary_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("Training CSV:"):
                return line.split("Training CSV:", 1)[1].strip()
    raise ValueError("Cannot find 'Training CSV:' in summary txt.")


def prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    drop_cols = [col for col in DROP_IF_PRESENT if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def reconstruct_X_explain(
    train_csv: str,
    label_col: str,
    background_samples: int,
    explain_samples: Optional[int],
) -> pd.DataFrame:
    np_mod = _require_numpy()
    pd_mod = _require_pandas()

    df = pd_mod.read_csv(train_csv)
    df = prepare_df(df, label_col)
    x_train = df.drop(columns=[label_col]).copy()

    np_mod.random.seed(42)
    if len(x_train) > background_samples:
        _ = np_mod.random.choice(len(x_train), size=background_samples, replace=False)

    if explain_samples is None:
        explain_samples = min(500, len(x_train))

    n_explain = min(explain_samples, len(x_train))
    if n_explain < len(x_train):
        explain_idx = np_mod.random.choice(len(x_train), size=n_explain, replace=False)
        return x_train.iloc[explain_idx].copy()
    return x_train.copy()


def load_shap_values(shap_values_csv: str) -> pd.DataFrame:
    pd_mod = _require_pandas()
    shap_df = pd_mod.read_csv(shap_values_csv)
    if len(shap_df.columns) > 0 and shap_df.columns[0].lower().startswith("unnamed"):
        shap_df = shap_df.drop(columns=[shap_df.columns[0]])
    return shap_df


def save_current_figure(
    out_path: str,
    export_formats: Sequence[str] = ("png",),
    dpi: int = 300,
    bbox_inches: Optional[str] = None,
) -> list[str]:
    plt_mod = _require_matplotlib_pyplot()
    base, ext = os.path.splitext(out_path)
    normalized = [fmt.lower().lstrip(".") for fmt in export_formats]
    if ext:
        default_format = ext.lower().lstrip(".")
        if default_format not in normalized:
            normalized.insert(0, default_format)
    saved_paths = []
    for fmt in normalized:
        target = f"{base}.{fmt}"
        plt_mod.savefig(target, dpi=dpi, bbox_inches=bbox_inches)
        saved_paths.append(target)
    return saved_paths


def save_beeswarm_plot(
    shap_values,
    x_features: pd.DataFrame,
    out_path: str,
    max_display: int,
    *,
    feature_name_formatter: Optional[Callable[[str], str]] = None,
    save_feature_name_map: bool = False,
    export_formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
    figsize: tuple[int, int] = (12, 8),
    plot_type: str = "dot",
) -> list[str]:
    np_mod = _require_numpy()
    pd_mod = _require_pandas()
    plt_mod = _require_matplotlib_pyplot()
    shap_mod = _require_shap()

    x_use = x_features.copy()
    if feature_name_formatter is not None:
        rename_map = {col: feature_name_formatter(col) for col in x_use.columns}
        x_use.rename(columns=rename_map, inplace=True)
        if save_feature_name_map:
            base, _ = os.path.splitext(out_path)
            (
                pd_mod.Series(rename_map, name="paper_name")
                .rename_axis("raw_name")
                .to_csv(f"{base}_feature_name_map.csv")
            )

    plt_mod.figure(figsize=figsize)
    shap_mod.summary_plot(
        np_mod.asarray(shap_values),
        x_use,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    plt_mod.tight_layout()
    saved_paths = save_current_figure(out_path, export_formats=export_formats, dpi=dpi)
    plt_mod.close()
    return saved_paths


def save_waterfall_plot(
    shap_values,
    feature_values,
    feature_names,
    base_value: float,
    out_path: str,
    max_display: int,
    *,
    title: Optional[str] = None,
    title_fontsize: float = 18.0,
    export_formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 150,
    figsize: tuple[int, int] = (12, 8),
    bbox_inches: str = "tight",
) -> list[str]:
    np_mod = _require_numpy()
    plt_mod = _require_matplotlib_pyplot()
    shap_mod = _require_shap()

    shap_values = np_mod.asarray(shap_values).reshape(1, -1)
    feature_values = np_mod.asarray(feature_values).reshape(1, -1)
    explanation = shap_mod.Explanation(
        values=shap_values,
        base_values=np_mod.array([base_value]),
        data=feature_values,
        feature_names=list(feature_names),
    )

    plt_mod.figure(figsize=figsize)
    shap_mod.plots.waterfall(explanation[0], show=False, max_display=max_display)
    if title:
        plt_mod.gcf().suptitle(title, fontsize=title_fontsize, y=0.98)
    plt_mod.tight_layout()
    saved_paths = save_current_figure(
        out_path,
        export_formats=export_formats,
        dpi=dpi,
        bbox_inches=bbox_inches,
    )
    plt_mod.close()
    return saved_paths
