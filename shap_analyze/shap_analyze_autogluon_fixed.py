"""
稳定版SHAP分析脚本（支持AutoGluon BAG树模型）。

主要功能：
- 加载AutoGluon模型与训练CSV，构建背景样本和解释样本。
- 针对主模型计算SHAP值，输出标准化CSV结果。
- 可选生成beeswarm、waterfall和紧凑版局部SHAP条形图，并保存分析摘要。

输出目录默认为 <model_dir>/shap_analysis。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plots.plotting_utils import (
    paper_friendly_name,
    save_beeswarm_plot,
    save_current_figure,
    save_waterfall_plot,
    short_feature_name,
)

np = None
pd = None
TabularPredictor = None
DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]

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


def _require_tabular_predictor():
    global TabularPredictor
    if TabularPredictor is None:
        from autogluon.tabular import TabularPredictor as _TabularPredictor
        TabularPredictor = _TabularPredictor
    return TabularPredictor


def _require_matplotlib_pyplot():
    import matplotlib.pyplot as _plt
    return _plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP analysis for AutoGluon TabularPredictor models (fixed for BAG models)."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="AutoGluon model directory (contains predictor.pkl)",
    )
    p.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training CSV file (used for SHAP background and analysis)",
    )
    p.add_argument(
        "--label",
        type=str,
        default="label",
        help="Label column name",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for SHAP results (default: <model_dir>/shap_analysis)",
    )
    p.add_argument(
        "--background_samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP (default: 100)",
    )
    p.add_argument(
        "--explain_samples",
        type=int,
        default=None,
        help="Number of samples to explain (default: all test samples or 500 if too large)",
    )
    p.add_argument(
        "--skip_neural_net",
        action="store_true",
        help="Skip neural network models (NeuralNetFastAI) in SHAP analysis",
    )
    p.add_argument(
        "--main_models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of main model names to analyze (default: auto-detect from ensemble)",
    )
    p.add_argument(
        "--plot_waterfall",
        action="store_true",
        help="Generate SHAP waterfall plots and compact local SHAP bars with balanced positive/negative samples",
    )
    p.add_argument(
        "--waterfall_samples",
        type=int,
        default=3,
        help="Total samples per category, split between positive and negative examples when possible (default: 3)",
    )
    p.add_argument(
        "--sample_filename",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional: additionally plot waterfall figures for one or more samples with these "
            "filenames (should match the CSV 'filename' column or the basename of 'image_path')."
        ),
    )
    p.add_argument(
        "--plot_beeswarm_for",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional: names of model(s) to generate SHAP beeswarm plots for, "
            "e.g. LightGBM_BAG_L1."
        ),
    )
    p.add_argument(
        "--top_features",
        type=int,
        default=5,
        help=(
            "Number of top features (by absolute SHAP value) to display in each "
            "waterfall and beeswarm plot (default: 5)."
        ),
    )
    return p.parse_args()


def _prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    drop_cols = [c for c in DROP_IF_PRESENT if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def _extract_ensemble_weights_from_log(log_path: str) -> Optional[Dict[str, float]]:
    """Extract ensemble weights from predictor_log.txt"""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for "Ensemble Weights:" line
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
            # Format: 'ModelName': 0.467
            model_match = re.search(r"'([^']+)':\s*([\d.]+)", item)
            if model_match:
                model_name = model_match.group(1)
                weight = float(model_match.group(2))
                weights[model_name] = weight

        return weights if weights else None
    except Exception as e:
        print(f"Warning: Failed to extract ensemble weights from log: {e}")
        return None


def _get_main_models(
    predictor: TabularPredictor,
    model_dir: str,
    main_models: Optional[List[str]] = None,
) -> List[str]:
    """Get list of main models from ensemble or use provided list"""
    if main_models is not None:
        return main_models

    # Try to get from log file
    log_path = os.path.join(model_dir, "logs", "predictor_log.txt")
    weights = _extract_ensemble_weights_from_log(log_path)
    if weights:
        print(f"Found ensemble weights from log: {weights}")
        return list(weights.keys())

    # Fallback: get from best model (WeightedEnsemble)
    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_model_name = leaderboard.iloc[0]["model"]
        print(f"Best model: {best_model_name}")

        # Try to get the ensemble model and extract sub-models
        try:
            if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                ensemble_model = predictor._trainer.load_model(best_model_name)
                if hasattr(ensemble_model, "model_names"):
                    return ensemble_model.model_names
        except Exception:
            pass

        # If we can't extract, use top 5 non-ensemble models
        print("Warning: Could not extract ensemble sub-models. Using top 5 non-ensemble models.")
        non_ensemble = leaderboard[~leaderboard["model"].str.contains("Ensemble", case=False)]
        return non_ensemble.head(5)["model"].tolist()
    except Exception as e:
        print(f"Warning: Could not determine main models automatically: {e}")
        print("Using default: top models from leaderboard")
        leaderboard = predictor.leaderboard(silent=True)
        return leaderboard.head(5)["model"].tolist()


def _get_preprocessed_features(
    model, X: pd.DataFrame, model_name: str, predictor: Optional[TabularPredictor] = None
) -> Optional[pd.DataFrame]:
    """
    Get preprocessed features from AutoGluon model.
    This is critical for BAG models which have feature engineering steps.
    Uses predictor's feature_generator which is the most reliable method.
    """
    # Method 1: Use predictor's feature_generator (most reliable)
    if predictor is not None:
        try:
            if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
                X_processed = predictor._learner.feature_generator.transform(X)
                if isinstance(X_processed, pd.DataFrame):
                    return X_processed
                elif isinstance(X_processed, np.ndarray):
                    # Try to get feature names from feature_generator
                    feature_names = None
                    if hasattr(predictor._learner.feature_generator, "feature_metadata_in"):
                        try:
                            feature_metadata = predictor._learner.feature_generator.feature_metadata_in
                            if hasattr(feature_metadata, "get_features"):
                                feature_names = feature_metadata.get_features()
                        except:
                            pass
                    
                    if feature_names is None or len(feature_names) != X_processed.shape[1]:
                        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                    
                    return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            print(f"    feature_generator.transform failed: {e}")
    
    # Method 2: Try fold model's _preprocess method (for BAG models)
    if hasattr(model, "models"):
        try:
            models_list = model.models
            if models_list and len(models_list) > 0:
                fold_name = models_list[0]
                if isinstance(fold_name, str) and predictor is not None:
                    # Load fold model
                    from autogluon.common.loaders import load_pkl
                    model_path = os.path.join(predictor.path, "models", model_name, fold_name, "model.pkl")
                    if os.path.exists(model_path):
                        fold_model = load_pkl.load(path=model_path)
                        if hasattr(fold_model, "_preprocess"):
                            try:
                                X_processed = fold_model._preprocess(X, fit=False)
                                if isinstance(X_processed, pd.DataFrame):
                                    return X_processed
                                elif isinstance(X_processed, np.ndarray):
                                    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                                    return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
                            except Exception as e:
                                print(f"    fold_model._preprocess failed: {e}")
        except Exception as e:
            print(f"    Error accessing fold models: {e}")
    
    # Method 3: Try model's _preprocess method
    if hasattr(model, "_preprocess"):
        try:
            X_processed = model._preprocess(X, fit=False)
            if isinstance(X_processed, pd.DataFrame):
                return X_processed
            elif isinstance(X_processed, np.ndarray):
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            print(f"    model._preprocess failed: {e}")
    
    # Method 4: Try preprocess method
    if hasattr(model, "preprocess"):
        try:
            X_processed = model.preprocess(X)
            if isinstance(X_processed, pd.DataFrame):
                return X_processed
            elif isinstance(X_processed, np.ndarray):
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            print(f"    model.preprocess failed: {e}")
    
    return None


def _get_tree_model_from_bag(
    model, model_name: str, predictor: TabularPredictor
) -> Optional[Tuple]:
    """
    Extract tree model and preprocessing function from BAG model.
    Returns (tree_model, preprocess_func) or None.
    """
    # Get fold model path
    if not hasattr(model, "models"):
        return None
    
    try:
        models_list = model.models
        if not models_list or len(models_list) == 0:
            return None
        
        fold_name = models_list[0]
        if not isinstance(fold_name, str):
            return None
        
        # Load fold model from file system
        from autogluon.common.loaders import load_pkl
        model_path = os.path.join(predictor.path, "models", model_name, fold_name, "model.pkl")
        if not os.path.exists(model_path):
            return None
        
        fold_model = load_pkl.load(path=model_path)
        
        # Extract tree model recursively
        tree_model = fold_model
        while tree_model is not None:
            if hasattr(tree_model, "_model") and tree_model._model is not None:
                tree_model = tree_model._model
            elif hasattr(tree_model, "model") and tree_model.model is not None:
                tree_model = tree_model.model
            else:
                break
        
        # Check if it's a tree model
        model_type_str = str(type(tree_model))
        if "lightgbm" in model_type_str.lower() or "xgboost" in model_type_str.lower() or "catboost" in model_type_str.lower():
            # Create preprocessing function using predictor's feature_generator
            def preprocess_func(X):
                return _get_preprocessed_features(fold_model, X, model_name, predictor)
            
            return (tree_model, preprocess_func)
        
    except Exception as e:
        print(f"    Error extracting tree model from BAG: {e}")
    
    return None


def _compute_shap_tree_bag(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, 
    model_name: str, predictor: TabularPredictor
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for BAG tree models.
    Uses the full AutoGluon model pipeline but extracts tree model for TreeExplainer.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Computing SHAP for BAG model {model_name}...")
    
    # Try to extract tree model and preprocessing function
    tree_result = _get_tree_model_from_bag(model, model_name, predictor)
    
    if tree_result is None:
        print(f"  Could not extract tree model, using KernelExplainer...")
        return _compute_shap_kernel(model, X_background, X_explain, model_name)
    
    tree_model, preprocess_func = tree_result
    print(f"  Extracted tree model type: {type(tree_model)}")
    
    # Get preprocessed features
    print(f"  Getting preprocessed features using predictor's feature_generator...")
    X_background_processed = preprocess_func(X_background)
    X_explain_processed = preprocess_func(X_explain)
    
    if X_background_processed is None or X_explain_processed is None:
        print(f"  Could not get preprocessed features, using KernelExplainer...")
        return _compute_shap_kernel(model, X_background, X_explain, model_name)
    
    print(f"  Preprocessed features: {X_background_processed.shape[1]} (from {X_background.shape[1]} original)")
    
    # Verify feature count matches model expectations
    if hasattr(tree_model, "num_feature"):
        expected_features = tree_model.num_feature()
        actual_features = X_explain_processed.shape[1]
        
        if actual_features != expected_features:
            print(f"  Feature mismatch: data has {actual_features}, model expects {expected_features}")
            print(f"  This suggests preprocessing didn't work correctly. Using KernelExplainer...")
            return _compute_shap_kernel(model, X_background, X_explain, model_name)
        else:
            print(f"  Feature count verified: {actual_features} features match model expectations")
    
    print(f"  Using TreeExplainer with {X_explain_processed.shape[1]} features...")
    
    # Convert to numpy arrays
    X_background_array = X_background_processed.values if isinstance(X_background_processed, pd.DataFrame) else X_background_processed
    X_explain_array = X_explain_processed.values if isinstance(X_explain_processed, pd.DataFrame) else X_explain_processed
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_explain_array, X_background_array)
    
    # Handle binary classification: shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class for binary
    
    # Ensure it's numpy array
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)
    
    # Create DataFrame with preprocessed feature names
    # Use the feature names from the preprocessed DataFrame
    if isinstance(X_explain_processed, pd.DataFrame):
        feature_names = X_explain_processed.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    # Ensure feature count matches
    if len(feature_names) != shap_values.shape[1]:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_explain.index)
    
    return shap_values, shap_df


def _compute_shap_kernel(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, model_name: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values using KernelExplainer (for non-tree models or when TreeExplainer fails)"""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Using KernelExplainer for {model_name}...")
    print(f"    Warning: KernelExplainer is slow for large datasets. Using {len(X_background)} background samples.")

    def model_wrapper(X):
        X_df = pd.DataFrame(X, columns=X_background.columns, index=range(len(X)))
        # AutoGluon model expects DataFrame
        try:
            proba = model.predict_proba(X_df)
            if isinstance(proba, pd.DataFrame):
                if 1 in proba.columns:
                    return proba[1].values  # Binary classification: positive class
                return proba.iloc[:, -1].values
            elif isinstance(proba, np.ndarray):
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]  # Binary classification: positive class
                return proba.flatten()
            else:
                return np.array(proba).flatten()
        except Exception as e:
            print(f"    Error in model_wrapper: {e}")
            raise

    explainer = shap.KernelExplainer(model_wrapper, X_background.values)
    shap_values = explainer.shap_values(X_explain.values, nsamples=100)

    feature_names = X_explain.columns.tolist()
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_explain.index)

    return shap_values, shap_df


def _is_tree_model(model_name: str) -> bool:
    """Check if model is a tree-based model"""
    tree_keywords = [
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "RandomForest",
        "ExtraTrees",
    ]
    return any(kw in model_name for kw in tree_keywords)


def _is_bag_model(model_name: str) -> bool:
    """Check if model is a BAG model"""
    return "_BAG_" in model_name


def _abbreviate_feature_name(feature_name: str, max_len: int = 24) -> str:
    pretty_name = paper_friendly_name(feature_name)
    if len(pretty_name) <= max_len:
        return pretty_name

    short_name = short_feature_name(feature_name)
    if len(short_name) <= max_len:
        return short_name

    return pretty_name[: max(1, max_len - 1)].rstrip() + "…"


def _save_compact_shap_bar_plot(
    shap_values,
    feature_names,
    out_path: str,
    max_display: int,
    *,
    title: Optional[str] = None,
    title_fontsize: float = 14.0,
    xlabel_fontsize: float = 11.0,
    ytick_fontsize: float = 10.0,
    export_formats=("png", "svg"),
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
) -> list[str]:
    np_mod = _require_numpy()
    plt_mod = _require_matplotlib_pyplot()

    shap_arr = np_mod.asarray(shap_values).reshape(-1)
    feature_names = list(feature_names)
    if len(shap_arr) != len(feature_names):
        raise ValueError("shap_values and feature_names must have the same length")

    max_display = max(1, min(int(max_display), len(shap_arr)))
    top_indices = np_mod.argsort(np_mod.abs(shap_arr))[-max_display:][::-1]
    display_values = shap_arr[top_indices]
    display_names = [feature_names[i] for i in top_indices]

    def _compact_label(raw_name: str) -> str:
        candidates = [
            paper_friendly_name(raw_name),
            short_feature_name(raw_name),
            _abbreviate_feature_name(raw_name, max_len=18),
        ]
        for candidate in candidates:
            candidate = candidate.replace("\n", " ").strip()
            if len(candidate) <= 18:
                return candidate
        return _abbreviate_feature_name(raw_name, max_len=18)

    display_labels = [_compact_label(name) for name in display_names]

    if figsize is None:
        height = max(2.2, 0.52 * len(display_labels) + 1.15)
        figsize = (3.15, height)

    fig, ax = plt_mod.subplots(figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np_mod.arange(len(display_labels))
    ax.set_ylim(len(display_labels) - 0.5, -0.5)

    pos_color = "#c44e52"
    neg_color = "#4c72b0"
    colors = [pos_color if value >= 0 else neg_color for value in display_values]
    ax.barh(y_pos, display_values, color=colors, height=0.88, edgecolor="none", linewidth=0, zorder=2)
    ax.axvline(0, color="#6f6f6f", lw=0.85, zorder=1)

    max_abs = float(np_mod.max(np_mod.abs(display_values))) if len(display_values) else 0.0
    if max_abs > 0:
        ax.set_xlim(-max_abs * 1.32, max_abs * 1.32)

    from matplotlib.transforms import blended_transform_factory

    label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, value, label in zip(y_pos, display_values, display_labels):
        if value < 0:
            x_pos = 0.985
            ha = "right"
        else:
            x_pos = 0.015
            ha = "left"
        ax.text(
            x_pos,
            y,
            label,
            transform=label_transform,
            ha=ha,
            va="center",
            fontsize=ytick_fontsize,
            color="#222222",
            clip_on=False,
            zorder=3,
        )

    wrapped_title = None
    if title:
        wrapped_title = "\n".join(
            textwrap.wrap(
                title,
                width=22,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        if not wrapped_title:
            wrapped_title = title

    if wrapped_title:
        ax.set_title(wrapped_title, fontsize=title_fontsize, pad=4, color="#111111")

    ax.set_xlabel("SHAP value", fontsize=xlabel_fontsize, labelpad=2, color="#222222")
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", linestyle="--", alpha=0.12, linewidth=0.5, color="#9a9a9a")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", labelsize=max(7, xlabel_fontsize - 1), colors="#222222", length=2.5, width=0.6)
    ax.margins(x=0.01, y=0.03)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.22)
    saved_paths = save_current_figure(out_path, export_formats=export_formats, dpi=dpi, bbox_inches="tight")
    plt_mod.close(fig)
    return saved_paths


def _plot_waterfall_samples(
    predictor: TabularPredictor,
    results: Dict,
    X_explain: pd.DataFrame,
    y_explain: pd.Series,
    output_dir: str,
    n_samples: int = 3,
    label_col: str = "label",
    sample_ids: Optional[pd.Series] = None,
    sample_filenames: Optional[List[str]] = None,
    n_top_features: int = 5,
) -> None:
    """
    Identify balanced positive/negative samples and plot SHAP waterfall plots.

    Args:
        predictor: AutoGluon predictor
        results: Dictionary containing SHAP results for each model
        X_explain: Features to explain
        y_explain: True labels
        output_dir: Output directory for plots
        n_samples: Total samples per category, split between positive and negative when possible
        label_col: Label column name
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: shap or matplotlib not available. Skipping waterfall plots.")
        return
    
    # Enlarge fonts for all elements in the generated figures
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )
    
    # Get predictions for all explained samples
    print(f"  Getting predictions for {len(X_explain)} samples...")
    y_pred = predictor.predict(X_explain)
    y_proba = predictor.predict_proba(X_explain)
    
    # For binary classification, get probability of positive class
    if isinstance(y_proba, pd.DataFrame):
        if 1 in y_proba.columns:
            proba_positive = y_proba[1].values
        else:
            proba_positive = y_proba.iloc[:, -1].values
    else:
        proba_positive = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    
    # Convert predictions to numpy array if needed
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_explain, pd.Series):
        y_explain = y_explain.values
    
    # Identify correct and incorrect predictions
    correct_mask = (y_pred == y_explain)
    incorrect_mask = ~correct_mask

    print(f"  Correct predictions: {correct_mask.sum()}/{len(y_explain)} ({correct_mask.sum()/len(y_explain)*100:.1f}%)")
    print(f"  Incorrect predictions: {incorrect_mask.sum()}/{len(y_explain)} ({incorrect_mask.sum()/len(y_explain)*100:.1f}%)")

    # ---- 1) 基于“分类效果”划分为最好/中等/最差三类样本 ----
    # 定义一个“质量分数”：正确且置信度高 -> 分数高；错误且置信度高 -> 分数低（取负）
    conf_pred = np.where(
        y_pred == 1,
        proba_positive,
        1 - proba_positive,
    )
    quality_score = np.where(
        y_pred == y_explain,
        conf_pred,
        -conf_pred,
    )

    y_true = np.asarray(y_explain)
    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]
    positive_target = (n_samples + 1) // 2
    negative_target = n_samples // 2

    def _sort_indices_by_score(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=int)
        return indices[np.argsort(scores[indices])[::-1]]

    def _take_middle(sorted_indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(sorted_indices) == 0:
            return np.array([], dtype=int)
        count = min(count, len(sorted_indices))
        center = len(sorted_indices) // 2
        start = max(0, center - count // 2)
        end = start + count
        if end > len(sorted_indices):
            end = len(sorted_indices)
            start = max(0, end - count)
        return sorted_indices[start:end]

    def _take_best(sorted_indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(sorted_indices) == 0:
            return np.array([], dtype=int)
        return sorted_indices[: min(count, len(sorted_indices))]

    def _take_worst(sorted_indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(sorted_indices) == 0:
            return np.array([], dtype=int)
        count = min(count, len(sorted_indices))
        return sorted_indices[-count:]

    def _sort_by_confidence(indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=int)
        return indices[np.argsort(conf_pred[indices])[::-1]]

    def _select_balanced_quality(indices: np.ndarray, count: int, mode: str) -> np.ndarray:
        sorted_indices = _sort_indices_by_score(indices, quality_score)
        if mode == "best":
            return _take_best(sorted_indices, count)
        if mode == "medium":
            return _take_middle(sorted_indices, count)
        if mode == "worst":
            return _take_worst(sorted_indices, count)
        raise ValueError(f"Unknown quality mode: {mode}")

    def _select_balanced_correct(indices: np.ndarray, count: int) -> np.ndarray:
        selected = indices[correct_mask[indices]]
        return _take_best(_sort_by_confidence(selected), count)

    pos_best = _select_balanced_quality(positive_indices, positive_target, "best")
    neg_best = _select_balanced_quality(negative_indices, negative_target, "best")
    pos_medium = _select_balanced_quality(positive_indices, positive_target, "medium")
    neg_medium = _select_balanced_quality(negative_indices, negative_target, "medium")
    pos_worst = _select_balanced_quality(positive_indices, positive_target, "worst")
    neg_worst = _select_balanced_quality(negative_indices, negative_target, "worst")
    pos_correct = _select_balanced_correct(positive_indices, positive_target)
    neg_correct = _select_balanced_correct(negative_indices, negative_target)

    best_indices = _sort_indices_by_score(np.concatenate([pos_best, neg_best]), quality_score)
    medium_indices = _sort_indices_by_score(np.concatenate([pos_medium, neg_medium]), quality_score)
    worst_indices = _sort_indices_by_score(np.concatenate([pos_worst, neg_worst]), quality_score)
    correct_indices = _sort_by_confidence(np.concatenate([pos_correct, neg_correct]))

    print(
        f"  Balanced waterfall targets per category - total: {n_samples}, positive: {positive_target}, negative: {negative_target}"
    )
    print(
        f"  Selected counts - best: {len(best_indices)}, medium: {len(medium_indices)}, worst: {len(worst_indices)}, "
        f"correct: {len(correct_indices)}"
    )

    if len(positive_indices) == 0:
        print("  Warning: No positive samples found. Positive half will be empty.")
    if len(negative_indices) == 0:
        print("  Warning: No negative samples found. Negative half will be empty.")
    if len(positive_indices) < positive_target:
        print("  Warning: Not enough positive samples to fill the positive half for every category.")
    if len(negative_indices) < negative_target:
        print("  Warning: Not enough negative samples to fill the negative half for every category.")
    if len(pos_correct) < positive_target or len(neg_correct) < negative_target:
        print("  Warning: Not enough correctly predicted positive/negative samples to fill the balanced correct category.")
    
    # Prepare aligned image IDs (if available) and container for selected samples
    sample_ids_array = None
    if sample_ids is not None:
        try:
            # Align sample_ids to X_explain order
            aligned_ids = sample_ids.loc[X_explain.index]
        except Exception:
            aligned_ids = sample_ids
        sample_ids_array = np.array(aligned_ids)
    waterfall_records = []

    # Find indices for user-specified filename(s) (if requested)
    specified_indices = None
    if sample_filenames is not None and sample_ids_array is not None:
        # 允许传入一个或多个文件名；如果只传一个也兼容
        if isinstance(sample_filenames, str):
            filenames_list = [sample_filenames]
        else:
            filenames_list = list(sample_filenames)

        collected_indices: List[int] = []
        for fname in filenames_list:
            try:
                # 优先直接匹配
                direct_matches = np.where(sample_ids_array == fname)[0]

                if len(direct_matches) == 0:
                    # 再用 basename 匹配一次
                    target_base = os.path.basename(str(fname))
                    bases = np.array([os.path.basename(str(x)) for x in sample_ids_array])
                    direct_matches = np.where(bases == target_base)[0]

                if len(direct_matches) == 0:
                    print(f"  Warning: sample_filename '{fname}' not found among explained samples.")
                else:
                    idx0 = int(direct_matches[0])
                    collected_indices.append(idx0)
                    print(f"  sample_filename '{fname}' matched explained index {idx0}.")
            except Exception as e:
                print(f"  Warning: failed to locate sample_filename '{fname}': {e}")

        if collected_indices:
            # 去重并保持索引为 numpy 数组
            specified_indices = np.array(sorted(set(collected_indices)), dtype=int)
    
    # Plot waterfall plots for each model
    for model_name, model_results in results.items():
        shap_values = model_results["shap_values"]
        shap_df = model_results["shap_df"]
        
        # Get preprocessed features for this model
        X_processed = None
        try:
            # Try to get preprocessed features using predictor's feature_generator
            if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
                X_processed = predictor._learner.feature_generator.transform(X_explain)
                if not isinstance(X_processed, pd.DataFrame):
                    X_processed = pd.DataFrame(X_processed, columns=shap_df.columns, index=X_explain.index)
        except Exception as e:
            print(f"    Warning: Could not get preprocessed features: {e}")
            X_processed = X_explain  # Fallback to original features
        
        def _plot_category(indices, category_label: str, filename_tag: str, title_tag: str) -> None:
            if indices is None or len(indices) == 0:
                return
            print(f"  Plotting {len(indices)} {category_label} samples for {model_name}...")
            for i, idx in enumerate(indices):
                try:
                    sample_shap = shap_values[idx]
                    sample_features = X_processed.iloc[idx] if X_processed is not None else X_explain.iloc[idx]

                    if sample_shap.ndim > 1:
                        sample_shap = sample_shap.flatten()
                    elif sample_shap.ndim == 0:
                        sample_shap = np.array([sample_shap])

                    max_display = max(1, int(n_top_features))
                    abs_shap = np.abs(sample_shap)
                    top_indices = np.argsort(abs_shap)[-max_display:][::-1]

                    top_shap = sample_shap[top_indices]
                    top_feature_names = [shap_df.columns[i] for i in top_indices]

                    top_feature_values = []
                    if isinstance(sample_features, pd.Series):
                        for name in top_feature_names:
                            if name in sample_features.index:
                                top_feature_values.append(sample_features[name])
                            else:
                                try:
                                    idx_in_shap = shap_df.columns.get_loc(name)
                                    if idx_in_shap < len(sample_features):
                                        top_feature_values.append(sample_features.iloc[idx_in_shap])
                                    else:
                                        top_feature_values.append(np.nan)
                                except (KeyError, IndexError):
                                    top_feature_values.append(np.nan)

                    nan_count = sum(1 for v in top_feature_values if pd.isna(v))
                    if nan_count > 0:
                        print(
                            f"    Warning: {nan_count} out of {len(top_feature_values)} "
                            f"top features have nan values"
                        )
                        print(f"    Top feature names: {top_feature_names[:3]}...")
                        if isinstance(sample_features, pd.Series):
                            print(f"    Sample features index (first 5): {list(sample_features.index[:5])}")
                    elif isinstance(sample_features, pd.DataFrame):
                        top_feature_values = [
                            sample_features.iloc[0, sample_features.columns.get_loc(name)]
                            if name in sample_features.columns
                            else np.nan
                            for name in top_feature_names
                        ]
                    else:
                        if isinstance(sample_features, np.ndarray):
                            top_feature_values = (
                                sample_features[top_indices]
                                if len(sample_features) > max(top_indices)
                                else [np.nan] * len(top_indices)
                            )
                        elif isinstance(sample_features, pd.Series):
                            top_feature_values = [
                                sample_features.iloc[j] if j < len(sample_features) else np.nan
                                for j in top_indices
                            ]
                        else:
                            top_feature_values = [np.nan] * len(top_indices)

                    display_shap = top_shap
                    display_feature_names = [_abbreviate_feature_name(name, max_len=26) for name in top_feature_names]

                    compact_feature_names = [paper_friendly_name(name) for name in top_feature_names]

                    display_feature_values = np.array(top_feature_values)
                    if display_feature_values.ndim > 1:
                        display_feature_values = display_feature_values.flatten()

                    base_value = float(np.mean(proba_positive))

                    waterfall_plot_file = os.path.join(
                        output_dir, f"{model_name}_waterfall_{filename_tag}_{i+1}.png"
                    )
                    waterfall_saved_paths = save_waterfall_plot(
                        display_shap,
                        display_feature_values,
                        display_feature_names,
                        base_value,
                        waterfall_plot_file,
                        max_display,
                        title=f"{model_name} {title_tag} SHAP",
                        title_fontsize=20,
                        export_formats=("png", "svg"),
                        dpi=150,
                        figsize=(13.0, 9.0),
                        bbox_inches="tight",
                    )

                    compact_bar_file = os.path.join(
                        output_dir, f"{model_name}_compact_shap_bar_{filename_tag}_{i+1}.png"
                    )
                    compact_bar_saved_paths = _save_compact_shap_bar_plot(
                        display_shap,
                        compact_feature_names,
                        compact_bar_file,
                        max_display,
                        title=f"{model_name} {title_tag} SHAP",
                        title_fontsize=10.5,
                        xlabel_fontsize=9.0,
                        ytick_fontsize=8.8,
                        export_formats=("png", "svg"),
                        dpi=300,
                        figsize=(3.15, 3.15 * 4 / 3),
                    )
                    print(
                        f"    Saved waterfall: {', '.join(waterfall_saved_paths)}; "
                        f"compact bar: {', '.join(compact_bar_saved_paths)}"
                    )

                    if sample_ids_array is not None:
                        img_name = sample_ids_array[idx] if idx < len(sample_ids_array) else None
                        base, _ = os.path.splitext(waterfall_plot_file)
                        waterfall_records.append(
                            {
                                "model": model_name,
                                "category": category_label,
                                "figure_index": i + 1,
                                "image_name": img_name,
                                "true_label": int(y_explain[idx]),
                                "pred_label": int(y_pred[idx]),
                                "prob_positive": float(proba_positive[idx]),
                                # 记录基础文件名（不含扩展名），实际生成 png/svg 两个文件，并额外输出 compact bar 图
                                "plot_file": os.path.basename(base),
                            }
                        )
                except Exception as e:
                    print(f"    Error plotting {category_label} sample {i+1}: {e}")
                    import traceback
                    traceback.print_exc()

        # 按正负例均衡绘制四个类别：best / medium / worst / correct
        _plot_category(best_indices, "best", "best", "Best-quality")
        _plot_category(medium_indices, "medium", "medium", "Medium-quality")
        _plot_category(worst_indices, "worst", "worst", "Worst-quality")
        _plot_category(correct_indices, "correct", "correct", "Correct")

        # 新增：用户指定文件名对应的样本
        if specified_indices is not None:
            _plot_category(specified_indices, "specified", "sample", "Specified-sample")
    
    print(f"  Waterfall and compact bar plots saved to: {output_dir}")
    
    # Save mapping from plotted waterfall figures to image names (only selected samples)
    if sample_ids_array is not None and waterfall_records:
        import csv
        csv_path = os.path.join(output_dir, "waterfall_sample_images.csv")
        fieldnames = [
            "model",
            "category",
            "figure_index",
            "image_name",
            "true_label",
            "pred_label",
            "prob_positive",
            "plot_file",
        ]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(waterfall_records)
            print(f"  Saved waterfall sample image names to: {csv_path}")
        except Exception as e:
            print(f"  Warning: failed to save waterfall sample image names: {e}")


def _compute_shap_for_model(
    predictor: TabularPredictor,
    model_name: str,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    skip_neural_net: bool = False,
) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """Compute SHAP values for a single model"""
    if skip_neural_net and "NeuralNet" in model_name:
        print(f"  Skipping {model_name} (neural network, --skip_neural_net enabled)")
        return None

    try:
        # Load the model
        model = None
        
        # Method 1: Try _trainer.load_model (most common in newer versions)
        if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
            try:
                model = predictor._trainer.load_model(model_name)
            except Exception:
                pass
        
        # Method 2: Try model_info and load from path
        if model is None and hasattr(predictor, "model_info"):
            try:
                model_info_attr = predictor.model_info
                model_info = None
                if callable(model_info_attr):
                    try:
                        model_info = model_info_attr()
                    except TypeError:
                        try:
                            model_info = model_info_attr(model=model_name)
                        except Exception:
                            model_info = None
                else:
                    model_info = model_info_attr

                selected_model_info = None
                if isinstance(model_info, dict):
                    if "path" in model_info:
                        selected_model_info = model_info
                    elif model_name in model_info:
                        selected_model_info = model_info[model_name]

                if isinstance(selected_model_info, dict) and "path" in selected_model_info:
                    from autogluon.common.loaders import load_pkl
                    model_path = selected_model_info["path"]
                    if not os.path.isabs(model_path):
                        model_path = os.path.join(predictor.path, model_path)
                    model = load_pkl.load(path=model_path)
            except Exception:
                pass
        
        if model is None:
            raise AttributeError(f"Could not load model {model_name} using any available method")
        
        print(f"  Loaded model type: {type(model)}")
            
    except Exception as e:
        print(f"  Warning: Could not load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    try:
        # For BAG tree models, use special handling
        if _is_bag_model(model_name) and _is_tree_model(model_name):
            return _compute_shap_tree_bag(model, X_background, X_explain, model_name, predictor)
        elif _is_tree_model(model_name):
            # Non-BAG tree model - use standard TreeExplainer approach
            # This would need similar preprocessing handling, but for now use KernelExplainer
            print(f"  Non-BAG tree model detected, using KernelExplainer for safety...")
            return _compute_shap_kernel(model, X_background, X_explain, model_name)
        else:
            # Neural network or other models
            return _compute_shap_kernel(model, X_background, X_explain, model_name)
    except Exception as e:
        print(f"  Error computing SHAP for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main() -> None:
    args = parse_args()

    np_mod = _require_numpy()
    pd_mod = _require_pandas()
    predictor_cls = _require_tabular_predictor()

    print(f"Loading predictor from: {args.model_dir}")
    predictor = predictor_cls.load(args.model_dir)

    print(f"Loading training data from: {args.train_csv}")
    raw_df = pd_mod.read_csv(args.train_csv)
    train_df = _prepare_df(raw_df.copy(), args.label)

    # Prepare mapping from row index to image name (if available)
    sample_ids_all: Optional[pd.Series] = None
    id_col = None
    if "filename" in raw_df.columns:
        id_col = "filename"
    elif "image_path" in raw_df.columns:
        id_col = "image_path"

    if id_col is not None:
        sample_ids_all = raw_df.loc[train_df.index, id_col].copy()
        if id_col == "image_path":
            sample_ids_all = sample_ids_all.astype(str).apply(os.path.basename)

    # Get main models
    main_models = _get_main_models(predictor, args.model_dir, args.main_models)
    print(f"\nMain models to analyze: {main_models}")

    # Prepare feature data (without label)
    X_train = train_df.drop(columns=[args.label]).copy()
    y_train = train_df[args.label].copy()

    # Sample background data
    np_mod.random.seed(42)  # For reproducibility
    if len(X_train) > args.background_samples:
        background_idx = np_mod.random.choice(
            len(X_train), size=args.background_samples, replace=False
        )
        X_background = X_train.iloc[background_idx].copy()
        print(f"Using {len(X_background)} background samples for SHAP")
    else:
        X_background = X_train.copy()
        print(f"Using all {len(X_background)} samples as background")

    # Determine samples to explain (use stratified sampling to ensure class balance)
    from sklearn.model_selection import train_test_split

    if args.explain_samples is not None:
        n_explain = min(args.explain_samples, len(X_train))
        if n_explain < len(X_train):
            # Use stratified sampling to maintain class distribution
            X_explain, _, y_explain, _ = train_test_split(
                X_train,
                y_train,
                test_size=1 - n_explain / len(X_train),
                stratify=y_train,
                random_state=42
            )
            X_explain = X_explain.copy()
        else:
            X_explain = X_train.copy()
            y_explain = y_train.copy()
    else:
        # Default: use all or cap at 500
        max_explain = 500
        if len(X_train) <= max_explain:
            X_explain = X_train.copy()
            y_explain = y_train.copy()
        else:
            # Use stratified sampling to maintain class distribution
            X_explain, _, y_explain, _ = train_test_split(
                X_train,
                y_train,
                test_size=1 - max_explain / len(X_train),
                stratify=y_train,
                random_state=42
            )
            X_explain = X_explain.copy()

    # Align sample IDs to explained samples (if available)
    sample_ids_explain: Optional[pd.Series] = None
    if sample_ids_all is not None:
        try:
            sample_ids_explain = sample_ids_all.loc[X_explain.index]
        except Exception:
            sample_ids_explain = sample_ids_all

    # Print class distribution in explained samples
    class_counts = y_explain.value_counts().sort_index()
    print(f"Explaining {len(X_explain)} samples")
    print(f"  Class distribution: {dict(class_counts)}")
    for cls, count in class_counts.items():
        pct = count / len(X_explain) * 100
        print(f"    Class {cls}: {count} samples ({pct:.1f}%)")

    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.model_dir, "shap_analysis")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Compute SHAP values for each main model
    results = {}
    shap_summary = []

    for model_name in main_models:
        print(f"\nAnalyzing model: {model_name}")
        result = _compute_shap_for_model(
            predictor, model_name, X_background, X_explain, args.skip_neural_net
        )

        if result is None:
            continue

        shap_values, shap_df = result
        results[model_name] = {
            "shap_values": shap_values,
            "shap_df": shap_df,
        }

        # Save SHAP values for this model
        model_output_file = os.path.join(output_dir, f"{model_name}_shap_values.csv")
        shap_df.to_csv(model_output_file, index=False)
        print(f"  Saved SHAP values to: {model_output_file}")

        # Compute summary statistics
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        shap_summary.append(
            {
                "model": model_name,
                "top_features": mean_abs_shap.head(20).to_dict(),
                "mean_abs_shap_sum": float(shap_df.abs().sum().sum()),
            }
        )

        # Save feature importance plot data
        importance_file = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
        importance_df = pd_mod.DataFrame(
            {
                "feature": mean_abs_shap.index,
                "mean_abs_shap": mean_abs_shap.values,
            }
        )
        importance_df.to_csv(importance_file, index=False)
        print(f"  Saved feature importance to: {importance_file}")

        # Optionally generate beeswarm plots for selected models
        if args.plot_beeswarm_for and model_name in args.plot_beeswarm_for:
            try:
                import shap
                import matplotlib.pyplot as plt
            except ImportError:
                print(
                    "  Warning: shap or matplotlib not available. "
                    f"Skipping beeswarm plot for {model_name}."
                )
            else:
                print(f"  Generating SHAP beeswarm plot for: {model_name}")
                shap_df_short = shap_df.copy()
                shap_df_short.columns = [
                    short_feature_name(col) for col in shap_df_short.columns
                ]
                beeswarm_max_display = max(1, int(args.top_features))
                beeswarm_path = os.path.join(output_dir, f"{model_name}_beeswarm.png")
                saved_paths = save_beeswarm_plot(
                    shap_values,
                    shap_df_short,
                    beeswarm_path,
                    beeswarm_max_display,
                    export_formats=("png", "svg", "pdf"),
                    dpi=150,
                    figsize=(10, 8),
                    plot_type="dot",
                )
                print(f"  Saved beeswarm plot to: {', '.join(saved_paths)}")

    # Aggregate SHAP values (weighted by ensemble weights if available)
    log_path = os.path.join(args.model_dir, "logs", "predictor_log.txt")
    weights = _extract_ensemble_weights_from_log(log_path)
    if weights and results:
        print("\nComputing weighted ensemble SHAP values...")
        weighted_shap_list = []
        # Get feature names from the first model's shap_df (they should all have the same features)
        feature_names = None
        expected_shape = None
        for model_name in weights.keys():
            if model_name in results:
                weight = weights[model_name]
                shap_values = results[model_name]["shap_values"]
                # Verify shape consistency
                if expected_shape is None:
                    expected_shape = shap_values.shape
                elif shap_values.shape != expected_shape:
                    print(f"  Warning: Shape mismatch for {model_name}: {shap_values.shape} vs {expected_shape}")
                    print(f"  Skipping this model in ensemble aggregation")
                    continue
                weighted_shap_list.append(shap_values * weight)
                # Get feature names from shap_df (preprocessed features)
                if feature_names is None:
                    feature_names = results[model_name]["shap_df"].columns.tolist()

        if weighted_shap_list:
            ensemble_shap = np_mod.sum(weighted_shap_list, axis=0)
            # Use feature names from shap_df (preprocessed features), not X_explain (original features)
            if feature_names is None:
                # Fallback: use original feature names if shap_df columns not available
                feature_names = X_explain.columns.tolist()
            # Ensure feature count matches
            if len(feature_names) != ensemble_shap.shape[1]:
                print(f"  Warning: Feature count mismatch: {len(feature_names)} names vs {ensemble_shap.shape[1]} values")
                feature_names = [f"feature_{i}" for i in range(ensemble_shap.shape[1])]
            ensemble_shap_df = pd_mod.DataFrame(
                ensemble_shap, columns=feature_names, index=X_explain.index
            )
            ensemble_file = os.path.join(output_dir, "WeightedEnsemble_L3_shap_values.csv")
            ensemble_shap_df.to_csv(ensemble_file, index=False)
            print(f"Saved ensemble SHAP values to: {ensemble_file}")

            # Ensemble feature importance
            ensemble_importance = ensemble_shap_df.abs().mean().sort_values(ascending=False)
            ensemble_importance_file = os.path.join(
                output_dir, "WeightedEnsemble_L3_feature_importance.csv"
            )
            ensemble_importance_df = pd_mod.DataFrame(
                {
                    "feature": ensemble_importance.index,
                    "mean_abs_shap": ensemble_importance.values,
                }
            )
            ensemble_importance_df.to_csv(ensemble_importance_file, index=False)
            print(f"Saved ensemble feature importance to: {ensemble_importance_file}")

    # Save summary
    summary_file = os.path.join(output_dir, "shap_analysis_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("SHAP Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model directory: {args.model_dir}\n")
        f.write(f"Training CSV: {args.train_csv}\n")
        f.write(f"Background samples: {len(X_background)}\n")
        f.write(f"Explained samples: {len(X_explain)}\n")
        f.write(f"Main models analyzed: {len(results)}\n\n")

        if weights:
            f.write("Ensemble weights:\n")
            for model_name, weight in weights.items():
                f.write(f"  {model_name}: {weight:.4f}\n")
            f.write("\n")

        for item in shap_summary:
            f.write(f"Model: {item['model']}\n")
            f.write(f"  Mean absolute SHAP sum: {item['mean_abs_shap_sum']:.4f}\n")
            f.write("  Top 10 features:\n")
            for feat, val in list(item["top_features"].items())[:10]:
                f.write(f"    {feat}: {val:.6f}\n")
            f.write("\n")

    print(f"\nSaved analysis summary to: {summary_file}")
    
    # Generate waterfall plots for typical correct and incorrect samples
    if args.plot_waterfall and results:
        print(f"\nGenerating SHAP waterfall plots...")
        _plot_waterfall_samples(
            predictor, results, X_explain, y_explain, output_dir, 
            args.waterfall_samples, args.label, sample_ids_explain, args.sample_filename
        )
    
    print(f"\nSHAP analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

