import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]

# Feature name simplification (参考 plot_beeswarm_batch.py)
# Use short names for waterfall display (max ~20 chars recommended)
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
    # ---- GLRLM (Run-Length Matrix) ----
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
    # ---- GLSZM (Size Zone Matrix) ----
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
    # ---- NGTDM ----
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
    """CamelCase -> space separated"""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).strip()


def _rewrite_tokens(phrase: str) -> str:
    """Rewrite tokens to be more readable"""
    tokens = phrase.split()
    out = []
    for t in tokens:
        t2 = TOKEN_REWRITE.get(t, t)
        out.append(t2)
    res = " ".join(out)
    return res


def _short_feature_name(col: str) -> str:
    """Convert feature name to short, readable format"""
    # 优先人工映射
    if col in MANUAL_MAP:
        return MANUAL_MAP[col]

    # 去掉原始前缀
    s = re.sub(r"^original_", "", col)

    # 拆 group + name
    m = re.match(r"^([A-Za-z0-9]+)_(.+)$", s)
    if m:
        group, name = m.group(1), m.group(2)
    else:
        group, name = "", s

    prefix = GROUP_PREFIX.get(group, group.upper() if group else "")

    # 名字转成更可读短语
    name_spaced = _split_camel(name)
    name_rw = _rewrite_tokens(name_spaced)
    name_rw = re.sub(r"\s+", " ", name_rw).strip()

    result = f"{prefix}: {name_rw}" if prefix else name_rw
    return _truncate_display_name(result)


def _truncate_display_name(name: str, max_len: int = 24) -> str:
    """Truncate long feature names for waterfall display."""
    if len(name) <= max_len:
        return name
    # Abbreviate common long phrases in texture names (TOKEN_REWRITE may use "intensity" for GrayLevel)
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
    }
    result = name
    for long_phrase, short in abbrev.items():
        result = result.replace(long_phrase, short)
    return result[:max_len] if len(result) > max_len else result


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
        help="Generate SHAP waterfall plots for typical correct and incorrect samples",
    )
    p.add_argument(
        "--waterfall_samples",
        type=int,
        default=3,
        help="Number of typical samples to plot for each category (correct/incorrect) (default: 3)",
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
    Identify typical correct and incorrect samples and plot SHAP waterfall plots.
    
    Args:
        predictor: AutoGluon predictor
        results: Dictionary containing SHAP results for each model
        X_explain: Features to explain
        y_explain: True labels
        output_dir: Output directory for plots
        n_samples: Number of samples to plot for each category
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
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
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

    sorted_idx = np.argsort(quality_score)[::-1]  # 从高到低
    n_total = len(sorted_idx)
    if n_total == 0:
        best_indices = medium_indices = worst_indices = np.array([], dtype=int)
    else:
        per_group = min(n_samples, max(1, n_total // 3)) if n_total >= 3 else min(n_samples, n_total)

        best_indices = sorted_idx[:per_group]
        worst_indices = sorted_idx[-per_group:] if per_group <= n_total else sorted_idx

        # 中等样本从中间区域取，不与 best / worst 重叠
        if n_total > 2 * per_group:
            mid_start = n_total // 2 - per_group // 2
            mid_start = max(per_group, mid_start)
            mid_end = min(n_total - per_group, mid_start + per_group)
            medium_indices = sorted_idx[mid_start:mid_start + per_group]
        else:
            medium_indices = np.array([], dtype=int)

    print(
        f"  Selected for quality groups - best: {len(best_indices)}, "
        f"medium: {len(medium_indices)}, worst: {len(worst_indices)}"
    )

    # Select typical samples for each category
    # For correct: high confidence (high probability for predicted class)
    # For incorrect: high confidence (high probability for wrong class)
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(correct_indices) == 0:
        print("  Warning: No correct predictions found. Cannot plot correct samples.")
    if len(incorrect_indices) == 0:
        print("  Warning: No incorrect predictions found. Cannot plot incorrect samples.")
    
    # For correct predictions: select samples with highest confidence
    # Confidence is the probability of the predicted class
    if len(correct_indices) > 0:
        correct_preds = y_pred[correct_indices]
        correct_confidences = np.where(
            correct_preds == 1,
            proba_positive[correct_indices],
            1 - proba_positive[correct_indices]
        )
        correct_top_idx = correct_indices[np.argsort(correct_confidences)[-n_samples:][::-1]]
    else:
        correct_top_idx = []
    
    # For incorrect predictions: select samples with highest confidence (wrong prediction)
    # Confidence is the probability of the incorrectly predicted class
    if len(incorrect_indices) > 0:
        incorrect_preds = y_pred[incorrect_indices]
        incorrect_confidences = np.where(
            incorrect_preds == 1,
            proba_positive[incorrect_indices],
            1 - proba_positive[incorrect_indices]
        )
        incorrect_top_idx = incorrect_indices[np.argsort(incorrect_confidences)[-n_samples:][::-1]]
    else:
        incorrect_top_idx = []
    
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
                    display_feature_names = [_short_feature_name(name) for name in top_feature_names]

                    display_feature_values = np.array(top_feature_values)
                    if display_feature_values.ndim > 1:
                        display_feature_values = display_feature_values.flatten()

                    base_value = float(np.mean(proba_positive))

                    shap_explanation = shap.Explanation(
                        values=display_shap.reshape(1, -1),
                        base_values=np.array([base_value]),
                        data=display_feature_values.reshape(1, -1),
                        feature_names=display_feature_names,
                    )

                    shap_explanation_single = shap_explanation[0]

                    plt.figure(figsize=(12, 8))
                    shap.plots.waterfall(shap_explanation_single, show=False, max_display=max_display)
                    # 统一 waterfall 图标题并加粗放大字号
                    # plt.title("(a) SHAP waterfall for a classification prediction", fontsize=20, fontweight="bold", loc='left')
                    plt.tight_layout()

                    plot_file = os.path.join(
                        output_dir, f"{model_name}_waterfall_{filename_tag}_{i+1}.png"
                    )
                    base, _ = os.path.splitext(plot_file)
                    saved_paths = []
                    for ext in ("png", "svg", "pdf"):
                        out_path = f"{base}.{ext}"
                        plt.savefig(out_path, dpi=150, bbox_inches="tight")
                        saved_paths.append(out_path)
                    plt.close()
                    print(f"    Saved: {', '.join(saved_paths)}")

                    if sample_ids_array is not None:
                        img_name = sample_ids_array[idx] if idx < len(sample_ids_array) else None
                        waterfall_records.append(
                            {
                                "model": model_name,
                                "category": category_label,
                                "figure_index": i + 1,
                                "image_name": img_name,
                                "true_label": int(y_explain[idx]),
                                "pred_label": int(y_pred[idx]),
                                "prob_positive": float(proba_positive[idx]),
                                # 记录基础文件名（不含扩展名），实际生成 png/svg/pdf 三个文件
                                "plot_file": os.path.basename(base),
                            }
                        )
                except Exception as e:
                    print(f"    Error plotting {category_label} sample {i+1}: {e}")
                    import traceback
                    traceback.print_exc()

        # 原先的正确/错误样本绘制仍然保留
        _plot_category(correct_top_idx, "correct", "correct", "Correct")
        _plot_category(incorrect_top_idx, "incorrect", "incorrect", "Incorrect")

        # 新增：分类效果最好/中等/最差三类样本
        _plot_category(best_indices, "best", "best", "Best-quality")
        _plot_category(medium_indices, "medium", "medium", "Medium-quality")
        _plot_category(worst_indices, "worst", "worst", "Worst-quality")

        # 新增：用户指定文件名对应的样本
        if specified_indices is not None:
            _plot_category(specified_indices, "specified", "sample", "Specified-sample")
    
    print(f"  Waterfall plots saved to: {output_dir}")
    
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
        if model is None and hasattr(predictor, "model_info") and model_name in predictor.model_info:
            try:
                model_info = predictor.model_info[model_name]
                if isinstance(model_info, dict) and "path" in model_info:
                    from autogluon.common.loaders import load_pkl
                    model_path = model_info["path"]
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

    print(f"Loading predictor from: {args.model_dir}")
    predictor = TabularPredictor.load(args.model_dir)

    print(f"Loading training data from: {args.train_csv}")
    raw_df = pd.read_csv(args.train_csv)
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
    np.random.seed(42)  # For reproducibility
    if len(X_train) > args.background_samples:
        background_idx = np.random.choice(
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
        importance_df = pd.DataFrame(
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
                plt.figure(figsize=(10, 8))
                # Use short feature names for beeswarm display
                shap_df_short = shap_df.copy()
                shap_df_short.columns = [
                    _short_feature_name(col) for col in shap_df_short.columns
                ]
                beeswarm_max_display = max(1, int(args.top_features))
                shap.summary_plot(
                    shap_values,
                    features=shap_df_short,
                    feature_names=shap_df_short.columns,
                    show=False,
                    max_display=beeswarm_max_display,
                )
                # 统一 beeswarm 图标题并增大字号、加粗
                # plt.title("(b) SHAP feature importance (beeswarm) for classification", fontsize=20, fontweight="bold", loc='left')
                plt.tight_layout()
                beeswarm_path = os.path.join(output_dir, f"{model_name}_beeswarm.png")
                base, _ = os.path.splitext(beeswarm_path)
                saved_paths = []
                for ext in ("png", "svg", "pdf"):
                    out_path = f"{base}.{ext}"
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")
                    saved_paths.append(out_path)
                plt.close()
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
            ensemble_shap = np.sum(weighted_shap_list, axis=0)
            # Use feature names from shap_df (preprocessed features), not X_explain (original features)
            if feature_names is None:
                # Fallback: use original feature names if shap_df columns not available
                feature_names = X_explain.columns.tolist()
            # Ensure feature count matches
            if len(feature_names) != ensemble_shap.shape[1]:
                print(f"  Warning: Feature count mismatch: {len(feature_names)} names vs {ensemble_shap.shape[1]} values")
                feature_names = [f"feature_{i}" for i in range(ensemble_shap.shape[1])]
            ensemble_shap_df = pd.DataFrame(
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
            ensemble_importance_df = pd.DataFrame(
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

