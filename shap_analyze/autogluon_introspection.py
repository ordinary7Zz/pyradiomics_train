from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


def extract_ensemble_weights_from_log(log_path: str) -> Optional[Dict[str, float]]:
    """Extract ensemble weights from predictor_log.txt."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"Ensemble Weights:\s*\{([^}]+)\}"
        matches = list(re.finditer(pattern, content))
        if not matches:
            return None

        weights_str = matches[-1].group(1)
        weights: Dict[str, float] = {}
        for item in weights_str.split(","):
            item = item.strip()
            if not item:
                continue
            model_match = re.search(r"'([^']+)':\s*([\d.]+)", item)
            if model_match:
                model_name = model_match.group(1)
                weight = float(model_match.group(2))
                weights[model_name] = weight

        return weights if weights else None
    except Exception as e:
        print(f"Warning: Failed to extract ensemble weights from log: {e}")
        return None


def extract_ensemble_weights_from_shap_summary(summary_path: str) -> Optional[Dict[str, float]]:
    """Parse ``Ensemble weights:`` block from shap_analysis_summary.txt."""
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        weights: Dict[str, float] = {}
        in_section = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Ensemble weights:"):
                in_section = True
                continue
            if not in_section:
                continue
            if not stripped:
                break
            model_match = re.match(r"([^:]+):\s*([\d.]+)", stripped)
            if model_match:
                weights[model_match.group(1).strip()] = float(model_match.group(2))
        return weights if weights else None
    except Exception as e:
        print(f"Warning: Failed to parse ensemble weights from summary: {e}")
        return None


def resolve_ensemble_weights(model_dir: str, output_dir: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Prefer last weights in predictor_log.txt; fall back to shap_analysis_summary.txt."""
    log_path = os.path.join(model_dir, "logs", "predictor_log.txt")
    weights = extract_ensemble_weights_from_log(log_path)
    if weights:
        return weights

    summary_paths: List[str] = []
    if output_dir:
        summary_paths.append(os.path.join(output_dir, "shap_analysis_summary.txt"))
    summary_paths.append(os.path.join(model_dir, "shap_analysis", "shap_analysis_summary.txt"))
    for path in summary_paths:
        weights = extract_ensemble_weights_from_shap_summary(path)
        if weights:
            return weights
    return None


def _models_in_leaderboard(predictor: Any, names: List[str]) -> List[str]:
    try:
        leaderboard = predictor.leaderboard(silent=True)
        available = set(leaderboard["model"].astype(str))
    except Exception:
        return list(names)
    kept = [n for n in names if n in available]
    missing = [n for n in names if n not in available]
    if missing:
        print(f"Warning: models not in leaderboard (skipped): {missing}")
    return kept


def load_autogluon_model(predictor: Any, model_name: str) -> Any:
    """Load a model from AutoGluon using the available predictor APIs."""
    model = None

    if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
        try:
            model = predictor._trainer.load_model(model_name)
        except Exception:
            pass

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

    return model


def get_main_models(
    predictor: Any,
    model_dir: str,
    main_models: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Get list of main models from ensemble or use provided list."""
    if main_models is not None:
        filtered = _models_in_leaderboard(predictor, main_models)
        return filtered if filtered else main_models

    weights = resolve_ensemble_weights(model_dir, output_dir)
    if weights:
        print(f"Auto-detected ensemble weights: {weights}")
        filtered = _models_in_leaderboard(predictor, list(weights.keys()))
        if filtered:
            return filtered

    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_model_name = leaderboard.iloc[0]["model"]
        print(f"Best model: {best_model_name}")

        try:
            if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                ensemble_model = predictor._trainer.load_model(best_model_name)
                if hasattr(ensemble_model, "model_names"):
                    return ensemble_model.model_names
        except Exception:
            pass

        print("Warning: Could not extract ensemble sub-models. Using top 5 non-ensemble models.")
        non_ensemble = leaderboard[~leaderboard["model"].str.contains("Ensemble", case=False)]
        return non_ensemble.head(5)["model"].tolist()
    except Exception as e:
        print(f"Warning: Could not determine main models automatically: {e}")
        print("Using default: top models from leaderboard")
        leaderboard = predictor.leaderboard(silent=True)
        return leaderboard.head(5)["model"].tolist()


def is_tree_model(model_name: str) -> bool:
    """Check if model is a tree-based model."""
    tree_keywords = [
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "RandomForest",
        "ExtraTrees",
    ]
    return any(kw in model_name for kw in tree_keywords)


def is_bag_model(model_name: str) -> bool:
    """Check if model is a BAG model."""
    return "_BAG_" in model_name
