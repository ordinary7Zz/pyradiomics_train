import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# Columns that are almost always non-feature metadata. These will be dropped from feature matrix.
# NOTE: we do NOT hard-drop "filename" anymore because many datasets use it as the sample id.
DROP_ALWAYS = ["image_path", "mask_path"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-1 SHAP asset export for AutoGluon TabularPredictor main models."
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
        help="CSV file used for SHAP background/explain and for deriving case tables.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="label",
        help="Label column name",
    )
    p.add_argument(
        "--id_cols",
        type=str,
        nargs="+",
        default=["image_name", "file_name", "filename", "id", "sample_id"],
        help="Candidate columns to use as sample id (first existing will be used). "
             "If none exist, DataFrame index will be used as sample_id.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <model_dir>/shap_analysis_assets)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling background/explain sets (default: 42).",
    )
    p.add_argument(
        "--background_samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP (default: 100).",
    )
    p.add_argument(
        "--explain_samples",
        type=int,
        default=None,
        help="Number of samples to explain (default: cap at 500).",
    )
    p.add_argument(
        "--max_explain",
        type=int,
        default=500,
        help="Cap for explain samples when --explain_samples is not provided (default: 500).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probability to class (default: 0.5).",
    )
    p.add_argument(
        "--skip_neural_net",
        action="store_true",
        help="Skip neural network models (NeuralNetFastAI) in SHAP analysis.",
    )
    p.add_argument(
        "--main_models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of main model names to analyze (default: auto-detect from ensemble).",
    )
    p.add_argument(
        "--kernel_nsamples",
        type=int,
        default=100,
        help="nsamples for KernelExplainer (default: 100).",
    )
    p.add_argument(
        "--topk_cases",
        type=int,
        default=30,
        help="How many representative cases to list per category (correct/incorrect) in assets (default: 30).",
    )
    return p.parse_args()


def _pick_id_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prepare_df(df: pd.DataFrame, label_col: str, id_col: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare dataframe for SHAP:
      - keep a sample_id series aligned with the prepared df
      - drop obvious non-feature columns
      - filter label != -1, cast label to int
    """
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    # sample_id series
    if id_col is not None:
        sample_id = df[id_col].astype(str).copy()
    else:
        sample_id = df.index.astype(str).to_series(index=df.index)

    # Drop non-feature columns (but keep id_col and label_col)
    drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    # Optionally drop filename if it is NOT the id_col (legacy behavior)
    if "filename" in df.columns and id_col != "filename":
        drop_cols.append("filename")
    if drop_cols:
        df = df.drop(columns=list(sorted(set(drop_cols))))

    # Filter invalid labels
    keep_mask = df[label_col] != -1
    df = df[keep_mask].copy()
    sample_id = sample_id[keep_mask].copy()

    df[label_col] = df[label_col].astype(int)
    return df, sample_id.reset_index(drop=True)


def _extract_ensemble_weights_from_log(log_path: str) -> Optional[Dict[str, float]]:
    """Extract ensemble weights from predictor_log.txt"""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

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

    log_path = os.path.join(model_dir, "logs", "predictor_log.txt")
    weights = _extract_ensemble_weights_from_log(log_path)
    if weights:
        print(f"Found ensemble weights from log: {weights}")
        return list(weights.keys())

    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_model_name = leaderboard.iloc[0]["model"]
        print(f"Best model: {best_model_name}")

        try:
            if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                ensemble_model = predictor._trainer.load_model(best_model_name)
                if hasattr(ensemble_model, "model_names"):
                    return list(ensemble_model.model_names)
        except Exception:
            pass

        print("Warning: Could not extract ensemble sub-models. Using top 5 non-ensemble models.")
        non_ensemble = leaderboard[~leaderboard["model"].str.contains("Ensemble", case=False)]
        return non_ensemble.head(5)["model"].tolist()
    except Exception as e:
        print(f"Warning: Could not determine main models automatically: {e}")
        leaderboard = predictor.leaderboard(silent=True)
        return leaderboard.head(5)["model"].tolist()


def _get_base_model(model):
    """Extract base model from AutoGluon wrapper (recursive)"""
    if model is None:
        return None
    if hasattr(model, "_model") and model._model is not None:
        return _get_base_model(model._model)
    if hasattr(model, "model") and model.model is not None:
        return _get_base_model(model.model)
    return model


def _is_tree_model(model_name: str) -> bool:
    tree_keywords = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "ExtraTrees"]
    return any(kw in model_name for kw in tree_keywords)


def _safe_float(x) -> Optional[float]:
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            # choose positive class if list-like with 2 entries
            if len(x) == 2:
                return float(x[1])
            return float(np.array(x).ravel()[0])
        return float(x)
    except Exception:
        return None


def _compute_shap_tree(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, model_name: str
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """Compute SHAP values for tree-based models using TreeExplainer"""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Using TreeExplainer for {model_name}...")
    base_model = _get_base_model(model)
    if base_model is None:
        raise ValueError(f"Could not extract base model from {model_name}")

    actual_model = None

    # For BAG models, take the first fold model as an approximation
    if hasattr(base_model, "models"):
        try:
            models_list = base_model.models
            if models_list and len(models_list) > 0:
                actual_model = _get_base_model(models_list[0])
        except Exception:
            pass

    if actual_model is None and hasattr(base_model, "_models"):
        try:
            models_list = base_model._models
            if models_list and len(models_list) > 0:
                actual_model = _get_base_model(models_list[0])
        except Exception:
            pass

    if actual_model is None:
        if hasattr(base_model, "_model") and base_model._model is not None:
            actual_model = _get_base_model(base_model._model)
        elif hasattr(base_model, "model") and base_model.model is not None:
            actual_model = _get_base_model(base_model.model)
        else:
            actual_model = base_model

    if actual_model is None:
        print("  Warning: Could not extract tree model, falling back to KernelExplainer...")
        return _compute_shap_kernel(model, X_background, X_explain, model_name, kernel_nsamples=100)

    try:
        explainer = shap.TreeExplainer(actual_model)
    except Exception as e:
        print(f"  Warning: TreeExplainer failed ({e}), falling back to KernelExplainer...")
        return _compute_shap_kernel(model, X_background, X_explain, model_name, kernel_nsamples=100)

    shap_values = explainer.shap_values(X_explain)

    expected_value = explainer.expected_value
    # Handle binary classification where shap_values/expected_value may be list-like
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        if len(expected_value) == 2:
            expected_value = expected_value[1]

    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    feature_names = X_explain.columns.tolist()
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    meta = {
        "explainer": "TreeExplainer",
        "expected_value": _safe_float(expected_value),
        # TreeExplainer may be in raw/log-odds depending on model; record best-effort info.
        "output_space": getattr(explainer, "model_output", None) or "treeexplainer_default",
    }
    return shap_values, shap_df, meta


def _compute_shap_kernel(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, model_name: str, kernel_nsamples: int
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """Compute SHAP values using KernelExplainer (for non-tree models)"""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Using KernelExplainer for {model_name}...")
    print(f"    Warning: KernelExplainer is slow. Background={len(X_background)}, explain={len(X_explain)}, nsamples={kernel_nsamples}")

    def model_wrapper(X):
        X_df = pd.DataFrame(X, columns=X_background.columns, index=range(len(X)))
        proba = model.predict_proba(X_df)
        if isinstance(proba, pd.DataFrame):
            if 1 in proba.columns:
                return proba[1].values
            return proba.iloc[:, -1].values
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.flatten()
        return np.array(proba).flatten()

    explainer = shap.KernelExplainer(model_wrapper, X_background.values)
    shap_values = explainer.shap_values(X_explain.values, nsamples=kernel_nsamples)

    feature_names = X_explain.columns.tolist()
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    meta = {
        "explainer": "KernelExplainer",
        "expected_value": _safe_float(explainer.expected_value),
        "output_space": "probability",  # wrapper returns probability-like output
        "kernel_nsamples": int(kernel_nsamples),
    }
    return np.array(shap_values), shap_df, meta


def _load_single_model(predictor: TabularPredictor, model_name: str):
    """Try multiple methods to load AutoGluon internal model object."""
    model = None
    if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
        try:
            model = predictor._trainer.load_model(model_name)
        except Exception:
            pass

    if model is None and hasattr(predictor, "_trainer"):
        trainer = predictor._trainer
        if hasattr(trainer, "trainer") and hasattr(trainer.trainer, "load_model"):
            try:
                model = trainer.trainer.load_model(model_name)
            except Exception:
                pass

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

    if model is None and hasattr(predictor, "_trainer"):
        try:
            trainer = predictor._trainer
            if hasattr(trainer, "trainer") and hasattr(trainer.trainer, "model_graph"):
                model_graph = trainer.trainer.model_graph
                if model_name in model_graph:
                    model = model_graph[model_name]
        except Exception:
            pass

    return model


def _predict_proba_pos(
    predictor: TabularPredictor,
    model_name: str,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict positive-class probability for X."""
    # Prefer predictor.predict_proba with model=... when available
    try:
        proba = predictor.predict_proba(X, model=model_name)
    except Exception:
        proba = None

    if proba is None:
        model = _load_single_model(predictor, model_name)
        if model is None:
            raise RuntimeError(f"Could not load model to predict_proba: {model_name}")
        proba = model.predict_proba(X)

    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            return proba[1].to_numpy()
        return proba.iloc[:, -1].to_numpy()
    if isinstance(proba, np.ndarray):
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.flatten()
    return np.array(proba).flatten()


def _confusion_group(y_true: np.ndarray, y_pred: np.ndarray) -> List[str]:
    out = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            out.append("TP")
        elif yt == 0 and yp == 0:
            out.append("TN")
        elif yt == 0 and yp == 1:
            out.append("FP")
        elif yt == 1 and yp == 0:
            out.append("FN")
        else:
            out.append("NA")
    return out


def _build_case_table(
    sample_id: pd.Series,
    y_true: pd.Series,
    p_pos: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_true_np = y_true.to_numpy().astype(int)
    y_pred = (p_pos >= threshold).astype(int)
    correct = (y_pred == y_true_np).astype(int)
    group = _confusion_group(y_true_np, y_pred)
    df = pd.DataFrame(
        {
            "sample_id": sample_id.astype(str).to_numpy(),
            "y_true": y_true_np,
            "p_pos": p_pos.astype(float),
            "y_pred": y_pred,
            "correct": correct,
            "group": group,
            "abs_margin_to_threshold": np.abs(p_pos - threshold),
        }
    )
    return df


def _select_representative_cases(case_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    """
    Produce a compact table of representative samples:
      - high-confidence errors: FP with largest p_pos, FN with smallest p_pos
      - borderline cases: smallest abs_margin_to_threshold (both correct & incorrect)
      - confident correct: TP with largest p_pos, TN with smallest p_pos
    """
    rows = []

    def add_subset(name: str, sub: pd.DataFrame):
        if sub is None or len(sub) == 0:
            return
        tmp = sub.copy()
        tmp.insert(0, "subset", name)
        rows.append(tmp)

    # High-confidence errors
    add_subset("FP_high_conf", case_df[case_df["group"] == "FP"].sort_values("p_pos", ascending=False).head(topk))
    add_subset("FN_high_conf", case_df[case_df["group"] == "FN"].sort_values("p_pos", ascending=True).head(topk))

    # Borderline (closest to threshold), include both correct & incorrect
    add_subset("borderline_all", case_df.sort_values("abs_margin_to_threshold", ascending=True).head(topk))

    # Confident correct
    add_subset("TP_confident", case_df[case_df["group"] == "TP"].sort_values("p_pos", ascending=False).head(topk))
    add_subset("TN_confident", case_df[case_df["group"] == "TN"].sort_values("p_pos", ascending=True).head(topk))

    if not rows:
        return pd.DataFrame(columns=["subset"] + list(case_df.columns))
    return pd.concat(rows, axis=0, ignore_index=True)


def _compute_shap_for_model(
    predictor: TabularPredictor,
    model_name: str,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    skip_neural_net: bool,
    kernel_nsamples: int,
) -> Optional[Tuple[np.ndarray, pd.DataFrame, Dict]]:
    """Compute SHAP values + meta for a single model"""
    if skip_neural_net and "NeuralNet" in model_name:
        print(f"  Skipping {model_name} (neural network, --skip_neural_net enabled)")
        return None

    model = _load_single_model(predictor, model_name)
    if model is None:
        print(f"  Warning: Could not load model {model_name}")
        return None

    try:
        if _is_tree_model(model_name):
            return _compute_shap_tree(model, X_background, X_explain, model_name)
        return _compute_shap_kernel(model, X_background, X_explain, model_name, kernel_nsamples=kernel_nsamples)
    except Exception as e:
        print(f"  Error computing SHAP for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main() -> None:
    args = parse_args()

    print(f"Loading predictor from: {args.model_dir}")
    predictor = TabularPredictor.load(args.model_dir)

    print(f"Loading data from: {args.train_csv}")
    raw_df = pd.read_csv(args.train_csv)

    id_col = _pick_id_column(raw_df, args.id_cols)
    if id_col:
        print(f"Using sample id column: {id_col}")
    else:
        print("No id column found. Using DataFrame index as sample_id.")

    train_df, sample_id = _prepare_df(raw_df, args.label, id_col)

    # Prepare feature data (without label)
    X_all = train_df.drop(columns=[args.label]).copy()
    y_all = train_df[args.label].copy().reset_index(drop=True)

    # Get main models
    main_models = _get_main_models(predictor, args.model_dir, args.main_models)
    print(f"\nMain models to analyze: {main_models}")

    # Sampling
    np.random.seed(args.seed)
    n_all = len(X_all)

    # Background
    if n_all > args.background_samples:
        background_idx = np.random.choice(n_all, size=args.background_samples, replace=False)
        X_background = X_all.iloc[background_idx].copy()
    else:
        background_idx = np.arange(n_all)
        X_background = X_all.copy()
    print(f"Using {len(X_background)} background samples for SHAP")

    # Explain set
    if args.explain_samples is not None:
        n_explain = min(args.explain_samples, n_all)
    else:
        n_explain = min(args.max_explain, n_all)

    if n_explain < n_all:
        explain_idx = np.random.choice(n_all, size=n_explain, replace=False)
        X_explain = X_all.iloc[explain_idx].copy()
        y_explain = y_all.iloc[explain_idx].copy().reset_index(drop=True)
        sample_id_explain = sample_id.iloc[explain_idx].copy().reset_index(drop=True)
    else:
        explain_idx = np.arange(n_all)
        X_explain = X_all.copy()
        y_explain = y_all.copy()
        sample_id_explain = sample_id.copy()

    print(f"Explaining {len(X_explain)} samples")

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.model_dir, "shap_analysis_assets")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset assets
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # X_explain with sample_id + y_true
    X_explain_out = X_explain.copy()
    X_explain_out.insert(0, "sample_id", sample_id_explain.astype(str).to_numpy())
    X_explain_out.insert(1, "y_true", y_explain.astype(int).to_numpy())
    x_explain_file = os.path.join(assets_dir, "X_explain.csv")
    X_explain_out.to_csv(x_explain_file, index=False)
    print(f"Saved X_explain assets to: {x_explain_file}")

    # Background indices and explain indices for reproducibility
    index_meta = {
        "seed": int(args.seed),
        "background_idx": [int(i) for i in np.array(background_idx).tolist()],
        "explain_idx": [int(i) for i in np.array(explain_idx).tolist()],
        "n_all": int(n_all),
    }
    with open(os.path.join(assets_dir, "sample_indices.json"), "w", encoding="utf-8") as f:
        json.dump(index_meta, f, indent=2)

    # Global metadata
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": os.path.abspath(args.model_dir),
        "train_csv": os.path.abspath(args.train_csv),
        "label_col": args.label,
        "id_col": id_col if id_col else "__index__",
        "background_samples": int(args.background_samples),
        "explain_samples": int(len(X_explain)),
        "threshold": float(args.threshold),
        "kernel_nsamples": int(args.kernel_nsamples),
        "skip_neural_net": bool(args.skip_neural_net),
        "main_models": list(main_models),
    }
    with open(os.path.join(assets_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Per-model outputs
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    expected_values: Dict[str, Dict] = {}
    shap_summary = []
    weights = _extract_ensemble_weights_from_log(os.path.join(args.model_dir, "logs", "predictor_log.txt"))

    results = {}

    for model_name in main_models:
        print(f"\nAnalyzing model: {model_name}")

        # predictions first (for case table)
        try:
            p_pos = _predict_proba_pos(predictor, model_name, X_explain)
        except Exception as e:
            print(f"  Warning: predict_proba failed for {model_name}: {e}")
            p_pos = None

        case_df = None
        if p_pos is not None:
            case_df = _build_case_table(sample_id_explain, y_explain, p_pos, args.threshold)
            case_file = os.path.join(models_dir, f"{model_name}_case_table.csv")
            case_df.to_csv(case_file, index=False)
            print(f"  Saved case table to: {case_file}")

            # Representative case list (assets for plotting stage)
            rep = _select_representative_cases(case_df, topk=args.topk_cases)
            rep_file = os.path.join(models_dir, f"{model_name}_representative_cases.csv")
            rep.to_csv(rep_file, index=False)
            print(f"  Saved representative cases to: {rep_file}")

            # Confusion counts
            counts = case_df["group"].value_counts(dropna=False).to_dict()
            counts_file = os.path.join(models_dir, f"{model_name}_confusion_counts.json")
            with open(counts_file, "w", encoding="utf-8") as f:
                json.dump({k: int(v) for k, v in counts.items()}, f, indent=2)

        # SHAP
        result = _compute_shap_for_model(
            predictor=predictor,
            model_name=model_name,
            X_background=X_background,
            X_explain=X_explain,
            skip_neural_net=args.skip_neural_net,
            kernel_nsamples=args.kernel_nsamples,
        )

        if result is None:
            continue

        shap_values, shap_df, shap_meta = result

        # attach sample_id for stable alignment
        shap_df_out = shap_df.copy()
        shap_df_out.insert(0, "sample_id", sample_id_explain.astype(str).to_numpy())

        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        shap_file = os.path.join(model_dir, f"{model_name}_shap_values.csv")
        shap_df_out.to_csv(shap_file, index=False)
        print(f"  Saved SHAP values to: {shap_file}")

        # feature importance
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        importance_df = pd.DataFrame({"feature": mean_abs_shap.index, "mean_abs_shap": mean_abs_shap.values})
        importance_file = os.path.join(model_dir, f"{model_name}_feature_importance.csv")
        importance_df.to_csv(importance_file, index=False)

        # expected_value meta
        expected_values[model_name] = shap_meta

        shap_summary.append(
            {
                "model": model_name,
                "explainer": shap_meta.get("explainer", "unknown"),
                "output_space": shap_meta.get("output_space", "unknown"),
                "expected_value": shap_meta.get("expected_value", None),
                "mean_abs_shap_sum": float(shap_df.abs().sum().sum()),
                "top_features": mean_abs_shap.head(20).to_dict(),
            }
        )
        results[model_name] = {"shap_values": shap_values}

    # Save expected values
    with open(os.path.join(models_dir, "expected_values.json"), "w", encoding="utf-8") as f:
        json.dump(expected_values, f, indent=2)

    # Weighted ensemble SHAP (optional; still an approximation)
    if weights and results:
        print("\nComputing weighted ensemble SHAP values (approx)...")
        weighted_shap_list = []
        for mn, w in weights.items():
            if mn in results:
                weighted_shap_list.append(results[mn]["shap_values"] * float(w))

        if weighted_shap_list:
            ensemble_shap = np.sum(weighted_shap_list, axis=0)
            ensemble_df = pd.DataFrame(ensemble_shap, columns=X_explain.columns)
            ensemble_df.insert(0, "sample_id", sample_id_explain.astype(str).to_numpy())
            ensemble_file = os.path.join(models_dir, "WeightedEnsemble_L3_shap_values.csv")
            ensemble_df.to_csv(ensemble_file, index=False)

            ensemble_importance = ensemble_df.drop(columns=["sample_id"]).abs().mean().sort_values(ascending=False)
            ensemble_imp_df = pd.DataFrame({"feature": ensemble_importance.index, "mean_abs_shap": ensemble_importance.values})
            ensemble_imp_file = os.path.join(models_dir, "WeightedEnsemble_L3_feature_importance.csv")
            ensemble_imp_df.to_csv(ensemble_imp_file, index=False)
            print(f"  Saved ensemble SHAP assets to: {ensemble_file}")

    # Save summary txt (stage-1)
    summary_file = os.path.join(output_dir, "shap_assets_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("SHAP Asset Export Summary (Stage-1)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model directory: {args.model_dir}\n")
        f.write(f"CSV: {args.train_csv}\n")
        f.write(f"Label column: {args.label}\n")
        f.write(f"ID column: {id_col if id_col else '__index__'}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Background samples: {len(X_background)}\n")
        f.write(f"Explain samples: {len(X_explain)}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"KernelExplainer nsamples: {args.kernel_nsamples}\n")
        f.write(f"Main models requested: {len(main_models)}\n")
        f.write(f"Main models completed: {len(shap_summary)}\n\n")

        if weights:
            f.write("Ensemble weights (from log):\n")
            for mn, w in weights.items():
                f.write(f"  {mn}: {float(w):.6f}\n")
            f.write("\n")

        for item in shap_summary:
            f.write(f"Model: {item['model']}\n")
            f.write(f"  Explainer: {item['explainer']}\n")
            f.write(f"  Output space: {item['output_space']}\n")
            f.write(f"  Expected value: {item['expected_value']}\n")
            f.write(f"  Mean |SHAP| sum: {item['mean_abs_shap_sum']:.4f}\n")
            f.write("  Top 10 features:\n")
            for feat, val in list(item["top_features"].items())[:10]:
                f.write(f"    {feat}: {val:.6f}\n")
            f.write("\n")

    print(f"\nSaved stage-1 asset summary to: {summary_file}")
    print(f"Stage-1 assets exported to: {output_dir}")


if __name__ == "__main__":
    main()
