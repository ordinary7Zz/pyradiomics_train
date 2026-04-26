import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP analysis for AutoGluon TabularPredictor main models."
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


def _get_base_model(model):
    """Extract base model from AutoGluon wrapper (recursive)"""
    if model is None:
        return None
    
    # Try to get the actual model recursively
    if hasattr(model, "_model") and model._model is not None:
        return _get_base_model(model._model)
    if hasattr(model, "model") and model.model is not None:
        return _get_base_model(model.model)
    
    return model


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


def _compute_shap_tree(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, model_name: str, predictor=None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for tree-based models using TreeExplainer"""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Using TreeExplainer for {model_name}...")
    base_model = _get_base_model(model)
    
    if base_model is None:
        raise ValueError(f"Could not extract base model from {model_name}")

    # For BAG models, we need to get one of the fold models
    actual_model = None
    fold_model_obj = None  # Store the fold model wrapper for preprocessing
    
    # Debug: print base model structure
    print(f"  Base model type: {type(base_model)}")
    
    # Try different attributes that BAG models might have
    # BAG models typically have a 'models' attribute containing fold models
    if hasattr(base_model, "models"):
        try:
            models_list = base_model.models
            print(f"  Found 'models' attribute with {len(models_list) if models_list else 0} items")
            if models_list and len(models_list) > 0:
                # Use the first fold model for SHAP
                fold_model = models_list[0]
                print(f"  First fold model type: {type(fold_model)}")
                
                # Check if fold_model is a string (model name) or actual model object
                if isinstance(fold_model, str):
                    # It's a model name, need to load the actual model
                    print(f"  Detected model name string: {fold_model}, loading actual model...")
                    if predictor is not None:
                        # Try multiple methods to load the model
                        fold_model_obj = None
                        
                        # Method 1: Try direct load with the fold name
                        try:
                            if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                                fold_model_obj = predictor._trainer.load_model(fold_model)
                                print(f"  Successfully loaded model using direct name")
                        except Exception as e:
                            print(f"  Method 1 failed: {e}")
                        
                        # Method 2: Try constructing full model name (BAG_L1_S1F1 format)
                        if fold_model_obj is None:
                            try:
                                full_name = f"{model_name}_{fold_model}"
                                if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                                    fold_model_obj = predictor._trainer.load_model(full_name)
                                    print(f"  Successfully loaded model using full name: {full_name}")
                            except Exception as e:
                                print(f"  Method 2 failed: {e}")
                        
                        # Method 3: Try using get_model() method if available
                        if fold_model_obj is None and hasattr(base_model, "get_model"):
                            try:
                                fold_model_obj = base_model.get_model(fold_model)
                                print(f"  Successfully got model using get_model() method")
                            except Exception as e:
                                print(f"  Method 3 failed: {e}")
                        
                        # Method 4: Try accessing child_models or _child_models
                        if fold_model_obj is None:
                            try:
                                for attr_name in ["child_models", "_child_models", "models_dict", "_models_dict"]:
                                    if hasattr(base_model, attr_name):
                                        try:
                                            child_models = getattr(base_model, attr_name)
                                            if isinstance(child_models, dict) and fold_model in child_models:
                                                fold_model_obj = child_models[fold_model]
                                                print(f"  Successfully got model from {attr_name}")
                                                break
                                        except Exception as e:
                                            print(f"  Method 4 ({attr_name}) failed: {e}")
                            except Exception as e:
                                print(f"  Method 4 overall failed: {e}")
                        
                        # Method 5: Try loading from file system path (models/<BAG_name>/<fold_name>/model.pkl)
                        if fold_model_obj is None and predictor is not None and hasattr(predictor, "path"):
                            try:
                                from autogluon.common.loaders import load_pkl
                                # Path format: <predictor.path>/models/<model_name>/<fold_name>/model.pkl
                                model_path = os.path.join(predictor.path, "models", model_name, fold_model, "model.pkl")
                                if os.path.exists(model_path):
                                    fold_model_obj = load_pkl.load(path=model_path)
                                    print(f"  Successfully loaded model from file path: {model_path}")
                                else:
                                    print(f"  Method 5: Model file not found at {model_path}")
                            except Exception as e:
                                print(f"  Method 5 failed: {e}")
                        
                        if fold_model_obj is not None:
                            actual_model = _get_base_model(fold_model_obj)
                            print(f"  Successfully extracted base model, type: {type(actual_model)}")
                        else:
                            print(f"  Warning: All methods failed to load model {fold_model}")
                            actual_model = None
                    else:
                        print(f"  Warning: Predictor not provided, cannot load model from name")
                        actual_model = None
                else:
                    # It's already a model object
                    actual_model = _get_base_model(fold_model)
                    print(f"  Extracted from fold model type: {type(actual_model)}")
        except Exception as e:
            print(f"  Warning: Error accessing 'models' attribute: {e}")
    
    # Try alternative attribute name
    if actual_model is None and hasattr(base_model, "_models"):
        try:
            models_list = base_model._models
            print(f"  Found '_models' attribute with {len(models_list) if models_list else 0} items")
            if models_list and len(models_list) > 0:
                fold_model = models_list[0]
                # Check if fold_model is a string (model name) or actual model object
                if isinstance(fold_model, str):
                    print(f"  Detected model name string in _models: {fold_model}, loading actual model...")
                    if predictor is not None:
                        try:
                            if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                                fold_model_obj = predictor._trainer.load_model(fold_model)
                                actual_model = _get_base_model(fold_model_obj)
                                print(f"  Successfully loaded model from predictor, type: {type(actual_model)}")
                            else:
                                fold_model_obj = None
                                actual_model = None
                        except Exception as e:
                            print(f"  Warning: Failed to load model {fold_model}: {e}")
                            fold_model_obj = None
                            actual_model = None
                    else:
                        fold_model_obj = None
                        actual_model = None
                else:
                    actual_model = _get_base_model(fold_model)
                    print(f"  Extracted from _models fold model type: {type(actual_model)}")
        except Exception as e:
            print(f"  Warning: Error accessing '_models' attribute: {e}")
    
    # If not a BAG model or extraction failed, try direct attributes and other methods
    if actual_model is None:
        # Try _model or model attributes
        if hasattr(base_model, "_model") and base_model._model is not None:
            print(f"  Trying base_model._model...")
            actual_model = _get_base_model(base_model._model)
        elif hasattr(base_model, "model") and base_model.model is not None:
            print(f"  Trying base_model.model...")
            actual_model = _get_base_model(base_model.model)
        
        # Try accessing child models directly if StackerEnsembleModel
        if actual_model is None and "StackerEnsembleModel" in str(type(base_model)):
            # Try to get the base model name and load it
            if predictor is not None and hasattr(base_model, "model_names"):
                try:
                    model_names = base_model.model_names
                    if model_names and len(model_names) > 0:
                        # Get the first base model name (should be like LightGBMXT)
                        base_model_name = model_names[0]
                        print(f"  Found base model name from model_names: {base_model_name}")
                        # Try to load a non-BAG version of this model
                        # Look for models without _BAG suffix
                        leaderboard = predictor.leaderboard(silent=True)
                        for candidate in leaderboard["model"]:
                            if base_model_name.replace("_BAG_L1", "") in candidate and "_BAG" not in candidate:
                                try:
                                    candidate_model = predictor._trainer.load_model(candidate)
                                    actual_model = _get_base_model(candidate_model)
                                    print(f"  Successfully loaded non-BAG model: {candidate}, type: {type(actual_model)}")
                                    break
                                except Exception:
                                    continue
                except Exception as e:
                    print(f"  Warning: Failed to extract from model_names: {e}")
        
        if actual_model is None:
            print(f"  Using base_model directly...")
            actual_model = base_model
    
    # Final check - if still None, try to use the model directly
    if actual_model is None:
        print(f"  Warning: Could not extract tree model, trying to use model directly...")
        actual_model = model
    
    # Verify we have a valid model
    if actual_model is None:
        raise ValueError(f"Could not extract valid tree model from {model_name}. Model type: {type(base_model)}")
    
    print(f"  Final extracted model type: {type(actual_model)}")
    
    # Additional check for common tree model types
    model_type_str = str(type(actual_model))
    if "NoneType" in model_type_str:
        # Fallback to KernelExplainer if we can't extract tree model
        print(f"  Warning: Could not extract tree model, falling back to KernelExplainer...")
        return _compute_shap_kernel(model, X_background, X_explain, model_name)

    # Try to use TreeExplainer
    # Note: The underlying model expects preprocessed features, but we have raw features
    # We need to get the preprocessing pipeline from AutoGluon model and apply it
    try:
        explainer = shap.TreeExplainer(actual_model)
        
        # Try to get the expected number of features from the model
        expected_features = None
        model_feature_names = None
        if hasattr(actual_model, "num_feature"):
            expected_features = actual_model.num_feature()
            print(f"  Model expects {expected_features} features")
        elif hasattr(actual_model, "feature_name_"):
            model_feature_names = actual_model.feature_name_()
            expected_features = len(model_feature_names)
            print(f"  Model expects {expected_features} features: {model_feature_names[:5] if len(model_feature_names) > 5 else model_feature_names}...")
        
        # Check if we have feature mismatch
        actual_features = X_explain.shape[1] if hasattr(X_explain, 'shape') else len(X_explain.columns) if hasattr(X_explain, 'columns') else None
        
        if expected_features is not None and actual_features is not None and actual_features != expected_features:
            print(f"  Feature mismatch detected (data has {actual_features}, model expects {expected_features})")
            print(f"  Attempting to get preprocessing pipeline from AutoGluon model...")
            
            # Try to get preprocessing pipeline from the base_model (StackerEnsembleModel)
            X_explain_processed = None
            X_background_processed = None
            
            # Method 1: Try to get the preprocessing pipeline from the fold model wrapper (PRIORITY)
            # This is more accurate as it uses the actual fold model's preprocessing
            if fold_model_obj is not None:
                try:
                    # The fold_model_obj might be wrapped and have preprocessing
                    fold_wrapper = fold_model_obj
                    if hasattr(fold_wrapper, "_preprocess"):
                        X_explain_processed = fold_wrapper._preprocess(X_explain, fit=False)
                        X_background_processed = fold_wrapper._preprocess(X_background, fit=False)
                        processed_features = X_explain_processed.shape[1] if hasattr(X_explain_processed, 'shape') else len(X_explain_processed.columns) if hasattr(X_explain_processed, 'columns') else None
                        print(f"  Successfully preprocessed using fold model wrapper _preprocess (shape: {X_explain_processed.shape}, features: {processed_features})")
                        # Verify feature count matches
                        if processed_features is not None and processed_features != expected_features:
                            print(f"  Warning: Preprocessed features ({processed_features}) still don't match expected ({expected_features}), will try other methods")
                            X_explain_processed = None
                            X_background_processed = None
                    elif hasattr(fold_wrapper, "preprocess"):
                        X_explain_processed = fold_wrapper.preprocess(X_explain)
                        X_background_processed = fold_wrapper.preprocess(X_background)
                        processed_features = X_explain_processed.shape[1] if hasattr(X_explain_processed, 'shape') else len(X_explain_processed.columns) if hasattr(X_explain_processed, 'columns') else None
                        print(f"  Successfully preprocessed using fold model wrapper preprocess (shape: {X_explain_processed.shape}, features: {processed_features})")
                        # Verify feature count matches
                        if processed_features is not None and processed_features != expected_features:
                            print(f"  Warning: Preprocessed features ({processed_features}) still don't match expected ({expected_features}), will try other methods")
                            X_explain_processed = None
                            X_background_processed = None
                except Exception as e:
                    print(f"  Fold model wrapper preprocessing failed: {e}")
            
            # Method 2: Try to use base_model's preprocessing methods
            if X_explain_processed is None and hasattr(base_model, "_preprocess"):
                try:
                    X_explain_processed = base_model._preprocess(X_explain, fit=False)
                    X_background_processed = base_model._preprocess(X_background, fit=False)
                    processed_features = X_explain_processed.shape[1] if hasattr(X_explain_processed, 'shape') else len(X_explain_processed.columns) if hasattr(X_explain_processed, 'columns') else None
                    print(f"  Successfully preprocessed using base_model._preprocess (shape: {X_explain_processed.shape}, features: {processed_features})")
                    # Verify feature count matches
                    if processed_features is not None and processed_features != expected_features:
                        print(f"  Warning: Preprocessed features ({processed_features}) still don't match expected ({expected_features}), will try other methods")
                        X_explain_processed = None
                        X_background_processed = None
                except Exception as e:
                    print(f"  base_model._preprocess failed: {e}")
            
            # Method 3: Try to use model's feature names to select features
            if X_explain_processed is None and model_feature_names is not None:
                try:
                    # Check if model feature names match DataFrame columns
                    if isinstance(X_explain, pd.DataFrame):
                        # Try to match feature names
                        available_features = set(X_explain.columns)
                        model_features_set = set(model_feature_names)
                        
                        if model_features_set.issubset(available_features):
                            # All model features are in the data, select them
                            X_explain_processed = X_explain[model_feature_names].values
                            X_background_processed = X_background[model_feature_names].values
                            print(f"  Successfully selected features using model feature names")
                        else:
                            print(f"  Warning: Model features not fully available in data")
                            print(f"    Model features: {len(model_features_set)}, Available: {len(available_features)}")
                            print(f"    Missing: {model_features_set - available_features}")
                    else:
                        print(f"  Warning: Cannot select features from non-DataFrame")
                except Exception as e:
                    print(f"  Feature selection failed: {e}")
            
            # If preprocessing succeeded, use processed data
            if X_explain_processed is not None and X_background_processed is not None:
                # Final verification: check if processed data has correct number of features
                processed_features = X_explain_processed.shape[1] if hasattr(X_explain_processed, 'shape') else len(X_explain_processed.columns) if hasattr(X_explain_processed, 'columns') else None
                if processed_features is not None and processed_features == expected_features:
                    print(f"  Using preprocessed data for TreeExplainer (verified: {processed_features} features)")
                    # Convert to numpy array if DataFrame
                    if isinstance(X_explain_processed, pd.DataFrame):
                        X_explain_processed = X_explain_processed.values
                    if isinstance(X_background_processed, pd.DataFrame):
                        X_background_processed = X_background_processed.values
                    shap_values = explainer.shap_values(X_explain_processed)
                else:
                    print(f"  Preprocessing verification failed: processed has {processed_features} features, expected {expected_features}")
                    print(f"  Falling back to KernelExplainer which uses AutoGluon's full pipeline...")
                    return _compute_shap_kernel(model, X_background, X_explain, model_name)
            else:
                # Preprocessing failed, fall back to KernelExplainer
                print(f"  Could not obtain preprocessing pipeline. Falling back to KernelExplainer...")
                print(f"  (KernelExplainer uses AutoGluon's full pipeline including preprocessing)")
                return _compute_shap_kernel(model, X_background, X_explain, model_name)
        else:
            # No feature mismatch, use original data
            # Convert to numpy array if DataFrame
            X_explain_array = X_explain.values if isinstance(X_explain, pd.DataFrame) else X_explain
            shap_values = explainer.shap_values(X_explain_array)
        
    except Exception as e:
        error_msg = str(e)
        if "number of features" in error_msg.lower() or "feature" in error_msg.lower():
            print(f"  Warning: Feature mismatch detected ({error_msg})")
            print(f"  Falling back to KernelExplainer which uses AutoGluon's full pipeline...")
            return _compute_shap_kernel(model, X_background, X_explain, model_name)
        else:
            print(f"  Warning: TreeExplainer failed ({e}), falling back to KernelExplainer...")
            return _compute_shap_kernel(model, X_background, X_explain, model_name)

    # Handle binary classification: shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class for binary

    # Ensure it's numpy array
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    feature_names = X_explain.columns.tolist()
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_explain.index)

    return shap_values, shap_df


def _compute_shap_kernel(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, model_name: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values using KernelExplainer (for non-tree models)"""
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
        # Try different methods to get the model based on AutoGluon version
        model = None
        
        # Method 1: Try _trainer.load_model (most common in newer versions)
        if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
            try:
                model = predictor._trainer.load_model(model_name)
            except Exception:
                pass
        
        # Method 2: Try _trainer.trainer.load_model
        if model is None and hasattr(predictor, "_trainer"):
            trainer = predictor._trainer
            if hasattr(trainer, "trainer") and hasattr(trainer.trainer, "load_model"):
                try:
                    model = trainer.trainer.load_model(model_name)
                except Exception:
                    pass
        
        # Method 3: Try model_info and load from path
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
        
        # Method 4: Try accessing through _trainer.trainer.model_graph
        if model is None and hasattr(predictor, "_trainer"):
            try:
                trainer = predictor._trainer
                if hasattr(trainer, "trainer") and hasattr(trainer.trainer, "model_graph"):
                    model_graph = trainer.trainer.model_graph
                    if model_name in model_graph:
                        model = model_graph[model_name]
            except Exception:
                pass
        
        if model is None:
            raise AttributeError(f"Could not load model {model_name} using any available method")
        
        # Debug: print model type
        print(f"  Loaded model type: {type(model)}")
        if hasattr(model, "__dict__"):
            print(f"  Model attributes: {[k for k in dir(model) if not k.startswith('__')][:10]}")
            
    except Exception as e:
        print(f"  Warning: Could not load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    try:
        if _is_tree_model(model_name):
            return _compute_shap_tree(model, X_background, X_explain, model_name, predictor)
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
    train_df = pd.read_csv(args.train_csv)
    train_df = _prepare_df(train_df, args.label)

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

    # Aggregate SHAP values (weighted by ensemble weights if available)
    log_path = os.path.join(args.model_dir, "logs", "predictor_log.txt")
    weights = _extract_ensemble_weights_from_log(log_path)
    if weights and results:
        print("\nComputing weighted ensemble SHAP values...")
        weighted_shap_list = []
        for model_name in weights.keys():
            if model_name in results:
                weight = weights[model_name]
                shap_values = results[model_name]["shap_values"]
                weighted_shap_list.append(shap_values * weight)

        if weighted_shap_list:
            ensemble_shap = np.sum(weighted_shap_list, axis=0)
            ensemble_shap_df = pd.DataFrame(
                ensemble_shap, columns=X_explain.columns, index=X_explain.index
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
    print(f"\nSHAP analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

