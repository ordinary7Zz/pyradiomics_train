from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plots.plotting_utils import paper_friendly_name, prepare_df, save_waterfall_plot
from shap_analyze.autogluon_introspection import get_main_models, load_autogluon_model
from shap_analyze.shap_compute import compute_shap_for_model
from shap_analyze.shap_local_plots import save_compact_shap_bar_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate waterfall and compact SHAP bar plots for one image from TRAIN_CSV."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="AutoGluon model directory")
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV file")
    parser.add_argument("--filename", type=str, required=True, help="Target filename to explain")
    parser.add_argument("--label", type=str, default="label", help="Label column name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <model_dir>/shap_analysis_single_image)",
    )
    parser.add_argument(
        "--background_samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP",
    )
    parser.add_argument(
        "--main_models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of main model names to analyze",
    )
    parser.add_argument(
        "--skip_neural_net",
        action="store_true",
        help="Skip neural network models (NeuralNetFastAI) in SHAP analysis",
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=5,
        help="Number of top features shown in waterfall and compact SHAP bar plots (default: 5)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Task name shown in the compact SHAP legend title",
    )
    parser.add_argument(
        "--positive_class_name",
        type=str,
        default=None,
        help="Human-readable name for label 1, shown in the compact SHAP x-axis label and legend",
    )
    parser.add_argument(
        "--negative_class_name",
        type=str,
        default=None,
        help="Human-readable name for label 0, shown in the compact SHAP legend",
    )
    parser.add_argument(
        "--output_space",
        type=str,
        default=None,
        help="SHAP output space used for labeling, e.g. probability or raw score",
    )
    return parser.parse_args()


def _basenameish(value: object) -> str:
    return os.path.basename(str(value))


def _safe_token(value: object) -> str:
    token = _basenameish(value)
    token = os.path.splitext(token)[0]
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", token)
    return token.strip("._") or "sample"


def _positive_probability(y_proba: object) -> np.ndarray:
    if isinstance(y_proba, pd.DataFrame):
        if 1 in y_proba.columns:
            return y_proba[1].to_numpy()
        return y_proba.iloc[:, -1].to_numpy()
    if isinstance(y_proba, np.ndarray):
        if y_proba.ndim == 2:
            return y_proba[:, 1]
        return y_proba.reshape(-1)
    return np.asarray(y_proba).reshape(-1)


def _resolve_target_row(raw_df: pd.DataFrame, train_df: pd.DataFrame, filename: str) -> tuple[object, str]:
    id_col: Optional[str] = None
    if "filename" in raw_df.columns:
        id_col = "filename"
    elif "image_path" in raw_df.columns:
        id_col = "image_path"

    if id_col is None:
        raise ValueError("TRAIN_CSV must contain a 'filename' or 'image_path' column.")

    sample_ids = raw_df.loc[train_df.index, id_col].astype(str)
    normalized_ids = sample_ids.map(_basenameish)
    query = str(filename)
    query_base = _basenameish(filename)
    matches = np.where((sample_ids == query) | (normalized_ids == query_base))[0]

    if len(matches) == 0:
        raise ValueError(f"filename '{filename}' was not found in TRAIN_CSV.")

    if len(matches) > 1:
        print(f"Warning: filename '{filename}' matched multiple rows; using the first match.")

    matched_pos = int(matches[0])
    target_index = train_df.index[matched_pos]
    matched_name = normalized_ids.iloc[matched_pos]
    return target_index, matched_name


def _get_sample_features(predictor, x_explain: pd.DataFrame, shap_df: pd.DataFrame):
    try:
        if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
            processed = predictor._learner.feature_generator.transform(x_explain)
            if isinstance(processed, pd.DataFrame):
                return processed.iloc[0]
            return np.asarray(processed)[0]
    except Exception as exc:
        print(f"Warning: could not transform features for display: {exc}")
    return x_explain.iloc[0]


def _extract_feature_values(sample_features, top_indices: np.ndarray, top_feature_names: list[str], shap_columns: list[str]):
    values = []
    if isinstance(sample_features, pd.Series):
        for name, idx in zip(top_feature_names, top_indices):
            if name in sample_features.index:
                values.append(sample_features[name])
            else:
                try:
                    pos = shap_columns.index(name)
                    values.append(sample_features.iloc[pos])
                except Exception:
                    values.append(np.nan)
    elif isinstance(sample_features, np.ndarray):
        for idx in top_indices:
            values.append(sample_features[idx] if idx < len(sample_features) else np.nan)
    else:
        for idx in top_indices:
            try:
                values.append(sample_features[idx])
            except Exception:
                values.append(np.nan)
    return values


def main() -> None:
    args = parse_args()

    print(f"Loading predictor from: {args.model_dir}")
    predictor = __import__("autogluon.tabular", fromlist=["TabularPredictor"]).TabularPredictor.load(args.model_dir)

    print(f"Loading training data from: {args.train_csv}")
    raw_df = pd.read_csv(args.train_csv)
    train_df = prepare_df(raw_df.copy(), args.label)

    target_index, target_name = _resolve_target_row(raw_df, train_df, args.filename)
    print(f"Selected target sample: {target_name} (row index: {target_index})")

    x_train = train_df.drop(columns=[args.label]).copy()
    y_train = train_df[args.label].copy()
    x_explain = x_train.loc[[target_index]].copy()
    y_explain = y_train.loc[[target_index]].copy()

    background_pool = x_train.drop(index=target_index, errors="ignore")
    if len(background_pool) == 0:
        background_pool = x_train.copy()

    np.random.seed(42)
    if len(background_pool) > args.background_samples:
        background_idx = np.random.choice(len(background_pool), size=args.background_samples, replace=False)
        x_background = background_pool.iloc[background_idx].copy()
    else:
        x_background = background_pool.copy()
    print(f"Using {len(x_background)} background samples")
    print(f"Explaining 1 sample with label={int(y_explain.iloc[0])}")

    main_models = get_main_models(predictor, args.model_dir, args.main_models)
    print(f"Main models to analyze: {main_models}")

    output_dir = args.output_dir or os.path.join(args.model_dir, "shap_analysis_single_image")
    os.makedirs(output_dir, exist_ok=True)
    waterfall_dir = os.path.join(output_dir, "waterfall")
    compact_bar_dir = os.path.join(output_dir, "compact_shap_bar")
    os.makedirs(waterfall_dir, exist_ok=True)
    os.makedirs(compact_bar_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    max_display = max(1, int(args.top_features))
    target_tag = _safe_token(target_name)

    for model_name in main_models:
        print(f"\nAnalyzing model: {model_name}")
        result = compute_shap_for_model(predictor, model_name, x_background, x_explain, args.skip_neural_net)
        if result is None:
            continue

        shap_values, shap_df = result
        sample_shap = np.asarray(shap_values[0]).reshape(-1)
        feature_names = shap_df.columns.tolist()
        if len(sample_shap) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch for {model_name}: {len(feature_names)} names vs {len(sample_shap)} SHAP values"
            )

        top_indices = np.argsort(np.abs(sample_shap))[-max_display:][::-1]
        top_shap = sample_shap[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]
        display_feature_names = [paper_friendly_name(name) for name in top_feature_names]

        sample_features = _get_sample_features(predictor, x_explain, shap_df)
        top_feature_values = _extract_feature_values(sample_features, top_indices, top_feature_names, feature_names)

        try:
            model = load_autogluon_model(predictor, model_name)
            model_proba = model.predict_proba(x_background)
        except Exception as exc:
            print(f"Warning: could not load model-specific predictor for base value; using predictor output instead: {exc}")
            model_proba = predictor.predict_proba(x_background)

        base_value = float(np.mean(_positive_probability(model_proba)))

        waterfall_path = os.path.join(waterfall_dir, f"{model_name}_waterfall_{target_tag}.png")
        waterfall_textless_path = os.path.join(waterfall_dir, f"{model_name}_waterfall_{target_tag}_textless.svg")
        waterfall_saved = save_waterfall_plot(
            top_shap,
            top_feature_values,
            display_feature_names,
            base_value,
            waterfall_path,
            max_display,
            textless_svg_path=waterfall_textless_path,
            title=f"{model_name} SHAP",
            title_fontsize=24,
            export_formats=("png", "svg"),
            dpi=150,
            figsize=(13.0, 9.0),
            bbox_inches="tight",
        )

        compact_bar_path = os.path.join(compact_bar_dir, f"{model_name}_compact_shap_bar_{target_tag}.png")
        compact_bar_textless_path = os.path.join(
            compact_bar_dir, f"{model_name}_compact_shap_bar_{target_tag}_textless.svg"
        )
        compact_bar_saved = save_compact_shap_bar_plot(
            top_shap,
            top_feature_names,
            compact_bar_path,
            max_display,
            task_name=args.task_name,
            positive_class_name=args.positive_class_name,
            negative_class_name=args.negative_class_name,
            output_space=args.output_space,
            textless_svg_path=compact_bar_textless_path,
            xlabel_fontsize=12.0,
            ytick_fontsize=11.0,
            export_formats=("png", "svg"),
            dpi=300,
            figsize=(3.15, 3.15 * 4 / 3),
        )

        print(f"Saved waterfall: {', '.join(waterfall_saved)}")
        print(f"Saved compact bar: {', '.join(compact_bar_saved)}")

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
