import math
from typing import Dict

import numpy as np
import pandas as pd


def get_positive_proba(proba: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.astype(float)
    if 1 in proba.columns:
        return proba[1].astype(float)
    return proba.iloc[:, -1].astype(float)


def _safe_auroc(y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")

    yy = y_true.to_numpy()
    ss = y_score.to_numpy()
    if len(pd.unique(yy)) != 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, ss))
    except Exception:
        return float("nan")


def _safe_auprc(y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return float("nan")

    yy = y_true.to_numpy()
    ss = y_score.to_numpy()
    if len(pd.unique(yy)) != 2:
        return float("nan")
    try:
        return float(average_precision_score(yy, ss))
    except Exception:
        return float("nan")


def _safe_metric(metric_fn, y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        return float(metric_fn(y_true.to_numpy(), y_pred.to_numpy()))
    except Exception:
        return float("nan")


def _specificity(y_true: pd.Series, y_pred: pd.Series) -> float:
    yy = y_true.to_numpy()
    pp = y_pred.to_numpy()
    if len(pd.unique(yy)) != 2:
        return float("nan")
    tn = int(((yy == 0) & (pp == 0)).sum())
    fp = int(((yy == 0) & (pp == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def compute_ece(y_true: pd.Series, y_score: pd.Series, n_bins: int = 10) -> float:
    yy = y_true.to_numpy()
    ss = y_score.to_numpy(dtype=float)
    if len(yy) == 0:
        return float("nan")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(ss)

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (ss >= left) & (ss <= right)
        else:
            mask = (ss >= left) & (ss < right)

        if not np.any(mask):
            continue

        bin_scores = ss[mask]
        bin_true = yy[mask]
        confidence = float(np.mean(bin_scores))
        accuracy = float(np.mean(bin_true))
        weight = float(len(bin_scores) / n)
        ece += abs(confidence - accuracy) * weight

    return float(ece)


def compute_binary_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> Dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception:
        accuracy_score = f1_score = precision_score = recall_score = None

    y_pred = (y_score >= float(threshold)).astype(int)

    metrics: Dict[str, float] = {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "acc": float("nan"),
        "prec": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "specificity": _specificity(y_true, y_pred),
        "ece": compute_ece(y_true, y_score, n_bins=ece_bins),
    }

    if accuracy_score is not None:
        metrics["acc"] = _safe_metric(accuracy_score, y_true, y_pred)
        metrics["prec"] = _safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, y_pred)
        metrics["recall"] = _safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred)
        metrics["f1"] = _safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_true, y_pred)

    return metrics


def bootstrap_metric_cis(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float = 0.5,
    ece_bins: int = 10,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    yy = pd.Series(y_true).reset_index(drop=True)
    ss = pd.Series(y_score).reset_index(drop=True)
    n = len(yy)
    metric_names = ["auroc", "auprc", "acc", "prec", "recall", "f1", "specificity", "ece"]
    samples = {name: [] for name in metric_names}

    if n == 0:
        result: Dict[str, float] = {}
        for name in metric_names:
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
        return result

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = yy.iloc[idx].reset_index(drop=True)
        s_boot = ss.iloc[idx].reset_index(drop=True)
        metric_values = compute_binary_metrics(y_boot, s_boot, threshold=threshold, ece_bins=ece_bins)
        for name in metric_names:
            value = metric_values.get(name, float("nan"))
            if not math.isnan(value):
                samples[name].append(float(value))

    alpha = (1.0 - ci_level) / 2.0
    result: Dict[str, float] = {}
    for name in metric_names:
        values = samples[name]
        if not values:
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
            continue
        result[f"{name}_ci_lower"] = float(np.quantile(values, alpha))
        result[f"{name}_ci_upper"] = float(np.quantile(values, 1.0 - alpha))
    return result
