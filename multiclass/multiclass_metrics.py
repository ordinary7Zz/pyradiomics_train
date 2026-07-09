import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# TIRADS 标签范围
TIRADS_CLASSES = [1, 2, 3, 4, 5]


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]
) -> Dict[int, Dict[str, int]]:
    """逐类统计 TP / TN / FP / FN（OvR 方式）。"""
    result: Dict[int, Dict[str, int]] = {}
    n = len(y_true)
    for cls in classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        tn = int(((y_true != cls) & (y_pred != cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        result[cls] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "support": int((y_true == cls).sum())}
    return result


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def _compute_class_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    return {
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
    }


def _compute_per_class_auroc(y_true: np.ndarray, y_score: np.ndarray, classes: List[int]) -> Dict[int, float]:
    """OvR AUROC per class."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return {cls: float("nan") for cls in classes}

    result = {}
    # y_score shape: (n_samples, n_classes), columns aligned with classes
    if y_score.ndim != 2 or y_score.shape[1] != len(classes):
        return {cls: float("nan") for cls in classes}

    for i, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if len(np.unique(y_bin)) < 2:
            result[cls] = float("nan")
            continue
        result[cls] = _safe_metric(roc_auc_score, y_bin, y_score[:, i])
    return result


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Confidence-ECE: 按最大预测概率分箱，比较置信度与准确率。"""
    if len(y_true) == 0 or n_bins <= 0:
        return float("nan")

    confidences = np.max(y_score, axis=1)
    predictions = np.argmax(y_score, axis=1)
    # 将 argmax 索引映射回实际类别标签（classes 已排序）
    accuracies = (predictions == y_true).astype(float)  # 注意：这里 predictions 是索引，需要对齐

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(accuracies[mask]))
        weight = float(len(confidences[mask]) / n)
        ece += abs(bin_conf - bin_acc) * weight
    return float(ece)


def compute_classwise_ece(
    y_true: np.ndarray, y_score: np.ndarray, classes: List[int], n_bins: int = 10
) -> Dict[int, float]:
    """逐类 ECE：对每一类，比较该类预测概率与真实属于该类的比例。"""
    result = {}
    if y_score.ndim != 2 or y_score.shape[1] != len(classes):
        return {cls: float("nan") for cls in classes}

    for i, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        probs = y_score[:, i]
        if len(np.unique(probs)) <= 1:
            result[cls] = float("nan")
            continue

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        n = len(probs)
        ece_val = 0.0
        for j in range(n_bins):
            left = edges[j]
            right = edges[j + 1]
            if j == n_bins - 1:
                mask = (probs >= left) & (probs <= right)
            else:
                mask = (probs >= left) & (probs < right)
            if not np.any(mask):
                continue
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(y_bin[mask]))
            weight = float(len(probs[mask]) / n)
            ece_val += abs(bin_conf - bin_acc) * weight
        result[cls] = float(ece_val)
    return result


def compute_multiclass_metrics(
    y_true: pd.Series,
    y_score_df: pd.DataFrame,
    y_pred: Optional[pd.Series] = None,
    classes: Optional[List[int]] = None,
    ece_bins: int = 10,
) -> Dict[str, float]:
    """
    计算多分类评估指标。

    Parameters
    ----------
    y_true: 真实标签 (1-5)
    y_score_df: predict_proba 输出，columns 为类别标签 [1,2,3,4,5]
    y_pred: 预测标签，若为 None 则按 argmax 计算
    classes: 类别列表，默认 [1,2,3,4,5]
    ece_bins: ECE 分箱数
    """
    if classes is None:
        classes = TIRADS_CLASSES

    yy = y_true.to_numpy(dtype=int)
    if y_pred is None:
        # 按 proba 的 argmax + classes 映射得到预测标签
        pred_idx = np.argmax(y_score_df.to_numpy(), axis=1)
        pp = np.array([classes[i] for i in pred_idx], dtype=int)
    else:
        pp = y_pred.to_numpy(dtype=int)

    ss = y_score_df.to_numpy(dtype=float)
    n_samples = len(yy)
    n_classes = len(classes)

    # --- 基础统计 ---
    metrics: Dict[str, float] = {"n_samples": float(n_samples), "n_classes": float(n_classes)}

    # --- 逐类混淆矩阵与指标 ---
    cm = _confusion_matrix_counts(yy, pp, classes)

    for cls in classes:
        cls_metrics = _compute_class_metrics(cm[cls])
        support = cm[cls]["support"]
        for k, v in cls_metrics.items():
            metrics[f"class_{cls}_{k}"] = v
        metrics[f"class_{cls}_support"] = float(support)

    # --- 宏平均 ---
    for metric_name in ["precision", "recall", "specificity", "f1"]:
        values = []
        for cls in classes:
            v = metrics.get(f"class_{cls}_{metric_name}", float("nan"))
            if not math.isnan(v):
                values.append(v)
        metrics[f"macro_{metric_name}"] = float(np.mean(values)) if values else float("nan")

    # --- 加权平均 ---
    total = sum(cm[cls]["support"] for cls in classes)
    for metric_name in ["precision", "recall", "specificity", "f1"]:
        weighted_sum = 0.0
        for cls in classes:
            v = metrics.get(f"class_{cls}_{metric_name}", float("nan"))
            s = cm[cls]["support"]
            if not math.isnan(v) and s > 0:
                weighted_sum += v * s
        metrics[f"weighted_{metric_name}"] = float(weighted_sum / total) if total > 0 else float("nan")

    # --- Accuracy ---
    try:
        from sklearn.metrics import accuracy_score
        metrics["accuracy"] = _safe_metric(accuracy_score, yy, pp)
    except Exception:
        metrics["accuracy"] = float("nan")

    # --- Balanced Accuracy ---
    try:
        from sklearn.metrics import balanced_accuracy_score
        metrics["balanced_accuracy"] = _safe_metric(balanced_accuracy_score, yy, pp)
    except Exception:
        recalls = [metrics.get(f"class_{cls}_recall", float("nan")) for cls in classes]
        valid_recalls = [r for r in recalls if not math.isnan(r)]
        metrics["balanced_accuracy"] = float(np.mean(valid_recalls)) if valid_recalls else float("nan")

    # --- Cohen's Kappa ---
    try:
        from sklearn.metrics import cohen_kappa_score
        metrics["kappa"] = _safe_metric(cohen_kappa_score, yy, pp)
    except Exception:
        metrics["kappa"] = float("nan")

    # --- Quadratic Weighted Kappa (QWK，适合有序分类) ---
    try:
        from sklearn.metrics import cohen_kappa_score as cks
        metrics["qwk"] = _safe_metric(cks, yy, pp, weights="quadratic")
    except Exception:
        metrics["qwk"] = float("nan")

    # --- OvR AUROC per class ---
    per_class_auroc = _compute_per_class_auroc(yy, ss, classes)
    auroc_values = []
    for cls in classes:
        v = per_class_auroc.get(cls, float("nan"))
        metrics[f"class_{cls}_auroc"] = v
        if not math.isnan(v):
            auroc_values.append(v)
    metrics["macro_auroc"] = float(np.mean(auroc_values)) if auroc_values else float("nan")

    # --- Confidence ECE ---
    metrics["ece"] = compute_ece(yy, ss, n_bins=ece_bins)

    # --- Classwise ECE ---
    classwise_ece = compute_classwise_ece(yy, ss, classes, n_bins=ece_bins)
    ece_values = []
    for cls in classes:
        v = classwise_ece.get(cls, float("nan"))
        metrics[f"class_{cls}_ece"] = v
        if not math.isnan(v):
            ece_values.append(v)
    if ece_values:
        metrics["macro_ece"] = float(np.mean(ece_values))
        # weighted ECE
        weighted_ece = 0.0
        for cls in classes:
            v = classwise_ece.get(cls, float("nan"))
            s = cm[cls]["support"]
            if not math.isnan(v) and s > 0:
                weighted_ece += v * s
        metrics["weighted_ece"] = float(weighted_ece / total) if total > 0 else float("nan")
    else:
        metrics["macro_ece"] = float("nan")
        metrics["weighted_ece"] = float("nan")

    return metrics


def bootstrap_multiclass_cis(
    y_true: pd.Series,
    y_score_df: pd.DataFrame,
    classes: Optional[List[int]] = None,
    ece_bins: int = 10,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap 置信区间估计。

    返回每个指标的 ci_lower / ci_upper。
    """
    if classes is None:
        classes = TIRADS_CLASSES

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    yy = y_true.reset_index(drop=True)
    ss = y_score_df.reset_index(drop=True)
    n = len(yy)

    # 确定所有 metric 名称（先跑一次全量获取 key 列表）
    ref_metrics = compute_multiclass_metrics(
        yy, ss, y_pred=None, classes=classes, ece_bins=ece_bins
    )
    # 排除 n_samples / n_classes 等固定值
    skip_keys = {"n_samples", "n_classes"}
    metric_names = sorted(k for k in ref_metrics if k not in skip_keys)

    samples: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = yy.iloc[idx].reset_index(drop=True)
        s_boot = ss.iloc[idx].reset_index(drop=True)
        m = compute_multiclass_metrics(
            y_boot, s_boot, y_pred=None, classes=classes, ece_bins=ece_bins
        )
        for name in metric_names:
            v = m.get(name, float("nan"))
            if not math.isnan(v):
                samples[name].append(float(v))

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
