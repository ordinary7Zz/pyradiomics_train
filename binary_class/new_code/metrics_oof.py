from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def aggregate_patient_mean(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    probs: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    buckets: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for pid, y, p in zip(patient_ids, labels, probs):
        buckets[str(pid)].append((int(y), float(p)))

    pids: List[str] = []
    y_true: List[int] = []
    y_score: List[float] = []
    for pid in sorted(buckets.keys()):
        pairs = buckets[pid]
        labels_set = {y for y, _ in pairs}
        if len(labels_set) > 1:
            raise ValueError(f"patient {pid} mixed labels: {labels_set}")
        pids.append(pid)
        y_true.append(pairs[0][0])
        y_score.append(float(np.mean([p for _, p in pairs])))
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), pids


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {"n": float(len(y_true))}
    if len(np.unique(y_true)) < 2:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
        return out
    out["AUROC"] = float(roc_auc_score(y_true, y_score))
    out["AUPRC"] = float(average_precision_score(y_true, y_score))
    return out


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cluster_ids: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    clusters = np.asarray(cluster_ids)
    unique = np.unique(clusters)
    if len(unique) < 2 or len(np.unique(y_true)) < 2:
        nan = (float("nan"), float("nan"))
        return {"AUROC": nan, "AUPRC": nan}

    aucs: List[float] = []
    aprs: List[float] = []
    alpha = (1 - ci) / 2
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in chosen])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
        aprs.append(average_precision_score(yt, ys))

    if not aucs:
        nan = (float("nan"), float("nan"))
        return {"AUROC": nan, "AUPRC": nan}
    return {
        "AUROC": (float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))),
        "AUPRC": (float(np.quantile(aprs, alpha)), float(np.quantile(aprs, 1 - alpha))),
    }
