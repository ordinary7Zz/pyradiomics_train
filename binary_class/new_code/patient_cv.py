from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Sequence, Tuple

from sklearn.model_selection import KFold, StratifiedKFold

from .packed_io import SampleRecord


def patient_majority_label(labels: Sequence[int]) -> int:
    if not labels:
        raise ValueError("empty labels")
    return Counter(int(x) for x in labels).most_common(1)[0][0]


def split_samples_by_patient_cv(
    samples: Sequence[SampleRecord],
    n_splits: int,
    seed: int,
) -> List[Tuple[List[SampleRecord], List[SampleRecord]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    pid_to_samples: dict[str, List[SampleRecord]] = defaultdict(list)
    for sample in samples:
        pid_to_samples[sample.patient_id].append(sample)

    patients = sorted(pid_to_samples.keys())
    if len(patients) < n_splits:
        raise ValueError(f"patient clusters {len(patients)} < n_splits {n_splits}")

    patient_labels = [
        patient_majority_label([s.label for s in pid_to_samples[pid]]) for pid in patients
    ]
    min_class = min(Counter(patient_labels).values())

    if min_class >= n_splits:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            patients, patient_labels
        )
    else:
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(patients)

    folds: List[Tuple[List[SampleRecord], List[SampleRecord]]] = []
    for train_idx, val_idx in split_iter:
        train_pids = {patients[i] for i in train_idx}
        val_pids = {patients[i] for i in val_idx}
        train_samples = [s for pid in train_pids for s in pid_to_samples[pid]]
        val_samples = [s for pid in val_pids for s in pid_to_samples[pid]]
        folds.append((train_samples, val_samples))
    return folds
