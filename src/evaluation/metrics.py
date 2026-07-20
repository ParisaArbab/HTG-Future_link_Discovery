from __future__ import annotations

from typing import Iterable
import numpy as np


def binary_metrics(labels: Iterable[int], scores: Iterable[float], threshold: float = 0.5):
    labels = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(scores), dtype=float)
    predictions = (scores >= threshold).astype(int)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def precision_at_k(labels: Iterable[int], scores: Iterable[float], k: int = 10) -> float:
    labels = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(scores), dtype=float)
    if len(labels) == 0:
        return 0.0
    k = min(k, len(labels))
    top_indices = np.argsort(scores)[::-1][:k]
    return float(labels[top_indices].mean())
