"""Common metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean() * 100.0)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred) * 100.0)


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores) * 100.0)


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    return float(sen * 100), float(spe * 100)
