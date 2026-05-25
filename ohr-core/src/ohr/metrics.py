"""Common multiclass metrics for the standalone OHR core."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a compact set of multiclass classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def confusion_matrix_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Return the confusion matrix as a labeled DataFrame."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
