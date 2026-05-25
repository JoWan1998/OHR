"""Input preprocessing utilities for the standalone OHR core."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


class IdentityPreprocessor:
    """Keep a scikit-learn-like interface when inputs are already prepared."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IdentityPreprocessor":
        """Return self without learning state."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a float32 view of the provided matrix."""
        return np.asarray(X, dtype=np.float32)


def _build_scaler(scaling: str) -> StandardScaler | RobustScaler:
    """Instantiate the requested numeric scaling strategy."""
    normalized = str(scaling).lower()
    if normalized == "standard":
        return StandardScaler()
    if normalized == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unsupported preprocessing.scaling='{scaling}'. "
        "Supported values are 'none', 'standard' and 'robust'."
    )


def build_input_preprocessor(
    scaling: str,
    preprocessed: bool,
    handle_missing: str = "median",
) -> IdentityPreprocessor | Pipeline:
    """Build the preprocessing stage that follows tabularization.

    Missing-value handling remains outside the tabularizer so the latter can
    focus purely on turning raw in-memory structures into a consistent table.
    """

    if preprocessed:
        return IdentityPreprocessor()

    normalized_missing = str(handle_missing).lower()
    steps: list[tuple[str, Any]] = []
    if normalized_missing == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    elif normalized_missing != "none":
        raise ValueError(
            f"Unsupported handle_missing='{handle_missing}'. Supported values are 'median' and 'none'."
        )

    normalized_scaling = str(scaling).lower()
    if normalized_scaling != "none":
        steps.append(("scaler", _build_scaler(normalized_scaling)))

    if not steps:
        return IdentityPreprocessor()
    return Pipeline(steps)
