"""Tabularization helpers for the standalone OHR core."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Tabularizer:
    """Convert raw in-memory inputs into a numeric tabular matrix.

    This component is intentionally lightweight. It does not know anything
    about datasets or domain-specific schemas. Its responsibility is limited to
    validating supported input kinds, selecting columns, enforcing a stable
    feature order, coercing values to numeric data and replacing infinities
    when configured to do so.
    """

    def __init__(
        self,
        enabled: bool = True,
        input_type: str = "tabular",
        replace_infinite: bool = True,
        drop_columns: list[str] | None = None,
        keep_columns: list[str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.input_type = str(input_type).lower()
        self.replace_infinite = bool(replace_infinite)
        self.drop_columns = list(drop_columns or [])
        self.keep_columns = list(keep_columns) if keep_columns is not None else None
        self.feature_names_: list[str] | None = None
        self.input_kind_: str | None = None
        self.selected_columns_: list[str] | None = None

    def _ensure_supported_container(self, X: np.ndarray | pd.DataFrame) -> None:
        """Reject container types that this lightweight OHR version does not support."""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                "OHR tabularization only supports pandas.DataFrame or numpy.ndarray inputs."
            )

    def _ensure_supported_input_type(self) -> None:
        """Validate the only input family supported by this first OHR version."""
        if self.input_type != "tabular":
            raise ValueError(
                f"Unsupported tabularizer input_type='{self.input_type}'. "
                "Only 'tabular' is supported in ohr-core."
            )

    def _ensure_non_empty_frame(self, frame: pd.DataFrame) -> None:
        """Reject empty tabular inputs before a schema reaches the model."""
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("OHR tabularization requires a non-empty DataFrame with features.")

    def _ensure_non_empty_array(self, numeric: np.ndarray) -> None:
        """Reject empty dense matrices before adaptation starts."""
        if numeric.size == 0 or numeric.shape[0] == 0 or numeric.shape[1] == 0:
            raise ValueError(
                "OHR tabularization requires a non-empty 2D array with at least one feature."
            )

    def _coerce_numeric(self, values: np.ndarray) -> np.ndarray:
        """Convert an array-like payload into float32 and normalize infinities."""
        numeric = np.asarray(values, dtype=np.float32)
        if self.replace_infinite:
            numeric[np.isinf(numeric)] = np.nan
        return numeric

    def _select_training_columns(self, frame: pd.DataFrame) -> list[str]:
        """Determine the feature schema learned from a training DataFrame."""
        if not self.enabled:
            return list(frame.columns)

        if self.keep_columns is not None:
            missing = [column for column in self.keep_columns if column not in frame.columns]
            if missing:
                raise KeyError(
                    "Missing required columns during tabularizer fit: "
                    + ", ".join(missing[:10])
                )
            return list(self.keep_columns)

        return [column for column in frame.columns if column not in set(self.drop_columns)]

    def _tabularize_dataframe(
        self,
        frame: pd.DataFrame,
        fit: bool,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert a DataFrame into a stable numeric matrix."""
        self._ensure_non_empty_frame(frame)
        if fit:
            selected_columns = self._select_training_columns(frame)
            if not selected_columns:
                raise ValueError("Tabularizer produced an empty feature schema")
            self.selected_columns_ = list(selected_columns)
            self.feature_names_ = list(selected_columns)
            self.input_kind_ = "dataframe"
        else:
            if self.feature_names_ is None:
                raise RuntimeError("Tabularizer must be fitted before calling transform()")
            missing = [column for column in self.feature_names_ if column not in frame.columns]
            if missing:
                raise KeyError(
                    "Missing required columns for OHR tabularization: "
                    + ", ".join(missing[:10])
                )
            selected_columns = list(self.feature_names_)

        numeric_frame = frame.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
        return self._coerce_numeric(numeric_frame.to_numpy()), list(selected_columns)

    def _tabularize_array(
        self,
        values: np.ndarray | Any,
        fit: bool,
    ) -> tuple[np.ndarray, list[str]]:
        """Validate and normalize a dense numeric matrix."""
        numeric = self._coerce_numeric(values)
        if numeric.ndim != 2:
            raise ValueError("Tabularizer expects a 2D array with shape [n_samples, n_features]")
        self._ensure_non_empty_array(numeric)

        if fit:
            feature_names = [f"feature_{index}" for index in range(numeric.shape[1])]
            self.feature_names_ = feature_names
            self.input_kind_ = "ndarray"
            self.selected_columns_ = None
            return numeric, feature_names

        if self.feature_names_ is None:
            raise RuntimeError("Tabularizer must be fitted before calling transform()")
        if numeric.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features and received {numeric.shape[1]}"
            )
        return numeric, list(self.feature_names_)

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Learn the tabular feature schema and return the numeric training matrix."""
        self._ensure_supported_input_type()
        self._ensure_supported_container(X)
        if isinstance(X, pd.DataFrame):
            return self._tabularize_dataframe(X, fit=True)
        return self._tabularize_array(X, fit=True)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Apply the learned tabular schema to new inputs."""
        self._ensure_supported_input_type()
        self._ensure_supported_container(X)
        if isinstance(X, pd.DataFrame):
            values, _ = self._tabularize_dataframe(X, fit=False)
            return values
        values, _ = self._tabularize_array(X, fit=False)
        return values
