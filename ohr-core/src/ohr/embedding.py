"""Embedding strategies that bridge tabular inputs and the OHR adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


class EmbeddingStage:
    """Apply the explicit embedding stage `h = E(x)` before the OHR adapter.

    Supported modes are intentionally separated for clean experimentation:

    - `fixed`: preserve or deterministically resize the feature dimension
    - `proportional`: derive the output dimension from the input width
    - `pca_based`: fit an explicit PCA model during training only

    The non-PCA modes support two deterministic resize strategies:

    - `random_orthogonal`: default, uses a seeded orthogonal basis slice so the
      embedding is more informative than simple truncation or zero padding
    - `identity_resize`: legacy mode that keeps the previous truncate/pad behavior
    """

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "fixed",
        projection_strategy: str = "random_orthogonal",
        output_dim: int | None = None,
        proportion: float = 1.0,
        explained_variance_ratio: float | None = None,
        whiten: bool = False,
        random_state: int = 42,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode).lower()
        self.projection_strategy = str(projection_strategy).lower()
        self.output_dim = int(output_dim) if output_dim is not None else None
        self.proportion = float(proportion)
        self.explained_variance_ratio = (
            float(explained_variance_ratio) if explained_variance_ratio is not None else None
        )
        self.whiten = bool(whiten)
        self.random_state = int(random_state)

        self.input_dim_: int | None = None
        self.output_dim_: int | None = None
        self.feature_names_: list[str] | None = None
        self.mode_: str | None = None
        self.projection_matrix_: np.ndarray | None = None
        self.pca_model_: PCA | None = None

    def _validate_mode(self) -> str:
        """Return the effective embedding mode for the current configuration."""
        if not self.enabled:
            return "fixed"
        if self.mode not in {"fixed", "proportional", "pca_based"}:
            raise ValueError(
                f"Unsupported embedding mode='{self.mode}'. "
                "Supported values are 'fixed', 'proportional' and 'pca_based'."
            )
        return self.mode

    def _validate_projection_strategy(self) -> str:
        """Return the deterministic resize strategy used by non-PCA embeddings."""
        if self.projection_strategy not in {"random_orthogonal", "identity_resize"}:
            raise ValueError(
                f"Unsupported embedding.projection_strategy='{self.projection_strategy}'. "
                "Supported values are 'random_orthogonal' and 'identity_resize'."
            )
        return self.projection_strategy

    def _validate_matrix(self, X: np.ndarray | Any) -> np.ndarray:
        """Convert the embedding input into a finite 2D float32 array when possible."""
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("EmbeddingStage expects a 2D array with shape [n_samples, n_features].")
        if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("EmbeddingStage requires a non-empty 2D array.")
        return matrix

    def _resolve_fixed_output_dim(self, input_dim: int) -> int:
        """Resolve the explicit fixed output dimension."""
        if self.output_dim is None:
            return int(input_dim)
        if self.output_dim <= 0:
            raise ValueError("embedding.output_dim must be greater than zero.")
        return int(self.output_dim)

    def _resolve_proportional_output_dim(self, input_dim: int) -> int:
        """Resolve the output dimension derived from the input width."""
        if self.proportion <= 0.0:
            raise ValueError("embedding.proportion must be greater than zero.")
        return max(1, int(math.ceil(self.proportion * input_dim)))

    def _build_identity_resize_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Build the legacy truncate/pad resize operator."""
        matrix = np.zeros((output_dim, input_dim), dtype=np.float32)
        diagonal_extent = min(input_dim, output_dim)
        indices = np.arange(diagonal_extent)
        matrix[indices, indices] = 1.0
        return matrix

    def _build_random_orthogonal_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Build a seeded rectangular projection from an orthogonal basis.

        For `output_dim <= input_dim` the resulting rows are orthonormal.
        For `output_dim > input_dim` the resulting columns are orthonormal.
        """
        basis_dim = max(input_dim, output_dim)
        rng = np.random.default_rng(self.random_state)
        gaussian_basis = rng.normal(size=(basis_dim, basis_dim)).astype(np.float64)
        orthogonal_basis, upper = np.linalg.qr(gaussian_basis)
        signs = np.sign(np.diag(upper))
        signs[signs == 0.0] = 1.0
        orthogonal_basis = orthogonal_basis * signs
        projection = orthogonal_basis[:output_dim, :input_dim]
        return projection.astype(np.float32)

    def _build_deterministic_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Build the configured deterministic non-PCA resize operator."""
        strategy = self._validate_projection_strategy()
        if strategy == "identity_resize":
            return self._build_identity_resize_projection(input_dim, output_dim)
        return self._build_random_orthogonal_projection(input_dim, output_dim)

    def _build_feature_names(self, prefix: str, output_dim: int) -> list[str]:
        """Build stable feature names for the embedded representation."""
        return [f"{prefix}_{index}" for index in range(output_dim)]

    def _ensure_fitted(self) -> None:
        """Fail fast when transform is requested before fitting."""
        if self.input_dim_ is None or self.output_dim_ is None or self.feature_names_ is None:
            raise RuntimeError("EmbeddingStage must be fitted before calling transform().")

    def _fit_fixed_or_proportional(
        self,
        matrix: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Fit the deterministic non-PCA embedding modes."""
        input_dim = int(matrix.shape[1])
        output_dim = (
            self._resolve_fixed_output_dim(input_dim)
            if mode == "fixed"
            else self._resolve_proportional_output_dim(input_dim)
        )

        projection_matrix = self._build_deterministic_projection(input_dim, output_dim)
        transformed = matrix @ projection_matrix.T

        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.mode_ = mode
        self.projection_matrix_ = projection_matrix
        self.pca_model_ = None
        self.feature_names_ = self._build_feature_names("embedding_feature", output_dim)
        return transformed.astype(np.float32), list(self.feature_names_)

    def _fit_pca_based(self, matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Fit the explicit PCA-based embedding strategy."""
        if not np.isfinite(matrix).all():
            raise ValueError(
                "embedding.mode='pca_based' requires finite inputs after preprocessing."
            )

        input_dim = int(matrix.shape[1])
        if self.explained_variance_ratio is not None:
            if not 0.0 < self.explained_variance_ratio <= 1.0:
                raise ValueError("embedding.explained_variance_ratio must be in the interval (0, 1].")
            n_components: int | float | None = self.explained_variance_ratio
        elif self.output_dim is not None:
            if self.output_dim <= 0:
                raise ValueError("embedding.output_dim must be greater than zero.")
            n_components = min(int(self.output_dim), int(matrix.shape[0]), input_dim)
        else:
            n_components = None

        pca_model = PCA(
            n_components=n_components,
            whiten=self.whiten,
            svd_solver="full",
            random_state=self.random_state,
        )
        transformed = pca_model.fit_transform(matrix).astype(np.float32)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)

        self.input_dim_ = input_dim
        self.output_dim_ = int(transformed.shape[1])
        self.mode_ = "pca_based"
        self.projection_matrix_ = None
        self.pca_model_ = pca_model
        self.feature_names_ = self._build_feature_names("pca_component", self.output_dim_)
        return transformed, list(self.feature_names_)

    def fit_transform(self, X: np.ndarray | Any) -> tuple[np.ndarray, list[str]]:
        """Fit the embedding strategy on training data and return embedded features."""
        matrix = self._validate_matrix(X)
        mode = self._validate_mode()

        if mode in {"fixed", "proportional"}:
            return self._fit_fixed_or_proportional(matrix, mode)
        return self._fit_pca_based(matrix)

    def transform(self, X: np.ndarray | Any) -> np.ndarray:
        """Apply the fitted embedding stage to inference inputs."""
        matrix = self._validate_matrix(X)
        self._ensure_fitted()
        if int(matrix.shape[1]) != int(self.input_dim_):
            raise ValueError(
                f"EmbeddingStage expected {self.input_dim_} input features and received {matrix.shape[1]}."
            )

        if self.mode_ in {"fixed", "proportional"}:
            assert self.projection_matrix_ is not None
            return (matrix @ self.projection_matrix_.T).astype(np.float32)

        if not np.isfinite(matrix).all():
            raise ValueError("embedding.mode='pca_based' requires finite inputs during transform().")
        if self.pca_model_ is None:
            raise RuntimeError("EmbeddingStage PCA model is missing.")
        return self.pca_model_.transform(matrix).astype(np.float32)
