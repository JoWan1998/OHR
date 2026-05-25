"""Projection modules for the OHR nucleus."""

from __future__ import annotations

import torch
from torch import nn


class OrthogonalProjection(nn.Module):
    """Apply the explicit projection `t = P h` with fixed or learnable matrices."""

    def __init__(self, embedding_dim: int, projection_type: str = "learnable") -> None:
        super().__init__()
        projection_type = projection_type.lower()
        self.projection_type = projection_type
        identity = torch.eye(embedding_dim)

        if projection_type in {"none", "identity"}:
            self.register_buffer("projection_matrix", identity)
        elif projection_type in {"fixed", "fixed_orthogonal"}:
            random_matrix = torch.randn(embedding_dim, embedding_dim)
            q_matrix, _ = torch.linalg.qr(random_matrix)
            self.register_buffer("projection_matrix", q_matrix)
        elif projection_type == "learnable":
            parameter = torch.empty(embedding_dim, embedding_dim)
            nn.init.orthogonal_(parameter)
            self.projection_matrix = nn.Parameter(parameter)
        else:
            raise ValueError(f"Unsupported projection type: {projection_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project adapter features into the orthogonal routing space."""
        return inputs @ self.projection_matrix.T

    def orthogonality_penalty(self) -> torch.Tensor:
        """Measure how far `P P^T` is from the identity matrix."""
        if self.projection_type in {"none", "identity"}:
            return torch.zeros(1, device=self.projection_matrix.device).squeeze()

        matrix = self.projection_matrix
        identity = torch.eye(matrix.shape[0], device=matrix.device)
        gram = matrix @ matrix.T
        return ((gram - identity) ** 2).mean()
