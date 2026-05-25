"""Adapter-side modules for the OHR nucleus."""

from __future__ import annotations

import torch
from torch import nn


class TabularAdapter(nn.Module):
    """Map embedded tabular inputs into the shared latent space used by OHR."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        adapter_type: str = "linear",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []
        adapter_type = adapter_type.lower()

        if adapter_type == "identity":
            self.net = nn.Identity() if input_dim == embedding_dim else nn.Linear(
                input_dim,
                embedding_dim,
            )
        elif adapter_type == "linear":
            self.net = nn.Linear(input_dim, embedding_dim)
        elif adapter_type == "mlp":
            layers: list[nn.Module] = []
            previous_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(previous_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                previous_dim = hidden_dim
            layers.append(nn.Linear(previous_dim, embedding_dim))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce the adapted representation consumed by projection and routing."""
        return self.net(inputs)


class FusionModule(nn.Module):
    """Keep a dedicated fusion stage even when the current setting is single-input."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the single-stream representation unchanged."""
        return inputs
