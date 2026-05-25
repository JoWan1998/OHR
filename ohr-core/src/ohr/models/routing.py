"""Routing primitives for probabilistic OHR assignment."""

from __future__ import annotations

import torch
from torch import nn


class ProbabilisticTreeRouter(nn.Module):
    """Binary tree router whose main mode is probability-based distributed assignment."""

    def __init__(
        self,
        embedding_dim: int,
        depth: int = 2,
        mode: str = "soft",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.mode = str(mode).lower()
        self.temperature = float(temperature)
        self.num_internal_nodes = (2**self.depth) - 1
        self.num_leaves = 2**self.depth
        self.router = nn.Linear(embedding_dim, self.num_internal_nodes)

        if self.mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported routing mode: {self.mode}")

    def set_temperature(self, temperature: float) -> None:
        """Update the routing temperature during training schedules."""
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError("Routing temperature must be greater than zero.")
        self.temperature = temperature

    def _routing_values(
        self,
        gate_logits: torch.Tensor,
        left_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Return soft probabilities or straight-through hard decisions."""
        if self.mode == "soft":
            return left_probabilities

        hard_decisions = (gate_logits >= 0.0).float()
        if self.training:
            return hard_decisions.detach() - left_probabilities.detach() + left_probabilities
        return hard_decisions

    def _propagate(
        self,
        left_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate probability mass through the tree and recover node/leaf usage."""
        batch_size = left_values.shape[0]
        total_nodes = (2 ** (self.depth + 1)) - 1
        reach_probabilities: dict[int, torch.Tensor] = {
            0: torch.ones(batch_size, device=left_values.device)
        }
        internal_reach: list[torch.Tensor] = []

        for node_index in range(self.num_internal_nodes):
            parent_reach = reach_probabilities[node_index]
            internal_reach.append(parent_reach)
            go_left = left_values[:, node_index]
            go_right = 1.0 - go_left
            reach_probabilities[(2 * node_index) + 1] = parent_reach * go_left
            reach_probabilities[(2 * node_index) + 2] = parent_reach * go_right

        leaf_start = self.num_internal_nodes
        leaf_probabilities = torch.stack(
            [reach_probabilities[index] for index in range(leaf_start, total_nodes)],
            dim=1,
        )
        node_reach_probabilities = torch.stack(internal_reach, dim=1)
        return leaf_probabilities, node_reach_probabilities

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produce node activations and the probability mass assigned to every leaf."""
        safe_temperature = max(self.temperature, 1e-6)
        gate_logits = self.router(inputs)
        left_probabilities = torch.sigmoid(gate_logits / safe_temperature)
        routing_values = self._routing_values(gate_logits, left_probabilities)
        leaf_probabilities, node_reach_probabilities = self._propagate(routing_values)

        return {
            "gate_logits": gate_logits,
            "left_probabilities": left_probabilities,
            "right_probabilities": 1.0 - left_probabilities,
            "routing_values": routing_values,
            "leaf_probabilities": leaf_probabilities,
            "node_reach_probabilities": node_reach_probabilities,
        }
