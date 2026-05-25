"""Classifier-side modules for the OHR nucleus."""

from __future__ import annotations

import torch
from torch import nn


class LeafExpert(nn.Module):
    """Emit multiclass logits for one specialized classifier expert."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        expert_type: str = "linear",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []
        expert_type = expert_type.lower()

        if expert_type == "linear":
            self.net = nn.Linear(input_dim, num_classes)
        elif expert_type == "mlp":
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
            layers.append(nn.Linear(previous_dim, num_classes))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported expert type: {expert_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate logits from one specialized classifier expert."""
        return self.net(inputs)


class ExpertBank(nn.Module):
    """Group the specialized experts so they can cooperate under routing."""

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        num_classes: int,
        expert_type: str = "linear",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                LeafExpert(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    expert_type=expert_type,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate all classifier experts over the same batch representation."""
        return torch.stack([expert(inputs) for expert in self.experts], dim=1)


class CooperativeOutputAggregator(nn.Module):
    """Combine expert outputs according to their routing probability mass."""

    def __init__(
        self,
        aggregation_type: str = "weighted_logits",
        inference_top_k: int | None = None,
        renormalize_after_top_k: bool = True,
    ) -> None:
        super().__init__()
        aggregation_type = aggregation_type.lower()
        if aggregation_type != "weighted_logits":
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        self.aggregation_type = aggregation_type
        self.inference_top_k = None if inference_top_k is None else int(inference_top_k)
        self.renormalize_after_top_k = bool(renormalize_after_top_k)

    def _apply_inference_top_k(self, leaf_probabilities: torch.Tensor) -> torch.Tensor:
        """Optionally prune routing mass to the top-k experts during inference only."""
        if self.training or self.inference_top_k is None:
            return leaf_probabilities

        safe_top_k = max(1, min(self.inference_top_k, int(leaf_probabilities.shape[1])))
        top_values, top_indices = torch.topk(leaf_probabilities, k=safe_top_k, dim=1)
        pruned = torch.zeros_like(leaf_probabilities)
        pruned.scatter_(1, top_indices, top_values)
        if self.renormalize_after_top_k:
            pruned = pruned / pruned.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return pruned

    def forward(
        self,
        leaf_probabilities: torch.Tensor,
        expert_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Aggregate experts cooperatively as `sum_L P(L|x) * Expert_L(.)`."""
        effective_leaf_probabilities = self._apply_inference_top_k(leaf_probabilities)
        weighted_expert_logits = effective_leaf_probabilities.unsqueeze(-1) * expert_logits
        classifier_logits = weighted_expert_logits.sum(dim=1)

        contribution_mass = weighted_expert_logits.abs().sum(dim=-1)
        contribution_total = contribution_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        relative_contributions = contribution_mass / contribution_total

        return {
            "logits": classifier_logits,
            "classifier_logits": classifier_logits,
            "effective_leaf_probabilities": effective_leaf_probabilities,
            "weighted_expert_logits": weighted_expert_logits,
            "relative_expert_contributions": relative_contributions,
        }
