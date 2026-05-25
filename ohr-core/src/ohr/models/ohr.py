"""Top-level OHR model composed from modular hive blocks."""

from __future__ import annotations

from typing import Any

from torch import nn

from ohr.models.adapter import FusionModule, TabularAdapter
from ohr.models.classifier import CooperativeOutputAggregator, ExpertBank
from ohr.models.projection import OrthogonalProjection
from ohr.models.routing import ProbabilisticTreeRouter


class OHRModel(nn.Module):
    """Orthogonal Honeycomb Routing over already structured numeric tabular inputs.

    This module is the numeric OHR nucleus. It does not perform raw-data
    tabularization or YAML handling. Its job starts once inputs have already
    been converted into a stable numeric table.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 256,
        adapter: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
        expert: dict[str, Any] | None = None,
        aggregator: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        adapter = adapter or {}
        projection = projection or {}
        routing = routing or {}
        expert = expert or {}
        aggregator = aggregator or {}

        self.depth = int(routing.get("depth", 2))
        self.routing_mode = str(routing.get("mode", "soft")).lower()
        self.temperature = float(routing.get("temperature", 1.0))
        self.num_internal_nodes = (2**self.depth) - 1
        self.num_leaves = 2**self.depth
        self.projection_apply_to = str(projection.get("apply_to", "fused")).lower()
        self.use_projected_features_for_experts = bool(
            expert.get("use_projected_features", True)
        )

        self.adapter = TabularAdapter(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            adapter_type=str(adapter.get("type", "linear")),
            hidden_dims=list(adapter.get("hidden_dims", [])),
            dropout=float(adapter.get("dropout", 0.0)),
        )
        self.fusion = FusionModule()
        self.projection = OrthogonalProjection(
            embedding_dim=embedding_dim,
            projection_type=str(projection.get("type", "learnable")),
        )
        self.router = ProbabilisticTreeRouter(
            embedding_dim=embedding_dim,
            depth=self.depth,
            mode=self.routing_mode,
            temperature=self.temperature,
        )
        self.experts = ExpertBank(
            num_experts=self.num_leaves,
            input_dim=embedding_dim,
            num_classes=num_classes,
            expert_type=str(expert.get("type", "linear")),
            hidden_dims=list(expert.get("hidden_dims", [])),
            dropout=float(expert.get("dropout", 0.0)),
        )
        self.aggregator = CooperativeOutputAggregator(
            aggregation_type=str(aggregator.get("type", "weighted_logits")),
            inference_top_k=aggregator.get("inference_top_k"),
            renormalize_after_top_k=bool(aggregator.get("renormalize_after_top_k", True)),
        )

    def forward(self, embedded_inputs):
        """Run the numeric OHR nucleus after tabularization, hygiene and embedding.

        The internal flow remains explicit:

        `embedded_input -> adapter -> projection -> routing -> classifier`
        """
        adapter_output = self.adapter(embedded_inputs)
        fused_features = self.fusion(adapter_output)

        if self.projection_apply_to == "adapter":
            projected_features = self.projection(adapter_output)
        else:
            projected_features = self.projection(fused_features)

        routing_outputs = self.router(projected_features)
        expert_inputs = projected_features if self.use_projected_features_for_experts else fused_features
        expert_logits = self.experts(expert_inputs)
        classifier_outputs = self.aggregator(
            routing_outputs["leaf_probabilities"],
            expert_logits,
        )

        return {
            "embedded_inputs": embedded_inputs,
            "adapter_output": adapter_output,
            "fused_features": fused_features,
            "projected_features": projected_features,
            "expert_inputs": expert_inputs,
            "expert_logits": expert_logits,
            "projection_penalty": self.projection.orthogonality_penalty(),
            **routing_outputs,
            **classifier_outputs,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_parameter_memory_mb(model: nn.Module) -> float:
    """Approximate parameter memory footprint in megabytes."""
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return float(total_bytes / (1024**2))
