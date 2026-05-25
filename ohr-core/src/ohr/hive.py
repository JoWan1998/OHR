"""Losses, metrics and inspection helpers for OHR as a cooperative hive."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_ENTROPY_EPS = 1e-12


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensors to numpy without modifying already materialized arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stable_binary_entropy_numpy(
    probabilities: np.ndarray,
    eps: float = _ENTROPY_EPS,
) -> np.ndarray:
    """Compute binary entropy in float64 while avoiding exact 0/1 boundaries."""
    safe_probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(safe_probabilities, eps, 1.0 - eps)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def _stable_binary_entropy_torch(
    probabilities: torch.Tensor,
    eps: float = _ENTROPY_EPS,
) -> torch.Tensor:
    """Compute binary entropy in float64 while avoiding exact 0/1 boundaries."""
    safe_probabilities = probabilities.to(dtype=torch.float64)
    clipped = safe_probabilities.clamp(min=eps, max=1.0 - eps)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))


def expert_diversity_loss(
    expert_logits: torch.Tensor,
    leaf_probabilities: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage experts to specialize by decorrelating their class signatures."""
    num_experts = expert_logits.shape[1]
    if num_experts < 2:
        return expert_logits.new_zeros(())

    expert_probabilities = F.softmax(expert_logits, dim=-1)
    usage = leaf_probabilities.sum(dim=0).clamp_min(eps)
    signatures = (leaf_probabilities.unsqueeze(-1) * expert_probabilities).sum(dim=0)
    signatures = signatures / usage.unsqueeze(-1)
    signatures = signatures - signatures.mean(dim=1, keepdim=True)
    signatures = F.normalize(signatures, p=2, dim=1, eps=eps)

    similarity = signatures @ signatures.T
    mask = ~torch.eye(num_experts, dtype=torch.bool, device=similarity.device)
    if not mask.any():
        return expert_logits.new_zeros(())
    return similarity[mask].pow(2).mean()


def load_balance_loss(leaf_probabilities: torch.Tensor) -> torch.Tensor:
    """Push the average expert usage toward a balanced allocation."""
    mean_usage = leaf_probabilities.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return (mean_usage - target).pow(2).mean()


def routing_entropy(
    left_probabilities: torch.Tensor,
    node_reach_probabilities: torch.Tensor,
    eps: float = _ENTROPY_EPS,
) -> torch.Tensor:
    """Measure routing uncertainty while weighting nodes by how often they are reached."""
    binary_entropy = _stable_binary_entropy_torch(left_probabilities, eps=eps)
    weighted_entropy = node_reach_probabilities.to(dtype=torch.float64) * binary_entropy
    return weighted_entropy.sum(dim=1).mean()


def compute_hive_metrics(outputs: dict[str, torch.Tensor | np.ndarray]) -> dict[str, Any]:
    """Summarize expert cooperation, routing spread and load balance for one evaluation run."""
    leaf_probabilities = np.asarray(_to_numpy(outputs["leaf_probabilities"]), dtype=np.float64)
    left_probabilities = np.asarray(_to_numpy(outputs["left_probabilities"]), dtype=np.float64)
    node_reach_probabilities = np.asarray(
        _to_numpy(outputs["node_reach_probabilities"]),
        dtype=np.float64,
    )

    mean_leaf_probability = leaf_probabilities.mean(axis=0)
    dominant_experts = leaf_probabilities.argmax(axis=1)
    expert_usage_frequency = (
        np.bincount(dominant_experts, minlength=leaf_probabilities.shape[1]) / leaf_probabilities.shape[0]
    )

    binary_entropy = _stable_binary_entropy_numpy(left_probabilities)
    weighted_entropy = node_reach_probabilities * binary_entropy

    leaf_distribution = mean_leaf_probability / np.clip(mean_leaf_probability.sum(), _ENTROPY_EPS, None)
    if len(leaf_distribution) <= 1:
        normalized_entropy = 1.0
        effective_experts = 1.0
        load_balance_mse = 0.0
    else:
        normalized_entropy = float(
            -np.sum(leaf_distribution * np.log(np.clip(leaf_distribution, _ENTROPY_EPS, None)))
            / np.log(len(leaf_distribution))
        )
        effective_experts = float(1.0 / np.sum(np.square(leaf_distribution)))
        load_balance_mse = float(
            np.mean(np.square(leaf_distribution - (1.0 / len(leaf_distribution))))
        )

    metrics: dict[str, Any] = {
        "expert_usage_frequency": expert_usage_frequency.tolist(),
        "mean_leaf_probability": mean_leaf_probability.tolist(),
        "routing_entropy": float(weighted_entropy.sum(axis=1).mean()),
        "load_balance_score": normalized_entropy,
        "load_balance_mse": load_balance_mse,
        "effective_experts": effective_experts,
        "mean_effective_depth": float(node_reach_probabilities.sum(axis=1).mean()),
        "mean_top_expert_probability": float(leaf_probabilities.max(axis=1).mean()),
    }

    if "projection_penalty" in outputs:
        projection_penalty = _to_numpy(outputs["projection_penalty"]).reshape(-1)
        metrics["mean_projection_penalty"] = float(projection_penalty.mean())

    return metrics


def build_sample_inspection(
    outputs: dict[str, torch.Tensor | np.ndarray],
    top_k: int = 3,
) -> dict[str, np.ndarray]:
    """Expose per-sample routing and expert contribution traces for interpretation."""
    leaf_probabilities = _to_numpy(outputs["leaf_probabilities"])
    relative_contributions = _to_numpy(outputs["relative_expert_contributions"])

    safe_top_k = max(1, min(int(top_k), leaf_probabilities.shape[1]))
    top_indices = np.argsort(leaf_probabilities, axis=1)[:, ::-1][:, :safe_top_k]
    top_probabilities = np.take_along_axis(leaf_probabilities, top_indices, axis=1)
    top_contributions = np.take_along_axis(relative_contributions, top_indices, axis=1)

    return {
        "leaf_probabilities": leaf_probabilities,
        "top_expert_indices": top_indices,
        "top_expert_probabilities": top_probabilities,
        "top_expert_relative_contributions": top_contributions,
        "relative_expert_contributions": relative_contributions,
        "weighted_expert_logits": _to_numpy(outputs["weighted_expert_logits"]),
        "expert_logits": _to_numpy(outputs["expert_logits"]),
        "gate_logits": _to_numpy(outputs["gate_logits"]),
        "gate_activation_magnitude": np.abs(_to_numpy(outputs["gate_logits"])),
        "left_probabilities": _to_numpy(outputs["left_probabilities"]),
        "node_reach_probabilities": _to_numpy(outputs["node_reach_probabilities"]),
    }
