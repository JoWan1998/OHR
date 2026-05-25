"""Internal execution helpers for the OHR training and evaluation flow."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ohr.hive import compute_hive_metrics, expert_diversity_loss, load_balance_loss
from ohr.metrics import compute_classification_metrics


def safe_stratify_labels(labels: np.ndarray) -> np.ndarray | None:
    """Return labels for stratification only when every class has enough support."""
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size == 0 or counts.min() < 2:
        return None
    return labels


def build_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """Wrap numpy arrays into a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def class_weights(
    y: np.ndarray,
    num_classes: int,
    strategy: str = "balanced",
    cap: float | None = None,
    rare_class_boost_factor: float = 1.0,
    rare_class_threshold_ratio: float = 0.5,
) -> torch.Tensor | None:
    """Estimate class weights for imbalanced training with simple strategies."""
    profile = class_weight_profile(
        y,
        num_classes,
        strategy=strategy,
        cap=cap,
        rare_class_boost_factor=rare_class_boost_factor,
        rare_class_threshold_ratio=rare_class_threshold_ratio,
    )
    if profile["weights"] is None:
        return None
    return torch.tensor(profile["weights"], dtype=torch.float32)


def class_weight_profile(
    y: np.ndarray,
    num_classes: int,
    strategy: str = "balanced",
    cap: float | None = None,
    rare_class_boost_factor: float = 1.0,
    rare_class_threshold_ratio: float = 0.5,
) -> dict[str, Any]:
    """Describe the effective class weighting profile used during training."""
    normalized_strategy = str(strategy).lower()
    boost_factor = float(rare_class_boost_factor)
    threshold_ratio = float(rare_class_threshold_ratio)
    if boost_factor < 1.0:
        raise ValueError("rare_class_boost_factor must be greater than or equal to one.")
    if threshold_ratio <= 0.0:
        raise ValueError("rare_class_threshold_ratio must be greater than zero.")

    raw_counts = np.bincount(y, minlength=num_classes).astype(np.int64)
    safe_counts = raw_counts.astype(np.float32)
    safe_counts[safe_counts == 0] = 1.0
    if normalized_strategy == "none":
        weights = None if boost_factor <= 1.0 else np.ones(num_classes, dtype=np.float32)
    elif normalized_strategy == "balanced":
        weights = safe_counts.sum() / (num_classes * safe_counts)
    elif normalized_strategy == "balanced_sqrt":
        weights = np.sqrt(safe_counts.sum() / (num_classes * safe_counts))
    else:
        raise ValueError(
            f"Unsupported class weighting strategy='{strategy}'. "
            "Supported values are 'none', 'balanced' and 'balanced_sqrt'."
        )

    positive_counts = raw_counts[raw_counts > 0]
    median_support = float(np.median(positive_counts)) if positive_counts.size else 0.0
    rare_support_threshold = (
        float(max(1.0, median_support * threshold_ratio)) if positive_counts.size else 0.0
    )
    rare_mask = np.zeros(num_classes, dtype=bool)
    if weights is not None and boost_factor > 1.0 and positive_counts.size:
        rare_mask = (raw_counts > 0) & (raw_counts <= rare_support_threshold)
        weights[rare_mask] *= boost_factor

    if weights is not None and cap is not None:
        weights = np.minimum(weights, float(cap))

    return {
        "strategy": normalized_strategy,
        "cap": None if cap is None else float(cap),
        "weights": None if weights is None else weights.astype(np.float32),
        "class_counts": raw_counts.astype(int).tolist(),
        "rare_class_boost_factor": boost_factor,
        "rare_class_threshold_ratio": threshold_ratio,
        "median_class_support": median_support,
        "rare_class_support_threshold": rare_support_threshold,
        "rare_class_indices": np.flatnonzero(rare_mask).astype(int).tolist(),
        "rare_class_counts": raw_counts[rare_mask].astype(int).tolist(),
    }


def _confidence_penalty(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Penalize overconfident predictions by encouraging output entropy."""
    probabilities = torch.softmax(logits, dim=1).clamp_min(eps)
    return (probabilities * torch.log(probabilities)).sum(dim=1).mean()


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weight_tensor: torch.Tensor | None,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    """Compute the configured classification loss."""
    if loss_name == "cross_entropy":
        return F.cross_entropy(
            logits,
            labels,
            weight=class_weight_tensor,
            label_smoothing=float(label_smoothing),
        )

    if loss_name == "focal_loss":
        per_sample_loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weight_tensor,
            label_smoothing=float(label_smoothing),
            reduction="none",
        )
        pt = torch.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        modulation = (1.0 - pt).pow(float(focal_gamma))
        return (modulation * per_sample_loss).mean()

    raise ValueError(
        f"Unsupported classification loss='{loss_name}'. "
        "Supported values are 'cross_entropy' and 'focal_loss'."
    )


def _ensure_finite_outputs(outputs: dict[str, torch.Tensor], context: str) -> None:
    """Fail fast when the model produces NaN or inf values."""
    for key, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not torch.isfinite(value).all():
            raise RuntimeError(f"OHR produced non-finite values in '{key}' during {context}.")


def compute_loss_terms(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    orthogonal_weight: float,
    diversity_weight: float,
    load_balance_weight: float,
    class_weight_tensor: torch.Tensor | None,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
    confidence_penalty_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute the full end-to-end objective and its tracked components."""
    classification_loss = _classification_loss(
        outputs["logits"],
        labels,
        class_weight_tensor=class_weight_tensor,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )
    projection_penalty = outputs["projection_penalty"]
    diversity_penalty = expert_diversity_loss(
        outputs["expert_logits"],
        outputs["leaf_probabilities"],
    )
    load_balance_penalty = load_balance_loss(outputs["leaf_probabilities"])
    confidence_penalty = _confidence_penalty(outputs["logits"])

    total_loss = classification_loss
    total_loss = total_loss + (orthogonal_weight * projection_penalty)
    total_loss = total_loss + (diversity_weight * diversity_penalty)
    total_loss = total_loss + (load_balance_weight * load_balance_penalty)
    total_loss = total_loss + (confidence_penalty_weight * confidence_penalty)

    return {
        "classification_loss": classification_loss,
        "projection_penalty": projection_penalty,
        "diversity_loss": diversity_penalty,
        "load_balance_loss": load_balance_penalty,
        "confidence_penalty": confidence_penalty,
        "total_loss": total_loss,
    }


def _accumulate_loss_terms(
    running: dict[str, float],
    loss_terms: dict[str, torch.Tensor],
    batch_n: int,
) -> None:
    """Accumulate weighted scalar losses over one epoch or evaluation loop."""
    running["classification_loss"] += float(loss_terms["classification_loss"].item()) * batch_n
    running["projection_penalty"] += float(loss_terms["projection_penalty"].item()) * batch_n
    running["diversity_loss"] += float(loss_terms["diversity_loss"].item()) * batch_n
    running["load_balance_loss"] += float(loss_terms["load_balance_loss"].item()) * batch_n
    running["confidence_penalty"] += float(loss_terms["confidence_penalty"].item()) * batch_n
    running["total_loss"] += float(loss_terms["total_loss"].item()) * batch_n
    running["n_samples"] += batch_n


def _finalize_running_losses(running: dict[str, float]) -> dict[str, float]:
    """Convert accumulated weighted losses into mean losses."""
    denominator = max(int(running["n_samples"]), 1)
    return {
        "classification_loss": running["classification_loss"] / denominator,
        "projection_penalty": running["projection_penalty"] / denominator,
        "diversity_loss": running["diversity_loss"] / denominator,
        "load_balance_loss": running["load_balance_loss"] / denominator,
        "confidence_penalty": running["confidence_penalty"] / denominator,
        "total_loss": running["total_loss"] / denominator,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    orthogonal_weight: float,
    diversity_weight: float,
    load_balance_weight: float,
    class_weight_tensor: torch.Tensor | None,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
    confidence_penalty_weight: float,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    """Train OHR for one epoch and return averaged losses."""
    model.train()
    running = {
        "classification_loss": 0.0,
        "projection_penalty": 0.0,
        "diversity_loss": 0.0,
        "load_balance_loss": 0.0,
        "confidence_penalty": 0.0,
        "total_loss": 0.0,
        "n_samples": 0,
    }

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        _ensure_finite_outputs(outputs, context="training")
        loss_terms = compute_loss_terms(
            outputs=outputs,
            labels=labels,
            orthogonal_weight=orthogonal_weight,
            diversity_weight=diversity_weight,
            load_balance_weight=load_balance_weight,
            class_weight_tensor=class_weight_tensor,
            loss_name=loss_name,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            confidence_penalty_weight=confidence_penalty_weight,
        )
        if not torch.isfinite(loss_terms["total_loss"]):
            raise RuntimeError("OHR produced a non-finite total loss during training.")
        loss_terms["total_loss"].backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
        optimizer.step()
        _accumulate_loss_terms(running, loss_terms, int(labels.shape[0]))

    return _finalize_running_losses(running)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    orthogonal_weight: float,
    diversity_weight: float,
    load_balance_weight: float,
    class_weight_tensor: torch.Tensor | None,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
    confidence_penalty_weight: float,
) -> dict[str, Any]:
    """Evaluate one loader and summarize predictive and hive-specific behavior."""
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    leaf_probabilities_parts: list[np.ndarray] = []
    left_probabilities_parts: list[np.ndarray] = []
    node_reach_parts: list[np.ndarray] = []
    projection_penalty_parts: list[np.ndarray] = []
    running = {
        "classification_loss": 0.0,
        "projection_penalty": 0.0,
        "diversity_loss": 0.0,
        "load_balance_loss": 0.0,
        "confidence_penalty": 0.0,
        "total_loss": 0.0,
        "n_samples": 0,
    }

    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _ensure_finite_outputs(outputs, context="evaluation")
            loss_terms = compute_loss_terms(
                outputs=outputs,
                labels=labels,
                orthogonal_weight=orthogonal_weight,
                diversity_weight=diversity_weight,
                load_balance_weight=load_balance_weight,
                class_weight_tensor=class_weight_tensor,
                loss_name=loss_name,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                confidence_penalty_weight=confidence_penalty_weight,
            )
            if not torch.isfinite(loss_terms["total_loss"]):
                raise RuntimeError("OHR produced a non-finite total loss during evaluation.")
            batch_n = int(labels.shape[0])
            _accumulate_loss_terms(running, loss_terms, batch_n)

            predictions = torch.argmax(outputs["logits"], dim=1)
            y_true_parts.append(labels.cpu().numpy())
            y_pred_parts.append(predictions.cpu().numpy())
            leaf_probabilities_parts.append(outputs["leaf_probabilities"].cpu().numpy())
            left_probabilities_parts.append(outputs["left_probabilities"].cpu().numpy())
            node_reach_parts.append(outputs["node_reach_probabilities"].cpu().numpy())
            projection_penalty_parts.append(
                np.full((batch_n,), float(loss_terms["projection_penalty"].item()), dtype=np.float32)
            )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    hive_payload = {
        "leaf_probabilities": np.concatenate(leaf_probabilities_parts, axis=0),
        "left_probabilities": np.concatenate(left_probabilities_parts, axis=0),
        "node_reach_probabilities": np.concatenate(node_reach_parts, axis=0),
        "projection_penalty": np.concatenate(projection_penalty_parts, axis=0),
    }

    return {
        "classification": compute_classification_metrics(y_true, y_pred),
        "hive": compute_hive_metrics(hive_payload),
        "losses": _finalize_running_losses(running),
    }


def collect_model_outputs(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    keys: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run inference and materialize the requested internal tensors batch by batch."""
    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            outputs = model(batch)
            for key in keys:
                value = outputs[key].detach().cpu().numpy()
                if np.asarray(value).ndim == 0:
                    value = np.full((len(batch),), float(value), dtype=np.float32)
                collected[key].append(value)

    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}
