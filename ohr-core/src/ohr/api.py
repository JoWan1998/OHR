from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ohr.engine import (
    build_loader,
    class_weight_profile,
    collect_model_outputs,
    evaluate_loader,
    safe_stratify_labels,
    train_one_epoch,
)
from ohr.embedding import EmbeddingStage
from ohr.hive import build_sample_inspection, compute_hive_metrics as summarize_hive_metrics
from ohr.metrics import compute_classification_metrics
from ohr.models.ohr import OHRModel, count_parameters, estimate_parameter_memory_mb
from ohr.preprocessing import build_input_preprocessor
from ohr.settings import (
    AdapterConfig,
    AggregatorConfig,
    EmbeddingConfig,
    ExpertConfig,
    OHRConfig,
    PreprocessingConfig,
    ProjectionConfig,
    RoutingConfig,
    TabularizerConfig,
    TrainingConfig,
    coerce_ohr_config,
    load_default_ohr_config,
    load_ohr_config,
)
from ohr.tabularizer import Tabularizer
from ohr.utils.io import load_json, save_json
from ohr.utils.paths import ensure_dir, resolve_path
from ohr.utils.seed import set_global_seed


@dataclass
class OHRHistory:
    """Keras-like training history container."""

    history: dict[str, list[float]]

    def to_frame(self) -> pd.DataFrame:
        """Represent the recorded training history as a pandas DataFrame."""
        return pd.DataFrame(self.history)


def _normalize_history_payload(payload: dict[str, Any]) -> dict[str, list[float]]:
    """Normalize stored history into the canonical dict-of-lists structure."""
    history = payload.get("history", {})
    if isinstance(history, dict):
        return history
    if isinstance(history, list):
        if not history:
            return {}
        keys = history[0].keys()
        return {key: [float(row[key]) for row in history if key in row] for key in keys}
    return {}


class OHRClassifier:
    """Stable façade for configurable OHR experiments on structured tabular data.

    The public pipeline intentionally keeps the architectural flow explicit:

    `raw inputs -> tabularizer -> preprocessing -> embedding -> adapter -> projection -> routing -> classifier`
    """

    def __init__(self, config: OHRConfig | dict[str, Any] | str | Path | None = None) -> None:
        self.config, self.config_source = coerce_ohr_config(config)
        self.config.validate()
        self.model: OHRModel | None = None
        self.tabularizer: Tabularizer | None = self._build_tabularizer()
        self.embedding: EmbeddingStage | None = self._build_embedding()
        self.preprocessor: Any | None = None
        self.label_encoder: LabelEncoder | None = None
        self.feature_names: list[str] | None = None
        self.tabular_feature_names: list[str] | None = None
        self.class_names: list[str] | None = None
        self.history: OHRHistory | None = None
        self.artifact_dir: Path | None = None
        self.metadata: dict[str, Any] = {}
        self._compile_config: dict[str, Any] = self._default_compile_config()

    def _default_compile_config(self) -> dict[str, Any]:
        """Build the compile-time defaults directly from the loaded training config."""
        return {
            "optimizer": "adamw",
            "loss": self.config.training.classification_loss,
            "metrics": [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
            ],
            "lr": self.config.training.lr,
            "weight_decay": self.config.training.weight_decay,
            "class_weighting": self.config.training.class_weighting,
            "orthogonal_regularization_weight": self.config.training.orthogonal_regularization_weight,
            "diversity_regularization_weight": self.config.training.diversity_regularization_weight,
            "load_balance_weight": self.config.training.load_balance_weight,
            "device": self.config.training.device,
            "end_to_end": self.config.training.end_to_end,
            "classification_loss": self.config.training.classification_loss,
            "label_smoothing": self.config.training.label_smoothing,
            "confidence_penalty_weight": self.config.training.confidence_penalty_weight,
            "inference_temperature": self.config.training.inference_temperature,
            "class_weight_cap": self.config.training.class_weight_cap,
            "rare_class_boost_factor": self.config.training.rare_class_boost_factor,
            "rare_class_threshold_ratio": self.config.training.rare_class_threshold_ratio,
            "focal_gamma": self.config.training.focal_gamma,
            "gradient_clip_norm": self.config.training.gradient_clip_norm,
        }

    def get_config(self) -> dict[str, Any]:
        """Return the serializable configuration for the classifier."""
        return {
            "ohr_config": self.config.to_dict(),
            "compile_config": dict(self._compile_config),
            "config_source": self.config_source,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OHRClassifier":
        """Instantiate the classifier from a serialized configuration."""
        instance = cls(OHRConfig.from_dict(config.get("ohr_config", {})))
        instance._compile_config.update(config.get("compile_config", {}))
        instance.config_source = str(config.get("config_source", instance.config_source))
        return instance

    def _build_tabularizer(self) -> Tabularizer:
        """Create the explicit tabulation stage used by the public pipeline."""
        return Tabularizer(**asdict(self.config.tabularizer))

    def _build_embedding(self) -> EmbeddingStage:
        """Create the explicit embedding stage inserted before the OHR adapter."""
        return EmbeddingStage(**asdict(self.config.embedding))

    def _validate_runtime_configuration(self) -> None:
        """Validate config after any in-memory user mutation and before runtime use."""
        self.config.validate()

    def _effective_adapter_hidden_dims(self) -> list[int]:
        """Return the adapter hidden dimensions that are actually used at runtime."""
        if self.config.adapter.type == "linear":
            return []
        return [int(dim) for dim in self.config.adapter.hidden_dims]

    def _effective_expert_hidden_dims(self) -> list[int]:
        """Return the expert hidden dimensions that are actually used at runtime."""
        if self.config.expert.type == "linear":
            return []
        return [int(dim) for dim in self.config.expert.hidden_dims]

    def _effective_orthogonal_regularization_weight(self, requested_weight: float) -> float:
        """Disable orthogonal regularization when the projection is an identity bypass."""
        if self.config.projection.type in {"identity", "none"}:
            return 0.0
        return float(requested_weight)

    def _build_resolved_config_snapshot(
        self,
        *,
        scaling_mode: str | None = None,
        seed: int | None = None,
        epochs_configured: int | None = None,
        orthogonal_weight_requested: float | None = None,
        orthogonal_weight_effective: float | None = None,
    ) -> dict[str, Any]:
        """Capture the effective configuration that the runtime is actually using."""
        resolved = copy.deepcopy(self.config.to_dict())
        resolved["seed"] = int(self.config.seed if seed is None else seed)
        resolved["preprocessing"]["scaling"] = (
            str(scaling_mode).lower()
            if scaling_mode is not None
            else str(resolved["preprocessing"]["scaling"]).lower()
        )
        resolved["adapter"]["hidden_dims"] = self._effective_adapter_hidden_dims()
        resolved["expert"]["hidden_dims"] = self._effective_expert_hidden_dims()
        resolved["training"]["lr"] = float(self._compile_config["lr"])
        resolved["training"]["weight_decay"] = float(self._compile_config["weight_decay"])
        resolved["training"]["class_weighting"] = str(self._compile_config["class_weighting"]).lower()
        resolved["training"]["device"] = str(self._compile_config["device"])
        resolved["training"]["epochs"] = int(
            self.config.training.epochs if epochs_configured is None else epochs_configured
        )
        resolved["training"]["classification_loss"] = str(
            self._compile_config["classification_loss"]
        ).lower()
        resolved["training"]["label_smoothing"] = float(self._compile_config["label_smoothing"])
        resolved["training"]["confidence_penalty_weight"] = float(
            self._compile_config["confidence_penalty_weight"]
        )
        resolved["training"]["inference_temperature"] = float(
            self._compile_config["inference_temperature"]
        )
        resolved["training"]["class_weight_cap"] = self._compile_config["class_weight_cap"]
        resolved["training"]["rare_class_boost_factor"] = float(
            self._compile_config["rare_class_boost_factor"]
        )
        resolved["training"]["rare_class_threshold_ratio"] = float(
            self._compile_config["rare_class_threshold_ratio"]
        )
        resolved["training"]["focal_gamma"] = float(self._compile_config["focal_gamma"])
        resolved["training"]["gradient_clip_norm"] = self._compile_config["gradient_clip_norm"]
        resolved["training"]["selection_metric"] = str(self.config.training.selection_metric).lower()
        requested = float(
            self._compile_config["orthogonal_regularization_weight"]
            if orthogonal_weight_requested is None
            else orthogonal_weight_requested
        )
        effective = self._effective_orthogonal_regularization_weight(requested)
        if orthogonal_weight_effective is not None:
            effective = float(orthogonal_weight_effective)
        resolved["training"]["orthogonal_regularization_weight_requested"] = requested
        resolved["training"]["orthogonal_regularization_weight"] = effective
        resolved["training"]["diversity_regularization_weight"] = float(
            self._compile_config["diversity_regularization_weight"]
        )
        resolved["training"]["load_balance_weight"] = float(
            self._compile_config["load_balance_weight"]
        )
        if self.embedding is not None and self.embedding.output_dim_ is not None:
            resolved["embedding"]["effective_output_dim"] = int(self.embedding.output_dim_)
            resolved["embedding"]["runtime_mode"] = str(self.embedding.mode_ or self.config.embedding.mode)
            resolved["embedding"]["pca_fitted"] = bool(self.embedding.pca_model_ is not None)
        resolved["routing"]["temperature_schedule"] = str(self.config.routing.temperature_schedule)
        resolved["routing"]["temperature_end"] = self.config.routing.temperature_end
        resolved["routing"]["temperature_schedule_epochs"] = self.config.routing.temperature_schedule_epochs
        resolved["training"]["regularization_schedule"] = str(
            self.config.training.regularization_schedule
        )
        resolved["training"]["regularization_warmup_epochs"] = int(
            self.config.training.regularization_warmup_epochs
        )
        return resolved

    def get_resolved_config(self) -> dict[str, Any]:
        """Return the effective runtime config, including compile-time overrides."""
        return self._build_resolved_config_snapshot(
            scaling_mode=self.metadata.get("scaling_mode", self.config.preprocessing.scaling),
            seed=int(self.metadata.get("seed", self.config.seed)),
            epochs_configured=self.metadata.get("epochs_configured"),
            orthogonal_weight_requested=self.metadata.get(
                "orthogonal_regularization_weight_requested",
                self._compile_config["orthogonal_regularization_weight"],
            ),
            orthogonal_weight_effective=self.metadata.get(
                "orthogonal_regularization_weight_effective",
            ),
        )

    def get_run_metadata(self) -> dict[str, Any]:
        """Return structured run metadata that external scripts can persist or inspect."""
        if self.metadata:
            metadata = copy.deepcopy(self.metadata)
        else:
            metadata = {
                "config_source": self.config_source,
                "seed": int(self.config.seed),
                "embedding_mode": self.config.embedding.mode,
                "tree_depth": int(self.config.routing.depth),
                "expert_type": self.config.expert.type,
                "projection_type": self.config.projection.type,
                "scaling_mode": self.config.preprocessing.scaling,
            }
        metadata.setdefault("resolved_config", self.get_resolved_config())
        metadata.setdefault("compile_config", dict(self._compile_config))
        return metadata

    def _resolve_regularization_factor(self, epoch: int) -> float:
        """Resolve a simple regularization schedule shared by OHR penalties."""
        schedule = str(self.config.training.regularization_schedule).lower()
        warmup_epochs = int(self.config.training.regularization_warmup_epochs)
        if schedule == "constant" or warmup_epochs <= 0:
            return 1.0
        return min(float(epoch) / float(max(warmup_epochs, 1)), 1.0)

    def _resolve_routing_temperature(self, epoch: int, fit_epochs: int) -> float:
        """Resolve the router temperature schedule for the current epoch."""
        start = float(self.config.routing.temperature)
        schedule = str(self.config.routing.temperature_schedule).lower()
        end = self.config.routing.temperature_end
        if schedule == "constant" or end is None:
            return start
        total_epochs = int(
            self.config.routing.temperature_schedule_epochs
            if self.config.routing.temperature_schedule_epochs is not None
            else fit_epochs
        )
        if total_epochs <= 1:
            return float(end)
        progress = min(max(float(epoch - 1) / float(total_epochs - 1), 0.0), 1.0)
        return float(start + progress * (float(end) - start))

    def _resolve_inference_temperature(self) -> float:
        """Return the calibrated probability temperature used at inference time."""
        return float(self._compile_config.get("inference_temperature", 1.0))

    def compile(
        self,
        optimizer: str | None = None,
        loss: str | None = None,
        metrics: list[str] | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        class_weighting: str | None = None,
        orthogonal_regularization: float | None = None,
        orthogonal_regularization_weight: float | None = None,
        diversity_regularization_weight: float | None = None,
        load_balance_weight: float | None = None,
        label_smoothing: float | None = None,
        confidence_penalty_weight: float | None = None,
        inference_temperature: float | None = None,
        class_weight_cap: float | None = None,
        rare_class_boost_factor: float | None = None,
        rare_class_threshold_ratio: float | None = None,
        focal_gamma: float | None = None,
        gradient_clip_norm: float | None = None,
        device: str | None = None,
    ) -> "OHRClassifier":
        """Configure training behavior similarly to Keras `compile()`."""
        self._validate_runtime_configuration()
        current = dict(self._compile_config)
        normalized_optimizer = str(current["optimizer"] if optimizer is None else optimizer).lower()
        normalized_loss = str(current["classification_loss"] if loss is None else loss).lower()
        if normalized_optimizer != "adamw":
            raise ValueError("Currently only optimizer='adamw' is supported")
        if normalized_loss not in {"cross_entropy", "focal_loss"}:
            raise ValueError("Currently only loss='cross_entropy' or loss='focal_loss' are supported")
        normalized_class_weighting = (
            current["class_weighting"] if class_weighting is None else str(class_weighting).lower()
        )
        if normalized_class_weighting not in {"balanced", "balanced_sqrt", "none"}:
            raise ValueError(
                "Currently only class_weighting='balanced', "
                "'balanced_sqrt' or class_weighting='none' are supported"
            )
        self._compile_config = {
            "optimizer": normalized_optimizer,
            "loss": normalized_loss,
            "metrics": metrics or current["metrics"],
            "lr": current["lr"] if lr is None else float(lr),
            "weight_decay": current["weight_decay"] if weight_decay is None else float(weight_decay),
            "class_weighting": normalized_class_weighting,
            "orthogonal_regularization_weight": (
                current["orthogonal_regularization_weight"]
                if orthogonal_regularization_weight is None and orthogonal_regularization is None
                else float(
                    orthogonal_regularization_weight
                    if orthogonal_regularization_weight is not None
                    else orthogonal_regularization
                )
            ),
            "diversity_regularization_weight": (
                current["diversity_regularization_weight"]
                if diversity_regularization_weight is None
                else float(diversity_regularization_weight)
            ),
            "load_balance_weight": (
                current["load_balance_weight"]
                if load_balance_weight is None
                else float(load_balance_weight)
            ),
            "classification_loss": normalized_loss,
            "label_smoothing": (
                current["label_smoothing"] if label_smoothing is None else float(label_smoothing)
            ),
            "confidence_penalty_weight": (
                current["confidence_penalty_weight"]
                if confidence_penalty_weight is None
                else float(confidence_penalty_weight)
            ),
            "inference_temperature": (
                current["inference_temperature"]
                if inference_temperature is None
                else float(inference_temperature)
            ),
            "class_weight_cap": (
                current["class_weight_cap"] if class_weight_cap is None else float(class_weight_cap)
            ),
            "rare_class_boost_factor": (
                current["rare_class_boost_factor"]
                if rare_class_boost_factor is None
                else float(rare_class_boost_factor)
            ),
            "rare_class_threshold_ratio": (
                current["rare_class_threshold_ratio"]
                if rare_class_threshold_ratio is None
                else float(rare_class_threshold_ratio)
            ),
            "focal_gamma": current["focal_gamma"] if focal_gamma is None else float(focal_gamma),
            "gradient_clip_norm": (
                current["gradient_clip_norm"]
                if gradient_clip_norm is None
                else float(gradient_clip_norm)
            ),
            "device": current["device"] if device is None else str(device),
            "end_to_end": True,
        }
        if not 0.0 <= float(self._compile_config["label_smoothing"]) < 1.0:
            raise ValueError("label_smoothing must be in the interval [0, 1).")
        if float(self._compile_config["confidence_penalty_weight"]) < 0.0:
            raise ValueError("confidence_penalty_weight must be non-negative.")
        if float(self._compile_config["inference_temperature"]) <= 0.0:
            raise ValueError("inference_temperature must be greater than zero.")
        if self._compile_config["class_weight_cap"] is not None and float(
            self._compile_config["class_weight_cap"]
        ) <= 0.0:
            raise ValueError("class_weight_cap must be greater than zero.")
        if float(self._compile_config["rare_class_boost_factor"]) < 1.0:
            raise ValueError("rare_class_boost_factor must be greater than or equal to one.")
        if float(self._compile_config["rare_class_threshold_ratio"]) <= 0.0:
            raise ValueError("rare_class_threshold_ratio must be greater than zero.")
        if float(self._compile_config["focal_gamma"]) < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if self._compile_config["gradient_clip_norm"] is not None and float(
            self._compile_config["gradient_clip_norm"]
        ) <= 0.0:
            raise ValueError("gradient_clip_norm must be greater than zero.")
        return self

    def build(
        self,
        input_dim: int,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> "OHRClassifier":
        """Explicitly build the underlying numeric OHR model."""
        self._validate_runtime_configuration()
        adapter_config = asdict(self.config.adapter)
        expert_config = asdict(self.config.expert)
        adapter_config["hidden_dims"] = self._effective_adapter_hidden_dims()
        expert_config["hidden_dims"] = self._effective_expert_hidden_dims()
        self.model = OHRModel(
            input_dim=input_dim,
            num_classes=num_classes,
            embedding_dim=self.config.embedding_dim,
            adapter=adapter_config,
            projection=asdict(self.config.projection),
            routing=asdict(self.config.routing),
            expert=expert_config,
            aggregator=asdict(self.config.aggregator),
        )
        self.feature_names = self.feature_names or [f"feature_{idx}" for idx in range(input_dim)]
        self.class_names = class_names or self.class_names or [str(idx) for idx in range(num_classes)]
        return self

    def _tabularize_training_inputs(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Apply the explicit `D_raw -> T(D_raw)` stage before model adaptation."""
        self.tabularizer = self._build_tabularizer()
        return self.tabularizer.fit_transform(X)

    def _tabularize_inference_inputs(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Apply the fitted tabularizer while preserving the learned schema."""
        if self.tabularizer is None:
            raise RuntimeError("The OHR tabularizer has not been fitted")
        return self.tabularizer.transform(X)

    def _resolve_scaling_mode(self, scale_features: bool | str | None) -> str:
        """Resolve per-call scaling overrides against configuration defaults."""
        if scale_features is None:
            resolved = str(self.config.preprocessing.scaling).lower()
        elif isinstance(scale_features, str):
            resolved = str(scale_features).lower()
        else:
            resolved = "standard" if bool(scale_features) else "none"
        if resolved not in {"none", "standard", "robust"}:
            raise ValueError(
                f"Unsupported preprocessing scaling override='{scale_features}'. "
                "Supported values are 'none', 'standard' and 'robust'."
            )
        return resolved

    def _count_samples(self, X: np.ndarray | pd.DataFrame) -> int:
        """Return the batch size for supported raw input containers."""
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            return int(len(X))
        raise TypeError(
            "OHRClassifier only supports pandas.DataFrame or numpy.ndarray feature inputs."
        )

    def _validate_feature_input(self, X: np.ndarray | pd.DataFrame, stage: str) -> None:
        """Reject empty or unsupported feature containers before tabularization."""
        count = self._count_samples(X)
        if count == 0:
            raise ValueError(f"OHR {stage} requires at least one input sample.")

    def _validate_target_vector(
        self,
        y: np.ndarray | pd.Series | list[Any],
        expected_length: int,
        stage: str,
    ) -> np.ndarray:
        """Validate labeled targets against the raw sample count."""
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError(f"OHR {stage} expects a one-dimensional target vector.")
        if y_array.size == 0:
            raise ValueError(f"OHR {stage} requires at least one target value.")
        if int(y_array.shape[0]) != int(expected_length):
            raise ValueError(
                f"OHR {stage} received {expected_length} samples but {y_array.shape[0]} targets."
            )
        return y_array

    def _ensure_fitted(self) -> None:
        """Fail fast when inference is requested before training or loading."""
        if (
            self.model is None
            or self.tabularizer is None
            or self.embedding is None
            or self.preprocessor is None
            or self.label_encoder is None
        ):
            raise RuntimeError("The OHR model has not been trained or loaded")

    def _transform_inputs(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
    ) -> np.ndarray:
        """Apply tabularization, hygiene and embedding before inference."""
        self._ensure_fitted()
        self._validate_feature_input(X, stage="inference")
        values = self._tabularize_inference_inputs(X)
        if preprocessed:
            preprocessed_values = values.astype(np.float32)
        else:
            preprocessed_values = self.preprocessor.transform(values).astype(np.float32)
        return self.embedding.transform(preprocessed_values).astype(np.float32)

    def _predict_outputs(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        preprocessed: bool = False,
        batch_size: int = 1024,
        keys: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Collect logits and optional internals in a single inference pass."""
        self._ensure_fitted()
        features = self._transform_inputs(X, preprocessed=preprocessed)
        device = torch.device(str(self._compile_config["device"]))

        requested_keys = list(keys or [])
        requested_keys.insert(0, "logits")
        if "leaf_probabilities" in requested_keys:
            requested_keys.append("effective_leaf_probabilities")
        materialized_keys = list(dict.fromkeys(requested_keys))
        outputs = collect_model_outputs(self.model, features, batch_size, materialized_keys, device)

        logits = outputs["logits"]
        inference_temperature = self._resolve_inference_temperature()
        scaled_logits = torch.from_numpy(logits.astype(np.float64)) / inference_temperature
        probabilities = torch.softmax(scaled_logits, dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)

        payload: dict[str, np.ndarray] = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }
        if self.label_encoder is not None:
            payload["predicted_labels"] = self.label_encoder.inverse_transform(predictions)

        for key, value in outputs.items():
            if key == "effective_leaf_probabilities":
                continue
            payload[key] = value

        if "leaf_probabilities" in outputs and "effective_leaf_probabilities" in outputs:
            payload["leaf_probabilities_raw"] = outputs["leaf_probabilities"]
            payload["leaf_probabilities"] = outputs["effective_leaf_probabilities"]
        return payload

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | list[Any],
        validation_data: tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series | list[Any]] | None = None,
        validation_split: float = 0.15,
        epochs: int | None = None,
        batch_size: int | None = None,
        preprocessed: bool = False,
        scale_features: bool | str | None = None,
        random_state: int | None = None,
    ) -> OHRHistory:
        """Train OHR from raw in-memory inputs.

        The conceptual flow remains explicit:

        `tabularizer -> preprocessing -> embedding -> adapter -> projection -> routing -> classifier`
        """
        self._validate_runtime_configuration()
        run_started_at_utc = datetime.now(timezone.utc).isoformat()
        seed = self.config.seed if random_state is None else int(random_state)
        set_global_seed(seed)

        self._validate_feature_input(X, stage="fit")
        raw_sample_count = self._count_samples(X)
        y_raw = self._validate_target_vector(y, expected_length=raw_sample_count, stage="fit")

        X_train_raw, feature_names = self._tabularize_training_inputs(X)
        if validation_data is None and not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1")
        if validation_data is not None:
            self._validate_feature_input(validation_data[0], stage="validation")
            self._validate_target_vector(
                validation_data[1],
                expected_length=self._count_samples(validation_data[0]),
                stage="validation",
            )

        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_raw)
        self.class_names = self.label_encoder.classes_.tolist()
        self.tabular_feature_names = feature_names

        resolved_scaling = self._resolve_scaling_mode(scale_features)
        preprocessor = build_input_preprocessor(
            scaling=resolved_scaling,
            preprocessed=preprocessed,
            handle_missing=self.config.preprocessing.handle_missing,
        )

        if validation_data is None:
            stratify = safe_stratify_labels(y_train_encoded)
            X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
                X_train_raw,
                y_train_encoded,
                test_size=validation_split,
                random_state=seed,
                stratify=stratify,
            )
        else:
            X_fit_raw = X_train_raw
            y_fit = y_train_encoded
            X_val_raw = self._tabularize_inference_inputs(validation_data[0])
            y_val = self.label_encoder.transform(np.asarray(validation_data[1]))

        self.preprocessor = preprocessor.fit(X_fit_raw)
        X_fit_preprocessed = self.preprocessor.transform(X_fit_raw).astype(np.float32)
        X_val_preprocessed = self.preprocessor.transform(X_val_raw).astype(np.float32)

        self.embedding = self._build_embedding()
        X_fit, embedding_feature_names = self.embedding.fit_transform(X_fit_preprocessed)
        X_val = self.embedding.transform(X_val_preprocessed).astype(np.float32)
        if self.config.embedding.mode == "pca_based" and self.embedding.pca_model_ is None:
            raise RuntimeError(
                "embedding.mode='pca_based' was requested but the PCA transformer was not fitted."
            )
        self.feature_names = embedding_feature_names

        self.build(
            input_dim=X_fit.shape[1],
            num_classes=len(self.class_names),
            class_names=self.class_names,
        )
        assert self.model is not None

        device = torch.device(str(self._compile_config["device"]))
        self.model.to(device)

        fit_batch_size = int(batch_size or self.config.training.batch_size)
        fit_epochs = int(epochs or self.config.training.epochs)
        patience = int(self.config.training.early_stopping_patience)
        lr = float(self._compile_config["lr"])
        weight_decay = float(self._compile_config["weight_decay"])
        requested_orthogonal_weight = float(self._compile_config["orthogonal_regularization_weight"])
        base_orthogonal_weight = self._effective_orthogonal_regularization_weight(
            requested_orthogonal_weight
        )
        base_diversity_weight = float(self._compile_config["diversity_regularization_weight"])
        base_load_balance_weight = float(self._compile_config["load_balance_weight"])
        class_weighting = str(self._compile_config["class_weighting"]).lower()
        class_weight_cap = self._compile_config["class_weight_cap"]
        rare_class_boost_factor = float(self._compile_config["rare_class_boost_factor"])
        rare_class_threshold_ratio = float(self._compile_config["rare_class_threshold_ratio"])
        class_weight_details = class_weight_profile(
            y_fit,
            len(self.class_names),
            strategy=class_weighting,
            cap=class_weight_cap,
            rare_class_boost_factor=rare_class_boost_factor,
            rare_class_threshold_ratio=rare_class_threshold_ratio,
        )
        class_weight_tensor = (
            None
            if class_weight_details["weights"] is None
            else torch.tensor(class_weight_details["weights"], dtype=torch.float32)
        )
        if class_weight_tensor is not None:
            class_weight_tensor = class_weight_tensor.to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        train_loader = build_loader(X_fit, y_fit, fit_batch_size, shuffle=True, seed=seed)
        val_loader = build_loader(X_val, y_val, fit_batch_size, shuffle=False, seed=seed + 1)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_f1 = float("-inf")
        best_selection_score = float("-inf")
        selection_metric = str(self.config.training.selection_metric).lower()
        wait = 0
        history_dict = {
            "epoch": [],
            "train_loss": [],
            "train_classification_loss": [],
            "train_projection_penalty": [],
            "train_diversity_loss": [],
            "train_load_balance_loss": [],
            "train_confidence_penalty": [],
            "val_accuracy": [],
            "val_precision_macro": [],
            "val_recall_macro": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
            "val_routing_entropy": [],
            "val_load_balance_score": [],
            "val_effective_experts": [],
            "val_mean_top_expert_probability": [],
            "val_projection_penalty": [],
            "val_diversity_loss": [],
            "val_load_balance_loss": [],
            "val_confidence_penalty": [],
            "routing_temperature": [],
            "orthogonal_regularization_weight_effective": [],
            "diversity_regularization_weight_effective": [],
            "load_balance_weight_effective": [],
        }
        best_epoch = 0

        for epoch in range(1, fit_epochs + 1):
            if hasattr(self.model, "router") and hasattr(self.model.router, "set_temperature"):
                self.model.router.set_temperature(self._resolve_routing_temperature(epoch, fit_epochs))
            regularization_factor = self._resolve_regularization_factor(epoch)
            orthogonal_weight = base_orthogonal_weight * regularization_factor
            diversity_weight = base_diversity_weight * regularization_factor
            load_balance_weight = base_load_balance_weight * regularization_factor
            train_summary = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                orthogonal_weight=orthogonal_weight,
                diversity_weight=diversity_weight,
                load_balance_weight=load_balance_weight,
                class_weight_tensor=class_weight_tensor,
                loss_name=str(self._compile_config["classification_loss"]).lower(),
                label_smoothing=float(self._compile_config["label_smoothing"]),
                focal_gamma=float(self._compile_config["focal_gamma"]),
                confidence_penalty_weight=float(self._compile_config["confidence_penalty_weight"]),
                gradient_clip_norm=self._compile_config["gradient_clip_norm"],
            )
            val_summary = evaluate_loader(
                model=self.model,
                loader=val_loader,
                device=device,
                orthogonal_weight=orthogonal_weight,
                diversity_weight=diversity_weight,
                load_balance_weight=load_balance_weight,
                class_weight_tensor=class_weight_tensor,
                loss_name=str(self._compile_config["classification_loss"]).lower(),
                label_smoothing=float(self._compile_config["label_smoothing"]),
                focal_gamma=float(self._compile_config["focal_gamma"]),
                confidence_penalty_weight=float(self._compile_config["confidence_penalty_weight"]),
            )

            history_dict["epoch"].append(float(epoch))
            history_dict["train_loss"].append(train_summary["total_loss"])
            history_dict["train_classification_loss"].append(train_summary["classification_loss"])
            history_dict["train_projection_penalty"].append(train_summary["projection_penalty"])
            history_dict["train_diversity_loss"].append(train_summary["diversity_loss"])
            history_dict["train_load_balance_loss"].append(train_summary["load_balance_loss"])
            history_dict["train_confidence_penalty"].append(train_summary["confidence_penalty"])
            history_dict["val_accuracy"].append(val_summary["classification"]["accuracy"])
            history_dict["val_precision_macro"].append(
                val_summary["classification"]["precision_macro"]
            )
            history_dict["val_recall_macro"].append(val_summary["classification"]["recall_macro"])
            history_dict["val_f1_macro"].append(val_summary["classification"]["f1_macro"])
            history_dict["val_f1_weighted"].append(val_summary["classification"]["f1_weighted"])
            history_dict["val_routing_entropy"].append(val_summary["hive"]["routing_entropy"])
            history_dict["val_load_balance_score"].append(val_summary["hive"]["load_balance_score"])
            history_dict["val_effective_experts"].append(val_summary["hive"]["effective_experts"])
            history_dict["val_mean_top_expert_probability"].append(
                val_summary["hive"]["mean_top_expert_probability"]
            )
            history_dict["val_projection_penalty"].append(val_summary["losses"]["projection_penalty"])
            history_dict["val_diversity_loss"].append(val_summary["losses"]["diversity_loss"])
            history_dict["val_load_balance_loss"].append(val_summary["losses"]["load_balance_loss"])
            history_dict["val_confidence_penalty"].append(val_summary["losses"]["confidence_penalty"])
            history_dict["routing_temperature"].append(float(self.model.router.temperature))
            history_dict["orthogonal_regularization_weight_effective"].append(orthogonal_weight)
            history_dict["diversity_regularization_weight_effective"].append(diversity_weight)
            history_dict["load_balance_weight_effective"].append(load_balance_weight)

            best_val_f1 = max(best_val_f1, float(val_summary["classification"]["f1_macro"]))
            current_selection_score = float(val_summary["classification"][selection_metric])
            if current_selection_score > best_selection_score:
                best_selection_score = current_selection_score
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        final_validation = evaluate_loader(
            model=self.model,
            loader=val_loader,
            device=device,
            orthogonal_weight=base_orthogonal_weight,
            diversity_weight=base_diversity_weight,
            load_balance_weight=base_load_balance_weight,
            class_weight_tensor=class_weight_tensor,
            loss_name=str(self._compile_config["classification_loss"]).lower(),
            label_smoothing=float(self._compile_config["label_smoothing"]),
            focal_gamma=float(self._compile_config["focal_gamma"]),
            confidence_penalty_weight=float(self._compile_config["confidence_penalty_weight"]),
        )

        self.history = OHRHistory(history=history_dict)
        run_finished_at_utc = datetime.now(timezone.utc).isoformat()
        self.metadata = {
            "config_source": self.config_source,
            "run_started_at_utc": run_started_at_utc,
            "run_finished_at_utc": run_finished_at_utc,
            "tabularizer": asdict(self.config.tabularizer),
            "preprocessing": asdict(self.config.preprocessing),
            "embedding": asdict(self.config.embedding),
            "tabularizer_input_kind": getattr(self.tabularizer, "input_kind_", None),
            "tabular_feature_names": self.tabular_feature_names,
            "embedding_dim": self.config.embedding_dim,
            "embedding_mode": self.config.embedding.mode,
            "embedding_runtime_mode": getattr(self.embedding, "mode_", None),
            "embedding_output_dim": getattr(self.embedding, "output_dim_", None),
            "effective_embedding_dim": getattr(self.embedding, "output_dim_", None),
            "pca_fitted": bool(getattr(self.embedding, "pca_model_", None) is not None),
            "routing_mode": self.config.routing.mode,
            "tree_depth": self.config.routing.depth,
            "expert_type": self.config.expert.type,
            "expert_hidden_dims_requested": [int(dim) for dim in self.config.expert.hidden_dims],
            "expert_hidden_dims_effective": self._effective_expert_hidden_dims(),
            "adapter_type": self.config.adapter.type,
            "adapter_hidden_dims_requested": [int(dim) for dim in self.config.adapter.hidden_dims],
            "adapter_hidden_dims_effective": self._effective_adapter_hidden_dims(),
            "scaling": resolved_scaling,
            "scaling_mode": resolved_scaling,
            "projection_type": self.config.projection.type,
            "projection_apply_to": self.config.projection.apply_to,
            "projection_penalty_active": self.config.projection.type not in {"identity", "none"},
            "use_projected_features_for_experts": self.config.expert.use_projected_features,
            "end_to_end": True,
            "n_parameters": count_parameters(self.model),
            "parameter_memory_mb": estimate_parameter_memory_mb(self.model),
            "compile_config": dict(self._compile_config),
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "scale_features": resolved_scaling != "none",
            "handle_missing": self.config.preprocessing.handle_missing,
            "orthogonal_regularization_weight_requested": requested_orthogonal_weight,
            "orthogonal_regularization_weight": base_orthogonal_weight,
            "orthogonal_regularization_weight_effective": history_dict[
                "orthogonal_regularization_weight_effective"
            ][-1],
            "diversity_regularization_weight": base_diversity_weight,
            "diversity_regularization_weight_effective": history_dict[
                "diversity_regularization_weight_effective"
            ][-1],
            "load_balance_weight": base_load_balance_weight,
            "load_balance_weight_effective": history_dict["load_balance_weight_effective"][-1],
            "epochs_configured": fit_epochs,
            "epochs_trained": len(history_dict["epoch"]),
            "early_stopping_patience": patience,
            "best_epoch": int(best_epoch or len(history_dict["epoch"])),
            "selection_metric": selection_metric,
            "best_selection_metric_value": float(best_selection_score),
            "best_val_f1_macro": float(best_val_f1),
            "early_stopped": bool(len(history_dict["epoch"]) < fit_epochs),
            "stop_reason": "early_stopping" if len(history_dict["epoch"]) < fit_epochs else "max_epochs",
            "seed": seed,
            "classification_loss": str(self._compile_config["classification_loss"]).lower(),
            "label_smoothing": float(self._compile_config["label_smoothing"]),
            "confidence_penalty_weight": float(self._compile_config["confidence_penalty_weight"]),
            "inference_temperature": float(self._compile_config["inference_temperature"]),
            "class_weighting": class_weighting,
            "class_weight_cap": class_weight_cap,
            "rare_class_boost_factor": rare_class_boost_factor,
            "rare_class_threshold_ratio": rare_class_threshold_ratio,
            "rare_class_support_threshold": float(
                class_weight_details["rare_class_support_threshold"]
            ),
            "rare_class_indices": list(class_weight_details["rare_class_indices"]),
            "rare_class_counts": list(class_weight_details["rare_class_counts"]),
            "train_class_counts": list(class_weight_details["class_counts"]),
            "class_weight_vector": (
                None
                if class_weight_details["weights"] is None
                else [float(value) for value in class_weight_details["weights"].tolist()]
            ),
            "focal_gamma": float(self._compile_config["focal_gamma"]),
            "gradient_clip_norm": self._compile_config["gradient_clip_norm"],
            "routing_temperature_initial": float(self.config.routing.temperature),
            "routing_temperature_final": float(self.model.router.temperature),
            "routing_temperature_schedule": str(self.config.routing.temperature_schedule),
            "final_metrics": (
                final_validation["classification"]
                | final_validation["hive"]
                | final_validation["losses"]
            ),
            "validation_metrics": (
                final_validation["classification"]
                | final_validation["hive"]
                | final_validation["losses"]
            ),
        }
        self.metadata["resolved_config"] = self._build_resolved_config_snapshot(
            scaling_mode=resolved_scaling,
            seed=seed,
            epochs_configured=fit_epochs,
            orthogonal_weight_requested=requested_orthogonal_weight,
            orthogonal_weight_effective=orthogonal_weight,
        )
        return self.history

    def summary(self) -> str:
        """Return a compact textual summary similar to Keras `summary()`."""
        if self.model is None:
            return (
                "OHRClassifier(not built)\n"
                f"flow=tabularizer -> preprocessing -> embedding -> adapter -> projection -> routing -> classifier\n"
                f"embedding_dim={self.config.embedding_dim}, "
                f"routing_mode={self.config.routing.mode}, tree_depth={self.config.routing.depth}, "
                f"expert={self.config.expert.type}, projection={self.config.projection.type}"
            )

        expert_source = "projected" if self.config.expert.use_projected_features else "fused"
        return "\n".join(
            [
                "OHRClassifier",
                "flow=tabularizer -> preprocessing -> embedding -> adapter -> projection -> routing -> classifier",
                "formula=T(D_raw) -> E(x) -> f(h) -> P(z) -> R(t) -> C(...)",
                (
                    "input_hygiene="
                    f"handle_missing:{self.config.preprocessing.handle_missing}, "
                    f"scaling:{self.metadata.get('scaling', self.config.preprocessing.scaling)}"
                ),
                f"tabular_input_dim={len(self.tabular_feature_names or [])}",
                f"embedded_input_dim={len(self.feature_names or [])}",
                f"num_classes={len(self.class_names or [])}",
                f"tabularizer_enabled={self.config.tabularizer.enabled}",
                f"tabularizer_input_type={self.config.tabularizer.input_type}",
                f"tabularizer_input_kind={getattr(self.tabularizer, 'input_kind_', None)}",
                f"embedding_mode={self.config.embedding.mode}",
                f"embedding_output_dim={getattr(self.embedding, 'output_dim_', None)}",
                f"scaling={self.metadata.get('scaling', self.config.preprocessing.scaling)}",
                f"embedding_dim={self.config.embedding_dim}",
                f"routing_mode={self.config.routing.mode}",
                f"tree_depth={self.config.routing.depth}",
                f"expert_type={self.config.expert.type}",
                f"classifier_input_source={expert_source}",
                f"projection_type={self.config.projection.type}",
                f"projection_apply_to={self.config.projection.apply_to}",
                f"classification_loss={self._compile_config['classification_loss']}",
                f"class_weighting={self._compile_config['class_weighting']}",
                f"rare_class_boost_factor={self._compile_config['rare_class_boost_factor']}",
                f"rare_class_threshold_ratio={self._compile_config['rare_class_threshold_ratio']}",
                f"inference_temperature={self._compile_config['inference_temperature']}",
                f"selection_metric={self.config.training.selection_metric}",
                f"routing_temperature_schedule={self.config.routing.temperature_schedule}",
                f"regularization_schedule={self.config.training.regularization_schedule}",
                f"orthogonal_regularization_weight={self._compile_config['orthogonal_regularization_weight']}",
                f"diversity_regularization_weight={self._compile_config['diversity_regularization_weight']}",
                f"load_balance_weight={self._compile_config['load_balance_weight']}",
                f"inference_top_k={self.config.aggregator.inference_top_k}",
                f"epochs_configured={self.metadata.get('epochs_configured')}",
                f"epochs_trained={self.metadata.get('epochs_trained')}",
                f"best_epoch={self.metadata.get('best_epoch')}",
                f"trainable_parameters={count_parameters(self.model)}",
                f"parameter_memory_mb={estimate_parameter_memory_mb(self.model):.4f}",
            ]
        )

    def predict_logits(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Return raw multiclass logits for the provided samples."""
        return self._predict_outputs(X, preprocessed=preprocessed, batch_size=batch_size)["logits"]

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Return class probabilities for the provided samples."""
        return self._predict_outputs(X, preprocessed=preprocessed, batch_size=batch_size)[
            "probabilities"
        ]

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Return encoded class indices predicted by the fitted classifier."""
        return self._predict_outputs(X, preprocessed=preprocessed, batch_size=batch_size)[
            "predictions"
        ]

    def predict_labels(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Return decoded class labels predicted by the fitted classifier."""
        self._ensure_fitted()
        predictions = self.predict(X, preprocessed=preprocessed, batch_size=batch_size)
        return self.label_encoder.inverse_transform(predictions)

    def compute_hive_metrics(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
    ) -> dict[str, Any]:
        """Compute internal routing and classifier-cooperation metrics for a batch."""
        outputs = self._predict_outputs(
            X,
            preprocessed=preprocessed,
            batch_size=batch_size,
            keys=[
                "leaf_probabilities",
                "left_probabilities",
                "node_reach_probabilities",
                "projection_penalty",
            ],
        )
        return summarize_hive_metrics(
            {
                "leaf_probabilities": outputs["leaf_probabilities"],
                "left_probabilities": outputs["left_probabilities"],
                "node_reach_probabilities": outputs["node_reach_probabilities"],
                "projection_penalty": outputs["projection_penalty"],
            }
        )

    def inspect_samples(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Expose per-sample routing probabilities and expert contributions."""
        outputs = self._predict_outputs(
            X,
            preprocessed=preprocessed,
            batch_size=batch_size,
            keys=[
                "leaf_probabilities",
                "left_probabilities",
                "node_reach_probabilities",
                "gate_logits",
                "relative_expert_contributions",
                "weighted_expert_logits",
                "expert_logits",
            ],
        )
        inspection = build_sample_inspection(outputs, top_k=top_k)
        inspection["top_expert_names"] = np.asarray(
            [[f"expert_{index}" for index in row] for row in inspection["top_expert_indices"]],
            dtype=object,
        )
        return inspection

    def get_routing_diagnostics(
        self,
        X: np.ndarray | pd.DataFrame,
        preprocessed: bool = False,
        batch_size: int = 1024,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Return per-sample routing traces plus aggregated hive diagnostics."""
        outputs = self._predict_outputs(
            X,
            preprocessed=preprocessed,
            batch_size=batch_size,
            keys=[
                "leaf_probabilities",
                "left_probabilities",
                "node_reach_probabilities",
                "gate_logits",
                "relative_expert_contributions",
                "weighted_expert_logits",
                "expert_logits",
                "projection_penalty",
            ],
        )
        inspection = build_sample_inspection(outputs, top_k=top_k)
        dominant_expert_indices = inspection["top_expert_indices"][:, 0]
        dominant_expert_probabilities = inspection["top_expert_probabilities"][:, 0]
        inspection["top_expert_names"] = np.asarray(
            [[f"expert_{index}" for index in row] for row in inspection["top_expert_indices"]],
            dtype=object,
        )
        inspection["dominant_expert_indices"] = dominant_expert_indices
        inspection["dominant_expert_names"] = np.asarray(
            [f"expert_{index}" for index in dominant_expert_indices],
            dtype=object,
        )
        inspection["probabilities"] = outputs["probabilities"]
        inspection["predictions"] = outputs["predictions"]
        inspection["predicted_labels"] = outputs["predicted_labels"]
        inspection["logits"] = outputs["logits"]
        inspection["dominant_expert_probabilities"] = dominant_expert_probabilities
        inspection["effective_depth_per_sample"] = inspection["node_reach_probabilities"].sum(axis=1)
        inspection["routing_metrics"] = summarize_hive_metrics(
            {
                "leaf_probabilities": outputs["leaf_probabilities"],
                "left_probabilities": outputs["left_probabilities"],
                "node_reach_probabilities": outputs["node_reach_probabilities"],
                "projection_penalty": outputs["projection_penalty"],
            }
        )
        return inspection

    def evaluate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | list[Any],
        preprocessed: bool = False,
        batch_size: int = 1024,
        include_internal_metrics: bool = True,
        return_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Compute classification and optional hive-behavior metrics on a labeled set."""
        self._ensure_fitted()
        self._validate_feature_input(X, stage="evaluation")
        y_true = self.label_encoder.transform(
            self._validate_target_vector(y, expected_length=self._count_samples(X), stage="evaluation")
        )
        requested_keys: list[str] = []
        if include_internal_metrics:
            requested_keys.extend(
                [
                    "leaf_probabilities",
                    "left_probabilities",
                    "node_reach_probabilities",
                    "projection_penalty",
                ]
            )
        outputs = self._predict_outputs(
            X,
            preprocessed=preprocessed,
            batch_size=batch_size,
            keys=requested_keys,
        )
        y_pred = outputs["predictions"]
        metrics: dict[str, Any] = compute_classification_metrics(y_true, y_pred)
        hive_metrics: dict[str, Any] = {}
        if include_internal_metrics:
            hive_metrics = summarize_hive_metrics(
                {
                    "leaf_probabilities": outputs["leaf_probabilities"],
                    "left_probabilities": outputs["left_probabilities"],
                    "node_reach_probabilities": outputs["node_reach_probabilities"],
                    "projection_penalty": outputs["projection_penalty"],
                }
            )
            metrics.update(hive_metrics)
        if return_artifacts:
            return {
                "classification": metrics,
                "predictions": outputs["predictions"],
                "predicted_labels": outputs["predicted_labels"],
                "probabilities": outputs["probabilities"],
                "encoded_targets": y_true,
                "targets": np.asarray(y, dtype=object),
                "hive": hive_metrics,
            }
        return metrics

    def save(self, path: str | Path) -> Path:
        """Persist the full OHR classifier state to a self-contained directory."""
        self._ensure_fitted()
        target = ensure_dir(path)
        self.artifact_dir = target

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "ohr_config": self.config.to_dict(),
                "input_dim": len(self.feature_names or []),
                "num_classes": len(self.class_names or []),
                "class_names": self.class_names,
            },
            target / "model.pt",
        )
        joblib.dump(self.tabularizer, target / "tabularizer.joblib")
        joblib.dump(self.embedding, target / "embedding.joblib")
        joblib.dump(self.preprocessor, target / "preprocessor.joblib")
        joblib.dump(self.label_encoder, target / "label_encoder.joblib")
        save_json(self.get_run_metadata(), target / "metadata.json")
        save_json({"history": self.history.history if self.history else {}}, target / "history.json")
        save_json(self.get_config(), target / "classifier_config.json")
        return target

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "OHRClassifier":
        """Load a saved OHR classifier from disk."""
        root = resolve_path(path)
        checkpoint = torch.load(root / "model.pt", map_location=map_location)

        config_path = root / "classifier_config.json"
        if config_path.exists():
            instance = cls.from_config(load_json(config_path))
        else:
            instance = cls(OHRConfig.from_dict(checkpoint.get("ohr_config", {})))

        ohr_config = checkpoint["ohr_config"]
        instance.model = OHRModel(
            input_dim=int(checkpoint["input_dim"]),
            num_classes=int(checkpoint["num_classes"]),
            embedding_dim=int(ohr_config.get("embedding_dim", 256)),
            adapter=ohr_config.get("adapter"),
            projection=ohr_config.get("projection"),
            routing=ohr_config.get("routing"),
            expert=ohr_config.get("expert"),
            aggregator=ohr_config.get("aggregator"),
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()

        tabularizer_path = root / "tabularizer.joblib"
        if tabularizer_path.exists():
            instance.tabularizer = joblib.load(tabularizer_path)
        embedding_path = root / "embedding.joblib"
        if embedding_path.exists():
            instance.embedding = joblib.load(embedding_path)
        instance.preprocessor = joblib.load(root / "preprocessor.joblib")
        instance.label_encoder = joblib.load(root / "label_encoder.joblib")
        instance.class_names = checkpoint.get("class_names")
        if instance.class_names is None and instance.label_encoder is not None:
            instance.class_names = instance.label_encoder.classes_.tolist()

        instance.metadata = load_json(root / "metadata.json") if (root / "metadata.json").exists() else {}
        if instance.metadata:
            instance.metadata.setdefault("scaling_mode", instance.metadata.get("scaling"))
            instance.metadata.setdefault(
                "effective_embedding_dim",
                instance.metadata.get("embedding_output_dim"),
            )
            if hasattr(instance.model, "router") and hasattr(instance.model.router, "set_temperature"):
                instance.model.router.set_temperature(
                    float(
                        instance.metadata.get(
                            "routing_temperature_final",
                            ohr_config.get("routing", {}).get("temperature", 1.0),
                        )
                    )
                )
        instance.feature_names = instance.metadata.get("feature_names")
        instance.tabular_feature_names = instance.metadata.get("tabular_feature_names")
        if instance.tabularizer is None:
            instance.tabularizer = instance._build_tabularizer()
        if instance.embedding is None:
            instance.embedding = instance._build_embedding()
        if instance.tabularizer.feature_names_ is None:
            if instance.tabular_feature_names is not None:
                instance.tabularizer.feature_names_ = list(instance.tabular_feature_names)
            else:
                instance.tabularizer.feature_names_ = [
                    f"feature_{index}" for index in range(int(checkpoint["input_dim"]))
                ]
        if instance.tabularizer.input_kind_ is None:
            instance.tabularizer.input_kind_ = instance.metadata.get("tabularizer_input_kind", "ndarray")
        if instance.embedding.input_dim_ is None:
            embedding_dim = int(checkpoint["input_dim"])
            instance.embedding.input_dim_ = embedding_dim
            instance.embedding.output_dim_ = embedding_dim
            instance.embedding.mode_ = "fixed"
            instance.embedding.projection_matrix_ = np.eye(embedding_dim, dtype=np.float32)
            instance.embedding.feature_names_ = instance.feature_names or [
                f"embedding_feature_{index}" for index in range(embedding_dim)
            ]

        instance.artifact_dir = root
        history_path = root / "history.json"
        if history_path.exists():
            history_payload = load_json(history_path)
            instance.history = OHRHistory(history=_normalize_history_payload(history_payload))
        instance.metadata.setdefault("resolved_config", instance.get_resolved_config())
        return instance


__all__ = [
    "AdapterConfig",
    "AggregatorConfig",
    "EmbeddingConfig",
    "ExpertConfig",
    "OHRClassifier",
    "OHRConfig",
    "OHRHistory",
    "PreprocessingConfig",
    "ProjectionConfig",
    "RoutingConfig",
    "TabularizerConfig",
    "TrainingConfig",
    "load_default_ohr_config",
    "load_ohr_config",
]
