"""Structured configuration objects and loaders for the OHR public API."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ohr.config import load_config, load_packaged_config


def _normalize_choice(name: str, value: str, allowed: set[str]) -> str:
    """Normalize lowercase enum-like values and fail early on unsupported options."""
    normalized = str(value).lower()
    if normalized not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {name}='{value}'. Supported values are {supported}.")
    return normalized


def _normalize_hidden_dims(name: str, hidden_dims: list[int]) -> list[int]:
    """Normalize hidden dimensions into a validated list of positive integers."""
    normalized = [int(dim) for dim in hidden_dims]
    if any(dim <= 0 for dim in normalized):
        raise ValueError(f"{name} must contain only positive integers.")
    return normalized


def _validate_probability_like(name: str, value: float, allow_one: bool = False) -> float:
    """Validate weights, dropout rates or explained variance thresholds."""
    normalized = float(value)
    upper_bound = 1.0 if allow_one else 1.0
    if allow_one:
        if not 0.0 < normalized <= upper_bound:
            raise ValueError(f"{name} must be in the interval (0, 1].")
    elif not 0.0 <= normalized < upper_bound:
        raise ValueError(f"{name} must be in the interval [0, 1).")
    return normalized


def _validate_non_negative(name: str, value: float) -> float:
    """Validate that scalar hyper-parameters stay non-negative."""
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be greater than or equal to zero.")
    return normalized


def _extract_ohr_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the effective OHR block from packaged, wrapped or flat configs."""
    if "model_defaults" in payload and isinstance(payload["model_defaults"], dict):
        extracted = dict(payload["model_defaults"])
        extracted["seed"] = int(payload.get("seed", extracted.get("seed", 42)))
        return extracted

    if "ohr" in payload and isinstance(payload["ohr"], dict):
        extracted = dict(payload["ohr"])
        extracted["seed"] = int(payload.get("seed", extracted.get("seed", 42)))
        return extracted

    extracted = dict(payload)
    extracted.pop("_config_path", None)
    return extracted


@dataclass
class TabularizerConfig:
    """Configuration for the explicit tabulation stage `x = T(D_raw)`."""

    enabled: bool = True
    input_type: str = "tabular"
    replace_infinite: bool = True
    drop_columns: list[str] = field(default_factory=list)
    keep_columns: list[str] | None = None

    def validate(self) -> "TabularizerConfig":
        """Validate the lightweight tabularizer contract."""
        self.input_type = _normalize_choice("tabularizer.input_type", self.input_type, {"tabular"})
        self.drop_columns = [str(column) for column in self.drop_columns]
        if self.keep_columns is not None:
            self.keep_columns = [str(column) for column in self.keep_columns]
        return self


@dataclass
class PreprocessingConfig:
    """Configuration for lightweight input hygiene after tabularization."""

    handle_missing: str = "median"
    scaling: str = "standard"

    def validate(self) -> "PreprocessingConfig":
        """Validate lightweight preprocessing choices."""
        self.handle_missing = _normalize_choice(
            "preprocessing.handle_missing",
            self.handle_missing,
            {"median", "none"},
        )
        self.scaling = _normalize_choice(
            "preprocessing.scaling",
            self.scaling,
            {"none", "standard", "robust"},
        )
        return self

    @property
    def scale_numeric(self) -> bool:
        """Backward-compatible alias for whether numeric scaling is enabled."""
        return str(self.scaling).lower() != "none"

    @scale_numeric.setter
    def scale_numeric(self, value: bool | str) -> None:
        """Allow legacy code to keep assigning the previous boolean flag."""
        if isinstance(value, str):
            self.scaling = str(value).lower()
        else:
            self.scaling = "standard" if bool(value) else "none"


@dataclass
class EmbeddingConfig:
    """Configuration for the explicit embedding stage `h = E(x)`."""

    enabled: bool = True
    mode: str = "fixed"
    projection_strategy: str = "random_orthogonal"
    output_dim: int | None = None
    proportion: float = 1.0
    explained_variance_ratio: float | None = None
    whiten: bool = False
    random_state: int = 42

    def validate(self) -> "EmbeddingConfig":
        """Validate embedding-stage hyper-parameters."""
        self.mode = _normalize_choice(
            "embedding.mode",
            self.mode,
            {"fixed", "proportional", "pca_based"},
        )
        self.projection_strategy = _normalize_choice(
            "embedding.projection_strategy",
            self.projection_strategy,
            {"random_orthogonal", "identity_resize"},
        )
        if self.output_dim is not None:
            self.output_dim = int(self.output_dim)
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("embedding.output_dim must be greater than zero.")
        self.proportion = float(self.proportion)
        if self.proportion <= 0.0:
            raise ValueError("embedding.proportion must be greater than zero.")
        if self.explained_variance_ratio is not None:
            self.explained_variance_ratio = _validate_probability_like(
                "embedding.explained_variance_ratio",
                self.explained_variance_ratio,
                allow_one=True,
            )
        self.random_state = int(self.random_state)
        return self


@dataclass
class AdapterConfig:
    """Parameters for the information-collection block."""

    type: str = "linear"
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.0

    def validate(self) -> "AdapterConfig":
        """Validate adapter hyper-parameters."""
        self.type = _normalize_choice("adapter.type", self.type, {"linear", "mlp"})
        self.hidden_dims = _normalize_hidden_dims("adapter.hidden_dims", self.hidden_dims)
        self.dropout = _validate_probability_like("adapter.dropout", self.dropout)
        return self


@dataclass
class ProjectionConfig:
    """Parameters for the explicit orthogonal organization stage."""

    type: str = "learnable"
    apply_to: str = "fused"

    def validate(self) -> "ProjectionConfig":
        """Validate projection settings before runtime."""
        self.type = _normalize_choice(
            "projection.type",
            self.type,
            {"learnable", "fixed", "fixed_orthogonal", "identity", "none"},
        )
        self.apply_to = _normalize_choice(
            "projection.apply_to",
            self.apply_to,
            {"adapter", "fused"},
        )
        return self


@dataclass
class RoutingConfig:
    """Parameters controlling distributed probabilistic routing."""

    mode: str = "soft"
    depth: int = 2
    temperature: float = 1.0
    temperature_schedule: str = "constant"
    temperature_end: float | None = None
    temperature_schedule_epochs: int | None = None

    def validate(self) -> "RoutingConfig":
        """Validate routing hyper-parameters."""
        self.mode = _normalize_choice("routing.mode", self.mode, {"soft", "hard"})
        self.depth = int(self.depth)
        if self.depth < 1:
            raise ValueError("routing.depth must be greater than or equal to one.")
        self.temperature = float(self.temperature)
        if self.temperature <= 0.0:
            raise ValueError("routing.temperature must be greater than zero.")
        self.temperature_schedule = _normalize_choice(
            "routing.temperature_schedule",
            self.temperature_schedule,
            {"constant", "linear"},
        )
        if self.temperature_end is not None:
            self.temperature_end = float(self.temperature_end)
            if self.temperature_end <= 0.0:
                raise ValueError("routing.temperature_end must be greater than zero.")
        if self.temperature_schedule_epochs is not None:
            self.temperature_schedule_epochs = int(self.temperature_schedule_epochs)
            if self.temperature_schedule_epochs < 1:
                raise ValueError(
                    "routing.temperature_schedule_epochs must be greater than or equal to one."
                )
        return self


@dataclass
class ExpertConfig:
    """Parameters for the specialized classifier bank."""

    type: str = "linear"
    hidden_dims: list[int] = field(default_factory=lambda: [128])
    dropout: float = 0.1
    use_projected_features: bool = True

    def validate(self) -> "ExpertConfig":
        """Validate classifier-expert hyper-parameters."""
        self.type = _normalize_choice("expert.type", self.type, {"linear", "mlp"})
        self.hidden_dims = _normalize_hidden_dims("expert.hidden_dims", self.hidden_dims)
        self.dropout = _validate_probability_like("expert.dropout", self.dropout)
        return self


@dataclass
class AggregatorConfig:
    """Parameters for the cooperative aggregation stage."""

    type: str = "weighted_logits"
    inference_top_k: int | None = None
    renormalize_after_top_k: bool = True

    def validate(self) -> "AggregatorConfig":
        """Validate the cooperative aggregation stage."""
        self.type = _normalize_choice("aggregator.type", self.type, {"weighted_logits"})
        if self.inference_top_k is not None:
            self.inference_top_k = int(self.inference_top_k)
            if self.inference_top_k < 1:
                raise ValueError("aggregator.inference_top_k must be greater than or equal to one.")
        return self


@dataclass
class TrainingConfig:
    """Optimization settings shared by the high-level OHR training API."""

    batch_size: int = 1024
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 5
    orthogonal_regularization_weight: float = 0.01
    diversity_regularization_weight: float = 0.01
    load_balance_weight: float = 0.01
    regularization_schedule: str = "constant"
    regularization_warmup_epochs: int = 0
    selection_metric: str = "f1_macro"
    class_weighting: str = "balanced"
    class_weight_cap: float | None = None
    rare_class_boost_factor: float = 1.0
    rare_class_threshold_ratio: float = 0.5
    classification_loss: str = "cross_entropy"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    confidence_penalty_weight: float = 0.0
    inference_temperature: float = 1.0
    gradient_clip_norm: float | None = None
    device: str = "cpu"
    end_to_end: bool = True

    def validate(self) -> "TrainingConfig":
        """Validate optimization hyper-parameters before runtime."""
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.early_stopping_patience = int(self.early_stopping_patience)
        if self.batch_size < 1:
            raise ValueError("training.batch_size must be greater than or equal to one.")
        if self.epochs < 1:
            raise ValueError("training.epochs must be greater than or equal to one.")
        if self.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be non-negative.")
        self.lr = float(self.lr)
        if self.lr <= 0.0:
            raise ValueError("training.lr must be greater than zero.")
        self.weight_decay = _validate_non_negative("training.weight_decay", self.weight_decay)
        self.orthogonal_regularization_weight = _validate_non_negative(
            "training.orthogonal_regularization_weight",
            self.orthogonal_regularization_weight,
        )
        self.diversity_regularization_weight = _validate_non_negative(
            "training.diversity_regularization_weight",
            self.diversity_regularization_weight,
        )
        self.load_balance_weight = _validate_non_negative(
            "training.load_balance_weight",
            self.load_balance_weight,
        )
        self.regularization_schedule = _normalize_choice(
            "training.regularization_schedule",
            self.regularization_schedule,
            {"constant", "linear_warmup"},
        )
        self.regularization_warmup_epochs = int(self.regularization_warmup_epochs)
        if self.regularization_warmup_epochs < 0:
            raise ValueError("training.regularization_warmup_epochs must be non-negative.")
        self.selection_metric = _normalize_choice(
            "training.selection_metric",
            self.selection_metric,
            {"accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"},
        )
        self.class_weighting = _normalize_choice(
            "training.class_weighting",
            self.class_weighting,
            {"balanced", "balanced_sqrt", "none"},
        )
        if self.class_weight_cap is not None:
            self.class_weight_cap = float(self.class_weight_cap)
            if self.class_weight_cap <= 0.0:
                raise ValueError("training.class_weight_cap must be greater than zero.")
        self.rare_class_boost_factor = float(self.rare_class_boost_factor)
        if self.rare_class_boost_factor < 1.0:
            raise ValueError("training.rare_class_boost_factor must be greater than or equal to one.")
        self.rare_class_threshold_ratio = float(self.rare_class_threshold_ratio)
        if self.rare_class_threshold_ratio <= 0.0:
            raise ValueError("training.rare_class_threshold_ratio must be greater than zero.")
        self.classification_loss = _normalize_choice(
            "training.classification_loss",
            self.classification_loss,
            {"cross_entropy", "focal_loss"},
        )
        self.focal_gamma = _validate_non_negative("training.focal_gamma", self.focal_gamma)
        self.label_smoothing = _validate_probability_like(
            "training.label_smoothing",
            self.label_smoothing,
        )
        self.confidence_penalty_weight = _validate_non_negative(
            "training.confidence_penalty_weight",
            self.confidence_penalty_weight,
        )
        self.inference_temperature = float(self.inference_temperature)
        if self.inference_temperature <= 0.0:
            raise ValueError("training.inference_temperature must be greater than zero.")
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
            if self.gradient_clip_norm <= 0.0:
                raise ValueError("training.gradient_clip_norm must be greater than zero.")
        self.device = str(self.device)
        if not self.device.strip():
            raise ValueError("training.device must be a non-empty string.")
        return self

    @property
    def orthogonal_regularization(self) -> float:
        """Backward-compatible alias for the orthogonality weight."""
        return self.orthogonal_regularization_weight

    @orthogonal_regularization.setter
    def orthogonal_regularization(self, value: float) -> None:
        """Allow legacy code to keep setting the old field name."""
        self.orthogonal_regularization_weight = float(value)


def _apply_legacy_preprocessing_mapping(
    tabularizer_payload: dict[str, Any],
    preprocessing_payload: dict[str, Any],
) -> None:
    """Map old preprocessing keys stored under `tabularizer` into the new block."""
    if "handle_missing" in tabularizer_payload and "handle_missing" not in preprocessing_payload:
        preprocessing_payload["handle_missing"] = tabularizer_payload.pop("handle_missing")
    if "scale_numeric" in tabularizer_payload and "scale_numeric" not in preprocessing_payload:
        preprocessing_payload["scale_numeric"] = tabularizer_payload.pop("scale_numeric")


def _normalize_preprocessing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy preprocessing flags into the current compact schema."""
    normalized = dict(payload)
    if "scale_numeric" in normalized and "scaling" not in normalized:
        normalized["scaling"] = "standard" if bool(normalized.pop("scale_numeric")) else "none"
    if "scaling" not in normalized:
        normalized["scaling"] = "standard"
    normalized["scaling"] = str(normalized["scaling"]).lower()
    return normalized


@dataclass
class OHRConfig:
    """Top-level configuration for the standalone OHR classifier."""

    embedding_dim: int = 256
    tabularizer: TabularizerConfig = field(default_factory=TabularizerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    def validate(self) -> "OHRConfig":
        """Validate and normalize the full OHR configuration tree."""
        self.embedding_dim = int(self.embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")
        self.seed = int(self.seed)
        self.tabularizer.validate()
        self.preprocessing.validate()
        self.embedding.validate()
        self.adapter.validate()
        self.projection.validate()
        self.routing.validate()
        self.expert.validate()
        self.aggregator.validate()
        self.training.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a plain dictionary with aliases."""
        return {
            "embedding_dim": self.embedding_dim,
            "routing_mode": self.routing.mode,
            "expert_type": self.expert.type,
            "orthogonal_projection": self.projection.type,
            "orthogonal_regularization_weight": self.training.orthogonal_regularization_weight,
            "diversity_regularization_weight": self.training.diversity_regularization_weight,
            "load_balance_weight": self.training.load_balance_weight,
            "use_projected_features_for_experts": self.expert.use_projected_features,
            "tree_depth": self.routing.depth,
            "tabularizer": asdict(self.tabularizer),
            "preprocessing": asdict(self.preprocessing),
            "embedding": asdict(self.embedding),
            "adapter": asdict(self.adapter),
            "projection": asdict(self.projection),
            "routing": asdict(self.routing),
            "expert": asdict(self.expert),
            "aggregator": asdict(self.aggregator),
            "training": asdict(self.training),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OHRConfig":
        """Instantiate OHRConfig from nested or flattened mappings."""
        tabularizer_payload = dict(payload.get("tabularizer", {}))
        preprocessing_payload = dict(payload.get("preprocessing", {}))
        _apply_legacy_preprocessing_mapping(tabularizer_payload, preprocessing_payload)
        preprocessing_payload = _normalize_preprocessing_payload(preprocessing_payload)

        embedding_payload = dict(payload.get("embedding", {}))
        adapter_payload = dict(payload.get("adapter", {}))
        projection_payload = dict(payload.get("projection", {}))
        routing_payload = dict(payload.get("routing", {}))
        expert_payload = dict(payload.get("expert", {}))
        aggregator_payload = dict(payload.get("aggregator", {}))
        training_payload = dict(payload.get("training", {}))

        if (
            "orthogonal_regularization" in training_payload
            and "orthogonal_regularization_weight" not in training_payload
        ):
            training_payload["orthogonal_regularization_weight"] = training_payload.pop(
                "orthogonal_regularization"
            )

        if "orthogonal_projection" in payload and "type" not in projection_payload:
            projection_payload["type"] = payload["orthogonal_projection"]
        if "routing_mode" in payload and "mode" not in routing_payload:
            routing_payload["mode"] = payload["routing_mode"]
        if "tree_depth" in payload and "depth" not in routing_payload:
            routing_payload["depth"] = payload["tree_depth"]
        if "expert_type" in payload and "type" not in expert_payload:
            expert_payload["type"] = payload["expert_type"]
        if (
            "use_projected_features_for_experts" in payload
            and "use_projected_features" not in expert_payload
        ):
            expert_payload["use_projected_features"] = payload["use_projected_features_for_experts"]
        if (
            "orthogonal_regularization_weight" in payload
            and "orthogonal_regularization_weight" not in training_payload
        ):
            training_payload["orthogonal_regularization_weight"] = payload[
                "orthogonal_regularization_weight"
            ]
        if (
            "orthogonal_regularization" in payload
            and "orthogonal_regularization_weight" not in training_payload
        ):
            training_payload["orthogonal_regularization_weight"] = payload[
                "orthogonal_regularization"
            ]
        if (
            "diversity_regularization_weight" in payload
            and "diversity_regularization_weight" not in training_payload
        ):
            training_payload["diversity_regularization_weight"] = payload[
                "diversity_regularization_weight"
            ]
        if "load_balance_weight" in payload and "load_balance_weight" not in training_payload:
            training_payload["load_balance_weight"] = payload["load_balance_weight"]

        config = cls(
            embedding_dim=int(payload.get("embedding_dim", 256)),
            tabularizer=TabularizerConfig(**tabularizer_payload),
            preprocessing=PreprocessingConfig(**preprocessing_payload),
            embedding=EmbeddingConfig(**embedding_payload),
            adapter=AdapterConfig(**adapter_payload),
            projection=ProjectionConfig(**projection_payload),
            routing=RoutingConfig(**routing_payload),
            expert=ExpertConfig(**expert_payload),
            aggregator=AggregatorConfig(**aggregator_payload),
            training=TrainingConfig(**training_payload),
            seed=int(payload.get("seed", 42)),
        )
        return config.validate()

    @classmethod
    def from_file(cls, path: str | Path) -> "OHRConfig":
        """Instantiate OHRConfig from an external YAML or JSON file."""
        payload = load_config(path)
        return cls.from_dict(_extract_ohr_payload(payload))

    @classmethod
    def from_packaged_defaults(cls) -> "OHRConfig":
        """Instantiate OHRConfig from the YAML defaults bundled in the package."""
        payload = load_packaged_config("default_ohr.yaml")
        return cls.from_dict(_extract_ohr_payload(payload))


def load_default_ohr_config() -> OHRConfig:
    """Return the packaged default OHR configuration as a dataclass."""
    return OHRConfig.from_packaged_defaults()


def load_ohr_config(path: str | Path | None = None) -> OHRConfig:
    """Load an external OHR config when provided, otherwise use the packaged default."""
    config, _ = coerce_ohr_config(path)
    return config


def coerce_ohr_config(
    config: OHRConfig | dict[str, Any] | str | Path | None,
) -> tuple[OHRConfig, str]:
    """Accept a config dataclass, dict or file path and normalize it to OHRConfig."""
    if config is None:
        return OHRConfig.from_packaged_defaults(), "package://ohr.resources.configs/default_ohr.yaml"

    if isinstance(config, OHRConfig):
        return copy.deepcopy(config), "in_memory_ohr_config"

    if isinstance(config, dict):
        return OHRConfig.from_dict(_extract_ohr_payload(config)), "in_memory_mapping"

    if isinstance(config, (str, Path)):
        loaded_payload = load_config(config)
        config_path = str(loaded_payload.get("_config_path", Path(config).resolve()))
        return OHRConfig.from_dict(_extract_ohr_payload(loaded_payload)), config_path

    raise TypeError(f"Unsupported OHR config input: {type(config)!r}")


__all__ = [
    "AdapterConfig",
    "AggregatorConfig",
    "EmbeddingConfig",
    "ExpertConfig",
    "OHRConfig",
    "PreprocessingConfig",
    "ProjectionConfig",
    "RoutingConfig",
    "TabularizerConfig",
    "TrainingConfig",
    "coerce_ohr_config",
    "load_default_ohr_config",
    "load_ohr_config",
]
