from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr import OHRClassifier, OHRConfig, load_default_ohr_config, load_ohr_config
from ohr.config import load_packaged_config


class TestPackagedConfig(unittest.TestCase):
    def test_packaged_default_config_can_be_loaded(self) -> None:
        config = load_packaged_config("default_ohr.yaml")
        self.assertEqual(config["model_defaults"]["embedding_dim"], 256)
        self.assertEqual(config["model_defaults"]["routing_mode"], "soft")
        self.assertEqual(config["model_defaults"]["orthogonal_projection"], "learnable")
        self.assertTrue(config["model_defaults"]["tabularizer"]["enabled"])
        self.assertEqual(config["model_defaults"]["preprocessing"]["handle_missing"], "median")
        self.assertEqual(config["model_defaults"]["preprocessing"]["scaling"], "standard")
        self.assertEqual(config["model_defaults"]["embedding"]["mode"], "fixed")
        self.assertEqual(
            config["model_defaults"]["embedding"]["projection_strategy"],
            "random_orthogonal",
        )
        self.assertEqual(config["model_defaults"]["routing"]["temperature_schedule"], "constant")
        self.assertEqual(config["model_defaults"]["training"]["classification_loss"], "cross_entropy")
        self.assertEqual(config["model_defaults"]["training"]["inference_temperature"], 1.0)

    def test_default_ohr_config_dataclass_can_be_loaded(self) -> None:
        config = load_default_ohr_config()
        self.assertIsInstance(config, OHRConfig)
        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.routing.mode, "soft")
        self.assertTrue(config.expert.use_projected_features)
        self.assertEqual(config.training.diversity_regularization_weight, 0.01)
        self.assertTrue(config.tabularizer.enabled)
        self.assertEqual(config.preprocessing.scaling, "standard")
        self.assertEqual(config.embedding.mode, "fixed")
        self.assertEqual(config.embedding.projection_strategy, "random_orthogonal")
        self.assertEqual(config.routing.temperature_schedule, "constant")
        self.assertEqual(config.training.classification_loss, "cross_entropy")
        self.assertEqual(config.training.inference_temperature, 1.0)

    def test_load_ohr_config_without_path_uses_packaged_defaults(self) -> None:
        config = load_ohr_config()

        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.routing.mode, "soft")
        self.assertTrue(config.tabularizer.enabled)

    def test_ohr_config_round_trip(self) -> None:
        config = OHRConfig()
        payload = config.to_dict()
        restored = OHRConfig.from_dict(payload)
        self.assertEqual(restored.embedding_dim, config.embedding_dim)
        self.assertEqual(restored.expert.type, config.expert.type)
        self.assertEqual(restored.routing.depth, config.routing.depth)
        self.assertEqual(restored.preprocessing.handle_missing, config.preprocessing.handle_missing)
        self.assertEqual(restored.preprocessing.scaling, config.preprocessing.scaling)
        self.assertEqual(
            restored.training.orthogonal_regularization_weight,
            config.training.orthogonal_regularization_weight,
        )

    def test_external_config_file_can_be_loaded(self) -> None:
        external_path = Path(__file__).resolve().parents[1] / "configs" / "soft_linear_d2.yaml"
        config = load_ohr_config(external_path)

        self.assertEqual(config.routing.mode, "soft")
        self.assertEqual(config.routing.depth, 2)
        self.assertEqual(config.expert.type, "linear")
        self.assertEqual(config.projection.type, "learnable")
        self.assertTrue(config.tabularizer.enabled)
        self.assertEqual(config.preprocessing.handle_missing, "median")
        self.assertEqual(config.preprocessing.scaling, "standard")
        self.assertEqual(config.embedding.mode, "proportional")

    def test_classifier_can_start_from_external_config_path(self) -> None:
        external_path = Path(__file__).resolve().parents[1] / "configs" / "soft_linear_d2.yaml"
        model = OHRClassifier(external_path)

        self.assertEqual(model.config.routing.mode, "soft")
        self.assertTrue(str(external_path.resolve()) in model.config_source)

    def test_external_ohr_wrapped_yaml_is_supported(self) -> None:
        yaml_payload = """
seed: 7
ohr:
  embedding_dim: 32
  routing_mode: soft
  tree_depth: 2
  expert_type: linear
  orthogonal_projection: fixed
  preprocessing:
    handle_missing: none
    scaling: robust
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "wrapped.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertEqual(config.embedding_dim, 32)
        self.assertEqual(config.projection.type, "fixed")
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.preprocessing.handle_missing, "none")
        self.assertEqual(config.preprocessing.scaling, "robust")

    def test_missing_preprocessing_block_uses_defaults(self) -> None:
        yaml_payload = """
embedding_dim: 16
routing_mode: soft
tree_depth: 2
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "legacy.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertTrue(config.tabularizer.enabled)
        self.assertEqual(config.preprocessing.handle_missing, "median")
        self.assertEqual(config.preprocessing.scaling, "standard")
        self.assertTrue(config.tabularizer.replace_infinite)

    def test_legacy_preprocessing_fields_inside_tabularizer_are_mapped(self) -> None:
        yaml_payload = """
embedding_dim: 16
tabularizer:
  enabled: true
  input_type: tabular
  replace_infinite: true
  handle_missing: none
  scale_numeric: false
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "legacy.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertEqual(config.preprocessing.handle_missing, "none")
        self.assertEqual(config.preprocessing.scaling, "none")
        self.assertNotIn("handle_missing", config.tabularizer.__dict__)

    def test_new_experimental_config_can_define_robust_scaling(self) -> None:
        yaml_payload = """
embedding_dim: 128
preprocessing:
  handle_missing: median
  scaling: robust
embedding:
  enabled: true
  mode: pca_based
  explained_variance_ratio: 0.95
routing:
  mode: soft
  depth: 3
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "robust.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertEqual(config.preprocessing.scaling, "robust")
        self.assertEqual(config.embedding.mode, "pca_based")
        self.assertEqual(config.routing.depth, 3)

    def test_embedding_block_can_be_loaded_from_external_yaml(self) -> None:
        yaml_payload = """
embedding_dim: 16
embedding:
  enabled: true
  mode: pca_based
  output_dim: 4
  whiten: true
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "embedding.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertTrue(config.embedding.enabled)
        self.assertEqual(config.embedding.mode, "pca_based")
        self.assertEqual(config.embedding.output_dim, 4)
        self.assertTrue(config.embedding.whiten)

    def test_embedding_projection_strategy_can_be_loaded_from_external_yaml(self) -> None:
        yaml_payload = """
embedding:
  enabled: true
  mode: fixed
  output_dim: 8
  projection_strategy: identity_resize
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "embedding_projection_strategy.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertEqual(config.embedding.projection_strategy, "identity_resize")

    def test_runtime_training_extensions_can_be_loaded_from_yaml(self) -> None:
        yaml_payload = """
training:
  class_weighting: balanced_sqrt
  rare_class_boost_factor: 2.5
  rare_class_threshold_ratio: 0.4
  classification_loss: focal_loss
  focal_gamma: 1.5
  label_smoothing: 0.05
  confidence_penalty_weight: 0.01
  inference_temperature: 1.25
  gradient_clip_norm: 2.0
  regularization_schedule: linear_warmup
  regularization_warmup_epochs: 3
  selection_metric: accuracy
routing:
  temperature: 1.4
  temperature_schedule: linear
  temperature_end: 0.8
  temperature_schedule_epochs: 5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "runtime_extensions.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertEqual(config.training.class_weighting, "balanced_sqrt")
        self.assertAlmostEqual(config.training.rare_class_boost_factor, 2.5)
        self.assertAlmostEqual(config.training.rare_class_threshold_ratio, 0.4)
        self.assertEqual(config.training.classification_loss, "focal_loss")
        self.assertAlmostEqual(config.training.focal_gamma, 1.5)
        self.assertAlmostEqual(config.training.label_smoothing, 0.05)
        self.assertAlmostEqual(config.training.confidence_penalty_weight, 0.01)
        self.assertAlmostEqual(config.training.inference_temperature, 1.25)
        self.assertAlmostEqual(config.training.gradient_clip_norm, 2.0)
        self.assertEqual(config.training.regularization_schedule, "linear_warmup")
        self.assertEqual(config.training.regularization_warmup_epochs, 3)
        self.assertEqual(config.training.selection_metric, "accuracy")
        self.assertEqual(config.routing.temperature_schedule, "linear")
        self.assertAlmostEqual(config.routing.temperature_end or 0.0, 0.8)
        self.assertEqual(config.routing.temperature_schedule_epochs, 5)

    def test_selection_metric_is_validated_early(self) -> None:
        yaml_payload = """
training:
  selection_metric: auc
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_selection_metric.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ohr_config(config_path)

    def test_aggregator_runtime_extensions_can_be_loaded_from_yaml(self) -> None:
        yaml_payload = """
aggregator:
  type: weighted_logits
  inference_top_k: 2
  renormalize_after_top_k: true
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "aggregator_runtime_extensions.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            config = load_ohr_config(config_path)

        self.assertEqual(config.aggregator.inference_top_k, 2)
        self.assertTrue(config.aggregator.renormalize_after_top_k)

    def test_invalid_preprocessing_scaling_is_rejected_early(self) -> None:
        yaml_payload = """
preprocessing:
  scaling: minmax
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_scaling.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ohr_config(config_path)

    def test_invalid_routing_depth_is_rejected_early(self) -> None:
        yaml_payload = """
routing:
  mode: soft
  depth: 0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_depth.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ohr_config(config_path)

    def test_invalid_runtime_training_extensions_are_rejected_early(self) -> None:
        yaml_payload = """
training:
  class_weighting: inverse_freq
  rare_class_boost_factor: 0.5
  classification_loss: hinge
routing:
  temperature_schedule: cosine
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_runtime_extensions.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ohr_config(config_path)

    def test_invalid_rare_class_threshold_ratio_is_rejected_early(self) -> None:
        yaml_payload = """
training:
  rare_class_threshold_ratio: 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_rare_class_threshold_ratio.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ohr_config(config_path)

    def test_invalid_embedding_projection_strategy_is_rejected_early(self) -> None:
        yaml_payload = """
embedding:
  projection_strategy: learned
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_embedding_projection_strategy.yaml"
            config_path.write_text(yaml_payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ohr_config(config_path)


if __name__ == "__main__":
    unittest.main()
