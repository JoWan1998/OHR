from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr import OHRClassifier, OHRConfig


class TestOHRClassifier(unittest.TestCase):
    def test_keras_like_workflow(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"f{i}" for i in range(6)])
        y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

        config = OHRConfig()
        config.embedding_dim = 16
        config.routing.depth = 2
        config.routing.mode = "soft"
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2
        config.training.diversity_regularization_weight = 0.02
        config.training.load_balance_weight = 0.02
        config.training.class_weighting = "balanced_sqrt"
        config.training.rare_class_boost_factor = 2.0
        config.training.rare_class_threshold_ratio = 0.5
        config.training.classification_loss = "focal_loss"
        config.training.focal_gamma = 1.5
        config.training.label_smoothing = 0.05
        config.training.confidence_penalty_weight = 0.01
        config.training.inference_temperature = 1.5
        config.training.gradient_clip_norm = 1.0
        config.training.regularization_schedule = "linear_warmup"
        config.training.regularization_warmup_epochs = 1
        config.preprocessing.scale_numeric = False
        config.preprocessing.handle_missing = "none"
        config.routing.temperature_schedule = "linear"
        config.routing.temperature_end = 0.8
        config.routing.temperature_schedule_epochs = 2

        model = OHRClassifier(config)
        model.compile(device="cpu")
        history = model.fit(X, y, validation_split=0.2, scale_features=None)
        metrics = model.evaluate(X, y)
        inspection = model.inspect_samples(X.iloc[:4], top_k=2)
        diagnostics = model.get_routing_diagnostics(X.iloc[:4], top_k=2)
        reordered_predictions = model.predict(X.iloc[:5][list(reversed(X.columns))])
        run_metadata = model.get_run_metadata()
        logits = model.predict_logits(X.iloc[:5])
        probabilities = model.predict_proba(X.iloc[:5])
        raw_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        raw_probabilities /= raw_probabilities.sum(axis=1, keepdims=True)
        scaled_logits = logits / 1.5
        scaled_probabilities = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        scaled_probabilities /= scaled_probabilities.sum(axis=1, keepdims=True)

        self.assertIn("val_f1_macro", history.history)
        self.assertIn("val_routing_entropy", history.history)
        self.assertIn("train_confidence_penalty", history.history)
        self.assertIn("routing_temperature", history.history)
        self.assertEqual(len(model.class_names), 3)
        self.assertEqual(model.predict(X.iloc[:5]).shape, (5,))
        self.assertEqual(probabilities.shape, (5, 3))
        self.assertTrue(np.allclose(probabilities, scaled_probabilities))
        self.assertFalse(np.allclose(probabilities, raw_probabilities))
        self.assertIn("routing_entropy", metrics)
        self.assertEqual(inspection["leaf_probabilities"].shape[0], 4)
        self.assertEqual(inspection["top_expert_indices"].shape[1], 2)
        self.assertIn("OHRClassifier", model.summary())
        self.assertIn("embedding_mode", model.summary())
        self.assertFalse(model.metadata["scale_features"])
        self.assertEqual(model.metadata["scaling"], "none")
        self.assertEqual(model.metadata["scaling_mode"], "none")
        self.assertEqual(model.metadata["embedding_mode"], "fixed")
        self.assertEqual(model.metadata["embedding_runtime_mode"], "fixed")
        self.assertEqual(model.metadata["epochs_configured"], 2)
        self.assertGreaterEqual(model.metadata["epochs_trained"], 1)
        self.assertEqual(model.metadata["effective_embedding_dim"], 6)
        self.assertEqual(model.metadata["classification_loss"], "focal_loss")
        self.assertEqual(model.metadata["class_weighting"], "balanced_sqrt")
        self.assertAlmostEqual(model.metadata["rare_class_boost_factor"], 2.0)
        self.assertAlmostEqual(model.metadata["rare_class_threshold_ratio"], 0.5)
        self.assertIn("rare_class_indices", model.metadata)
        self.assertAlmostEqual(model.metadata["label_smoothing"], 0.05)
        self.assertAlmostEqual(model.metadata["confidence_penalty_weight"], 0.01)
        self.assertAlmostEqual(model.metadata["inference_temperature"], 1.5)
        self.assertAlmostEqual(model.metadata["gradient_clip_norm"], 1.0)
        self.assertEqual(model.metadata["routing_temperature_schedule"], "linear")
        self.assertGreaterEqual(model.metadata["best_epoch"], 1)
        self.assertLessEqual(model.metadata["best_epoch"], model.metadata["epochs_trained"])
        self.assertIn("run_started_at_utc", model.metadata)
        self.assertIn("run_finished_at_utc", model.metadata)
        self.assertIn("resolved_config", model.metadata)
        self.assertIn("final_metrics", model.metadata)
        self.assertIn("stop_reason", model.metadata)
        self.assertIn("resolved_config", run_metadata)
        self.assertEqual(run_metadata["seed"], 42)
        self.assertEqual(
            run_metadata["resolved_config"]["training"]["classification_loss"],
            "focal_loss",
        )
        self.assertEqual(
            run_metadata["resolved_config"]["training"]["class_weighting"],
            "balanced_sqrt",
        )
        self.assertEqual(
            run_metadata["resolved_config"]["training"]["rare_class_boost_factor"],
            2.0,
        )
        self.assertEqual(
            run_metadata["resolved_config"]["routing"]["temperature_schedule"],
            "linear",
        )
        self.assertEqual(diagnostics["dominant_expert_indices"].shape, (4,))
        self.assertEqual(diagnostics["dominant_expert_probabilities"].shape, (4,))
        self.assertEqual(diagnostics["effective_depth_per_sample"].shape, (4,))
        self.assertIn("routing_entropy", diagnostics["routing_metrics"])
        self.assertTrue(np.array_equal(reordered_predictions, model.predict(X.iloc[:5])))

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "saved_ohr"
            model.save(save_dir)
            loaded = OHRClassifier.load(save_dir)
            self.assertEqual(loaded.predict(X.iloc[:5]).shape, (5,))
            self.assertEqual(len(loaded.class_names), 3)
            self.assertIn("validation_metrics", loaded.metadata)
            self.assertEqual(loaded.metadata["embedding_mode"], "fixed")
            self.assertEqual(loaded.metadata["scaling"], "none")
            self.assertIn("resolved_config", loaded.metadata)
            self.assertTrue(np.array_equal(loaded.predict(X.iloc[:5]), model.predict(X.iloc[:5])))
            self.assertTrue(
                np.array_equal(
                    loaded.predict(X.iloc[:5][list(reversed(X.columns))]),
                    model.predict(X.iloc[:5]),
                )
            )

    def test_ndarray_workflow_uses_packaged_defaults(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(90, 5)).astype(np.float32)
        y = np.asarray(["zero"] * 30 + ["one"] * 30 + ["two"] * 30)

        model = OHRClassifier()
        model.config.embedding_dim = 12
        model.config.training.epochs = 2
        model.config.training.batch_size = 16
        model.config.training.early_stopping_patience = 2
        model.compile(device="cpu")

        history = model.fit(X, y, validation_split=0.2, scale_features=None)
        metrics = model.evaluate(X, y)

        self.assertIn("train_loss", history.history)
        self.assertEqual(model.predict(X[:6]).shape, (6,))
        self.assertIn("accuracy", metrics)
        self.assertEqual(model.metadata["scaling"], "standard")

    def test_fit_rejects_empty_inputs_and_mismatched_labels(self) -> None:
        model = OHRClassifier()

        with self.assertRaises(ValueError):
            model.fit(pd.DataFrame(columns=["a", "b"]), np.asarray([]))

        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with self.assertRaises(ValueError):
            model.fit(X, np.asarray(["only_one"]))

    def test_external_yaml_workflow_runs_on_synthetic_dataframe(self) -> None:
        rng = np.random.default_rng(21)
        X = pd.DataFrame(rng.normal(size=(45, 4)), columns=["d", "c", "b", "a"])
        y = np.asarray(["alpha"] * 15 + ["beta"] * 15 + ["gamma"] * 15)
        config_path = Path(__file__).resolve().parents[1] / "configs" / "soft_linear_d2.yaml"

        model = OHRClassifier(config_path)
        model.config.embedding_dim = 10
        model.config.training.epochs = 2
        model.config.training.batch_size = 8
        model.config.training.early_stopping_patience = 2
        model.compile(device="cpu")

        history = model.fit(X, y, validation_split=0.2, scale_features="robust")
        predictions = model.predict(X.iloc[:4])

        self.assertIn("train_loss", history.history)
        self.assertEqual(predictions.shape, (4,))
        self.assertEqual(model.metadata["embedding_mode"], "proportional")
        self.assertEqual(model.metadata["scaling"], "robust")

    def test_identity_projection_disables_effective_orthogonal_regularization(self) -> None:
        rng = np.random.default_rng(8)
        X = pd.DataFrame(rng.normal(size=(45, 4)), columns=["a", "b", "c", "d"])
        y = np.asarray(["alpha"] * 15 + ["beta"] * 15 + ["gamma"] * 15)

        config = OHRConfig()
        config.embedding_dim = 12
        config.routing.depth = 2
        config.projection.type = "identity"
        config.training.epochs = 2
        config.training.batch_size = 8
        config.training.early_stopping_patience = 2

        model = OHRClassifier(config)
        model.compile(device="cpu", orthogonal_regularization_weight=0.5)
        model.fit(X, y, validation_split=0.2, scale_features=None)

        self.assertEqual(model.metadata["orthogonal_regularization_weight_requested"], 0.5)
        self.assertEqual(model.metadata["orthogonal_regularization_weight_effective"], 0.0)
        self.assertFalse(model.metadata["projection_penalty_active"])

    def test_pca_embedding_reports_fitted_transformer(self) -> None:
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(60, 8)), columns=[f"f{i}" for i in range(8)])
        y = np.asarray(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

        config = OHRConfig()
        config.embedding_dim = 16
        config.embedding.mode = "pca_based"
        config.embedding.explained_variance_ratio = 0.95
        config.routing.depth = 2
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2

        model = OHRClassifier(config)
        model.compile(device="cpu")
        model.fit(X, y, validation_split=0.2, scale_features=None)

        self.assertTrue(model.metadata["pca_fitted"])
        self.assertEqual(model.metadata["embedding_runtime_mode"], "pca_based")
        self.assertGreaterEqual(model.metadata["effective_embedding_dim"], 1)
        self.assertEqual(
            model.metadata["resolved_config"]["embedding"]["runtime_mode"],
            "pca_based",
        )

    def test_same_seed_produces_reproducible_predictions(self) -> None:
        rng = np.random.default_rng(13)
        X = pd.DataFrame(rng.normal(size=(75, 5)), columns=[f"f{i}" for i in range(5)])
        y = np.asarray(["zero"] * 25 + ["one"] * 25 + ["two"] * 25)

        config = OHRConfig()
        config.embedding_dim = 10
        config.routing.depth = 2
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2
        config.seed = 123

        first = OHRClassifier(config)
        first.compile(device="cpu")
        first.fit(X, y, validation_split=0.2, scale_features=None)
        first_predictions = first.predict_labels(X.iloc[:10])

        second = OHRClassifier(config)
        second.compile(device="cpu")
        second.fit(X, y, validation_split=0.2, scale_features=None)
        second_predictions = second.predict_labels(X.iloc[:10])

        self.assertTrue(np.array_equal(first_predictions, second_predictions))
        self.assertEqual(first.metadata["seed"], 123)
        self.assertEqual(second.metadata["seed"], 123)

    def test_compile_rejects_invalid_runtime_overrides(self) -> None:
        model = OHRClassifier()

        with self.assertRaises(ValueError):
            model.compile(class_weighting="inverse_freq")

        with self.assertRaises(ValueError):
            model.compile(loss="hinge")

        with self.assertRaises(ValueError):
            model.compile(rare_class_boost_factor=0.5)

        with self.assertRaises(ValueError):
            model.compile(rare_class_threshold_ratio=0.0)

    def test_evaluate_can_return_artifacts_in_one_call(self) -> None:
        rng = np.random.default_rng(101)
        X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"f{i}" for i in range(6)])
        y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

        config = OHRConfig()
        config.embedding_dim = 12
        config.routing.depth = 2
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2

        model = OHRClassifier(config)
        model.compile(device="cpu")
        model.fit(X, y, validation_split=0.2, scale_features=None)

        result = model.evaluate(
            X.iloc[:8],
            y[:8],
            batch_size=4,
            include_internal_metrics=True,
            return_artifacts=True,
        )

        self.assertIn("classification", result)
        self.assertIn("predictions", result)
        self.assertIn("probabilities", result)
        self.assertIn("hive", result)
        self.assertEqual(result["predictions"].shape, (8,))
        self.assertEqual(result["probabilities"].shape, (8, 3))
        self.assertEqual(result["encoded_targets"].shape, (8,))

    def test_routing_diagnostics_include_prediction_artifacts(self) -> None:
        rng = np.random.default_rng(102)
        X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"f{i}" for i in range(6)])
        y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

        config = OHRConfig()
        config.embedding_dim = 12
        config.routing.depth = 2
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2

        model = OHRClassifier(config)
        model.compile(device="cpu")
        model.fit(X, y, validation_split=0.2, scale_features=None)

        diagnostics = model.get_routing_diagnostics(X.iloc[:6], top_k=2, batch_size=3)

        self.assertIn("probabilities", diagnostics)
        self.assertIn("predictions", diagnostics)
        self.assertIn("predicted_labels", diagnostics)
        self.assertEqual(diagnostics["probabilities"].shape, (6, 3))
        self.assertEqual(diagnostics["predictions"].shape, (6,))
        self.assertEqual(diagnostics["predicted_labels"].shape, (6,))

    def test_fit_records_selection_metric_metadata(self) -> None:
        rng = np.random.default_rng(103)
        X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"f{i}" for i in range(6)])
        y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

        config = OHRConfig()
        config.embedding_dim = 12
        config.routing.depth = 2
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2
        config.training.selection_metric = "accuracy"

        model = OHRClassifier(config)
        model.compile(device="cpu")
        model.fit(X, y, validation_split=0.2, scale_features=None)

        self.assertEqual(model.metadata["selection_metric"], "accuracy")
        self.assertIn("best_selection_metric_value", model.metadata)
        self.assertEqual(
            model.metadata["resolved_config"]["training"]["selection_metric"],
            "accuracy",
        )

    def test_inference_top_k_keeps_probabilities_well_formed(self) -> None:
        rng = np.random.default_rng(104)
        X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"f{i}" for i in range(6)])
        y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

        config = OHRConfig()
        config.embedding_dim = 12
        config.routing.depth = 2
        config.training.epochs = 2
        config.training.batch_size = 16
        config.training.early_stopping_patience = 2
        config.aggregator.inference_top_k = 1

        model = OHRClassifier(config)
        model.compile(device="cpu")
        model.fit(X, y, validation_split=0.2, scale_features=None)
        probabilities = model.predict_proba(X.iloc[:10], batch_size=5)

        self.assertEqual(probabilities.shape, (10, 3))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), np.ones(10)))
        self.assertEqual(
            model.metadata["resolved_config"]["aggregator"]["inference_top_k"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
