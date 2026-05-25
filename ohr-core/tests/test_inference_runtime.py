from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr import OHRClassifier, OHRConfig


class TestOHRInferenceRuntime(unittest.TestCase):
    def _fit_small_model(self) -> tuple[OHRClassifier, pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(201)
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
        return model, X, y

    def test_evaluate_with_artifacts_collects_outputs_once(self) -> None:
        model, X, y = self._fit_small_model()
        from ohr import api as ohr_api

        original = ohr_api.collect_model_outputs
        with patch("ohr.api.collect_model_outputs") as mocked_collect:
            mocked_collect.side_effect = original
            model.evaluate(
                X.iloc[:8],
                y[:8],
                batch_size=4,
                include_internal_metrics=True,
                return_artifacts=True,
            )

        self.assertEqual(mocked_collect.call_count, 1)


if __name__ == "__main__":
    unittest.main()
