from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.engine import class_weights


class TestEngineClassWeights(unittest.TestCase):
    def test_class_weights_preserve_none_strategy_when_no_rare_boost_is_requested(self) -> None:
        y = np.asarray([0] * 10 + [1] * 2 + [2] * 1, dtype=np.int64)

        weights = class_weights(y, num_classes=3, strategy="none")

        self.assertIsNone(weights)

    def test_class_weights_can_emphasize_rare_classes_even_without_base_weighting(self) -> None:
        y = np.asarray([0] * 10 + [1] * 2 + [2] * 1, dtype=np.int64)

        weights = class_weights(
            y,
            num_classes=3,
            strategy="none",
            rare_class_boost_factor=3.0,
            rare_class_threshold_ratio=0.5,
        )

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertTrue(np.allclose(weights.numpy(), np.asarray([1.0, 1.0, 3.0], dtype=np.float32)))

    def test_class_weights_boost_rare_classes_after_balanced_sqrt_weighting(self) -> None:
        y = np.asarray([0] * 10 + [1] * 2 + [2] * 1, dtype=np.int64)

        baseline = class_weights(y, num_classes=3, strategy="balanced_sqrt")
        boosted = class_weights(
            y,
            num_classes=3,
            strategy="balanced_sqrt",
            rare_class_boost_factor=2.0,
            rare_class_threshold_ratio=0.5,
        )

        assert baseline is not None
        assert boosted is not None
        self.assertGreater(float(boosted[2].item()), float(baseline[2].item()))
        self.assertTrue(np.allclose(boosted[:2].numpy(), baseline[:2].numpy()))


if __name__ == "__main__":
    unittest.main()
