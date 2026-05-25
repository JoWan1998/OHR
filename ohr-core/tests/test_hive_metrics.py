from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.hive import compute_hive_metrics, routing_entropy


class TestHiveMetrics(unittest.TestCase):
    def test_internal_hive_metrics_are_computed_from_routing_outputs(self) -> None:
        payload = {
            "leaf_probabilities": np.asarray(
                [
                    [0.70, 0.20, 0.10, 0.00],
                    [0.15, 0.65, 0.10, 0.10],
                    [0.10, 0.15, 0.70, 0.05],
                    [0.05, 0.10, 0.20, 0.65],
                ],
                dtype=np.float32,
            ),
            "left_probabilities": np.asarray(
                [
                    [0.8, 0.7, 0.3],
                    [0.6, 0.4, 0.5],
                    [0.3, 0.2, 0.8],
                    [0.2, 0.1, 0.9],
                ],
                dtype=np.float32,
            ),
            "node_reach_probabilities": np.asarray(
                [
                    [1.0, 0.8, 0.2],
                    [1.0, 0.6, 0.4],
                    [1.0, 0.3, 0.7],
                    [1.0, 0.2, 0.8],
                ],
                dtype=np.float32,
            ),
            "projection_penalty": np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
        }

        metrics = compute_hive_metrics(payload)

        self.assertEqual(len(metrics["expert_usage_frequency"]), 4)
        self.assertEqual(len(metrics["mean_leaf_probability"]), 4)
        self.assertGreater(metrics["routing_entropy"], 0.0)
        self.assertGreater(metrics["effective_experts"], 1.0)
        self.assertGreaterEqual(metrics["load_balance_score"], 0.0)
        self.assertLessEqual(metrics["load_balance_score"], 1.0)
        self.assertAlmostEqual(metrics["mean_projection_penalty"], 0.025, places=6)

    def test_routing_entropy_is_finite_for_extreme_probabilities(self) -> None:
        left_probabilities = torch.tensor(
            [
                [0.0, 1.0, 1e-15, 1.0 - 1e-15],
                [1.0, 0.0, 1e-30, 1.0],
            ],
            dtype=torch.float32,
        )
        node_reach_probabilities = torch.ones_like(left_probabilities)

        entropy = routing_entropy(left_probabilities, node_reach_probabilities)

        self.assertTrue(torch.isfinite(entropy))
        self.assertGreaterEqual(float(entropy), 0.0)

    def test_compute_hive_metrics_keeps_routing_entropy_finite_at_extremes(self) -> None:
        payload = {
            "leaf_probabilities": np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "left_probabilities": np.asarray(
                [
                    [0.0, 1.0, 1e-15],
                    [1.0, 0.0, 1.0 - 1e-15],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "node_reach_probabilities": np.ones((4, 3), dtype=np.float32),
        }

        metrics = compute_hive_metrics(payload)

        self.assertTrue(np.isfinite(metrics["routing_entropy"]))
        self.assertTrue(np.all(np.isfinite(np.asarray(metrics["mean_leaf_probability"], dtype=np.float64))))
        self.assertTrue(np.isfinite(metrics["effective_experts"]))
        self.assertGreaterEqual(metrics["routing_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
