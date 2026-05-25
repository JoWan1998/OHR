from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.models.blocks import CooperativeOutputAggregator
from ohr.models.routing import ProbabilisticTreeRouter


class TestRoutingAndAggregation(unittest.TestCase):
    def test_soft_routing_produces_probabilities_that_sum_to_one_per_leaf_set(self) -> None:
        router = ProbabilisticTreeRouter(embedding_dim=5, depth=3, mode="soft", temperature=1.0)
        outputs = router(torch.randn(7, 5))

        self.assertEqual(outputs["leaf_probabilities"].shape, (7, 8))
        self.assertTrue(torch.all(outputs["left_probabilities"] >= 0.0))
        self.assertTrue(torch.all(outputs["left_probabilities"] <= 1.0))
        self.assertTrue(
            torch.allclose(
                outputs["leaf_probabilities"].sum(dim=1),
                torch.ones(7),
                atol=1e-5,
            )
        )

    def test_weighted_aggregation_matches_manual_expert_combination(self) -> None:
        aggregator = CooperativeOutputAggregator()
        leaf_probabilities = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
        expert_logits = torch.tensor(
            [[[1.0, 2.0, -1.0], [0.0, 4.0, 3.0]]],
            dtype=torch.float32,
        )

        outputs = aggregator(leaf_probabilities, expert_logits)
        expected = (0.7 * expert_logits[:, 0, :]) + (0.3 * expert_logits[:, 1, :])

        self.assertTrue(torch.allclose(outputs["logits"], expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
