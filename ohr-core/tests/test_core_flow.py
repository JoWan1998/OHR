from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.embedding import EmbeddingStage
from ohr import load_default_ohr_config
from ohr.models.ohr import OHRModel
from ohr.tabularizer import Tabularizer


class TestOHRCoreFlow(unittest.TestCase):
    def test_tabularizer_embedding_adapter_projection_chain_runs_with_dataframe_input(self) -> None:
        frame = pd.DataFrame(
            {
                "feature_b": [1.0, 2.0, 3.0],
                "feature_a": [4.0, 5.0, 6.0],
                "drop_me": [7.0, 8.0, 9.0],
            }
        )
        tabularizer = Tabularizer(keep_columns=["feature_a", "feature_b"])
        matrix, feature_names = tabularizer.fit_transform(frame)
        embedding = EmbeddingStage(mode="fixed", output_dim=4)
        embedded_matrix, embedded_feature_names = embedding.fit_transform(matrix)

        model = OHRModel(
            input_dim=embedded_matrix.shape[1],
            num_classes=3,
            embedding_dim=8,
            adapter={"type": "linear", "hidden_dims": [], "dropout": 0.0},
            projection={"type": "learnable", "apply_to": "fused"},
            routing={"mode": "soft", "depth": 2, "temperature": 1.0},
            expert={"type": "linear", "hidden_dims": [16], "dropout": 0.0},
            aggregator={"type": "weighted_logits"},
        )

        outputs = model(torch.from_numpy(embedded_matrix))

        self.assertEqual(feature_names, ["feature_a", "feature_b"])
        self.assertEqual(embedded_feature_names, [f"embedding_feature_{i}" for i in range(4)])
        self.assertEqual(outputs["embedded_inputs"].shape, (3, 4))
        self.assertEqual(outputs["adapter_output"].shape, (3, 8))
        self.assertEqual(outputs["projected_features"].shape, (3, 8))
        self.assertEqual(outputs["classifier_logits"].shape, (3, 3))
        self.assertIn("projection_penalty", outputs)
        self.assertEqual(outputs["projection_penalty"].ndim, 0)

    def test_ohr_forward_runs_from_default_config_with_dummy_array(self) -> None:
        config = load_default_ohr_config()
        config.embedding_dim = 12
        config.routing.depth = 2

        dummy_inputs = np.random.default_rng(3).normal(size=(5, 4)).astype(np.float32)
        model = OHRModel(
            input_dim=dummy_inputs.shape[1],
            num_classes=4,
            embedding_dim=config.embedding_dim,
            adapter=config.adapter.__dict__,
            projection=config.projection.__dict__,
            routing=config.routing.__dict__,
            expert=config.expert.__dict__,
            aggregator=config.aggregator.__dict__,
        )

        outputs = model(torch.from_numpy(dummy_inputs))

        self.assertEqual(outputs["leaf_probabilities"].shape, (5, 4))
        self.assertEqual(outputs["expert_logits"].shape, (5, 4, 4))
        self.assertTrue(
            torch.allclose(
                outputs["leaf_probabilities"].sum(dim=1),
                torch.ones(5),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
