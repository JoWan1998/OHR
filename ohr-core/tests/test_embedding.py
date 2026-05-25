from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.embedding import EmbeddingStage


class TestEmbeddingStage(unittest.TestCase):
    def test_fixed_embedding_uses_random_orthogonal_projection_by_default(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        )
        embedding = EmbeddingStage(mode="fixed", output_dim=2, random_state=11)

        transformed, feature_names = embedding.fit_transform(matrix)

        self.assertEqual(transformed.shape, (2, 2))
        self.assertFalse(np.allclose(transformed, matrix[:, :2]))
        self.assertIsNone(embedding.pca_model_)
        self.assertIsNotNone(embedding.projection_matrix_)
        self.assertTrue(
            np.allclose(
                embedding.projection_matrix_ @ embedding.projection_matrix_.T,
                np.eye(2, dtype=np.float32),
                atol=1e-5,
            )
        )
        self.assertEqual(feature_names, ["embedding_feature_0", "embedding_feature_1"])

    def test_fixed_embedding_can_use_legacy_identity_resize_projection(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        )
        embedding = EmbeddingStage(
            mode="fixed",
            output_dim=2,
            projection_strategy="identity_resize",
        )

        transformed, _ = embedding.fit_transform(matrix)

        self.assertTrue(np.allclose(transformed, matrix[:, :2]))

    def test_proportional_embedding_derives_dimension_from_input_width(self) -> None:
        matrix = np.random.default_rng(7).normal(size=(3, 10)).astype(np.float32)
        embedding = EmbeddingStage(mode="proportional", proportion=0.4, random_state=19)

        transformed, _ = embedding.fit_transform(matrix)

        self.assertEqual(transformed.shape, (3, 4))
        self.assertEqual(embedding.output_dim_, 4)
        self.assertFalse(np.allclose(transformed, matrix[:, :4]))

        reused = EmbeddingStage(mode="proportional", proportion=0.4, random_state=19)
        repeated, _ = reused.fit_transform(matrix)
        self.assertTrue(np.allclose(transformed, repeated))

    def test_pca_based_embedding_fits_and_reuses_pca_model(self) -> None:
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(20, 6)).astype(np.float32)
        embedding = EmbeddingStage(mode="pca_based", output_dim=3, whiten=True)

        transformed, feature_names = embedding.fit_transform(matrix)
        reused = embedding.transform(matrix[:5])

        self.assertEqual(transformed.shape, (20, 3))
        self.assertEqual(reused.shape, (5, 3))
        self.assertIsNotNone(embedding.pca_model_)
        self.assertEqual(feature_names, ["pca_component_0", "pca_component_1", "pca_component_2"])

    def test_pca_based_embedding_rejects_non_finite_inputs(self) -> None:
        matrix = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        embedding = EmbeddingStage(mode="pca_based", output_dim=1)

        with self.assertRaises(ValueError):
            embedding.fit_transform(matrix)


if __name__ == "__main__":
    unittest.main()
