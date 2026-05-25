from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.preprocessing import IdentityPreprocessor, build_input_preprocessor


class TestPreprocessing(unittest.TestCase):
    def test_standard_scaling_builds_pipeline(self) -> None:
        preprocessor = build_input_preprocessor(
            scaling="standard",
            preprocessed=False,
            handle_missing="median",
        )
        self.assertEqual(preprocessor.named_steps["scaler"].__class__.__name__, "StandardScaler")

    def test_robust_scaling_builds_pipeline(self) -> None:
        preprocessor = build_input_preprocessor(
            scaling="robust",
            preprocessed=False,
            handle_missing="median",
        )
        self.assertEqual(preprocessor.named_steps["scaler"].__class__.__name__, "RobustScaler")

    def test_none_scaling_can_reduce_to_identity(self) -> None:
        preprocessor = build_input_preprocessor(
            scaling="none",
            preprocessed=False,
            handle_missing="none",
        )
        self.assertIsInstance(preprocessor, IdentityPreprocessor)

    def test_preprocessed_inputs_skip_additional_steps(self) -> None:
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        preprocessor = build_input_preprocessor(
            scaling="robust",
            preprocessed=True,
            handle_missing="median",
        )
        transformed = preprocessor.transform(matrix)
        self.assertTrue(np.allclose(transformed, matrix))

    def test_invalid_scaling_fails_with_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            build_input_preprocessor(
                scaling="minmax",
                preprocessed=False,
                handle_missing="median",
            )


if __name__ == "__main__":
    unittest.main()
