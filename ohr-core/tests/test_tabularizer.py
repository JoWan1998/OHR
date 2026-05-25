from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohr.tabularizer import Tabularizer


class TestTabularizer(unittest.TestCase):
    def test_basic_dataframe_tabularization_preserves_numeric_values(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
        tabularizer = Tabularizer()

        values, feature_names = tabularizer.fit_transform(frame)

        self.assertEqual(feature_names, ["a", "b"])
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.allclose(values, np.asarray([[1.0, 3.5], [2.0, 4.5]], dtype=np.float32)))

    def test_dataframe_columns_are_reordered_by_keep_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "b": [1.0, 2.0],
                "a": [3.0, 4.0],
                "c": [5.0, 6.0],
            }
        )
        tabularizer = Tabularizer(keep_columns=["a", "b"])

        values, feature_names = tabularizer.fit_transform(frame)

        self.assertEqual(feature_names, ["a", "b"])
        self.assertTrue(np.allclose(values, np.asarray([[3.0, 1.0], [4.0, 2.0]], dtype=np.float32)))

    def test_drop_columns_removes_unwanted_features(self) -> None:
        frame = pd.DataFrame({"a": [1.0], "b": [2.0], "drop_me": [3.0]})
        tabularizer = Tabularizer(drop_columns=["drop_me"])

        values, feature_names = tabularizer.fit_transform(frame)

        self.assertEqual(feature_names, ["a", "b"])
        self.assertEqual(values.shape, (1, 2))

    def test_replace_infinite_converts_values_to_nan(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.inf], "b": [-np.inf, 2.0]})
        tabularizer = Tabularizer(replace_infinite=True)

        values, _ = tabularizer.fit_transform(frame)

        self.assertTrue(np.isnan(values[1, 0]))
        self.assertTrue(np.isnan(values[0, 1]))

    def test_ndarray_generates_default_feature_names(self) -> None:
        values = np.ones((3, 4), dtype=np.float32)
        tabularizer = Tabularizer()

        matrix, feature_names = tabularizer.fit_transform(values)

        self.assertEqual(matrix.shape, (3, 4))
        self.assertEqual(feature_names, ["feature_0", "feature_1", "feature_2", "feature_3"])

    def test_transform_raises_when_required_dataframe_columns_are_missing(self) -> None:
        train_frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        test_frame = pd.DataFrame({"a": [3.0]})
        tabularizer = Tabularizer()
        tabularizer.fit_transform(train_frame)

        with self.assertRaises(KeyError):
            tabularizer.transform(test_frame)

    def test_transform_requires_fit_before_inference(self) -> None:
        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        tabularizer = Tabularizer()

        with self.assertRaises(RuntimeError):
            tabularizer.transform(frame)

    def test_invalid_array_shape_is_rejected(self) -> None:
        tabularizer = Tabularizer()

        with self.assertRaises(ValueError):
            tabularizer.fit_transform(np.ones(4, dtype=np.float32))

    def test_empty_dataframe_is_rejected(self) -> None:
        frame = pd.DataFrame(columns=["a", "b"])
        tabularizer = Tabularizer()

        with self.assertRaises(ValueError):
            tabularizer.fit_transform(frame)

    def test_unsupported_container_type_is_rejected(self) -> None:
        tabularizer = Tabularizer()

        with self.assertRaises(TypeError):
            tabularizer.fit_transform([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
