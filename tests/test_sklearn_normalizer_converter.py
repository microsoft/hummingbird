"""
Tests sklearn Normalizer converter
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.preprocessing import Normalizer

import hummingbird.ml


class TestSklearnNormalizer(unittest.TestCase):
    def test_normalizer_converter(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            pytorch_model = hummingbird.ml.convert(model, "pytorch")

            self.assertIsNotNone(pytorch_model)
            np.testing.assert_allclose(
                model.transform(data), pytorch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
            )

    def test_normalizer_converter_raises_wrong_type(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)

        model = Normalizer(norm="invalid")
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertRaises(RuntimeError, pytorch_model.operator_map.SklearnNormalizer, torch.from_numpy(data))


if __name__ == "__main__":
    unittest.main()
