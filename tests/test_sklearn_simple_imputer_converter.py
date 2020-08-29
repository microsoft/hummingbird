"""
Tests sklearn Normalizer converter
"""
import unittest
import warnings

import numpy as np
import torch

from sklearn.impute import SimpleImputer

import hummingbird.ml


class TestSklearnSimpleImputer(unittest.TestCase):
    def test_simple_imputer_float_inputs(self):
        model = SimpleImputer(strategy="mean", fill_value="nan")
        data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
        model.fit(data)
        data_tensor = torch.from_numpy(data)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
        )


if __name__ == "__main__":
    unittest.main()
