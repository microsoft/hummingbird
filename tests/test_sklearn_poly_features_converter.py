"""
Tests sklearn Binarizer converter
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

import hummingbird.ml


class TestSklearnPolynomialFeatures(unittest.TestCase):
    def test_sklearn_polynomial_featurizer(self):
        data = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]], dtype=np.float32)
        data_tensor = torch.from_numpy(data)
        model = PolynomialFeatures(degree=2, include_bias=True, order="F").fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
        )


if __name__ == "__main__":
    unittest.main()
