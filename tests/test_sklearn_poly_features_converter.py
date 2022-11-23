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
    def _test_sklearn_polynomial_featurizer(self, data, model):

        data_tensor = torch.from_numpy(data)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(
            model.transform(data),
            torch_model.transform(data_tensor),
            rtol=1e-06,
            atol=1e-06,
        )

    def test_sklearn_poly_feat_with_bias(self):
        data = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]], dtype=np.float32)
        model = PolynomialFeatures(degree=2, include_bias=True, order="F").fit(data)
        self._test_sklearn_polynomial_featurizer(data, model)

    def test_sklearn_poly_feat_with_no_bias(self):
        data = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]], dtype=np.float32)
        model = PolynomialFeatures(degree=2, include_bias=False, order="F").fit(data)
        self._test_sklearn_polynomial_featurizer(data, model)

    # TODO: interaction is not currently supported (bug)
    # def test_sklearn_poly_feat_with_interaction_and_bias(self):
    #    data = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]], dtype=np.float32)
    #    model = PolynomialFeatures(degree=2, include_bias=True, order="F", interaction_only=True).fit(data)
    #    self._test_sklearn_polynomial_featurizer(data, model)

    # def test_sklearn_poly_feat_with_interaction_and_no_bias(self):
    #    data = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]], dtype=np.float32)
    #    model = PolynomialFeatures(degree=2, include_bias=False, order="F", interaction_only=True).fit(data)
    #    self._test_sklearn_polynomial_featurizer(data, model)

    def test_sklearn_poly_featurizer_raises(self):
        data = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]], dtype=np.float32)

        # TODO: delete when implemented
        model = PolynomialFeatures(degree=4, include_bias=True, order="F").fit(data)
        self.assertRaises(NotImplementedError, hummingbird.ml.convert, model, "torch")

        # TODO: delete when implemented
        model = PolynomialFeatures(degree=2, interaction_only=True, order="F").fit(data)
        self.assertRaises(NotImplementedError, hummingbird.ml.convert, model, "torch")


if __name__ == "__main__":
    unittest.main()
