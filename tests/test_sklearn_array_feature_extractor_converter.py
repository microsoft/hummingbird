"""
Tests sklearn Normalizer converter
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.datasets import load_digits

import hummingbird.ml


class TestSklearnArrayFeatureExtractor(unittest.TestCase):

    # tests VarianceThreshold converter (convert_sklearn_variance_threshold)
    def test_variance_threshold(self):

        X = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)
        selector = VarianceThreshold()
        selector.fit_transform(X)
        data_tensor = torch.from_numpy(X)

        torch_model = hummingbird.ml.convert(selector, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(
            selector.transform(X), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
        )

    # tests SelectKBest converter (convert_sklearn_select_k_best) with mutual_info_classif
    def test_k_best_mutual_info_classif(self):

        X, y = load_digits(return_X_y=True)
        selector = SelectKBest(mutual_info_classif, k=20)
        selector.fit_transform(X, y)
        data_tensor = torch.from_numpy(X)

        torch_model = hummingbird.ml.convert(selector, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(
            selector.transform(X), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
        )

    # # tests SelectKBest converter (convert_sklearn_select_k_best) with chi2
    # def test_k_best_chi2(self):
    # ### TODO: This is failing, need to fixup convert_sklearn_select_k_best for chi2

    #     X, y = load_digits(return_X_y=True)
    #     selector = SelectKBest(chi2, k=20)
    #     selector.fit_transform(X, y)
    #     data_tensor = torch.from_numpy(X)

    #     torch_model = hummingbird.ml.convert(selector, "torch")

    #     self.assertIsNotNone(torch_model)
    #     np.testing.assert_allclose(
    #         selector.transform(X), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
    #     )


if __name__ == "__main__":
    unittest.main()
