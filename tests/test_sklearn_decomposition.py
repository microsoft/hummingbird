"""
Tests sklearn matrix decomposition converters
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.decomposition import FastICA, KernelPCA, PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

import hummingbird.ml


class TestSklearnMatrixDecomposition(unittest.TestCase):
    def _fit_model_pca(self, model):
        data = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        model.fit(X_train)
        return model, X_test.astype(np.float32)

    def test_pca_converter(self):
        model, X_test = self._fit_model_pca(PCA(n_components=2, random_state=42, whiten=True))
        X_test = X_test.astype("float32")

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.transform(X_test), torch_model.transform(X_test), rtol=1e-6, atol=1e-6)

    def test_kernel_pca_converter(self):
        for kernel in ["linear"]:
            model, X_test = self._fit_model_pca(KernelPCA(n_components=3, random_state=42, kernel=kernel))
            X_test = X_test.astype("float32")

            torch_model = hummingbird.ml.convert(model, "torch")
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.transform(X_test), torch_model.transform(X_test), rtol=1e-6, atol=1e-6)

    def test_fast_ica_converter(self):
        model, X_test = self._fit_model_pca(FastICA(n_components=2, random_state=42, whiten=True))
        X_test = X_test.astype("float32")

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.transform(X_test), torch_model.transform(X_test), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
