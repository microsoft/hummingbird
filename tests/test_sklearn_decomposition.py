"""
Tests sklearn matrix decomposition converters
"""
import unittest
import warnings
import sys
from distutils.version import LooseVersion

import numpy as np
import torch
import sklearn
from sklearn.decomposition import FastICA, KernelPCA, PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


import hummingbird.ml


class TestSklearnMatrixDecomposition(unittest.TestCase):
    def _fit_model_pca(self, model, precompute=False):
        data = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        X_test = X_test.astype("float32")
        if precompute:
            # For precompute we use a linear kernel
            model.fit(np.dot(X_train, X_train.T))
            X_test = np.dot(X_test, X_train.T)
        else:
            model.fit(X_train)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.transform(X_test), torch_model.transform(X_test), rtol=1e-6, atol=2 * 1e-5)

    # PCA n_components none
    def test_pca_converter_none(self):
        self._fit_model_pca(PCA(n_components=None))

    # PCA n_componenets two
    def test_pca_converter_two(self):
        self._fit_model_pca(PCA(n_components=2))

    # PCA n_componenets mle and whiten true
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < LooseVersion("0.23.2"),
        reason="With Sklearn version < 0.23.2 returns ValueError: math domain error (https://github.com/scikit-learn/scikit-learn/issues/4441)",
    )
    def test_pca_converter_mle_whiten(self):
        self._fit_model_pca(PCA(n_components="mle", whiten=True))

    # PCA n_componenets mle and solver full
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < LooseVersion("0.23.2"),
        reason="With Sklearn version < 0.23.2 returns ValueError: math domain error (https://github.com/scikit-learn/scikit-learn/issues/4441)",
    )
    def test_pca_converter_mle_full(self):
        self._fit_model_pca(PCA(n_components="mle", svd_solver="full"))

    # PCA n_componenets none and solver arpack
    def test_pca_converter_none_arpack(self):
        self._fit_model_pca(PCA(n_components=None, svd_solver="arpack"))

    # PCA n_componenets none and solver randomized
    def test_pca_converter_none_randomized(self):
        self._fit_model_pca(PCA(n_components=None, svd_solver="randomized"))

    # KernelPCA linear kernel
    def test_kernel_pca_converter_linear(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="linear"))

    # KernelPCA linear kernel with inverse transform
    def test_kernel_pca_converter_linear_fit_inverse_transform(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="linear", fit_inverse_transform=True))

    # KernelPCA poly kernel
    def test_kernel_pca_converter_poly(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="poly", degree=2))

    # KernelPCA poly kernel coef0
    def test_kernel_pca_converter_poly_coef0(self):
        self._fit_model_pca(KernelPCA(n_components=10, kernel="poly", degree=3, coef0=10))

    # KernelPCA poly kernel with inverse transform
    def test_kernel_pca_converter_poly_fit_inverse_transform(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="poly", degree=3, fit_inverse_transform=True))

    # KernelPCA poly kernel
    def test_kernel_pca_converter_rbf(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="rbf"))

    # KernelPCA sigmoid kernel
    def test_kernel_pca_converter_sigmoid(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="sigmoid"))

    # KernelPCA cosine kernel
    def test_kernel_pca_converter_cosine(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="cosine"))

    # KernelPCA precomputed kernel
    def test_kernel_pca_converter_precomputed(self):
        self._fit_model_pca(KernelPCA(n_components=5, kernel="precomputed"), precompute=True)

    # FastICA converter with n_components none
    # @unittest.skipIf(
    #     LooseVersion(sklearn.__version__) < LooseVersion("0.23.2"),
    #     reason="With Sklearn version < 0.23.2 returns ValueError: array must not contain infs or NaNs",
    # )
    def test_fast_ica_converter_none(self):
        self._fit_model_pca(FastICA(n_components=None, whiten=True))

    # FastICA converter with n_components 3
    def test_fast_ica_converter_3(self):
        self._fit_model_pca(FastICA(n_components=3))

    # FastICA converter with n_components 3 whiten
    def test_fast_ica_converter_3_whiten(self):
        self._fit_model_pca(FastICA(n_components=3, whiten=True))

    # FastICA converter with n_components 3 deflation algorithm
    def test_fast_ica_converter_3_deflation(self):
        self._fit_model_pca(FastICA(n_components=3, algorithm="deflation"))

    # FastICA converter with n_components 3 fun exp
    def test_fast_ica_converter_3_exp(self):
        self._fit_model_pca(FastICA(n_components=3, fun="exp"))

    # FastICA converter with n_components 3 fun cube
    def test_fast_ica_converter_3_cube(self):
        self._fit_model_pca(FastICA(n_components=3, fun="cube"))

    # FastICA converter with n_components 3 fun custom
    def test_fast_ica_converter_3_custom(self):
        def my_g(x):
            return x ** 3, (3 * x ** 2).mean(axis=-1)

        self._fit_model_pca(FastICA(n_components=3, fun=my_g))

    # TruncatedSVD converter with n_components 3
    def test_truncated_svd_converter_3(self):
        self._fit_model_pca(TruncatedSVD(n_components=3))

    # TruncatedSVD converter with n_components 3 algorithm arpack
    def test_truncated_svd_converter_3_arpack(self):
        self._fit_model_pca(TruncatedSVD(n_components=3, algorithm="arpack"))


if __name__ == "__main__":
    unittest.main()
