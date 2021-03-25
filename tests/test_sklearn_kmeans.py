"""
Tests sklearn KMeans converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans

import hummingbird.ml
from hummingbird.ml import constants
from hummingbird.ml._utils import tvm_installed


class TestSklearnKMeans(unittest.TestCase):
    # KMeans test function to be parameterized
    def _test_kmeans(self, n_clusters, algorithm="full", backend="torch", extra_config={}):
        model = KMeans(n_clusters=n_clusters, algorithm=algorithm, random_state=0)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)

        model.fit(X)
        torch_model = hummingbird.ml.convert(model, backend, X, extra_config=extra_config)
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # KMeans 2 full
    def test_kmeans_2_full(self):
        self._test_kmeans(2)

    # KMeans 5 full
    def test_kmeans_5_full(self):
        self._test_kmeans(5)

    # KMeans 2 auto
    def test_kmeans_2_auto(self):
        self._test_kmeans(2, "auto")

    # KMeans 5 full
    def test_kmeans_5_auto(self):
        self._test_kmeans(5, "auto")

    # KMeans 2 elkan
    def test_kmeans_2_elkan(self):
        self._test_kmeans(2, "elkan")

    # KMeans 5 elkan
    def test_kmeans_5_elkan(self):
        self._test_kmeans(5, "elkan")


if __name__ == "__main__":
    unittest.main()
