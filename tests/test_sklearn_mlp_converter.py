"""
Tests sklearn MLP models (MLPClassifier, MLPRegressor) converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier, MLPRegressor

import hummingbird.ml


class TestSklearnMLPClassifier(unittest.TestCase):

    # MLPClassifier test function to be parameterized
    def _test_mlp_classifer(self, num_classes, activation="relu", labels_shift=0):
        model = MLPClassifier(hidden_layer_sizes=(100, 100, 50,), activation=activation)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # MLPClassifier binary
    def test_mlp_classifer_bi(self):
        self._test_mlp_classifer(2)

    # MLPClassifier multi-class
    def test_mlp_classifer_multi(self):
        self._test_mlp_classifer(3)

    # MLPClassifier multi-class w/ shifted labels
    def test_mlp_classifer_multi_shifted_labels(self):
        self._test_mlp_classifer(3, labels_shift=3)

    #  MLPClassifier multi-class w/ tanh activation
    def test_mlp_classifer_multi_logistic(self):
        self._test_mlp_classifer(3, activation="tanh")

    #  MLPClassifier multi-class w/ logistic activation
    def test_mlp_classifer_multi_tanh(self):
        self._test_mlp_classifer(3, activation="logistic")

    #  MLPClassifier multi-class w/ identity activation
    def test_mlp_classifer_multi_identity(self):
        self._test_mlp_classifer(3, activation="identity")

    # MLPRegressor test function to be parameterized
    def _test_mlp_regressor(self, activation="relu"):
        model = MLPRegressor(hidden_layer_sizes=(100, 100, 50,), activation=activation)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.rand(100)

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # MLPRegressor
    def test_mlp_regressor(self):
        self._test_mlp_regressor()

    #  MLPRegressor w/ tanh activation
    def test_mlp_regressor_multi_logistic(self):
        self._test_mlp_regressor(activation="tanh")

    #  MLPRegressor w/ logistic activation
    def test_mlp_regressor_multi_tanh(self):
        self._test_mlp_regressor(activation="logistic")

    #  MLPRegressor multi-class w/ identity activation
    def test_mlp_regressor_multi_identity(self):
        self._test_mlp_regressor(activation="identity")


if __name__ == "__main__":
    unittest.main()
