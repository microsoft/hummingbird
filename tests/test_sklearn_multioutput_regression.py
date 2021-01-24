"""
Tests sklearn MultioutputRegressor converters
"""
import unittest
import warnings
import sys
from distutils.version import LooseVersion

import numpy as np
import torch
import sklearn
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

import hummingbird.ml

import random
random.seed(2021)


class TestSklearnMultioutputRegressor(unittest.TestCase):
    # Test MultiOutputRegressor with different child learners
    def test_sklearn_multioutput_regressor(self):
        for n_targets in [2, 3, 4]:
            for model_class in [DecisionTreeRegressor, ExtraTreesRegressor, RandomForestRegressor, LinearRegression]:
                model = MultiOutputRegressor(model_class())
                X, y = datasets.make_regression(n_samples=50, n_features=10, n_informative=5, n_targets=n_targets, random_state=2020)
                X = X.astype('float32')
                y = y.astype('float32')
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(model, "torch")
                self.assertTrue(torch_model is not None)
                np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-5, atol=1e-5)

    # Test RegressorChain with different child learners
    def test_sklearn_regressor_chain(self):
        for n_targets in [2, 3, 4]:
            for model_class in [DecisionTreeRegressor, ExtraTreesRegressor, RandomForestRegressor, LinearRegression]:
                order = [i for i in range(n_targets)]
                random.shuffle(order)
                model = RegressorChain(model_class(), order=order)
                X, y = datasets.make_regression(n_samples=50, n_features=10, n_informative=5, n_targets=n_targets, random_state=2021)
                X = X.astype('float32')
                y = y.astype('float32')
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(model, "torch")
                self.assertTrue(torch_model is not None)
                np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
