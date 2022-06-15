"""
Tests Sklearn HistGradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

import hummingbird.ml
from tree_utils import gbdt_implementation_map


class TestSklearnHistGradientBoostingClassifier(unittest.TestCase):
    def _run_GB_trees_classifier_converter(self, num_classes, extra_config={}, labels_shift=0):
        warnings.filterwarnings("ignore")
        for max_depth in [2, 3, 8, 10, 12, None]:
            model = HistGradientBoostingClassifier(max_iter=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    def _run_GB_trees_regressor_converter(self, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [2, 3, 8, 10, 12, None]:
            model = HistGradientBoostingRegressor(max_iter=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200).astype(np.float32)
            y = np.random.normal(size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    def test_float64_GB_trees_regressor_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [2, 3, 8, 10, 12, None]:
            model = HistGradientBoostingRegressor(max_iter=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.normal(size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config={})
            self.assertIsNotNone(torch_model)
            model.predict(X)
            torch_model.predict(X)


if __name__ == "__main__":
    unittest.main()
