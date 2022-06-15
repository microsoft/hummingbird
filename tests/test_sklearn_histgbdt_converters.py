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
