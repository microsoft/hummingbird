"""
Tests Sklearn RandomForest, DecisionTree, ExtraTrees converters.
"""
import unittest
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression, make_classification

import hummingbird.ml
from hummingbird.ml import constants
from tree_utils import dt_implementation_map


class TestSklearnTreeConverter(unittest.TestCase):
    # Check tree implementation
    def test_random_forest_implementation(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(1, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=1)

        model = RandomForestClassifier(n_estimators=1, max_depth=1)
        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch", extra_config={constants.FINE_TUNE: True})
        self.assertIsNotNone(torch_model)
        self.assertTrue(str(type(list(torch_model.model._operators)[0])) == dt_implementation_map["gemm_fine_tune"])

    # Fine tune random forest classifier.
    def test_random_forest_classifier_fine_tune(self):
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=4, n_redundant=0, random_state=0, shuffle=False
        )

        model = RandomForestClassifier()
        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch", X, extra_config={constants.FINE_TUNE: True})
        self.assertIsNotNone(torch_model)

        # Do fine tuning


if __name__ == "__main__":
    unittest.main()
