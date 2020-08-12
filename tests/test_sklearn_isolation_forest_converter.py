"""
Tests Sklearn IsolationForest converter.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

import hummingbird.ml
from tree_utils import iforest_implementation_map


class TestIsolationForestConverter(unittest.TestCase):
    # Check tree implementation
    def test_iforest_implementation(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(10, 1)
        X = np.array(X, dtype=np.float32)
        model = IsolationForest(n_estimators=1, max_samples=2)
        for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
            model.fit(X)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config={"tree_implementation": extra_config_param})
            self.assertIsNotNone(torch_model)
            self.assertEqual(
                str(type(list(torch_model.operator_map.values())[0])), iforest_implementation_map[extra_config_param]
            )

    def _run_isolation_forest_converter(self, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_samples in [2 ** 1, 2 ** 3, 2 ** 8, 2 ** 10, 2 ** 12]:
            model = IsolationForest(n_estimators=10, max_samples=max_samples)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            model.fit(X)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.decision_function(X), torch_model.decision_function(X), rtol=1e-06, atol=1e-06)
            np.testing.assert_allclose(model.score_samples(X), torch_model.score_samples(X), rtol=1e-06, atol=1e-06)
            np.testing.assert_array_equal(model.predict(X), torch_model.predict(X))

    # Isolation Forest
    def test_isolation_forest_converter(self):
        self._run_isolation_forest_converter()

    # Gemm Isolation Forest
    def test_isolation_forest_gemm_converter(self):
        self._run_isolation_forest_converter(extra_config={"tree_implementation": "gemm"})

    # Tree_trav Isolation Forest
    def test_isolation_forest_tree_trav_converter(self):
        self._run_isolation_forest_converter(extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav Isolation Forest
    def test_isolation_forest_perf_tree_trav_converter(self):
        self._run_isolation_forest_converter(extra_config={"tree_implementation": "perf_tree_trav"})

    # Float 64 data tests
    def test_float64_isolation_forest_converter(self):
        warnings.filterwarnings("ignore")
        for max_samples in [2 ** 1, 2 ** 3, 2 ** 8, 2 ** 10, 2 ** 12]:
            model = IsolationForest(n_estimators=10, max_samples=max_samples)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            model.fit(X)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config={})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.decision_function(X), torch_model.decision_function(X), rtol=1e-06, atol=1e-06)
            np.testing.assert_allclose(model.score_samples(X), torch_model.score_samples(X), rtol=1e-06, atol=1e-06)
            np.testing.assert_array_equal(model.predict(X), torch_model.predict(X))


if __name__ == "__main__":
    unittest.main()
