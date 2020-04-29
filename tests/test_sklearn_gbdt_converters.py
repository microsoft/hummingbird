"""
Tests Sklearn GradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

from hummingbird import convert_sklearn
from tree_utils import gbdt_implementation_map


class TestSklearnGradientBoostingClassifier(unittest.TestCase):
    # Check tree implementation
    def test_gbdt_implementation(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(10, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)

        for model in [GradientBoostingClassifier(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                pytorch_model = convert_sklearn(model, extra_config={"tree_implementation": extra_config_param})
                self.assertTrue(pytorch_model is not None)
                self.assertTrue(
                    str(type(list(pytorch_model.operator_map.values())[0])) == gbdt_implementation_map[extra_config_param]
                )

    def _run_GB_trees_classifier_converter(self, num_classes, extra_config={}, labels_shift=0):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model.fit(X, y)
            pytorch_model = convert_sklearn(model, extra_config=extra_config)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(
                model.predict_proba(X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-06, atol=1e-06
            )

    # Binary classifier
    def test_GBDT_classifier_binary_converter(self):
        self._run_GB_trees_classifier_converter(2)

    # Gemm classifier
    def test_GBDT_gemm_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier
    def test_GBDT_tree_trav_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier
    def test_GBDT_perf_tree_trav_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Multi classifier
    def test_GBDT_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3)

    # Gemm multi classifier
    def test_GBDT_gemm_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # Tree_trav multi classifier
    def test_GBDT_tree_trav_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav multi classifier
    def test_GBDT_perf_tree_trav_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    # Shifted classes
    def test_GBDT_shifted_labels_converter(self):
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "gemm"})
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "tree_trav"})
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "perf_tree_trav"})

    def test_zero_init_GB_trees_classifier_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth, init="zero")
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(3, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(model)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(
                model.predict_proba(X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-06, atol=1e-06
            )

    # Failure Cases
    def test_sklearn_random_forest_classifier_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(RuntimeError, convert_sklearn, model, [])


if __name__ == "__main__":
    unittest.main()
