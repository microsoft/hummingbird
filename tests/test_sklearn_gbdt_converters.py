"""
Tests Sklearn GradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

from hummingbird import convert_sklearn


class TestSklearnGradientBoostingClassifier(unittest.TestCase):
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

    # binary classifier
    def test_GBDT_classifier_binary_converter(self):
        self._run_GB_trees_classifier_converter(2)

    # gemm classifier
    def test_GBDT_gemm_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # tree_trav classifier
    def test_GBDT_tree_trav_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # perf_tree_trav classifier
    def test_GBDT_perf_tree_trav_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # multi classifier
    def test_GBDT_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3)

    # gemm multi classifier
    def test_GBDT_gemm_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # tree_trav multi classifier
    def test_GBDT_tree_trav_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # perf_tree_trav multi classifier
    def test_GBDT_perf_tree_trav_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    # shifted classes
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

    def test_GB_trees_classifier_converter_predict(self):
        warnings.filterwarnings("ignore")

        model = GradientBoostingClassifier(n_estimators=10, max_depth=8)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model)
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(
            model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-06, atol=1e-06
        )


if __name__ == "__main__":
    unittest.main()
