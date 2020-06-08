"""
Tests Sklearn HistGradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

import hummingbird.ml
from tree_utils import gbdt_implementation_map


class TestSklearnHistGradientBoostingClassifier(unittest.TestCase):
    def _run_GB_trees_classifier_converter(self, num_classes, extra_config={}, labels_shift=0):
        warnings.filterwarnings("ignore")
        for max_depth in [2, 3, 8, 10, 12, None]:
            model = HistGradientBoostingClassifier(max_iter=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model.fit(X, y)
            pytorch_model = hummingbird.ml.convert(model, "pytorch", extra_config=extra_config)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), pytorch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Binary classifier
    def test_HistGBDT_classifier_binary_converter(self):
        self._run_GB_trees_classifier_converter(2)

    # Gemm classifier
    def test_HistGBDT_gemm_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier
    def test_HistGBDT_tree_trav_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier
    def test_HistGBDT_perf_tree_trav_classifier_converter(self):
        self._run_GB_trees_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Multi classifier
    def test_HistGBDT_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3)

    # Gemm multi classifier
    def test_HistGBDT_gemm_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # Tree_trav multi classifier
    def test_HistGBDT_tree_trav_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav multi classifier
    def test_HistGBDT_perf_tree_trav_multi_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    # Shifted classes
    def test_HistGBDT_shifted_labels_converter(self):
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "gemm"})
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "tree_trav"})
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Failure Cases
    def test_HistGBDT_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = HistGradientBoostingClassifier(max_iter=10).fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "pytorch")


if __name__ == "__main__":
    unittest.main()
