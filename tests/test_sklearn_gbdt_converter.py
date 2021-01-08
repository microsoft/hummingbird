"""
Tests Sklearn GradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import hummingbird.ml
from tree_utils import gbdt_implementation_map


class TestSklearnGradientBoostingConverter(unittest.TestCase):
    # Check tree implementation
    def test_gbdt_implementation(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(10, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)

        for model in [
            GradientBoostingClassifier(n_estimators=1, max_depth=1),
            GradientBoostingRegressor(n_estimators=1, max_depth=1),
        ]:

            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(model, "torch", extra_config={"tree_implementation": extra_config_param})
                self.assertIsNotNone(torch_model)
                self.assertEqual(str(type(list(torch_model.model._operators)[0])), gbdt_implementation_map[extra_config_param])

    def _run_GB_trees_classifier_converter(self, num_classes, extra_config={}, labels_shift=0):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    def _run_GB_trees_regressor_converter(self, extra_config=None):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200).astype(np.float32)
            y = np.random.normal(size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config or {})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

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

    # Regressor
    def test_GBDT_regressor_converter(self):
        self._run_GB_trees_regressor_converter()

    # Gemm regressor
    def test_GBDT_gemm_regressor_converter(self):
        self._run_GB_trees_regressor_converter(extra_config={"tree_implementation": "gemm"})

    # Tree_trav regressor
    def test_GBDT_tree_trav_regressor_converter(self):
        self._run_GB_trees_regressor_converter(extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav regressor
    def test_GBDT_perf_tree_trav_regressor_converter(self):
        self._run_GB_trees_regressor_converter(extra_config={"tree_implementation": "perf_tree_trav"})

    def test_zero_init_GB_trees_classifier_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth, init="zero")
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(3, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch")
            self.assertIsNotNone(torch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    def test_zero_init_GB_trees_regressor_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth, init="zero")
            np.random.seed(0)
            X = np.random.rand(100, 200).astype(np.float32)
            y = np.random.normal(size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch")
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Float 64 data tests
    def test_float64_GB_trees_classifier_converter(self):
        warnings.filterwarnings("ignore")
        num_classes = 3
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config={})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    def test_float64_GB_trees_regressor_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.normal(size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config={})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Failure Cases
    def test_sklearn_GBDT_classifier_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch")


if __name__ == "__main__":
    unittest.main()
