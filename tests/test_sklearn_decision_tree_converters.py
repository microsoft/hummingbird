"""
Tests Sklearn RandomForest, DecisionTree, ExtraTrees converters.
"""
import unittest
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import hummingbird.ml
from hummingbird.ml.exceptions import MissingConverter
from tree_utils import dt_implementation_map


class TestSklearnRandomForestConverter(unittest.TestCase):
    # Check tree implementation
    def test_random_forest_implementation(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(1, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=1)

        for model in [RandomForestClassifier(n_estimators=1, max_depth=1), RandomForestRegressor(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(model, "torch", extra_config={"tree_implementation": extra_config_param})
                self.assertTrue(torch_model is not None)
                self.assertTrue(
                    str(type(list(torch_model.operator_map.values())[0])) == dt_implementation_map[extra_config_param]
                )

    def _run_random_forest_classifier_converter(self, num_classes, extra_config={}, labels_shift=0):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Binary classifier
    def test_random_forest_classifier_binary_converter(self):
        self._run_random_forest_classifier_converter(2)

    # Gemm classifier
    def test_random_forest_gemm_classifier_converter(self):
        self._run_random_forest_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier
    def test_random_forest_tree_trav_classifier_converter(self):
        self._run_random_forest_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier
    def test_random_forest_perf_tree_trav_classifier_converter(self):
        self._run_random_forest_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Multi classifier
    def test_random_forest_multi_classifier_converter(self):
        self._run_random_forest_classifier_converter(3)

    # Gemm multi classifier
    def test_random_forest_gemm_multi_classifier_converter(self):
        self._run_random_forest_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # Tree_trav multi classifier
    def test_random_forest_tree_trav_multi_classifier_converter(self):
        self._run_random_forest_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav multi classifier
    def test_random_forest_perf_tree_trav_multi_classifier_converter(self):
        self._run_random_forest_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    # Shifted classes
    def test_random_forest_classifier_shifted_labels_converter(self):
        self._run_random_forest_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "gemm"})
        self._run_random_forest_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "tree_trav"})
        self._run_random_forest_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "perf_tree_trav"})

    def _run_random_forest_regressor_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = RandomForestRegressor(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Regressor
    def test_random_forest_regressor_converter(self):
        self._run_random_forest_regressor_converter(1000)

    # Gemm regressor
    def test_random_forest_gemm_regressor_converter(self):
        self._run_random_forest_regressor_converter(1000, extra_config={"tree_implementation": "gemm"})

    # Tree_trav regressor
    def test_random_forest_tree_trav_regressor_converter(self):
        self._run_random_forest_regressor_converter(1000, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav regressor
    def test_random_forest_perf_tree_trav_regressor_converter(self):
        self._run_random_forest_regressor_converter(1000, extra_config={"tree_implementation": "perf_tree_trav"})

    # Used for DecisionTreeClassifier and ExtraTreesClassifier
    def _run_test_other_trees_classifier(self, model):
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    def test_decision_tree_classifier_converter(self):
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = DecisionTreeClassifier(max_depth=max_depth)
            self._run_test_other_trees_classifier(model)

    def test_extra_trees_classifier_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = ExtraTreesClassifier(n_estimators=10, max_depth=max_depth)
            self._run_test_other_trees_classifier(model)

    # Small tree
    def test_random_forest_classifier_single_node_tree_converter(self):
        warnings.filterwarnings("ignore")
        for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
            X = np.random.rand(1, 1)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(1, size=1)
            model = RandomForestClassifier(n_estimators=1).fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config={"tree_implementation": extra_config_param})
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Failure Cases
    def test_random_forest_classifier_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = RandomForestClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch")

    def test_random_forest_classifier_raises_wrong_extra_config(self):
        warnings.filterwarnings("ignore")
        X = np.array(np.random.rand(100, 200), dtype=np.float32)
        y = np.random.randint(3, size=100)
        model = RandomForestClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(
            MissingConverter, hummingbird.ml.convert, model, "torch", extra_config={"tree_implementation": "nonsense"}
        )


if __name__ == "__main__":
    unittest.main()
