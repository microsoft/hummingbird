"""
Tests Sklearn RandomForest, DecisionTree, ExtraTrees converters.
"""
import unittest
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import hummingbird.ml
from hummingbird.ml.exceptions import MissingConverter
from tree_utils import dt_implementation_map


class TestSklearnTreeConverter(unittest.TestCase):
    # Check tree implementation
    def test_random_forest_implementation(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(1, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=1)

        for model in [RandomForestClassifier(n_estimators=1, max_depth=1), RandomForestRegressor(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                pytorch_model = hummingbird.ml.convert(
                    model, "pytorch", extra_config={"tree_implementation": extra_config_param}
                )
                self.assertTrue(pytorch_model is not None)
                self.assertTrue(
                    str(type(list(pytorch_model.operator_map.values())[0])) == dt_implementation_map[extra_config_param]
                )

    # Used for classification tests
    def _run_tree_classification_converter(self, model_type, num_classes, extra_config={}, labels_shift=0, **kwargs):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model = model_type(max_depth=max_depth, **kwargs)
            model.fit(X, y)
            pytorch_model = hummingbird.ml.convert(model, "pytorch", extra_config=extra_config)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), pytorch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Random forest binary classifier
    def test_random_forest_classifier_binary_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 2, n_estimators=10)

    # Random forest gemm classifier
    def test_random_forest_gemm_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 2, extra_config={"tree_implementation": "gemm"}, n_estimators=10)

    # Random forest tree_trav classifier
    def test_random_forest_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 2, extra_config={"tree_implementation": "tree_trav"}, n_estimators=10)

    # Random forest perf_tree_trav classifier
    def test_random_forest_perf_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 2, extra_config={"tree_implementation": "perf_tree_trav"}, n_estimators=10)

    # Random forest multi classifier
    def test_random_forest_multi_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, n_estimators=10)

    # Random forest gemm multi classifier
    def test_random_forest_gemm_multi_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, extra_config={"tree_implementation": "gemm"}, n_estimators=10)

    # Random forest tree_trav multi classifier
    def test_random_forest_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, extra_config={"tree_implementation": "tree_trav"}, n_estimators=10)

    # Random forest perf_tree_trav multi classifier
    def test_random_forest_perf_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, extra_config={"tree_implementation": "perf_tree_trav"}, n_estimators=10)

    # Random forest gemm classifier shifted classes
    def test_random_forest_gemm_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, labels_shift=2, extra_config={"tree_implementation": "gemm"}, n_estimators=10)

    # Random forest tree_trav classifier shifted classes
    def test_random_forest_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, labels_shift=2, extra_config={"tree_implementation": "tree_trav"}, n_estimators=10)

    # Random forest perf_tree_trav classifier shifted classes
    def test_random_forest_perf_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, labels_shift=2, extra_config={"tree_implementation": "perf_tree_trav"}, n_estimators=10)

    # Used for regression tests
    def _run_tree_regressor_converter(self, model_type, num_classes, extra_config={}, **kwargs):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = model_type(max_depth=max_depth, **kwargs)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            pytorch_model = hummingbird.ml.convert(model, "pytorch", extra_config=extra_config)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict(X), pytorch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Random forest regressor
    def test_random_forest_regressor_converter(self):
        self._run_tree_regressor_converter(RandomForestRegressor, 1000, n_estimators=10)

    # Random forest gemm regressor
    def test_random_forest_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(RandomForestRegressor, 1000, extra_config={"tree_implementation": "gemm"}, n_estimators=10)

    # Random forest tree_trav regressor
    def test_random_forest_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(RandomForestRegressor, 1000, extra_config={"tree_implementation": "tree_trav"}, n_estimators=10)

    # Random forest perf_tree_trav regressor
    def test_random_forest_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(RandomForestRegressor, 1000, extra_config={"tree_implementation": "perf_tree_trav"}, n_estimators=10)

    # Extra trees regressor
    def test_extra_trees_regressor_converter(self):
        self._run_tree_regressor_converter(ExtraTreesRegressor, 1000, n_estimators=10)

    # Extra trees gemm regressor
    def test_extra_trees_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(ExtraTreesRegressor, 1000, extra_config={"tree_implementation": "gemm"}, n_estimators=10)

    # Extra trees tree_trav regressor
    def test_extra_trees_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(ExtraTreesRegressor, 1000, extra_config={"tree_implementation": "tree_trav"}, n_estimators=10)

    # Extra trees perf_tree_trav regressor
    def test_extra_trees_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(ExtraTreesRegressor, 1000, extra_config={"tree_implementation": "perf_tree_trav"}, n_estimators=10)

    # Decision tree regressor
    def test_decision_tree_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000)

    # Decision tree gemm regressor
    def test_decision_tree_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000, extra_config={"tree_implementation": "gemm"})

    # Decision tree tree_trav regressor
    def test_decision_tree_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000, extra_config={"tree_implementation": "tree_trav"})

    # Decision tree perf_tree_trav regressor
    def test_decision_tree_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000, extra_config={"tree_implementation": "perf_tree_trav"})

    # Decision tree classifier
    def test_decision_tree_classifier_converter(self):
        self._run_tree_classification_converter(DecisionTreeClassifier, 3)

    # Extra trees classifier
    def test_extra_trees_classifier_converter(self):
        self._run_tree_classification_converter(ExtraTreesClassifier, 3, n_estimators=10)

    # Used for small tree tests
    def _run_random_forest_classifier_single_node_tree_converter(self, extra_config={}):
        warnings.filterwarnings("ignore")
        X = np.random.rand(1, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(1, size=1)
        model = RandomForestClassifier(n_estimators=1).fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch", extra_config=extra_config)
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), pytorch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Small tree gemm implementation
    def test_random_forest_gemm_classifier_single_node_tree_converter(self):
        self._run_random_forest_classifier_single_node_tree_converter(extra_config={"tree_implementation": "gemm"})

    # Small tree tree_trav implementation
    def test_random_forest_tree_trav_classifier_single_node_tree_converter(self):
        self._run_random_forest_classifier_single_node_tree_converter(extra_config={"tree_implementation": "tree_trav"})

    # Small tree perf_tree_trav implementation
    def test_random_forest_perf_tree_trav_classifier_single_node_tree_converter(self):
        self._run_random_forest_classifier_single_node_tree_converter(extra_config={"tree_implementation": "perf_tree_trav"})

    # Failure Cases
    def test_random_forest_classifier_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = RandomForestClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "pytorch")

    def test_random_forest_classifier_raises_wrong_extra_config(self):
        warnings.filterwarnings("ignore")
        X = np.array(np.random.rand(100, 200), dtype=np.float32)
        y = np.random.randint(3, size=100)
        model = RandomForestClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(
            MissingConverter, hummingbird.ml.convert, model, "pytorch", extra_config={"tree_implementation": "nonsense"}
        )


if __name__ == "__main__":
    unittest.main()
