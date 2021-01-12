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
from hummingbird.ml._utils import tvm_installed
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

        for model in [RandomForestClassifier(n_estimators=1, max_depth=1), RandomForestRegressor(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(
                    model, "torch", extra_config={constants.TREE_IMPLEMENTATION: extra_config_param}
                )
                self.assertIsNotNone(torch_model)
                self.assertTrue(str(type(list(torch_model.model._operators)[0])) == dt_implementation_map[extra_config_param])

    # Used for classification tests
    def _run_tree_classification_converter(
        self, model_type, num_classes, backend="torch", extra_config={}, labels_shift=0, **kwargs
    ):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model = model_type(max_depth=max_depth, **kwargs)
            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, backend, X, extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Random forest binary classifier
    def test_random_forest_classifier_binary_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 2, n_estimators=10)

    # Random forest gemm classifier
    def test_random_forest_gemm_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 2, extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav classifier
    def test_random_forest_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 2, extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Random forest perf_tree_trav classifier
    def test_random_forest_perf_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 2, extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}, n_estimators=10
        )

    # Random forest multi classifier
    def test_random_forest_multi_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, n_estimators=10)

    # Random forest gemm multi classifier
    def test_random_forest_gemm_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 3, extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav multi classifier
    def test_random_forest_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 3, extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Random forest perf_tree_trav multi classifier
    def test_random_forest_perf_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 3, extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}, n_estimators=10
        )

    # Random forest gemm classifier shifted classes
    def test_random_forest_gemm_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 3, labels_shift=2, extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav classifier shifted classes
    def test_random_forest_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"},
            n_estimators=10,
        )

    # Random forest perf_tree_trav classifier shifted classes
    def test_random_forest_perf_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"},
            n_estimators=10,
        )

    # Used for regression tests
    def _run_tree_regressor_converter(self, model_type, num_classes, backend="torch", extra_config={}, **kwargs):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = model_type(max_depth=max_depth, **kwargs)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, backend, X, extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Random forest regressor
    def test_random_forest_regressor_converter(self):
        self._run_tree_regressor_converter(RandomForestRegressor, 1000, n_estimators=10)

    # Random forest gemm regressor
    def test_random_forest_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav regressor
    def test_random_forest_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Random forest perf_tree_trav regressor
    def test_random_forest_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}, n_estimators=10
        )

    # Extra trees regressor
    def test_extra_trees_regressor_converter(self):
        self._run_tree_regressor_converter(ExtraTreesRegressor, 1000, n_estimators=10)

    # Extra trees gemm regressor
    def test_extra_trees_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Extra trees tree_trav regressor
    def test_extra_trees_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Extra trees perf_tree_trav regressor
    def test_extra_trees_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}, n_estimators=10
        )

    # Decision tree regressor
    def test_decision_tree_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000)

    # Decision tree gemm regressor
    def test_decision_tree_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "gemm"})

    # Decision tree tree_trav regressor
    def test_decision_tree_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}
        )

    # Decision tree perf_tree_trav regressor
    def test_decision_tree_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor, 1000, extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}
        )

    # Decision tree classifier
    def test_decision_tree_classifier_converter(self):
        self._run_tree_classification_converter(DecisionTreeClassifier, 3)

    # Extra trees classifier
    def test_extra_trees_classifier_converter(self):
        self._run_tree_classification_converter(ExtraTreesClassifier, 3, n_estimators=10)

    # Used for small tree tests
    def _run_random_forest_classifier_single_node_tree_converter(self, extra_config={}):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(1, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(1, size=1)
        model = RandomForestClassifier(n_estimators=1).fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Small tree gemm implementation
    def test_random_forest_gemm_classifier_single_node_tree_converter(self):
        self._run_random_forest_classifier_single_node_tree_converter(extra_config={constants.TREE_IMPLEMENTATION: "gemm"})

    # Small tree tree_trav implementation
    def test_random_forest_tree_trav_classifier_single_node_tree_converter(self):
        self._run_random_forest_classifier_single_node_tree_converter(
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}
        )

    # Small tree perf_tree_trav implementation
    def test_random_forest_perf_tree_trav_classifier_single_node_tree_converter(self):
        self._run_random_forest_classifier_single_node_tree_converter(
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}
        )

    # Another small tree tests
    def test_random_forest_classifier_small_tree_converter(self):
        seed = 0
        np.random.seed(seed=0)
        N = 9
        X = np.random.randn(N, 8)
        y = np.random.randint(low=0, high=2, size=N)
        model = RandomForestClassifier(random_state=seed)
        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Float 64 classification test helper
    def _run_float64_tree_classification_converter(self, model_type, num_classes, extra_config={}, labels_shift=0, **kwargs):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model = model_type(max_depth=max_depth, **kwargs)
            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Random forest binary classifier (float64 data)
    def test_float64_random_forest_classifier_binary_converter(self):
        self._run_float64_tree_classification_converter(RandomForestClassifier, 2, n_estimators=10)

    # Decision tree classifier (float64 data)
    def test_float64_decision_tree_classifier_converter(self):
        self._run_float64_tree_classification_converter(DecisionTreeClassifier, 3)

    # Extra trees classifier (float64 data)
    def test_float64_extra_trees_classifier_converter(self):
        self._run_float64_tree_classification_converter(ExtraTreesClassifier, 3, n_estimators=10)

    # Float 64 regression tests helper
    def _run_float64_tree_regressor_converter(self, model_type, num_classes, extra_config={}, **kwargs):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = model_type(max_depth=max_depth, **kwargs)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Random forest regressor (float64 data)
    def test_float64_random_forest_regressor_converter(self):
        self._run_float64_tree_regressor_converter(RandomForestRegressor, 1000, n_estimators=10)

    # Decision tree regressor (float64 data)
    def test_float64_decision_tree_regressor_converter(self):
        self._run_float64_tree_regressor_converter(DecisionTreeRegressor, 1000)

    # Extra trees regressor (float64 data)
    def test_float64_extra_trees_regressor_converter(self):
        self._run_float64_tree_regressor_converter(ExtraTreesRegressor, 1000, n_estimators=10)

    # Failure Cases
    def test_random_forest_classifier_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = RandomForestClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch")

    def test_random_forest_classifier_raises_wrong_extra_config(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.array(np.random.rand(100, 200), dtype=np.float32)
        y = np.random.randint(3, size=100)
        model = RandomForestClassifier(n_estimators=10).fit(X, y)
        self.assertRaises(
            MissingConverter, hummingbird.ml.convert, model, "torch", extra_config={constants.TREE_IMPLEMENTATION: "nonsense"}
        )

    # Test trees with TorchScript backend
    # Random forest binary classifier
    def test_random_forest_ts_classifier_binary_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 2, "torch.jit", n_estimators=10)

    # Random forest gemm classifier
    def test_random_forest_ts_gemm_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 2, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav classifier
    def test_random_forest_ts_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 2, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Random forest perf_tree_trav classifier
    def test_random_forest_ts_perf_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            2,
            "torch.jit",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"},
            n_estimators=10,
        )

    # Random forest multi classifier
    def test_random_forest_ts_multi_classifier_converter(self):
        self._run_tree_classification_converter(RandomForestClassifier, 3, "torch.jit", n_estimators=10)

    # Random forest gemm multi classifier
    def test_random_forest_ts_gemm_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 3, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav multi classifier
    def test_random_forest_ts_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier, 3, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Random forest perf_tree_trav multi classifier
    def test_random_forest_ts_perf_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "torch.jit",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"},
            n_estimators=10,
        )

    # Random forest gemm classifier shifted classes
    def test_random_forest_ts_gemm_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "torch.jit",
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "gemm"},
            n_estimators=10,
        )

    # Random forest tree_trav classifier shifted classes
    def test_random_forest_ts_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "torch.jit",
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"},
            n_estimators=10,
        )

    # Random forest perf_tree_trav classifier shifted classes
    def test_random_forest_ts_perf_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "torch.jit",
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"},
            n_estimators=10,
        )

    # Random forest regressor
    def test_random_forest_ts_regressor_converter(self):
        self._run_tree_regressor_converter(RandomForestRegressor, 1000, "torch.jit", n_estimators=10)

    # Random forest gemm regressor
    def test_random_forest_ts_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor, 1000, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Random forest tree_trav regressor
    def test_random_forest_ts_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor,
            1000,
            "torch.jit",
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"},
            n_estimators=10,
        )

    # Random forest perf_tree_trav regressor
    def test_random_forest_ts_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor,
            1000,
            "torch.jit",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"},
            n_estimators=10,
        )

    # Extra trees regressor
    def test_extra_trees_ts_regressor_converter(self):
        self._run_tree_regressor_converter(ExtraTreesRegressor, 1000, "torch.jit", n_estimators=10)

    # Extra trees gemm regressor
    def test_extra_trees_ts_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor, 1000, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "gemm"}, n_estimators=10
        )

    # Extra trees tree_trav regressor
    def test_extra_trees_ts_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor, 1000, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}, n_estimators=10
        )

    # Extra trees perf_tree_trav regressor
    def test_extra_trees_ts_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor,
            1000,
            "torch.jit",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"},
            n_estimators=10,
        )

    # Decision tree regressor
    def test_decision_tree_ts_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000, "torch.jit")

    # Decision tree gemm regressor
    def test_decision_tree_ts_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor, 1000, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "gemm"}
        )

    # Decision tree tree_trav regressor
    def test_decision_tree_ts_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor, 1000, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "tree_trav"}
        )

    # Decision tree perf_tree_trav regressor
    def test_decision_tree_ts_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor, 1000, "torch.jit", extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav"}
        )

    # Decision tree classifier
    def test_decision_tree_ts_classifier_converter(self):
        self._run_tree_classification_converter(
            DecisionTreeClassifier, 3, "torch.jit",
        )

    # Extra trees classifier
    def test_extra_trees_ts_classifier_converter(self):
        self._run_tree_classification_converter(ExtraTreesClassifier, 3, "torch.jit", n_estimators=10)

    # Test trees with TVM backend
    # Random forest gemm classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_gemm_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            2,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "gemm", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest tree_trav classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            2,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest perf_tree_trav classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_perf_tree_trav_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            2,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest gemm multi classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_gemm_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "gemm", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest tree_trav multi classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest perf_tree_trav multi classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_perf_tree_trav_multi_classifier_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest gemm classifier shifted classes
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_gemm_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "tvm",
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "gemm", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest tree_trav classifier shifted classes
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "tvm",
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest perf_tree_trav classifier shifted classes
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_perf_tree_trav_classifier_shifted_labels_converter(self):
        self._run_tree_classification_converter(
            RandomForestClassifier,
            3,
            "tvm",
            labels_shift=2,
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav", constants.TVM_MAX_FUSE_DEPTH: 10},
            n_estimators=10,
        )

    # Random forest gemm regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "gemm", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest tree_trav regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Random forest perf_tree_trav regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_random_forest_tvm_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            RandomForestRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav", constants.TVM_MAX_FUSE_DEPTH: 10},
            n_estimators=10,
        )

    # Extra trees gemm regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_extra_trees_tvm_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "gemm", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Extra trees tree_trav regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_extra_trees_tvm_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
            n_estimators=10,
        )

    # Extra trees perf_tree_trav regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_extra_trees_tvm_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            ExtraTreesRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav", constants.TVM_MAX_FUSE_DEPTH: 10},
            n_estimators=10,
        )

    # Decision tree regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_decision_tree_tvm_regressor_converter(self):
        self._run_tree_regressor_converter(DecisionTreeRegressor, 1000, "tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

    # Decision tree gemm regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_decision_tree_tvm_gemm_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "gemm", constants.TVM_MAX_FUSE_DEPTH: 30},
        )

    # Decision tree tree_trav regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_decision_tree_tvm_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "tree_trav", constants.TVM_MAX_FUSE_DEPTH: 30},
        )

    # Decision tree perf_tree_trav regressor
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_decision_tree_tvm_perf_tree_trav_regressor_converter(self):
        self._run_tree_regressor_converter(
            DecisionTreeRegressor,
            1000,
            "tvm",
            extra_config={constants.TREE_IMPLEMENTATION: "perf_tree_trav", constants.TVM_MAX_FUSE_DEPTH: 10},
        )

    # Decision tree classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_decision_tree_tvm_classifier_converter(self):
        self._run_tree_classification_converter(
            DecisionTreeClassifier, 3, "tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30}
        )

    # Extra trees classifier
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_extra_trees_tvm_classifier_converter(self):
        self._run_tree_classification_converter(
            ExtraTreesClassifier, 3, "tvm", n_estimators=10, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30}
        )


if __name__ == "__main__":
    unittest.main()
