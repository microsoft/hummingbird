"""
Tests LightGBM converters.
"""
import unittest
import warnings

import numpy as np

import hummingbird.ml
from hummingbird.ml._utils import lightgbm_installed, onnx_runtime_installed, tvm_installed
from tree_utils import gbdt_implementation_map
from sklearn.datasets import make_classification, make_regression

if lightgbm_installed():
    import lightgbm as lgb


class TestLGBMConverter(unittest.TestCase):
    # Check tree implementation
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_implementation(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(10, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)

        for model in [lgb.LGBMClassifier(n_estimators=1, max_depth=1), lgb.LGBMRegressor(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(model, "torch", extra_config={"tree_implementation": extra_config_param})
                self.assertIsNotNone(torch_model)
                self.assertEqual(str(type(list(torch_model.model._operators)[0])), gbdt_implementation_map[extra_config_param])

    def _run_lgbm_classifier_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Binary classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_binary_classifier_converter(self):
        self._run_lgbm_classifier_converter(2)

    # Gemm classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_gemm_classifier_converter(self):
        self._run_lgbm_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_tree_trav_classifier_converter(self):
        self._run_lgbm_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_perf_tree_trav_classifier_converter(self):
        self._run_lgbm_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Multi classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3)

    # Gemm multi classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_gemm_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # Tree_trav multi classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_tree_trav_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav multi classifier
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_perf_tree_trav_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    def _run_lgbm_ranker_converter(self, num_classes, extra_config={}, label_gain=None):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMRanker(n_estimators=10, max_depth=max_depth, label_gain=label_gain)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y, group=[X.shape[0]], eval_set=[(X, y)], eval_group=[X.shape[0]])

            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Ranker - small, no label gain
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_binary_ranker_converter_no_label(self):
        self._run_lgbm_ranker_converter(30)

    # Ranker
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_binary_ranker_converter(self):
        self._run_lgbm_ranker_converter(1000, label_gain=list(range(1000)))

    # Gemm ranker
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_gemm_ranker_converter(self):
        self._run_lgbm_ranker_converter(1000, extra_config={"tree_implementation": "gemm"}, label_gain=list(range(1000)))

    # Tree_trav ranker
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_tree_trav_ranker_converter(self):
        self._run_lgbm_ranker_converter(1000, extra_config={"tree_implementation": "tree_trav"}, label_gain=list(range(1000)))

    # Perf_tree_trav ranker
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_perf_tree_trav_ranker_converter(self):
        self._run_lgbm_ranker_converter(
            1000, extra_config={"tree_implementation": "perf_tree_trav"}, label_gain=list(range(1000))
        )

    def _run_lgbm_regressor_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Regressor
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_binary_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000)

    # Gemm regressor
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_gemm_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "gemm"})

    # Tree_trav regressor
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_tree_trav_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav regressor
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_perf_tree_trav_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "perf_tree_trav"})

    # Float 64 classification test helper
    def _run_float64_lgbm_classifier_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Gemm classifier (float64 data)
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_float64_lgbm_gemm_classifier_converter(self):
        self._run_float64_lgbm_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier (float64 data)
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_float64_lgbm_tree_trav_classifier_converter(self):
        self._run_float64_lgbm_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier (float64 data)
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_float64_lgbm_perf_tree_trav_classifier_converter(self):
        self._run_float64_lgbm_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Float 64 regression test helper
    def _run_float64_lgbm_regressor_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Gemm regressor (float64 data)
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_float64_lgbm_gemm_regressor_converter(self):
        self._run_float64_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "gemm"})

    # Tree_trav regressor (float64 data)
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_float64_lgbm_tree_trav_regressor_converter(self):
        self._run_float64_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav regressor (float64 data)
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_float64_lgbm_perf_tree_trav_regressor_converter(self):
        self._run_float64_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "perf_tree_trav"})

    # Random forest in lgbm
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_classifier_random_forest(self):
        warnings.filterwarnings("ignore")

        model = lgb.LGBMClassifier(boosting_type="rf", n_estimators=128, max_depth=5, subsample=0.3, bagging_freq=1)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test Tweedie loss in lgbm
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_tweedie(self):
        warnings.filterwarnings("ignore")
        model = lgb.LGBMRegressor(objective="tweedie", n_estimators=2, max_depth=5)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(100, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Missing values test.
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_run_lgbm_classifier_w_missing_vals_converter(self):
        warnings.filterwarnings("ignore")
        for extra_config_param in ["gemm", "tree_trav", "perf_tree_trav"]:
            for missing in [None, np.nan]:
                for model_class, n_classes in zip([lgb.LGBMClassifier, lgb.LGBMClassifier, lgb.LGBMRegressor], [2, 3, None]):
                    model = model_class(use_missing=True, zero_as_missing=False)
                    # Missing values during training + inference.
                    if model_class == lgb.LGBMClassifier:
                        X, y = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_repeated=0, n_classes=n_classes, random_state=2021)
                    else:
                        X, y = make_regression(n_samples=100, n_features=3, n_informative=3, random_state=2021)
                    X[:25][y[:25] == 0, 0] = np.nan if missing is None else missing
                    model.fit(X, y)
                    torch_model = hummingbird.ml.convert(model, "torch", X, extra_config={"tree_implementation": extra_config_param})
                    self.assertIsNotNone(torch_model)
                    if model_class == lgb.LGBMClassifier:
                        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)
                    else:
                        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

                    # Missing values during only inference.
                    model = model_class(use_missing=True, zero_as_missing=False)
                    if model_class == lgb.LGBMClassifier:
                        X, y = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_repeated=0, n_classes=n_classes, random_state=2021)
                    else:
                        X, y = make_regression(n_samples=100, n_features=3, n_informative=3, random_state=2021)
                    model.fit(X, y)
                    torch_model = hummingbird.ml.convert(model, "torch", X, extra_config={"tree_implementation": extra_config_param})
                    X[:25][y[:25] == 0, 0] = np.nan if missing is None else missing
                    self.assertIsNotNone(torch_model)
                    if model_class == lgb.LGBMClassifier:
                        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)
                    else:
                        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Backend tests.
    # Test TorchScript backend regression.
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_regressor_converter_torchscript(self):
        warnings.filterwarnings("ignore")

        for max_depth in [1, 3, 8, 10, 12]:
            model = lgb.LGBMRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(1000, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torchscript", X, extra_config={})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test TorchScript backend classification.
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_classifier_converter_torchscript(self):
        warnings.filterwarnings("ignore")

        for max_depth in [1, 3, 8, 10, 12]:
            model = lgb.LGBMClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(2, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torchscript", X, extra_config={})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Check that we can export into ONNX.
    @unittest.skipIf(not onnx_runtime_installed(), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_onnx(self):
        warnings.filterwarnings("ignore")

        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)

        # Create ONNX model
        onnx_model = hummingbird.ml.convert(model, "onnx", X)

        np.testing.assert_allclose(onnx_model.predict(X).flatten(), model.predict(X))

    # TVM backend tests.
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_lightgbm_tvm_regressor(self):
        warnings.filterwarnings("ignore")

        for tree_implementation in ["gemm", "tree_trav", "perf_tree_trav"]:
            X = [[0, 1], [1, 1], [2, 0]]
            X = np.array(X, dtype=np.float32)
            y = np.array([100, -10, 50], dtype=np.float32)
            model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
            model.fit(X, y)

            # Create TVM model.
            tvm_model = hummingbird.ml.convert(model, "tvm", X, extra_config={"tree_implementation": tree_implementation})

            # Check results.
            np.testing.assert_allclose(tvm_model.predict(X), model.predict(X))

    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM installed")
    def test_lightgbm_tvm_classifier(self):
        warnings.filterwarnings("ignore")

        for tree_implementation in ["gemm", "tree_trav", "perf_tree_trav"]:
            X = [[0, 1], [1, 1], [2, 0]]
            X = np.array(X, dtype=np.float32)
            y = np.array([0, 1, 0], dtype=np.float32)
            model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
            model.fit(X, y)

            # Create TVM model.
            tvm_model = hummingbird.ml.convert(model, "tvm", X, extra_config={"tree_implementation": tree_implementation})

            # Check results.
            np.testing.assert_allclose(tvm_model.predict(X), model.predict(X))
            np.testing.assert_allclose(tvm_model.predict_proba(X), model.predict_proba(X))

    # Test TVM with large input datasets.
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM installed")
    def test_lightgbm_tvm_classifier_large_dataset(self):
        warnings.filterwarnings("ignore")

        for tree_implementation in ["gemm", "tree_trav", "perf_tree_trav"]:
            size = 200000
            X = np.random.rand(size, 28)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(2, size=size)
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=3)
            model.fit(X, y)

            # Create TVM model.
            tvm_model = hummingbird.ml.convert(model, "tvm", X, extra_config={"tree_implementation": tree_implementation})

            # Check results.
            np.testing.assert_allclose(tvm_model.predict(X), model.predict(X))
            np.testing.assert_allclose(tvm_model.predict_proba(X), model.predict_proba(X), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
