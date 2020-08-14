"""
Tests LightGBM converters.
"""
import unittest
import warnings

import numpy as np

import hummingbird.ml
from hummingbird.ml._utils import lightgbm_installed, onnx_runtime_installed
from tree_utils import gbdt_implementation_map

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
                self.assertEqual(
                    str(type(list(torch_model.model._operator_map.values())[0])), gbdt_implementation_map[extra_config_param]
                )

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

    # Test TorchScript backend regression.
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_regressor_converter_torchscript(self):
        warnings.filterwarnings("ignore")
        import torch

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
        import torch

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
    @unittest.skipIf(not (onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_onnx(self):
        import onnxruntime as ort

        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)

        # Create ONNX model
        onnx_model = hummingbird.ml.convert(model, "onnx", X)

        # Get the predictions for the ONNX-ML model
        onnx_pred = onnx_model.predict(X)

        np.testing.assert_allclose(onnx_pred[0].flatten(), model.predict(X))


if __name__ == "__main__":
    unittest.main()
