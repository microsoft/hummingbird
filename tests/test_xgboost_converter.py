"""
Tests XGBoost converters.
"""
import sys
import unittest
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import hummingbird.ml
from hummingbird.ml._utils import xgboost_installed, tvm_installed, pandas_installed, is_on_github_actions
from hummingbird.ml import constants
from tree_utils import gbdt_implementation_map

if xgboost_installed():
    import xgboost as xgb
    from xgboost import testing as tm

if pandas_installed():
    import pandas as pd


class TestXGBoostConverter(unittest.TestCase):
    # Check tree implementation
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_implementation(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(1, 1)
        X = np.array(X, dtype=np.float32)
        y = np.array([0], dtype=int)

        for model in [xgb.XGBClassifier(n_estimators=1, max_depth=1), xgb.XGBRegressor(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                torch_model = hummingbird.ml.convert(
                    model, "torch", X[0:1], extra_config={"tree_implementation": extra_config_param}
                )
                self.assertIsNotNone(torch_model)
                self.assertEqual(str(type(list(torch_model.model._operators)[0])), gbdt_implementation_map[extra_config_param])

    def _run_xgb_classifier_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", [], extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Binary classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_binary_classifier_converter(self):
        self._run_xgb_classifier_converter(2)

    # Gemm classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_gemm_classifier_converter(self):
        self._run_xgb_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_tree_trav_classifier_converter(self):
        self._run_xgb_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_perf_tree_trav_classifier_converter(self):
        self._run_xgb_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Multi classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_multi_classifier_converter(self):
        self._run_xgb_classifier_converter(3)

    # Gemm multi classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_gemm_multi_classifier_converter(self):
        self._run_xgb_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # Tree_trav multi classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_tree_trav_multi_classifier_converter(self):
        self._run_xgb_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav multi classifier
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_perf_tree_trav_multi_classifier_converter(self):
        self._run_xgb_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    def _run_xgb_ranker_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBRanker(n_estimators=10, max_depth=max_depth, ndcg_exp_gain=True)  # ndcg is default
            X, y, q, w = tm.make_ltr(n_samples=1024, n_features=num_classes, n_query_groups=3, max_rel=3)

            model.fit(X, y, qid=q, sample_weight=w, eval_set=[(X, y)],
                      eval_qid=(q,), sample_weight_eval_set=(w,), verbose=True)

            torch_model = hummingbird.ml.convert(model, "torch", X, extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Ranker
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_binary_ranker_converter(self):
        self._run_xgb_ranker_converter(1000)

    # Gemm ranker
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_gemm_ranker_converter(self):
        self._run_xgb_ranker_converter(1000, extra_config={"tree_implementation": "gemm"})

    # Tree_trav ranker
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_tree_trav_ranker_converter(self):
        self._run_xgb_ranker_converter(1000, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav ranker
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_perf_tree_trav_ranker_converter(self):
        self._run_xgb_ranker_converter(1000, extra_config={"tree_implementation": "perf_tree_trav"})

    def _run_xgb_regressor_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", X, extra_config=extra_config)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Regressor
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_binary_regressor_converter(self):
        self._run_xgb_regressor_converter(1000)

    # Gemm regressor
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_gemm_regressor_converter(self):
        self._run_xgb_regressor_converter(1000, extra_config={"tree_implementation": "gemm"})

    # Tree_trav regressor
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_tree_trav_regressor_converter(self):
        self._run_xgb_regressor_converter(1000, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav regressor
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_perf_tree_trav_regressor_converter(self):
        self._run_xgb_regressor_converter(1000, extra_config={"tree_implementation": "perf_tree_trav"})

    # Float 64 data tests
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_float64_xgb_classifier_converter(self):
        warnings.filterwarnings("ignore")
        num_classes = 3
        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", [])
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_float64_xgb_ranker_converter(self):
        warnings.filterwarnings("ignore")
        num_classes = 3
        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBRanker(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y, group=[X.shape[0]])

            torch_model = hummingbird.ml.convert(model, "torch", X[0:1])
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_float64_xgb_regressor_converter(self):
        warnings.filterwarnings("ignore")
        num_classes = 3
        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            torch_model = hummingbird.ml.convert(model, "torch", X[0:1])
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Small tree.
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_run_xgb_classifier_converter(self):
        warnings.filterwarnings("ignore")
        for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
            model = xgb.XGBClassifier(n_estimators=1, max_depth=1)
            np.random.seed(0)
            X = np.random.rand(1, 1)
            X = np.array(X, dtype=np.float32)
            y = np.array([0], dtype=int)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch", [], extra_config={"tree_implementation": extra_config_param})
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test xgboost with pandas.
    @unittest.skipIf(not xgboost_installed() or not pandas_installed(), reason="test requires XGBoost and Pandas installed")
    def test_run_xgb_pandas(self):
        cali = fetch_california_housing()
        data = pd.DataFrame(cali.data)
        data.columns = cali.feature_names

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        # Split the data into training and testing dataset by taking train_size as 75%
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        model = xgb.XGBRegressor(colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)
        model.fit(X_train, y_train)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.predict(X_test), torch_model.predict(X_test), rtol=1e-06, atol=1e-06)

    # Torchscript backends.
    # Test TorchScript backend regression.
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_regressor_converter_torchscript(self):
        warnings.filterwarnings("ignore")
        import torch

        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBRegressor(n_estimators=10, max_depth=max_depth, objective="rank:pairwise")
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(1000, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torchscript", X)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test TorchScript backend classification.
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    def test_xgb_classifier_converter_torchscript(self):
        warnings.filterwarnings("ignore")
        import torch

        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBClassifier(n_estimators=10, max_depth=max_depth, objective="rank:pairwise")
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(2, size=100)

            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torchscript", X)
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # TVM backend tests.
    # TVM backend regression.
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_xgb_regressor_converter_tvm(self):
        warnings.filterwarnings("ignore")
        import torch

        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBRegressor(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(1000, size=100)

            model.fit(X, y)

            tvm_model = hummingbird.ml.convert(model, "tvm", X, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})
            self.assertIsNotNone(tvm_model)
            np.testing.assert_allclose(model.predict(X), tvm_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test TVM backend classification.
    @unittest.skipIf(not xgboost_installed(), reason="XGBoost test requires XGBoost installed")
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_xgb_classifier_converter_tvm(self):
        warnings.filterwarnings("ignore")
        import torch

        for max_depth in [1, 3, 8, 10, 12]:
            model = xgb.XGBClassifier(n_estimators=10, max_depth=max_depth)
            np.random.seed(0)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(2, size=100)

            model.fit(X, y)

            tvm_model = hummingbird.ml.convert(model, "tvm", X, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})
            self.assertIsNotNone(tvm_model)
            np.testing.assert_allclose(model.predict_proba(X), tvm_model.predict_proba(X), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
