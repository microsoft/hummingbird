"""
Tests LightGBM converters.
"""
import unittest
import warnings

import numpy as np
import lightgbm as lgb

import hummingbird.ml
from tree_utils import gbdt_implementation_map


class TestLGBMConverter(unittest.TestCase):
    # Check tree implementation
    def test_lgbm_implementation(self):
        warnings.filterwarnings("ignore")
        X = np.random.rand(10, 1)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)

        for model in [lgb.LGBMClassifier(n_estimators=1, max_depth=1), lgb.LGBMRegressor(n_estimators=1, max_depth=1)]:
            for extra_config_param in ["tree_trav", "perf_tree_trav", "gemm"]:
                model.fit(X, y)

                pytorch_model = model.to('pytorch', extra_config={"tree_implementation": extra_config_param})
                self.assertTrue(pytorch_model is not None)
                self.assertTrue(
                    str(type(list(pytorch_model.operator_map.values())[0])) == gbdt_implementation_map[extra_config_param]
                )

    def _run_lgbm_classifier_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            pytorch_model = model.to('pytorch', extra_config=extra_config)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(
                model.predict_proba(X), pytorch_model.predict_proba(X), rtol=1e-06, atol=1e-06
            )

    # Binary classifier
    def test_lgbm_binary_classifier_converter(self):
        self._run_lgbm_classifier_converter(2)

    # Gemm classifier
    def test_lgbm_gemm_classifier_converter(self):
        self._run_lgbm_classifier_converter(2, extra_config={"tree_implementation": "gemm"})

    # Tree_trav classifier
    def test_lgbm_tree_trav_classifier_converter(self):
        self._run_lgbm_classifier_converter(2, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav classifier
    def test_lgbm_perf_tree_trav_classifier_converter(self):
        self._run_lgbm_classifier_converter(2, extra_config={"tree_implementation": "perf_tree_trav"})

    # Multi classifier
    def test_lgbm_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3)

    # Gemm multi classifier
    def test_lgbm_gemm_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3, extra_config={"tree_implementation": "gemm"})

    # Tree_trav multi classifier
    def test_lgbm_tree_trav_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav multi classifier
    def test_lgbm_perf_tree_trav_multi_classifier_converter(self):
        self._run_lgbm_classifier_converter(3, extra_config={"tree_implementation": "perf_tree_trav"})

    def _run_lgbm_regressor_converter(self, num_classes, extra_config={}):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = lgb.LGBMRegressor(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            pytorch_model = model.to('pytorch', extra_config=extra_config)
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(
                model.predict(X), pytorch_model.predict(X), rtol=1e-06, atol=1e-06
            )

    # Regressor
    def test_lgbm_binary_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000)

    # Gemm regressor
    def test_lgbm_gemm_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "gemm"})

    # Tree_trav regressor
    def test_lgbm_tree_trav_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "tree_trav"})

    # Perf_tree_trav regressor
    def test_lgbm_perf_tree_trav_regressor_converter(self):
        self._run_lgbm_regressor_converter(1000, extra_config={"tree_implementation": "perf_tree_trav"})


if __name__ == "__main__":
    unittest.main()
