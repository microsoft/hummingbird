"""
Tests sklearn TruncatedSVD converter
"""
import unittest

import numpy as np
import torch
import xgboost as xgb
from hummingbird import convert_sklearn
from hummingbird.common.data_types import Float32TensorType


class TestXGBoostConverter(unittest.TestCase):

    def _run_xgb_classifier_converter(self, num_classes, extra_config={}):
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = xgb.XGBClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)

            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 200]))],
                extra_config=extra_config
            )
            self.assertTrue(pytorch_model is not None)
            self.assertTrue(np.allclose(model.predict_proba(
                X), pytorch_model(torch.from_numpy(X))[1].data.numpy()))

    # binary classifier
    def test_xgb_binary_classifier_converter(self):
        self._run_xgb_classifier_converter(2)

    # multi classifier
    def test_xgb_multi_classifier_converter(self):
        self._run_xgb_classifier_converter(3)

    # batch classifier
    def test_xgb_batch_classifier_converter(self):
        self._run_xgb_classifier_converter(3, extra_config={"tree_implementation": "batch"})

    # beam classifier
    def test_xgb_beam_classifier_converter(self):
        self._run_xgb_classifier_converter(3, extra_config={"tree_implementation": "beam"})

    # beam++ classifier
    def test_xgb_beampp_classifier_converter(self):
        self._run_xgb_classifier_converter(3, extra_config={"tree_implementation": "beam++"})

    # TODO THIS ONE FAILS
    def _run_xgb_regressor_converter(self, num_classes, extra_config={}):
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = xgb.XGBRegressor(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 200]))],
                extra_config=extra_config
            )
            self.assertTrue(pytorch_model is not None)
            self.assertTrue(np.allclose(model.predict(X), pytorch_model(
                torch.from_numpy(X)).view(-1).numpy()))

    # binary regressor
    def test_xgb_binary_regressor_converter(self):
        self._run_xgb_regressor_converter(2)

    # multi regressor
    def test_xgb_multi_regressor_converter(self):
        self._run_xgb_regressor_converter(3)

    # batch regressor
    def test_xgb_batch_regressor_converter(self):
        self._run_xgb_regressor_converter(3, extra_config={"tree_implementation": "batch"})

    # beam regressor
    def test_xgb_beam_regressor_converter(self):
        self._run_xgb_regressor_converter(3, extra_config={"tree_implementation": "beam"})

    # beam++ regressor
    def test_xgb_beampp_regressor_converter(self):
        self._run_xgb_regressor_converter(3, extra_config={"tree_implementation": "beam++"})


if __name__ == "__main__":
    unittest.main()
