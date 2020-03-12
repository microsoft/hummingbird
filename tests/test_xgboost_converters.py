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

    # TODO implement multi-class support
    def test_xgb_binary_classifier_converter(self):
        for max_depth in [3, 7, 11]:
            model = xgb.XGBClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(2, size=100)

            model.fit(X, y)

            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 200]))]
            )
            self.assertTrue(pytorch_model is not None)
            self.assertTrue(np.allclose(model.predict_proba(
                X), pytorch_model(torch.from_numpy(X))[1].data.numpy()))


if __name__ == "__main__":
    unittest.main()
