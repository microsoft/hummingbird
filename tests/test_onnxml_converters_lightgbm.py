"""
Tests lightgbm->onnxmltools->hb conversion for lightgbm models.
"""
import unittest

import sys
import os
import pickle
import numpy as np
import lightgbm as lgb
import torch
import onnx
import onnxruntime as ort

from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm
from hummingbird.ml import convert
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_ml_tools_installed


def test_lgbm(X, model):
    assert onnx_ml_tools_installed()

    # Create ONNX-ML model
    onnx_ml_model = convert_lightgbm(
        model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=9
    )

    # Create ONNX model
    extra_config = {}
    extra_config[constants.TREE_IMPLEMENTATION] = "tree_trav"
    onnx_model = convert(onnx_ml_model, "onnx", X[0:1], extra_config)

    # Get the predictions for the ONNX-ML model
    session = ort.InferenceSession(onnx_ml_model.SerializeToString())
    output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
    onnx_ml_pred = [[] for i in range(len(output_names))]
    inputs = {session.get_inputs()[0].name: X}
    pred = session.run(output_names, inputs)
    for i in range(len(output_names)):
        onnx_ml_pred[i].extend(pred[i])

    # Get the predictions for the ONNX model
    session = ort.InferenceSession(onnx_model.SerializeToString())
    onnx_pred = [[] for i in range(len(output_names))]
    pred = session.run(output_names, inputs)
    for i in range(len(output_names)):
        onnx_pred[i].extend(pred[i])

    return onnx_ml_pred, onnx_pred, output_names


def test_regressor(X, model):
    onnx_ml_pred, onnx_pred, output_names = test_lgbm(X, model)

    # Check that predicted values match
    np.testing.assert_allclose(onnx_ml_pred[0], onnx_pred[0], rtol=1e-05, atol=1e-05)


def test_classifier(X, model):
    onnx_ml_pred, onnx_pred, output_names = test_lgbm(X, model)

    # Check that predicted values match
    for i in range(len(output_names)):
        if output_names[i] == "probabilities":
            onnx_ml_prob = list(map(lambda x: list(x.values()), onnx_ml_pred[i]))
            np.testing.assert_allclose(onnx_ml_prob, onnx_pred[i], rtol=1e-05, atol=1e-05)
        else:
            np.testing.assert_allclose(onnx_ml_pred[i], onnx_pred[i], rtol=1e-05, atol=1e-05)


class TestLightGBMConverter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLightGBMConverter, self).__init__(*args, **kwargs)

    # def test_lgbm_onnxml_model_regressor(self):
    #     self.n_features = 28
    #     self.n_total = 1000
    #     self.X = np.random.rand(self.n_total,self.n_features)
    #     self.X = np.array(self.X, dtype=np.float32)
    #     self.y = np.random.randint(1000, size=self.n_total)

    #     # Create LightGBM model
    #     model = lgb.LGBMRegressor()
    #     model.fit(self.X, self.y)
    #     test_regressor(self.X, model)

    def test_lightgbm_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        test_regressor(X, model)

    # def test_lightgbm_regressor1(self):
    #     model = lgb.LGBMRegressor(n_estimators=1, min_child_samples=1)
    #     X = [[0, 1], [1, 1], [2, 0]]
    #     X = np.array(X, dtype=np.float32)
    #     y = np.array([100, -10, 50], dtype=np.float32)
    #     model.fit(X, y)
    #     test_regressor(X, model)

    # def test_lightgbm_regressor2(self):
    #     model = lgb.LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
    #     X = [[0, 1], [1, 1], [2, 0]]
    #     X = np.array(X, dtype=np.float32)
    #     y = np.array([100, -10, 50], dtype=np.float32)
    #     model.fit(X, y)
    #     test_regressor(X, model)

    # def test_lightgbm_booster_regressor(self):
    #     X = [[0, 1], [1, 1], [2, 0]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 1.1]
    #     data = lgb.Dataset(X, label=y)
    #     model = lgb.train({'boosting_type': 'gbdt', 'objective': 'regression',
    #                             'n_estimators': 3, 'min_child_samples': 1, 'max_depth': 1},
    #                            data)
    #     test_regressor(X, model)

    # def test_lgbm_onnxml_model_binary(self):
    #     self.n_features = 28
    #     self.n_total = 1000
    #     self.X = np.random.rand(self.n_total,self.n_features)
    #     self.X = np.array(self.X, dtype=np.float32)
    #     self.y = np.random.randint(2, size=self.n_total)

    #     # Create LightGBM model
    #     model = lgb.LGBMClassifier()
    #     model.fit(self.X, self.y)
    #     test_classifier(self.X, model)

    # def test_lightgbm_classifier(self):
    #     model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
    #     X = [[0, 1], [1, 1], [2, 0]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 0]
    #     model.fit(X, y)
    #     test_classifier(X, model)

    # def test_lightgbm_classifier_zipmap(self):
    #     X = [[0, 1], [1, 1], [2, 0], [1, 2]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 0, 1]
    #     model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
    #     model.fit(X, y)
    #     test_classifier(X, model)

    # def test_lightgbm_booster_classifier(self):
    #     X = [[0, 1], [1, 1], [2, 0], [1, 2]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 0, 1]
    #     data = lgb.Dataset(X, label=y)
    #     model = lgb.train({'boosting_type': 'gbdt', 'objective': 'binary',
    #                             'n_estimators': 3, 'min_child_samples': 1},
    #                            data)
    #     test_classifier(X, model)

    # def test_lightgbm_booster_classifier_zipmap(self):
    #     X = [[0, 1], [1, 1], [2, 0], [1, 2]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 0, 1]
    #     data = lgb.Dataset(X, label=y)
    #     model = lgb.train({'boosting_type': 'gbdt', 'objective': 'binary',
    #                             'n_estimators': 3, 'min_child_samples': 1},
    #                            data)
    #     test_classifier(X, model)

    # def test_lgbm_onnxml_model_multi(self):
    #     self.n_features = 28
    #     self.n_total = 1000
    #     self.X = np.random.rand(self.n_total,self.n_features)
    #     self.X = np.array(self.X, dtype=np.float32)
    #     self.y = np.random.randint(3, size=self.n_total)

    #     # Create LightGBM model
    #     model = lgb.LGBMClassifier()
    #     model.fit(self.X, self.y)
    #     test_classifier(self.X, model)

    # def test_lightgbm_classifier_multi(self):
    #     model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
    #     X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 2, 1, 1, 2]
    #     model.fit(X, y)
    #     test_classifier(X, model)

    # def test_lightgbm_booster_multi_classifier(self):
    #     X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
    #     X = np.array(X, dtype=np.float32)
    #     y = [0, 1, 0, 1, 2, 2]
    #     data = lgb.Dataset(X, label=y)
    #     model = lgb.train({'boosting_type': 'gbdt', 'objective': 'multiclass',
    #                             'n_estimators': 3, 'min_child_samples': 1, 'num_class': 3},
    #                            data)
    #     test_classifier(X, model)


if __name__ == "__main__":
    unittest.main()
