"""
Tests lightgbm->onnxmltools->hb conversion for lightgbm models.
"""
import unittest

import sys
import os
import pickle
import numpy as np
import lightgbm as lgb
import onnxruntime as ort

from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm
from hummingbird.ml import convert
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_installed


class TestONNXConverterLightGBM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestONNXConverterLightGBM, self).__init__(*args, **kwargs)

    # Base test implementation comparing ONNXML and ONNX models.
    def _test_lgbm(self, X, model, extra_config={}):
        # Create ONNX-ML model
        onnx_ml_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=9
        )

        # Create ONNX model
        onnx_model = convert(onnx_ml_model, "onnx", X, extra_config)

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

    # Utility function for testing regression models.
    def _test_regressor(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_lgbm(X, model, extra_config)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[0], onnx_pred[0], rtol=rtol, atol=atol)

    # Utility function for testing classification models.
    def _test_classifier(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_lgbm(X, model, extra_config)

        # Check that predicted values match
        labels = []
        probabilities = []
        for i in range(len(output_names)):
            if type(onnx_ml_pred[i][0]) is dict:
                probabilities.append(list(map(lambda x: list(x.values()), onnx_ml_pred[i])))
            else:
                labels.append(onnx_ml_pred[i])
            if onnx_pred[i][0].dtype == np.dtype("int64"):
                labels.append(onnx_pred[i])
            else:
                probabilities.append(onnx_pred[i])
        np.testing.assert_allclose(labels[0], labels[1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(probabilities[0], probabilities[1], rtol=rtol, atol=atol)

    # Check that ONNXML models can only target the ONNX backend.
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_pytorch(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=9
        )

        self.assertRaises(RuntimeError, convert, onnx_ml_model, "torch")

    # Check conveter with extra configs.
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_pytorch_extra_config(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=9
        )

        # Create ONNX model
        model_name = "hummingbird.ml.test.lightgbm"
        extra_config = {}
        extra_config[constants.ONNX_OUTPUT_MODEL_NAME] = model_name
        extra_config[constants.ONNX_INITIAL_TYPES] = [("input", FloatTensorType([X.shape[0], X.shape[1]]))]
        onnx_model = convert(onnx_ml_model, "onnx", extra_config=extra_config)

        assert onnx_model.graph.name == model_name

    # Basic regression test.
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lgbm_onnxml_model_regressor(self):
        n_features = 28
        n_total = 1000
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(1000, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        self._test_regressor(X, model, rtol=1e-02, atol=1e-02)  # Lower tolerance to avoid random errors

    # Regression test with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 1 estimator (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_regressor1(self):
        model = lgb.LGBMRegressor(n_estimators=1, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 2 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_regressor2(self):
        model = lgb.LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with gbdt boosting type (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_booster_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 1.1]
        data = lgb.Dataset(X, label=y)
        model = lgb.train(
            {"boosting_type": "gbdt", "objective": "regression", "n_estimators": 3, "min_child_samples": 1, "max_depth": 1},
            data,
        )
        self._test_regressor(X, model)

    # Binary classication test.
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lgbm_onnxml_model_binary(self):
        n_features = 28
        n_total = 1000
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        self._test_classifier(X, model, rtol=1e-02, atol=1e-02)  # Lower tolerance to avoid random errors

    # Binary classification test with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_classifier(self):
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators zipmap (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_booster_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        data = lgb.Dataset(X, label=y)
        model = lgb.train({"boosting_type": "gbdt", "objective": "binary", "n_estimators": 3, "min_child_samples": 1}, data)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type zipmap (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_booster_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        data = lgb.Dataset(X, label=y)
        model = lgb.train({"boosting_type": "gbdt", "objective": "binary", "n_estimators": 3, "min_child_samples": 1}, data)
        self._test_classifier(X, model)

    # Multiclass classification test.
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lgbm_onnxml_model_multi(self):
        n_features = 28
        n_total = 1000
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        self._test_classifier(X, model, rtol=1e-02, atol=1e-02)  # Lower tolerance to avoid random errors

    # Multiclass classification test with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    def test_lightgbm_classifier_multi(self):
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators and selecting boosting type (taken from ONNXMLTOOLS).
    @unittest.skipIf(not (onnx_ml_tools_installed and onnx_installed), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS")
    @unittest.skipIf(
        True, reason='ONNXMLTOOLS fails with "ValueError: unsupported LightGbm objective: multiclass num_class:3"'
    )
    def test_lightgbm_booster_multi_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1, 2, 2]
        data = lgb.Dataset(X, label=y)
        model = lgb.train(
            {"boosting_type": "gbdt", "objective": "multiclass", "n_estimators": 3, "min_child_samples": 1, "num_class": 3},
            data,
        )
        self._test_classifier(X, model)


if __name__ == "__main__":
    unittest.main()
