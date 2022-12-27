"""
Tests lightgbm->onnxmltools->hb conversion for lightgbm models.
"""
import unittest
import warnings

import sys
import os
import pickle
import numpy as np
from onnxconverter_common.data_types import FloatTensorType

from hummingbird.ml import convert
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed

if lightgbm_installed():
    import lightgbm as lgb
if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_lightgbm


class TestONNXLightGBMConverter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestONNXLightGBMConverter, self).__init__(*args, **kwargs)

    # Base test implementation comparing ONNXML and ONNX models.
    def _test_lgbm(self, X, model, extra_config={}):
        # Create ONNX-ML model
        onnx_ml_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([None, X.shape[1]]))], target_opset=9
        )

        # Create ONNX model
        onnx_model = convert(onnx_ml_model, "onnx", extra_config=extra_config)

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        onnx_ml_pred = [[] for i in range(len(output_names))]
        inputs = {session.get_inputs()[0].name: X}
        pred = session.run(output_names, inputs)
        for i in range(len(output_names)):
            if "label" in output_names[i]:
                onnx_ml_pred[1] = pred[i]
            else:
                onnx_ml_pred[0] = pred[i]

        # Get the predictions for the ONNX model
        onnx_pred = [[] for i in range(len(output_names))]
        if len(output_names) == 1:  # regression
            onnx_pred = onnx_model.predict(X)
        else:  # classification
            onnx_pred[0] = onnx_model.predict_proba(X)
            onnx_pred[1] = onnx_model.predict(X)

        return onnx_ml_pred, onnx_pred, output_names

    # Utility function for testing regression models.
    def _test_regressor(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_lgbm(X, model, extra_config)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[0].ravel(), onnx_pred, rtol=rtol, atol=atol)

    # Utility function for testing classification models.
    def _test_classifier(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        for n in [2, len(X)]:
            onnx_ml_pred, onnx_pred, _ = self._test_lgbm(X[:n], model, extra_config)

            np.testing.assert_allclose(onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol)  # labels
            np.testing.assert_allclose(
                list(map(lambda x: list(x.values()), onnx_ml_pred[0])), onnx_pred[0], rtol=rtol, atol=atol
            )  # probs

            np.testing.assert_equal(n, len(onnx_pred[0]))  # pred count

    # Check that ONNXML models can also target other backends.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_onnx_pytorch(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([None, X.shape[1]]))], target_opset=9
        )

        pt_model = convert(onnx_ml_model, "torch", X)
        assert pt_model

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        onnx_ml_pred = [[] for i in range(len(output_names))]
        inputs = {session.get_inputs()[0].name: X}
        onnx_ml_pred = session.run(output_names, inputs)

        np.testing.assert_allclose(onnx_ml_pred[0].flatten(), pt_model.predict(X))

    # Basic regression test.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_onnxml_model_regressor(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(n_total, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        import platform

        # TODO bug on newer macOS versions?
        if platform.system() == "Darwin":
            self._test_regressor(X, model, rtol=1e-05, atol=1e-04)
        else:
            self._test_regressor(X, model)

    # Regression test with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_regressor(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 1 estimator (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_regressor1(self):
        warnings.filterwarnings("ignore")
        model = lgb.LGBMRegressor(n_estimators=1, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 2 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_regressor2(self):
        warnings.filterwarnings("ignore")
        model = lgb.LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with gbdt boosting type (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_booster_regressor(self):
        warnings.filterwarnings("ignore")
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
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_onnxml_model_binary(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classication test with float64.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_onnxml_model_binary_float64(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMClassifier()
        model.fit(X, y)

        onnx_model = convert(model, "onnx", X)

        np.testing.assert_allclose(model.predict(X), onnx_model.predict(X))

    # Binary classification test with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_classifier(self):
        warnings.filterwarnings("ignore")
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators zipmap (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_classifier_zipmap(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_booster_classifier(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        data = lgb.Dataset(X, label=y)
        model = lgb.train({"boosting_type": "gbdt", "objective": "binary", "n_estimators": 3, "min_child_samples": 1}, data)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type zipmap (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_booster_classifier_zipmap(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        data = lgb.Dataset(X, label=y)
        model = lgb.train({"boosting_type": "gbdt", "objective": "binary", "n_estimators": 3, "min_child_samples": 1}, data)
        self._test_classifier(X, model)

    # Multiclass classification test.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lgbm_onnxml_model_multi(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=n_total)

        # Create LightGBM model
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_classifier_multi(self):
        warnings.filterwarnings("ignore")
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators and selecting boosting type (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_booster_multi_classifier(self):
        warnings.filterwarnings("ignore")
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
