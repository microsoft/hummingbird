"""
Tests onnxml Linear converters
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import convert

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorType_onnx


class TestONNXLinear(unittest.TestCase):
    def _test_linear(self, classes):
        """
        This helper function tests conversion of `ai.onnx.ml.LinearClassifier`
        which is created from a scikit-learn LogisticRegression.

        This tests `convert_onnx_linear_model` in `hummingbird.ml.operator_converters.onnxml_linear`
        """
        n_features = 20
        n_total = 100
        np.random.seed(0)
        warnings.filterwarnings("ignore")
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(classes, size=n_total)

        # Create SKL model for testing
        model = LogisticRegression(solver="liblinear", multi_class="ovr", fit_intercept=True)
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", X)

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

        return onnx_ml_pred, onnx_pred

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test ai.onnx.ml.LinearClassifier with 2 classes
    def test_logistic_regression_onnxml_binary(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_linear(2)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol)  # labels
        np.testing.assert_allclose(
            list(map(lambda x: list(x.values()), onnx_ml_pred[0])), onnx_pred[0], rtol=rtol, atol=atol
        )  # probs

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test ai.onnx.ml.LinearClassifier with 3 classes
    def test_logistic_regression_onnxml_multi(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_linear(3)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol)  # labels
        np.testing.assert_allclose(
            list(map(lambda x: list(x.values()), onnx_ml_pred[0])), onnx_pred[0], rtol=rtol, atol=atol
        )  # probs

    def _test_regressor(self, values):
        """
        This helper function tests conversion of `ai.onnx.ml.LinearRegressor`
        which is created from a scikit-learn LinearRegression.

        This tests `convert_onnx_linear_regression_model` in `hummingbird.ml.operator_converters.onnxml_linear`
        """
        n_features = 20
        n_total = 100
        np.random.seed(0)
        warnings.filterwarnings("ignore")
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(values, size=n_total)

        # Create SKL model for testing
        model = LinearRegression()
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", X)

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: X}
        onnx_ml_pred = session.run(output_names, inputs)

        # Get the predictions for the ONNX model
        onnx_pred = onnx_model.predict(X)

        return onnx_ml_pred, onnx_pred

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test ai.onnx.ml.LinearRegressor with 2 values
    def test_linear_regression_onnxml_small(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_regressor(2)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[0].ravel(), onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test ai.onnx.ml.LinearRegressor with 100 values
    def test_linear_regression_onnxml_large(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_regressor(100)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[0].ravel(), onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test for malformed model/problem with parsing
    def test_onnx_linear_converter_raises_rt(self):
        n_features = 20
        n_total = 100
        np.random.seed(0)
        warnings.filterwarnings("ignore")
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=n_total)
        model = LinearRegression()
        model.fit(X, y)

        # generate test input
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])
        onnx_ml_model.graph.node[0].attribute[0].name = "".encode()

        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", X)


if __name__ == "__main__":
    unittest.main()
