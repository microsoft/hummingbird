"""
Tests onnxml SV converters
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC, NuSVC

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import convert

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorType_onnx


class TestONNXSVC(unittest.TestCase):
    def _test_sv(self, classes, mode="torch"):
        """
        This helper function tests conversion of `ai.onnx.ml.SVMClassifier`
        which is created from a scikit-learn SVC.

        This then calls either "_to_onnx" or "_to_torch"
        """
        n_features = 20
        n_total = 100
        np.random.seed(0)
        warnings.filterwarnings("ignore")
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(classes, size=n_total)

        # Create SKL model for testing
        model = SVC()
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

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

        model = convert(onnx_ml_model, mode, X)

        if mode == "torch":
            pred = model.predict(X)

        else:
            # Get the predictions for the ONNX model
            pred = [[] for i in range(len(output_names))]
            if len(output_names) == 1:  # regression
                pred = model.predict(X)
            else:  # classification
                pred[0] = model.predict_proba(X)
                pred[1] = model.predict(X)

        return onnx_ml_pred, pred

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test ai.onnx.ml.SVMClassifier with 2 classes for onnxml-> pytorch
    def test_logistic_regression_onnxml_binary_torch(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, pred = self._test_sv(2)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[1], pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # test ai.onnx.ml.SVMClassifier with 3 classes for onnxml-> pytorch
    def test_logistic_regression_onnxml_multi_torch(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, pred = self._test_sv(3)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[1], pred, rtol=rtol, atol=atol)

    # TODO: There is a bug with ORT:
    # onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented:
    # [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for the node Gemm_8:Gemm(11)
    # @unittest.skipIf(
    #     not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    # )
    # # test ai.onnx.ml.SVMClassifier with 2 classes
    # def test_logistic_regression_onnxml_binary_onnx(self, rtol=1e-06, atol=1e-06):
    #     onnx_ml_pred, onnx_pred = self._test_sv(2, mode="onnx")

    #     # Check that predicted values match
    #     np.testing.assert_allclose(onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol)  # labels
    #     np.testing.assert_allclose(
    #         list(map(lambda x: list(x.values()), onnx_ml_pred[0])), onnx_pred[0], rtol=rtol, atol=atol
    #     )


if __name__ == "__main__":
    unittest.main()
