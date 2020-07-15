"""
Tests onnxml scaler converter
"""
import unittest
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import convert

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorType_onnx


class TestONNXScaler(unittest.TestCase):
    def _test_standar_scaler_converter(self, with_mean, with_std):
        warnings.filterwarnings("ignore")
        X = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)

        # Create SKL model for testing
        model = StandardScaler(with_mean=with_mean, with_std=with_std)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", X)
        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        onnx_ml_pred = [[] for i in range(len(output_names))]
        inputs = {session.get_inputs()[0].name: X}
        onnx_ml_pred = session.run(output_names, inputs)

        # Get the predictions for the ONNX model
        session = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_pred = [[] for i in range(len(output_names))]
        onnx_pred = session.run(output_names, inputs)

        return onnx_ml_pred, onnx_pred

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # Test StandardScaler with_mean=True, with_std=True
    def test_standard_scaler_onnx_tt(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_standar_scaler_converter(True, True)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # Test StandardScaler with_mean=True, with_std=False
    def test_standard_scaler_onnx_tf(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_standar_scaler_converter(True, False)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test StandardScaler with_mean=False, with_std=False
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_standard_scaler_onnx_ff(self, rtol=1e-06, atol=1e-06):
        onnx_ml_pred, onnx_pred = self._test_standar_scaler_converter(False, False)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_scaler_converter_raises_rt(self):
        warnings.filterwarnings("ignore")
        X = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)

        # Create SKL model for testing
        model = StandardScaler()
        model.fit(X)

        # generate test input
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])
        print(onnx_ml_model.graph.node[0].attribute[0].name)
        onnx_ml_model.graph.node[0].attribute[0].name = "".encode()

        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", X)


if __name__ == "__main__":
    unittest.main()
