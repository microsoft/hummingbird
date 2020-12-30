"""
Tests onnxml scaler converter
"""
import unittest
import warnings

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
import torch

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed
from hummingbird.ml import convert

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert.common.data_types import DoubleTensorType


class TestONNXScaler(unittest.TestCase):
    def _test_scaler_converter(self, model):
        warnings.filterwarnings("ignore")
        X = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType(X.shape))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", X)
        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: X}
        onnx_ml_pred = session.run(output_names, inputs)[0]

        # Get the predictions for the ONNX model
        onnx_pred = onnx_model.transform(X)

        return onnx_ml_pred, onnx_pred

    # Test StandardScaler with_mean=True, with_std=True
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_standard_scaler_onnx_tt(self, rtol=1e-06, atol=1e-06):
        model = StandardScaler(with_mean=True, with_std=True)
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test StandardScaler with_mean=True, with_std=False
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_standard_scaler_onnx_tf(self, rtol=1e-06, atol=1e-06):
        model = StandardScaler(with_mean=True, with_std=False)
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test StandardScaler with_mean=False, with_std=False
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_standard_scaler_onnx_ff(self, rtol=1e-06, atol=1e-06):
        model = StandardScaler(with_mean=False, with_std=False)
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test RobustScaler with with_centering=True
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_robust_scaler_onnx_t(self, rtol=1e-06, atol=1e-06):
        model = RobustScaler(with_centering=True)
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test RobustScaler with with_centering=False
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_robust_scaler_onnx_f(self, rtol=1e-06, atol=1e-06):
        model = RobustScaler(with_centering=False)
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test MaxAbsScaler
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_max_abs_scaler_onnx(self, rtol=1e-06, atol=1e-06):
        model = MaxAbsScaler()
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test MinMaxScaler
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_min_max_scaler_onnx(self, rtol=1e-06, atol=1e-06):
        model = MinMaxScaler()
        onnx_ml_pred, onnx_pred = self._test_scaler_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test that malformed models throw an exception
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_scaler_converter_raises_rt_onnx(self):
        warnings.filterwarnings("ignore")
        X = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)

        # Create SKL model for testing
        model = StandardScaler()
        model.fit(X)

        # Generate test input
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType(X.shape))])
        print(onnx_ml_model.graph.node[0].attribute[0].name)
        onnx_ml_model.graph.node[0].attribute[0].name = "".encode()

        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", X)

    # Test with float64
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_scaler_converter_float_64(self):
        warnings.filterwarnings("ignore")
        X = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float64)

        # Create SKL model for testing
        model = StandardScaler()
        model.fit(X)

        # Generate test input
        onnx_ml_model = convert_sklearn(model, initial_types=[("double_input", DoubleTensorType(X.shape))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", X)
        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: X}
        onnx_ml_pred = session.run(output_names, inputs)[0]

        # Get the predictions for the ONNX model
        onnx_pred = onnx_model.transform(X)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
