"""
Tests onnxml scaler converter
"""
import unittest
import warnings

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import convert

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import Int32TensorType as IntTensorType_onnx


class TestONNXOneHotEncoder(unittest.TestCase):
    def _test_ohe_converter(self, model, X):
        warnings.filterwarnings("ignore")

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("int_input", IntTensorType_onnx(X.shape))])

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

    # Test StandardScaler with_mean=True, with_std=True
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_one_hot_encoder_onnx_int(self, rtol=1e-06, atol=1e-06):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3]], dtype=np.int32)
        model.fit(data)
        onnx_ml_pred, onnx_pred = self._test_ohe_converter(model, data)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test StandardScaler with_mean=True, with_std=True
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_one_hot_encoder_onnx_int_2d(self, rtol=1e-06, atol=1e-06):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int32)
        model.fit(data)
        onnx_ml_pred, onnx_pred = self._test_ohe_converter(model, data)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
