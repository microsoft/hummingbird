"""
Tests onnxml Imputer converter
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.impute import SimpleImputer

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import convert

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorType_onnx


class TestONNXImputer(unittest.TestCase):
    def _test_imputer_converter(self, model):
        warnings.filterwarnings("ignore")
        X = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

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

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_imputer_const(self, rtol=1e-06, atol=1e-06):
        model = SimpleImputer(strategy="constant")
        onnx_ml_pred, onnx_pred = self._test_imputer_converter(model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_imputer_const_nan0(self, rtol=1e-06, atol=1e-06):
        model = SimpleImputer(strategy="constant", fill_value=0)
        onnx_ml_pred, onnx_pred = self._test_imputer_converter(model=model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_imputer_mean(self, rtol=1e-06, atol=1e-06):
        model = SimpleImputer(strategy="mean", fill_value="nan")
        onnx_ml_pred, onnx_pred = self._test_imputer_converter(model=model)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_imputer_converter_raises_rt(self):
        warnings.filterwarnings("ignore")
        model = SimpleImputer(strategy="mean", fill_value="nan")
        X = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])
        onnx_ml_model.graph.node[0].attribute[0].name = "".encode()

        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", X)


if __name__ == "__main__":
    unittest.main()
