"""
Tests onnxml OneHotEncoderconverter
"""
from distutils.version import LooseVersion
import unittest
import warnings

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import convert

if onnx_ml_tools_installed():
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import Int32TensorType as IntTensorType_onnx
    from onnxmltools.convert.common.data_types import Int64TensorType as LongTensorType_onnx
    from onnxmltools.convert.common.data_types import StringTensorType as StringTensorType_onnx
    from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorType_onnx

if onnx_runtime_installed():
    import onnxruntime as ort


class TestONNXOneHotEncoder(unittest.TestCase):

    # Test OneHotEncoder with ints
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_one_hot_encoder_onnx_int(self, rtol=1e-06, atol=1e-06):
        model = OneHotEncoder()
        X = np.array([[1, 2, 3]], dtype=np.int32)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("int_input", IntTensorType_onnx(X.shape))])

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
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test OneHotEncoder with 2 inputs
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_one_hot_encoder_onnx2(self, rtol=1e-06, atol=1e-06):
        model = OneHotEncoder()
        X = np.array([[1, 2, 3], [2, 1, 3]], dtype=np.int32)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("int_input", IntTensorType_onnx(X.shape))])

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
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test OneHotEncoder with int64
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_one_hot_encoder_onnx_int64(self, rtol=1e-06, atol=1e-06):
        model = OneHotEncoder()
        X = np.array([[1, 2, 3]], dtype=np.int64)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("int_input", LongTensorType_onnx(X.shape))])

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
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

    # Test OneHotEncoder with strings. This test only works with pytorch >= 1.8
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(
        LooseVersion(torch.__version__) < LooseVersion("1.8.0"),
        reason="PyTorch exporter returns an error until version 1.8.0",
    )
    def test_model_one_hot_encoder_string(self):
        model = OneHotEncoder()
        data = [["a", "r", "x"], ["a", "r", "x"], ["aaaa", "r", "x"], ["a", "r", "xx"]]
        model.fit(data)

        onnx_ml_model = convert_sklearn(model, initial_types=[("input", StringTensorType_onnx([4, 3]))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", data)

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: data}
        onnx_ml_pred = session.run(output_names, inputs)

        # Get the predictions for the ONNX model
        onnx_pred = onnx_model.transform(data)

        return onnx_ml_pred, onnx_pred

    # Test OneHotEncoder failcase when input data type is not supported
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_ohe_string_raises_type_error_onnx(self):
        warnings.filterwarnings("ignore")
        model = OneHotEncoder()
        data = [["a", "r", "x"], ["a", "r", "x"], ["aaaa", "r", "x"], ["a", "r", "xx"]]
        model.fit(data)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("input", StringTensorType_onnx([4, 3]))])

        # Create ONNX model by calling converter, should raise error for strings
        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx")


if __name__ == "__main__":
    unittest.main()
