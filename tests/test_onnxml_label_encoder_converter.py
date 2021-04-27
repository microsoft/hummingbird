"""
Tests onnxml LabelEncoder converter
"""
from distutils.version import LooseVersion
import unittest
import warnings

import numpy as np
from sklearn.preprocessing import LabelEncoder
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


class TestONNXLabelEncoder(unittest.TestCase):

    # Test LabelEncoder with longs
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    def test_model_label_encoder_int_onnxml(self):
        model = LabelEncoder()
        X = np.array([1, 4, 5, 2, 0, 2], dtype=np.int64)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(model, initial_types=[("input", LongTensorType_onnx(X.shape))])

        # Create ONNX model by calling converter
        onnx_model = convert(onnx_ml_model, "onnx", X)

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: X}
        onnx_ml_pred = np.array(session.run(output_names, inputs)).ravel()

        # Get the predictions for the ONNX model
        onnx_pred = onnx_model.transform(X).ravel()

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=1e-06, atol=1e-06)

    # Test LabelEncoder with strings on Pytorch >=1.8.0
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(
        LooseVersion(torch.__version__) < LooseVersion("1.8.0"),
        reason="PyTorch exporter don't support nonzero until version 1.8.0",
    )
    def test_model_label_encoder_str_onnxml(self):
        model = LabelEncoder()
        data = [
            "paris",
            "milan",
            "amsterdam",
            "tokyo",
        ]
        model.fit(data)

        onnx_ml_model = convert_sklearn(model, initial_types=[("input", StringTensorType_onnx([4]))])

        onnx_model = convert(onnx_ml_model, "onnx", data)

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: data}
        onnx_ml_pred = session.run(output_names, inputs)

        # Get the predictions for the ONNX model
        onnx_pred = onnx_model.transform(data)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[0], onnx_pred, rtol=1e-06, atol=1e-06)

    # Test LabelEncoder String failcase for torch < 1.8.0
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(
        LooseVersion(torch.__version__) >= LooseVersion("1.8.0"),
        reason="PyTorch exporter supports nonzero only from version 1.8.0 and should fail on older versions",
    )
    def test_le_string_raises_rt_onnx(self):
        warnings.filterwarnings("ignore")
        model = LabelEncoder()
        data = [
            "paris",
            "milan",
            "amsterdam",
            "tokyo",
        ]
        model.fit(data)

        onnx_ml_model = convert_sklearn(model, initial_types=[("input", StringTensorType_onnx([4]))])

        # Create ONNX model by calling converter, should raise error for strings
        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", data)

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
    )
    # if the model is corrupt, we should get a RuntimeError
    def test_onnx_label_encoder_converter_raises_rt(self):
        warnings.filterwarnings("ignore")
        model = LabelEncoder()
        X = np.array([1, 4, 5, 2, 0, 2], dtype=np.int64)
        model.fit(X)

        # generate test input
        onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])
        onnx_ml_model.graph.node[0].attribute[0].name = "".encode()

        self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", X)


if __name__ == "__main__":
    unittest.main()
