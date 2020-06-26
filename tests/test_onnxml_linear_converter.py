"""
Tests lightgbm->onnxmltools->hb conversion for lightgbm models.
"""
import unittest

import sys
import os
import pickle
import numpy as np
from onnxconverter_common.data_types import FloatTensorType

from hummingbird.ml import convert
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_installed

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC, NuSVC

if onnx_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_sklearn
if skltoonnx_installed():
    from skl2onnx import convert_sklearn as convert_sklearn_onnx


class TestONNXConverterLightGBM(unittest.TestCase):
    class TestOnnxmlLinearClassifiers(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            super(TestOnnxmlLinearClassifiers, self).__init__(*args, **kwargs)
            self.n_features = 20
            self.n_total = 100
            self.X = np.random.rand(self.n_total, self.n_features)
            self.X = np.array(self.X, dtype=np.float32)
            self.y2 = np.random.randint(2, size=self.n_total)
            self.y3 = np.random.randint(3, size=self.n_total)

        def _logistic_regression_onnxml(self, y):
            model = LogisticRegression(solver="liblinear", multi_class="ovr", fit_intercept=True)
            model.fit(self.X, y)

            # From Sklearn to ONNX-ml to get the input
            onnx_ml_model = convert_sklearn_onnx(
                model, initial_types=[("float_input", FloatTensorType_onnx([-1, self.n_features]))]
            )

            # call parsing functions
            new_pytorch_model = Onnx2PyTorchModel(onnx_ml_model).as_pytorch()

            # Now call the new Pytorch model (converted from onnxml) with the data
            test_run = new_pytorch_model(torch.from_numpy(self.X))[1].data.numpy()

            # compare converted onnx model to pytorch (should be same as above)
            pytorch_model = convert_sklearn(model, [("input", Float32TensorType([-1, self.n_features]))])
            self.assertTrue(np.allclose(pytorch_model(torch.from_numpy(self.X))[1].data.numpy(), test_run))

        # Check that ONNXML models can only target the ONNX backend.
        @unittest.skipIf(
            not (onnx_ml_tools_installed() and onnx_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
        )
        def test_logistic_regression_onnxml_binary(self):
            self._logistic_regression_onnxml(self.y2)

        # Check that ONNXML models can only target the ONNX backend.
        @unittest.skipIf(
            not (onnx_ml_tools_installed() and onnx_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
        )
        def test_logistic_regression_onnxml_multi(self):
            self._logistic_regression_onnxml(self.y3)


if __name__ == "__main__":
    unittest.main()
