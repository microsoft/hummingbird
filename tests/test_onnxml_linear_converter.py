"""
Tests lightgbm->onnxmltools->hb conversion for lightgbm models.
"""
<<<<<<< HEAD


import unittest
import sys
import os
import numpy as np
=======
import unittest

import sys
import os
import pickle
import numpy as np
from onnxconverter_common.data_types import FloatTensorType

from hummingbird.ml import convert
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_installed
>>>>>>> pausing for now

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC, NuSVC

<<<<<<< HEAD
from hummingbird.ml import convert
from hummingbird.ml._utils import onnx_installed, onnx_ml_tools_installed, skl2onnx_installed

if skl2onnx_installed():
    # these 2 imports __must__ come before torch, or it will segfault
    from skl2onnx import convert_sklearn as convert_sklearn_onnx
    from skl2onnx.common.data_types import FloatTensorType as FloatTensorType_onnx
=======
>>>>>>> pausing for now
if onnx_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_sklearn
<<<<<<< HEAD

# this __must__ come __after__ skl2onnx import
import torch


class TestOnnxmlLinearClassifiers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOnnxmlLinearClassifiers, self).__init__(*args, **kwargs)

    def _test_regressor(self, classes):
        n_features = 20
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(classes, size=n_total)
        model = LogisticRegression(solver="liblinear", multi_class="ovr", fit_intercept=True)
        model.fit(X, y)

        # From Sklearn to ONNX-ml to get the input
        onnx_ml_model = convert_sklearn_onnx(model, initial_types=[("float_input", FloatTensorType_onnx([-1, n_features]))])

        # Create ONNX model
        onnx_model = convert(onnx_ml_model, "onnx", X)

        # Now call the new Pytorch model (converted from onnxml) with the data
        test_run = onnx_model(torch.from_numpy(self.X))[1].data.numpy()

        # compare converted onnx model to pytorch (should be same as above)
        pytorch_model = convert_sklearn(model)
        self.assertTrue(np.allclose(pytorch_model(torch.from_numpy(self.X))[1].data.numpy(), test_run))

        #### TODO: Pausing for now until we decide on target output

        # Get the predictions for the ONNX-ML model
        # session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        # output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        # onnx_ml_pred = [[] for i in range(len(output_names))]
        # inputs = {session.get_inputs()[0].name: X}
        # pred = session.run(output_names, inputs)
        # for i in range(len(output_names)):
        #     if output_names[i] == "label":
        #         onnx_ml_pred[1] = pred[i]
        #     else:
        #         onnx_ml_pred[0] = pred[i]

        # # Get the predictions for the ONNX model
        # session = ort.InferenceSession(onnx_model.SerializeToString())
        # onnx_pred = [[] for i in range(len(output_names))]
        # pred = session.run(output_names, inputs)
        # for i in range(len(output_names)):
        #     if output_names[i] == "label":
        #         onnx_pred[1] = pred[i]
        #     else:
        #         onnx_pred[0] = pred[i]

        # return onnx_ml_pred, onnx_pred, output_names
        #######

    # Check that ONNXML models can only target the ONNX backend.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_installed() and skl2onnx_installed()),
        reason="ONNXML-linear tests require ONNX, ORT and SKL2ONNX",
    )
    def test_logistic_regression_onnxml_binary(self):
        self._test_regressor(2)

    # Check that ONNXML models can only target the ONNX backend.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_installed() and skl2onnx_installed()),
        reason="ONNXML-linear tests require ONNX, ORT and SKL2ONNX",
    )
    def test_logistic_regression_onnxml_multi(self):
        self._test_regressor(3)
=======
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
>>>>>>> pausing for now


if __name__ == "__main__":
    unittest.main()
