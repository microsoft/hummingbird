# """
# Tests onnxml Normalizer converter
# """
# import unittest
# import warnings

# import numpy as np
# import torch
# from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV
# from sklearn.svm import LinearSVC, SVC, NuSVC

# from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
# from hummingbird.ml import convert

# if onnx_runtime_installed():
#     import onnxruntime as ort
# if onnx_ml_tools_installed():
#     from onnxmltools import convert_sklearn
#     from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorType_onnx


# class TestSklearnNormalizer(unittest.TestCase):
#     def _test_regressor(self, classes):
#         n_features = 20
#         n_total = 100
#         np.random.seed(0)
#         warnings.filterwarnings("ignore")
#         X = np.random.rand(n_total, n_features)
#         X = np.array(X, dtype=np.float32)
#         y = np.random.randint(classes, size=n_total)

#         # Create SKL model for testing
#         model = LogisticRegression(solver="liblinear", multi_class="ovr", fit_intercept=True)
#         model.fit(X, y)

#         # Create ONNX-ML model
#         onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

#         # Create ONNX model by calling converter
#         onnx_model = convert(onnx_ml_model, "onnx", X)
#         # Get the predictions for the ONNX-ML model
#         session = ort.InferenceSession(onnx_ml_model.SerializeToString())
#         output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
#         onnx_ml_pred = [[] for i in range(len(output_names))]
#         inputs = {session.get_inputs()[0].name: X}
#         onnx_ml_pred = session.run(output_names, inputs)

#         # Get the predictions for the ONNX model
#         session = ort.InferenceSession(onnx_model.SerializeToString())
#         onnx_pred = [[] for i in range(len(output_names))]
#         onnx_pred = session.run(output_names, inputs)

#         return onnx_ml_pred, onnx_pred

#     @unittest.skipIf(
#         not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
#     )
#     def test_logistic_regression_onnxml_binary(self, rtol=1e-06, atol=1e-06):
#         onnx_ml_pred, onnx_pred = self._test_regressor(2)

#         # Check that predicted values match
#         np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

#     @unittest.skipIf(
#         not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
#     )
#     def test_logistic_regression_onnxml_multi(self, rtol=1e-06, atol=1e-06):
#         onnx_ml_pred, onnx_pred = self._test_regressor(3)

#         # Check that predicted values match
#         np.testing.assert_allclose(onnx_ml_pred, onnx_pred, rtol=rtol, atol=atol)

#     # @unittest.skipIf(
#     #     not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test requires ONNX, ORT and ONNXMLTOOLS"
#     # )
#     # def test_onnx_linear_model_converter_raises_rt(self):
#     #     np.random.seed(0)
#     #     warnings.filterwarnings("ignore")
#     #     X = np.random.rand(20, 20)
#     #     X = np.array(X, dtype=np.float32)
#     #     # class size 1 should fail
#     #     y = np.random.randint(1, size=20)

#     #     # Create SKL model for testing
#     #     model = LogisticRegression(solver="liblinear", multi_class="ovr", fit_intercept=True)
#     #     model.fit(X, y)

#     #     # Create ONNX-ML model
#     #     onnx_ml_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType_onnx(X.shape))])

#     #     self.assertRaises(RuntimeError, convert, onnx_ml_model, "onnx", X)


# if __name__ == "__main__":
#     unittest.main()
