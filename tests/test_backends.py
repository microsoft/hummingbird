"""
Tests Sklearn GradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from onnxconverter_common.data_types import FloatTensorType

import hummingbird.ml
from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, tvm_installed
from hummingbird.ml.exceptions import MissingBackend

if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_sklearn


class TestBackends(unittest.TestCase):
    # Test backends are browsable
    def test_backends(self):
        warnings.filterwarnings("ignore")
        self.assertTrue(len(hummingbird.ml.backends) > 0)

    # Test backends are not case sensitive
    def test_backends_case_sensitive(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tOrCh")
        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test pytorch is still a valid backend name
    def test_backends_pytorch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "pytOrCh")
        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test not supported backends
    def test_unsupported_backend(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test backends are not case sensitive
        self.assertRaises(MissingBackend, hummingbird.ml.convert, model, "scala")

    # Test torchscript requires test_data
    def test_torchscript_test_data(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test torcscript requires test_input
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch.jit")

    # Test TVM requires test_data
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_test_data(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test tvm requires test_input
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "tvm")

    # Test onnx requires test_data or initial_types
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_test_data(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=11
        )

        # Test onnx requires test_data
        self.assertRaises(RuntimeError, hummingbird.ml.convert, onnx_ml_model, "onnx")


if __name__ == "__main__":
    unittest.main()
