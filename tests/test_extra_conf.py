"""
Tests extra configurations.
"""
from distutils.version import LooseVersion
import psutil
import unittest
import warnings
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import torch

import hummingbird.ml
from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed
from hummingbird.ml import constants


class TestExtraConf(unittest.TestCase):
    # Test default number of threads. It will only work on mac after 1.6 https://github.com/pytorch/pytorch/issues/43036
    @unittest.skipIf(
        sys.platform == "darwin" and LooseVersion(torch.__version__) <= LooseVersion("1.6.0"),
        reason="PyTorch has a bug on mac related to multi-threading",
    )
    def test_torch_deafault_n_threads(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")

        self.assertIsNotNone(hb_model)
        self.assertTrue(torch.get_num_threads() == psutil.cpu_count(logical=False))
        self.assertTrue(torch.get_num_interop_threads() == 1)

    # Test one thread in pytorch.
    def test_torch_one_thread(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.N_THREADS: 1})

        self.assertIsNotNone(hb_model)
        self.assertTrue(torch.get_num_threads() == 1)
        self.assertTrue(torch.get_num_interop_threads() == 1)

    # Test default number of threads onnx.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_deafault_n_threads(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X)

        self.assertIsNotNone(hb_model)
        self.assertTrue(hb_model._session.get_session_options().intra_op_num_threads == psutil.cpu_count(logical=False))
        self.assertTrue(hb_model._session.get_session_options().inter_op_num_threads == 1)

    # Test one thread onnx.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_one_thread(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X, extra_config={constants.N_THREADS: 1})

        self.assertIsNotNone(hb_model)
        self.assertTrue(hb_model._session.get_session_options().intra_op_num_threads == 1)
        self.assertTrue(hb_model._session.get_session_options().inter_op_num_threads == 1)

    # Test pytorch regressor with batching.
    def test_torch_regression_batch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test pytorch classifier with batching.
    def test_torch_classification_batch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test pytorch classifier with batching.
    def test_torch_iforest_batch(self):
        warnings.filterwarnings("ignore")
        num_classes = 2
        model = IsolationForest(n_estimators=10, max_samples=2)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.decision_function(X), hb_model.decision_function(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.score_samples(X), hb_model.score_samples(X), rtol=1e-06, atol=1e-06)

    # Test pytorch regressor with batching and uneven rows.
    def test_torch_batch_regression_uneven(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(105, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=105)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test pytorch classification with batching and uneven rows.
    def test_torch_batch_classification_uneven(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(105, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=105)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test pytorch transform with batching and uneven rows.
    def test_torch_batch_transform(self):
        warnings.filterwarnings("ignore")
        model = StandardScaler(with_mean=True, with_std=True)
        np.random.seed(0)
        X = np.random.rand(105, 200)
        X = np.array(X, dtype=np.float32)

        model.fit(X)

        hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.transform(X), hb_model.transform(X), rtol=1e-06, atol=1e-06)

    # Test torchscript regression with batching.
    def test_torchscript_regression_batch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(103, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=103)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test torchscript classification with batching.
    def test_torchscript_classification_batch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(103, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=103)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test torchscript iforest with batching.
    def test_torchscript_iforest_batch(self):
        warnings.filterwarnings("ignore")
        num_classes = 2
        model = IsolationForest(n_estimators=10, max_samples=2)
        np.random.seed(0)
        X = np.random.rand(103, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=103)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.decision_function(X), hb_model.decision_function(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.score_samples(X), hb_model.score_samples(X), rtol=1e-06, atol=1e-06)

    # Test torchscript transform with batching and uneven rows.
    def test_torchscript_batch_transform(self):
        warnings.filterwarnings("ignore")
        model = StandardScaler(with_mean=True, with_std=True)
        np.random.seed(0)
        X = np.random.rand(101, 200)
        X = np.array(X, dtype=np.float32)

        model.fit(X)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.transform(X), hb_model.transform(X), rtol=1e-06, atol=1e-06)

    # Test onnx transform with batching and uneven rows.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_batch_transform(self):
        warnings.filterwarnings("ignore")
        model = StandardScaler(with_mean=True, with_std=True)
        np.random.seed(0)
        X = np.random.rand(101, 200)
        X = np.array(X, dtype=np.float32)

        model.fit(X)

        hb_model = hummingbird.ml.convert(model, "onnx", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.transform(X), hb_model.transform(X), rtol=1e-06, atol=1e-06)

    # Test onnx regression with batching.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_regression_batch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingRegressor(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(103, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=103)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)

    # Test onnx classification with batching.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_classification_batch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(103, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=103)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test onnx iforest with batching.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_iforest_batch(self):
        warnings.filterwarnings("ignore")
        num_classes = 2
        model = IsolationForest(n_estimators=10, max_samples=2)
        np.random.seed(0)
        X = np.random.rand(103, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=103)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X, extra_config={constants.BATCH_SIZE: 10})

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict(X), hb_model.predict(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.decision_function(X), hb_model.decision_function(X), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(model.score_samples(X), hb_model.score_samples(X), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
