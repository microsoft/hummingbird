"""
Tests extra configurations.
"""
from distutils.version import LooseVersion
import psutil
import unittest
import warnings
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from onnxconverter_common.data_types import FloatTensorType
import torch

import hummingbird.ml
from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed, lightgbm_installed
from hummingbird.ml import constants

if lightgbm_installed():
    import lightgbm as lgb
if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_lightgbm


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

    # Check converter whit model name set as extra_config.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    @unittest.skipIf(not lightgbm_installed(), reason="LightGBM test requires LightGBM installed")
    def test_lightgbm_pytorch_extra_config(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model = lgb.LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=9
        )

        # Create ONNX model
        model_name = "hummingbird.ml.test.lightgbm"
        onnx_model = hummingbird.ml.convert(onnx_ml_model, "onnx", extra_config={constants.ONNX_OUTPUT_MODEL_NAME: model_name})

        assert onnx_model.model.graph.name == model_name


if __name__ == "__main__":
    unittest.main()
