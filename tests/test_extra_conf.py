"""
Tests extra configurations.
"""
from distutils.version import LooseVersion

try:
    import psutil
except ImportError:
    psutil = None
import unittest
import warnings
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
    @unittest.skipIf(psutil is None, reason="psutil is not installed")
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

    # # Test one thread in pytorch.
    # def test_torch_one_thread(self):
    #     warnings.filterwarnings("ignore")
    #     max_depth = 10
    #     num_classes = 2
    #     model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
    #     np.random.seed(0)
    #     X = np.random.rand(100, 200)
    #     X = np.array(X, dtype=np.float32)
    #     y = np.random.randint(num_classes, size=100)

    #     model.fit(X, y)

    #     hb_model = hummingbird.ml.convert(model, "torch", extra_config={constants.BATCH_SIZE: 10})

    #     self.assertIsNotNone(hb_model)

    #     hb_model.predict(X)


if __name__ == "__main__":
    unittest.main()
