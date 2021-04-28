"""
Tests sklearn Normalizer converter
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.preprocessing import Normalizer

import hummingbird.ml
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_runtime_installed, tvm_installed


class TestSklearnNormalizer(unittest.TestCase):
    def test_normalizer_converter(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            torch_model = hummingbird.ml.convert(model, "torch")

            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(
                model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
            )

    def test_normalizer_converter_raises_wrong_type(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)

        model = Normalizer(norm="invalid")
        model.fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertRaises(RuntimeError, torch_model.transform, torch.from_numpy(data))

    # Float 64 data tests
    def test_float64_normalizer_converter(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data_tensor = torch.from_numpy(data)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            torch_model = hummingbird.ml.convert(model, "torch")

            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(
                model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
            )

    # ONNX backend
    @unittest.skipIf(not (onnx_runtime_installed()), reason="ONNX test requires ONNX and  ORT")
    def test_normalizer_converter_onnx(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            hb_model = hummingbird.ml.convert(model, "onnx", data)

            self.assertIsNotNone(hb_model)
            np.testing.assert_allclose(
                model.transform(data), hb_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
            )

    # TVM backend
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_normalizer_converter_tvm(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            torch_model = hummingbird.ml.convert(model, "tvm", data, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(
                model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
            )


if __name__ == "__main__":
    unittest.main()
