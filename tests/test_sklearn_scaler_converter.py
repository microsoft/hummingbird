"""
Tests scikit scaler converter.
"""
import unittest
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, MinMaxScaler, StandardScaler

import hummingbird.ml
from hummingbird.ml._utils import tvm_installed
from hummingbird.ml import constants


class TestSklearnScalerConverter(unittest.TestCase):
    def test_robust_scaler_floats(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = RobustScaler(with_centering=False, with_scaling=False)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = RobustScaler(with_centering=False, with_scaling=True)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = RobustScaler(with_centering=True, with_scaling=False)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = RobustScaler(with_centering=True, with_scaling=True)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    def test_max_abs_scaler_floats(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = MaxAbsScaler()
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    def test_min_max_scaler_floats(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = MinMaxScaler()
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    def test_standard_scaler_floats(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = StandardScaler(with_mean=False, with_std=False)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = StandardScaler(with_mean=True, with_std=False)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = StandardScaler(with_mean=True, with_std=True)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    # Float 64 data tests
    def test_float64_robust_scaler_floats(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data_tensor = torch.from_numpy(data)

        model = RobustScaler(with_centering=False, with_scaling=False)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    # Tests with TVM backend
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_standard_scaler_floats_tvm(self):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = StandardScaler(with_mean=False, with_std=False)
        model.fit(data)
        tvm_model = hummingbird.ml.convert(model, "tvm", data, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})
        self.assertIsNotNone(tvm_model)
        np.testing.assert_allclose(model.transform(data), tvm_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = StandardScaler(with_mean=True, with_std=False)
        model.fit(data)
        tvm_model = hummingbird.ml.convert(model, "tvm", data, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})
        self.assertIsNotNone(tvm_model)
        np.testing.assert_allclose(model.transform(data), tvm_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

        model = StandardScaler(with_mean=True, with_std=True)
        model.fit(data)
        tvm_model = hummingbird.ml.convert(model, "tvm", data, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})
        self.assertIsNotNone(tvm_model)
        np.testing.assert_allclose(model.transform(data), tvm_model.transform(data_tensor), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
