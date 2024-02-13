"""
Tests scikit scaler converter.
"""
import unittest
import numpy as np
import sys
import torch
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, MinMaxScaler, StandardScaler

import hummingbird.ml
from hummingbird.ml._utils import tvm_installed, is_on_github_actions
from hummingbird.ml import constants


class TestSklearnScalerConverter(unittest.TestCase):
    def _test_robust_scaler_floats(self, with_centering, with_scaling, backend="torch"):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, backend, data)
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    def _test_standard_scaler_floats(self, with_mean, with_std, backend="torch"):
        # Generate a random 2D array with values in [0, 1000)
        np.random.seed(0)
        data = np.random.rand(100, 200) * 1000
        data = np.array(data, dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        model = StandardScaler(with_mean=with_mean, with_std=with_std)
        model.fit(data)
        torch_model = hummingbird.ml.convert(model, backend, data)
        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    def test_robust_scaler_floats_torch_false_false(self):
        self._test_robust_scaler_floats(False, False)

    def test_robust_scaler_floats_torch_true_false(self):
        self._test_robust_scaler_floats(True, False)

    def test_robust_scaler_floats_torch_falser_true(self):
        self._test_robust_scaler_floats(False, True)

    def test_robust_scaler_floats_torch_true_true(self):
        self._test_robust_scaler_floats(True, True)

    def test_standard_scaler_floats_torch_false_false(self):
        self._test_standard_scaler_floats(False, False)

    def test_standard_scaler_floats_torch_true_false(self):
        self._test_standard_scaler_floats(True, False)

    def test_standard_scaler_floats_torch_true_true(self):
        self._test_standard_scaler_floats(True, True)

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
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_standard_scaler_floats_tvm_false_false(self):
        self._test_standard_scaler_floats(False, False, "tvm")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_standard_scaler_floats_tvm_true_false(self):
        self._test_standard_scaler_floats(True, False, "tvm")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_standard_scaler_floats_tvm_true_true(self):
        self._test_standard_scaler_floats(True, True, "tvm")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_robust_scaler_floats_tvm_false_false(self):
        self._test_robust_scaler_floats(False, False, "tvm")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_robust_scaler_floats_tvm_true_false(self):
        self._test_robust_scaler_floats(True, False, "tvm")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_robust_scaler_floats_tvm_false_true(self):
        self._test_robust_scaler_floats(False, True, "tvm")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    @unittest.skipIf(
        ((sys.platform == "linux") and is_on_github_actions()),
        reason="This test is flaky on Ubuntu on GitHub Actions. See https://github.com/microsoft/hummingbird/pull/709 for more info.",
    )
    def test_robust_scaler_floats_tvm_true_true(self):
        self._test_robust_scaler_floats(True, True, "tvm")


if __name__ == "__main__":
    unittest.main()
