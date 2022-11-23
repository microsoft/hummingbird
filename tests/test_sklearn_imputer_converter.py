"""
Tests sklearn Imputers: MissingIndicator and SimpleImputer
"""
import unittest
import warnings

import numpy as np
import torch

from sklearn.impute import MissingIndicator, SimpleImputer

try:
    from sklearn.preprocessing import Imputer
except ImportError:
    # Imputer was deprecate in sklearn >= 0.22
    Imputer = None

from hummingbird.ml._utils import onnx_runtime_installed, tvm_installed
import hummingbird.ml


class TestSklearnSimpleImputer(unittest.TestCase):
    def _test_simple_imputer(self, model, data, backend):

        model.fit(data)

        hb_model = hummingbird.ml.convert(model, backend, data)

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(
            model.transform(data),
            hb_model.transform(data),
            rtol=1e-06,
            atol=1e-06,
        )

    def test_simple_imputer_float_inputs(self):
        model = SimpleImputer(strategy="mean", fill_value="nan")
        data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)

        for backend in ["torch", "torch.jit"]:
            self._test_simple_imputer(model, data, backend)

    def test_simple_imputer_no_nan_inputs(self):
        model = SimpleImputer(missing_values=0, strategy="most_frequent")
        data = np.array([[1, 2], [1, 3], [7, 6]], dtype=np.float32)

        for backend in ["torch", "torch.jit"]:
            self._test_simple_imputer(model, data, backend)

    def test_simple_imputer_nan_to_0(self):
        model = SimpleImputer(strategy="constant", fill_value=0)
        data = np.array([[1, 2], [1, 3], [7, 6]], dtype=np.float32)

        for backend in ["torch", "torch.jit"]:
            self._test_simple_imputer(model, data, backend)

    # TVM tests
    @unittest.skipIf(not (tvm_installed()), reason="TVM test requires TVM")
    def test_simple_imputer_float_inputs_tvm(self):
        model = SimpleImputer(strategy="mean", fill_value="nan")
        data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)

        self._test_simple_imputer(model, data, "tvm")

    @unittest.skipIf(not (tvm_installed()), reason="TVM test requires TVM")
    def test_simple_imputer_no_nan_inputs_tvm(self):
        model = SimpleImputer(missing_values=0, strategy="most_frequent")
        data = np.array([[1, 2], [1, 3], [7, 6]], dtype=np.float32)

        self._test_simple_imputer(model, data, "tvm")

    @unittest.skipIf(not (tvm_installed()), reason="TVM test requires TVM")
    def test_simple_imputer_nan_to_0_tvm(self):
        model = SimpleImputer(strategy="constant", fill_value=0)
        data = np.array([[1, 2], [1, 3], [7, 6]], dtype=np.float32)

        self._test_simple_imputer(model, data, "tvm")


class TestSklearnImputer(unittest.TestCase):
    def _test_imputer(self, model, data):

        data_tensor = torch.from_numpy(data)
        model.fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertIsNotNone(torch_model)
        np.testing.assert_allclose(
            model.transform(data),
            torch_model.transform(data_tensor),
            rtol=1e-06,
            atol=1e-06,
        )

    @unittest.skipIf(Imputer is None, reason="Imputer was deprecated in scikit-learn >= 0.22")
    def test_imputer_float_inputs(self):
        model = Imputer(strategy="mean")
        data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)

        self._test_imputer(model, data)

    @unittest.skipIf(Imputer is None, reason="Imputer was deprecated in scikit-learn >= 0.22")
    def test_imputer_no_nan_inputs(self):
        model = Imputer(missing_values=0, strategy="most_frequent")
        data = np.array([[1, 2], [1, 3], [7, 6]], dtype=np.float32)

        self._test_imputer(model, data)


class TestSklearnMissingIndicator(unittest.TestCase):
    def _test_sklearn_missing_indic(self, model, data, backend):
        data_tensor = torch.from_numpy(data)
        hb_model = hummingbird.ml.convert(model, backend, data)

        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(
            model.transform(data),
            hb_model.transform(data_tensor),
            rtol=1e-06,
            atol=1e-06,
        )

    def test_missing_indicator_float_inputs(self):
        for features in ["all", "missing-only"]:
            model = MissingIndicator(features=features)
            data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
            model.fit(data)

            for backend in ["torch", "torch.jit"]:
                self._test_sklearn_missing_indic(model, data, backend)

    def test_missing_indicator_float_inputs_isnan_false(self):
        for features in ["all", "missing-only"]:
            model = MissingIndicator(features=features, missing_values=0)
            data = np.array([[1, 2], [0, 3], [7, 6]], dtype=np.float32)
            model.fit(data)

            for backend in ["torch", "torch.jit"]:
                self._test_sklearn_missing_indic(model, data, backend)

    # TVM tests
    @unittest.skipIf(not (tvm_installed()), reason="TVM test requires TVM")
    def test_missing_indicator_float_inputs_tvm(self):
        for features in ["all", "missing-only"]:
            model = MissingIndicator(features=features)
            data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
            model.fit(data)

            self._test_sklearn_missing_indic(model, data, "tvm")

    @unittest.skipIf(not (tvm_installed()), reason="TVM test requires TVM")
    def test_missing_indicator_float_inputs_isnan_false_tvm(self):
        for features in ["all", "missing-only"]:
            model = MissingIndicator(features=features, missing_values=0)
            data = np.array([[1, 2], [0, 3], [7, 6]], dtype=np.float32)
            model.fit(data)

            self._test_sklearn_missing_indic(model, data, "tvm")


if __name__ == "__main__":
    unittest.main()
