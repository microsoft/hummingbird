"""
Tests sklearn Normalizer converter
"""
import unittest
import warnings

import numpy as np
import torch

from sklearn.impute import MissingIndicator

import hummingbird.ml


class TestSklearnMissingIndicator(unittest.TestCase):
    def _test_sklearn_missing_indic(self, model, data):
        data_tensor = torch.from_numpy(data)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(torch_model)

        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
        )

    def test_missing_indicator_float_inputs(self):
        for features in ["all", "missing-only"]:
            model = MissingIndicator(features=features)
            data = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
            model.fit(data)
            self._test_sklearn_missing_indic(model, data)

    def test_missing_indicator_float_inputs_isnan_false(self):
        for features in ["all", "missing-only"]:
            model = MissingIndicator(features=features, missing_values=0)
            data = np.array([[1, 2], [0, 3], [7, 6]], dtype=np.float32)
            model.fit(data)
            self._test_sklearn_missing_indic(model, data)


if __name__ == "__main__":
    unittest.main()
