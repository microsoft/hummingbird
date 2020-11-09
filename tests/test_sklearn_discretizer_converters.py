"""
Tests sklearn discretizer converters: Binarizer, KBinsDiscretizer
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
from sklearn.datasets import load_breast_cancer

import hummingbird.ml


class TestSklearnDiscretizers(unittest.TestCase):
    # Test Binarizer on dummy data
    def test_binarizer_converter(self):
        data = np.array([[1, 2, -3], [4, -3, 0], [0, 1, 4], [0, -5, 6]], dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        for threshold in [0.0, 1.0, -2.0]:
            model = Binarizer(threshold=threshold)
            model.fit(data)

            torch_model = hummingbird.ml.convert(model, "torch")
            self.assertIsNotNone(torch_model)
            np.testing.assert_allclose(
                model.transform(data), torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06,
            )

    def _test_k_bin_discretizer_base(self, data):
        data_tensor = torch.from_numpy(data)

        for n_bins in [2, 3, 5, 10, 20]:
            for encode in ["ordinal", "onehot", "onehot-dense"]:
                model = KBinsDiscretizer(n_bins=n_bins, encode=encode)
                model.fit(data)

                torch_model = hummingbird.ml.convert(model, "torch")
                self.assertIsNotNone(torch_model)

                if encode == "onehot":
                    sk_output = model.transform(data).todense()
                else:
                    sk_output = model.transform(data)

                np.testing.assert_allclose(sk_output, torch_model.transform(data_tensor), rtol=1e-06, atol=1e-06)

    # Test KBinDiscretizer on dummy data
    def test_k_bins_discretizer_converter_dummy_data(self):
        data = np.array([[1, 2, -3], [4, -3, 0], [0, 1, 4], [0, -5, 6]], dtype=np.float32)
        self._test_k_bin_discretizer_base(data)

    # Test KBinDiscretizer on breast cancer data
    def test_k_bins_discretizer_converter_breast_cancer_data(self):
        self._test_k_bin_discretizer_base(load_breast_cancer().data)


if __name__ == "__main__":
    unittest.main()
