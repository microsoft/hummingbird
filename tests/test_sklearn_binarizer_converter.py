"""
Tests sklearn Binarizer converter
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.preprocessing import Binarizer

import hummingbird.ml


class TestSklearnBinarizer(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
