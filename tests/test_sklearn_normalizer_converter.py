"""
Tests sklearn Normalizer converter
"""
import unittest

import numpy as np
import torch
from sklearn.preprocessing import Normalizer

import hummingbird.ml


class TestSklearnNormalizer(unittest.TestCase):
    def test_normalizer_converter(self):
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.float32)
        data_tensor = torch.from_numpy(data)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            pytorch_model = hummingbird.ml.convert(model, "pytorch")

            self.assertIsNotNone(pytorch_model)
            np.testing.assert_allclose(model.transform(data), pytorch_model.operator_map.SklearnNormalizer(data_tensor))


if __name__ == "__main__":
    unittest.main()
