"""
Tests sklearn OneHotEncoder converter
"""
import unittest

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import hummingbird.ml


class TestSklearnOneHotEncoderConverter(unittest.TestCase):
    def test_model_one_hot_encoder_int(self):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int64)
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "pytorch", np.array([4, 3], dtype=np.int64))

        self.assertIsNotNone(pytorch_model)
        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_one_hot_encoder_int32(self):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int32)
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "pytorch", np.array([4, 3], dtype=np.int32))

        self.assertIsNotNone(pytorch_model)
        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_one_hot_encoder_string(self):
        model = OneHotEncoder()
        data = [["a", "r", "x"], ["a", "r", "x"], ["aaaa", "r", "x"], ["a", "r", "xx"]]
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "pytorch")
        self.assertTrue(pytorch_model is not None)

        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_one_hot_encoder_string_not_mod4_len(self):
        model = OneHotEncoder()
        data = [["a", "r", "x"], ["a", "r", "x"], ["aaaaa", "r", "x"], ["a", "r", "xx"]]
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "pytorch")
        self.assertTrue(pytorch_model is not None)

        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    # TorchScript tests.
    def test_model_one_hot_encoder_ts_int(self):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int64)
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "torchscript", np.array([[4, 3, 1]], dtype=np.int64))

        self.assertIsNotNone(pytorch_model)
        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_one_hot_encoder_ts_int32(self):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int32)
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "torchscript", np.array([[4, 3, 1]], dtype=np.int32))

        self.assertIsNotNone(pytorch_model)
        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_one_hot_encoder_ts_string(self):
        model = OneHotEncoder()
        data = np.array([["a", "r", "x"], ["a", "r", "x"], ["aaaa", "r", "x"], ["a", "r", "xx"]])
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "torchscript", data)
        self.assertTrue(pytorch_model is not None)

        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_one_hot_encoder_ts_string_not_mod4_len(self):
        model = OneHotEncoder()
        data = [["a", "r", "x"], ["a", "r", "x"], ["aaaaa", "r", "x"], ["a", "r", "xx"]]
        model.fit(data)

        pytorch_model = hummingbird.ml.convert(model, "torchscript", data)
        self.assertTrue(pytorch_model is not None)

        np.testing.assert_allclose(model.transform(data).todense(), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
