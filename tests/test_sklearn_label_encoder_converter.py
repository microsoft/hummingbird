"""
Tests sklearn LabelEncoder converter
"""

import unittest

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import hummingbird.ml
from onnxconverter_common.data_types import Int32TensorType


class TestSklearnLabelEncoderConverter(unittest.TestCase):
    def test_model_label_encoder(self):
        model = LabelEncoder()
        data = np.array([1, 4, 5, 2, 0, 2], dtype=np.int32)
        model.fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.transform(data), torch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_label_encoder_int64_range(self):
        limits = np.iinfo(np.int64)
        data = np.array([0, 2**32, 2**32 + 1, -(2**32), limits.min, limits.max, 2**32], dtype=np.int64)
        model = LabelEncoder().fit(data)

        for backend in ("torch", "torch.jit"):
            with self.subTest(backend=backend):
                converted = hummingbird.ml.convert(model, backend, data, extra_config={"n_threads": 1})
                np.testing.assert_array_equal(model.transform(data), converted.transform(data))

    def test_model_label_encoder_str(self):
        model = LabelEncoder()
        data = [
            "paris",
            "tokyo",
            "amsterdam",
            "tokyo",
        ]
        model.fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")

        np.testing.assert_allclose(model.transform(data), torch_model.transform(data), rtol=1e-06, atol=1e-06)

    # if the user gives unseen string input, we should get a failed assert
    def test_skl_label_encoder_converter_raises_err(self):
        model = LabelEncoder()
        data = [
            "paris",
            "tokyo",
            "amsterdam",
            "tokyo",
        ]
        model.fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")

        # this isn't in the input data and should give an error.
        data[0] = "milan"

        self.assertRaises(AssertionError, torch_model.transform, data)


if __name__ == "__main__":
    unittest.main()
