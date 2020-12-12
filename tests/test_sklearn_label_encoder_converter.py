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

        pytorch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.transform(data), pytorch_model.transform(data), rtol=1e-06, atol=1e-06)

    def test_model_label_encoder_str(self):
        model = LabelEncoder()
        data = [
            "paris",
            "tokyo",
            "amsterdam",
            "tokyo",
        ]
        model.fit(data)

        # max word length is the smallest number which is divisible by 4 and larger than or equal to the length of any word
        max_word_length = 4
        num_columns = 4
        pytorch_model = hummingbird.ml.convert(model, "torch")

        pytorch_input = torch.from_numpy(np.array(data, dtype="|S" + str(max_word_length)).view(np.int32)).view(
            -1, num_columns, max_word_length // 4
        )
        np.testing.assert_allclose(model.transform(data), pytorch_model.transform(pytorch_input), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
