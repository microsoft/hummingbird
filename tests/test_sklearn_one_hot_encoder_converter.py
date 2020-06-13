"""
Tests sklearn OneHotEncoder converter
"""
import unittest

import numpy as np
import torch
from skl2pytorch import convert_sklearn
from skl2pytorch.common.data_types import Int64TensorType, Int32TensorType
from sklearn.preprocessing import OneHotEncoder


class TestSklearnOneHotEncoderConverter(unittest.TestCase):
    def test_model_one_hot_encoder_int(self):
        model = OneHotEncoder()
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int64)
        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Int64TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data).todense(), pytorch_model(torch.from_numpy(data)).data.numpy()))

    def test_model_one_hot_encoder_string(self):
        model = OneHotEncoder()
        data = [["a", "r", "x"], ["a", "r", "x"], ["aaaa", "r", "x"], ["a", "r", "xx"]]
        model.fit(data)

        # max word length is the smallest number which is divisible by 4 and larger than or equal to the length of any word
        max_word_length = 4
        num_columns = 3
        pytorch_model = convert_sklearn(model, [("input", Int32TensorType([4, 3, max_word_length // 4]))])
        self.assertTrue(pytorch_model is not None)

        pytorch_input = torch.from_numpy(np.array(data, dtype="|S" + str(max_word_length)).view(np.int32)).view(
            -1, num_columns, max_word_length // 4
        )
        self.assertTrue(np.allclose(model.transform(data).todense(), pytorch_model(pytorch_input).data.numpy()))


if __name__ == "__main__":
    unittest.main()
