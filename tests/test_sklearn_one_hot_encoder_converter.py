"""
Tests sklearn OneHotEncoder converter
"""
import unittest

import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
import hummingbird.ml

from packaging.version import Version, parse


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

    @unittest.skipIf(parse(sklearn.__version__) < Version("1.1"), "Skipping test because sklearn version is too old.")
    def test_infrequent_if_exists_str(self):

        # This test is a copy of the test in sklearn.
        # https://github.com/scikit-learn/scikit-learn/blob/
        #       ecb9a70e82d4ee352e2958c555536a395b53d2bd/sklearn/preprocessing/tests/test_encoders.py#L868

        X_train = np.array([["a"] * 5 + ["b"] * 2000 + ["c"] * 10 + ["d"] * 3]).T
        model = OneHotEncoder(
            categories=[["a", "b", "c", "d"]],
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
            min_frequency=15,

        ).fit(X_train)
        np.testing.assert_array_equal(model.infrequent_categories_, [["a", "c", "d"]])

        pytorch_model = hummingbird.ml.convert(model, "torch", device="cpu")
        self.assertIsNotNone(pytorch_model)

        X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
        expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
        orig = model.transform(X_test)
        np.testing.assert_allclose(expected, orig)

        hb = pytorch_model.transform(X_test)

        np.testing.assert_allclose(orig, hb, rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(orig.shape, hb.shape, rtol=1e-06, atol=1e-06)

    @unittest.skipIf(parse(sklearn.__version__) < Version("1.1"), "Skipping test because sklearn version is too old.")
    def test_infrequent_if_exists_int(self):

        X_train = np.array([[1] * 5 + [2] * 2000 + [3] * 10 + [4] * 3]).T
        model = OneHotEncoder(
            categories=[[1, 2, 3, 4]],
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
            min_frequency=15,
        ).fit(X_train)
        np.testing.assert_array_equal(model.infrequent_categories_, [[1, 3, 4]])

        pytorch_model = hummingbird.ml.convert(model, "torch", device="cpu")
        self.assertIsNotNone(pytorch_model)

        X_test = [[2], [1], [3], [4], [5]]
        expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
        orig = model.transform(X_test)
        np.testing.assert_allclose(expected, orig)

        hb = pytorch_model.transform(X_test)

        np.testing.assert_allclose(orig, hb, rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(orig.shape, hb.shape, rtol=1e-06, atol=1e-06)

    # @unittest.skipIf(parse(sklearn.__version__) < Version("1.1"), "Skipping test because sklearn version is too old.")
    # def test_2d_infrequent(self):

    #     X_train = np.array([[10.0, 1.0]] * 10
    #                         + [[22.0, 1.0]] * 2
    #                         + [[10.0, 1.0]] * 3
    #                         + [[14.0, 2.0]] * 1)
    #     ohe = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=0.49).fit(X_train)
    #     #ohe = OneHotEncoder(sparse_output=False).fit(X_train)

    #     hb = hummingbird.ml.convert(ohe, "pytorch", device="cpu")

    #     np.testing.assert_allclose(ohe.transform(X_train), hb.transform(X_train), rtol=1e-06, atol=1e-06)

    # @unittest.skipIf(parse(sklearn.__version__) < Version("1.1"), "Skipping test because sklearn version is too old.")
    # def test_user_provided_example(self):

    #     from sklearn.impute import SimpleImputer
    #     from sklearn.pipeline import Pipeline

    #     X_train = np.array([[22.0, 1.0, 0.0, 1251.0, 123.0, 124.0, 123.0, 0, 0, 0, 0, 0, 0, 0, 0] * 10
    #                         + [10.0, 1.0, 0.0, 1251.0, 123.0, 124.0, 134.0, 0, 0, 0, 0, 0, 0, 0, 0] * 2
    #                         + [14.0, 1.0, 0.0, 1251.0, 123.0, 124.0, 134.0, 0, 0, 0, 0, 0, 0, 0, 0] * 3
    #                         + [12.0, 2.0, 0.0, 1251.0, 123.0, 124.0, 134.0, 0, 0, 0, 0, 0, 0, 0, 0] * 1])
    #     pipe = Pipeline(
    #         [
    #             ("imputer", SimpleImputer(strategy="most_frequent")),
    #             ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=9)),
    #         ],
    #         verbose=True,
    #     ).fit(X_train)

    #     hb = hummingbird.ml.convert(pipe, "pytorch", device="cpu")

    #     np.testing.assert_allclose(pipe.transform(X_train), hb.transform(X_train), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
