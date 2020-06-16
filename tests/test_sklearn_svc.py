"""
Tests sklearn SVC classifiers (LinearSVC, SVC, NuSVC) converters.
"""
import unittest

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC, NuSVC

import hummingbird.ml


class TestSklearnSVC(unittest.TestCase):
    def _test_linear_svc(self, num_classes, labels_shift=0):
        model = LinearSVC()
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model.predict(X), rtol=1e-5, atol=1e-6)

    # LinearSVC binary
    def test_linear_svc_bi(self):
        self._test_linear_svc(2)

    # LinearSVC multiclass
    def test_linear_svc_multi(self):
        self._test_linear_svc(3)

    # LinearSVC with class labels shifted
    def test_linear_svc_shifted(self):
        self._test_linear_svc(3, labels_shift=2)

    # SVC test function to be parameterized
    def _test_svc(self, num_classes, kernel="rbf", gamma=None, labels_shift=0):

        if gamma:
            model = SVC(kernel=kernel, gamma=gamma)
        else:
            model = SVC(kernel=kernel)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model.predict(X), rtol=1e-5, atol=1e-6)

    # SVC binary
    def test_svc_bi(self):
        self._test_svc(2)

    # SVC multiclass
    def test_svc_multi(self):
        self._test_svc(3)

    # SVC linear kernel
    def test_svc_linear(self):
        self._test_svc(2, kernel="linear")

    # SVC sigmoid kernel
    def test_svc_sigmoid(self):
        self._test_svc(3, kernel="sigmoid")

    # SVC poly kernel
    def test_svc_poly(self):
        self._test_svc(3, kernel="poly")

    # SVC with class labels shifted
    def test_svc_shifted(self):
        self._test_svc(3, labels_shift=2)

    # SVC with different gamma (default=’scale’)
    def test_svc_gamma(self):
        self._test_svc(3, gamma="auto")

    # NuSVC test function to be parameterized
    def _test_nu_svc(self, num_classes):
        model = NuSVC()
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model.predict(X), rtol=1e-4, atol=1e-6)

    # NuSVC binary
    def test_nu_svc_bi(self):
        self._test_nu_svc(2)

    # NuSVC multiclass
    def test_nu_svc_multi(self):
        self._test_nu_svc(3)

    # assert fail on unsupported kernel
    def test_sklearn_linear_model_raises_wrong_type(self):

        np.random.seed(0)
        X = np.random.rand(10, 10)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)
        model = SVC(kernel="precomputed").fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "pytorch")


if __name__ == "__main__":
    unittest.main()
