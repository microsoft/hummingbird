"""
Tests sklearn linear classifiers converter.
"""
import unittest

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC, NuSVC

import hummingbird.ml


class TestSklearnLinearClassifiers(unittest.TestCase):
    def test_linear_svc(self):
        """
        TODO: this may have a bug.
        """
        model = LinearSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_linear_svc_multi(self):
        model = LinearSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_svc(self):
        model = SVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_svc_multi(self):
        model = SVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_nu_svc(self):
        model = NuSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-4, atol=1e-6)

    def test_nu_svc_multi(self):
        model = NuSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = hummingbird.ml.convert(model, "pytorch")

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
