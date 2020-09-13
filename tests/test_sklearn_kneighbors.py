"""
Tests sklearn linear classifiers (LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV) converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

import hummingbird.ml


class TestSklearnKNeighborsClassifiers(unittest.TestCase):

    def test_kneighbors_classifier(self, num_classes=2):
        model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        print(model.kneighbors(X))

        # torch_model = hummingbird.ml.convert(model, "torch")

        # self.assertTrue(torch_model is not None)
        # np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
