"""
Tests Sklearn GradientBoostingClassifier converters.
"""
import unittest
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import hummingbird.ml
from hummingbird.ml.exceptions import MissingBackend


class TestBackends(unittest.TestCase):
    # Test backends are browsable
    def test_backends(self):
        warnings.filterwarnings("ignore")
        self.assertTrue(len(hummingbird.ml.backends) > 0)

    # Test backends are not case sensitive
    def test_backends_case_sensitive(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tOrCh")
        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test pytorch is still a valid backend name
    def test_backends_pytorch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "pytOrCh")
        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test not supported backends
    def test_unsupported_backend(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test backends are not case sensitive
        self.assertRaises(MissingBackend, hummingbird.ml.convert, model, "scala")


if __name__ == "__main__":
    unittest.main()
