"""
Tests sklearn linear classifiers (LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV) converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV

import hummingbird.ml


class TestSklearnLinearClassifiers(unittest.TestCase):

    # LogisticRegression test function to be parameterized
    def _test_logistic_regression(self, num_classes, solver="liblinear", multi_class="auto", labels_shift=0):
        if num_classes > 2:
            model = LogisticRegression(solver=solver, multi_class=multi_class, fit_intercept=True)
        else:
            model = LogisticRegression(solver="liblinear", fit_intercept=True)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # LogisticRegression binary
    def test_logistic_regression_bi(self):
        self._test_logistic_regression(2)

    # LogisticRegression multiclass with auto
    def test_logistic_regression_multi_auto(self):
        self._test_logistic_regression(3)

    # LogisticRegression with class labels shifted
    def test_logistic_regression_shifted_classes(self):
        self._test_logistic_regression(3, labels_shift=2)

    # LogisticRegression with multi+ovr
    def test_logistic_regression_multi_ovr(self):
        self._test_logistic_regression(3, multi_class="ovr")

    # LogisticRegression with multi+multinomial
    def test_logistic_regression_multi_multin(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(3, multi_class="multinomial", solver="sag")

    # LinearRegression test function to be parameterized
    def _test_linear_regression(self, y_input):
        model = LinearRegression()

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = y_input

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # LinearRegression with ints
    def test_linear_regression_int(self):
        np.random.seed(0)
        self._test_linear_regression(np.random.randint(2, size=100))

    # LinearRegression with floats
    def test_linear_regression_float(self):
        np.random.seed(0)
        self._test_linear_regression(np.random.rand(100))

    # LogisticRegressionCV test function to be parameterized
    def _test_logistic_regression_cv(self, num_classes, solver="liblinear", multi_class="auto", labels_shift=0):
        if num_classes > 2:
            model = LogisticRegressionCV(solver=solver, multi_class=multi_class, fit_intercept=True)
        else:
            model = LogisticRegressionCV(solver="liblinear", fit_intercept=True)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # LogisticRegressionCV with 2 classes
    def test_logistic_regression_cv_bi(self):
        self._test_logistic_regression_cv(2)

    # LogisticRegressionCV with 3 classes
    def test_logistic_regression_cv_multi(self):
        self._test_logistic_regression_cv(3)

    # LogisticRegressionCV with shifted classes
    def test_logistic_regression_cv_shifted_classes(self):
        self._test_logistic_regression_cv(3, labels_shift=2)

    # LogisticRegressionCV with multi+ovr
    def test_logistic_regression_cv_multi_ovr(self):
        self._test_logistic_regression_cv(3, multi_class="ovr")

    # LogisticRegressionCV with multi+multinomial
    def test_logistic_regression_cv_multi_multin(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression_cv(3, multi_class="multinomial", solver="sag")

    # SGDClassifier test function to be parameterized
    def _test_sgd_classifier(self, num_classes):

        model = SGDClassifier(loss="log")

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # SGDClassifier with 2 classes
    def test_sgd_classifier_bi(self):
        self._test_sgd_classifier(2)

    # SGDClassifier with 3 classes
    def test_sgd_classifier_multi(self):
        self._test_sgd_classifier(3)

    # Failure Cases
    def test_sklearn_linear_model_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = SGDClassifier().fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch")

    # Float 64 data tests
    def test_float64_linear_regression(self):
        model = LinearRegression()

        np.random.seed(0)
        X = np.random.rand(100, 200)
        y = np.random.randint(2, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    def test_float64_sgd_classifier(self):

        model = SGDClassifier(loss="log")

        np.random.seed(0)
        num_classes = 3
        X = np.random.rand(100, 200)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
