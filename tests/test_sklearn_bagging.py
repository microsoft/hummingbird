import unittest
import numpy as np

from sklearn.svm import SVC, LinearSVR
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import hummingbird.ml
from hummingbird.ml import constants


class TestSklearnBagging(unittest.TestCase):
    def test_bagging_svc_1(self):
        X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
        clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)

        clf.fit(X, y)

        hb_model = hummingbird.ml.convert(clf, "torch")

        np.testing.assert_allclose(
            clf.predict_proba([[0, 0, 0, 0]]), hb_model.predict_proba(np.array([[0, 0, 0, 0]])), rtol=1e-06, atol=1e-06,
        )

    def test_bagging_svc(self):
        X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
        clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)

        clf.fit(X, y)

        hb_model = hummingbird.ml.convert(clf, "torch")

        np.testing.assert_allclose(
            clf.predict_proba([[0, 0, 0, 0]]), hb_model.predict_proba(np.array([[0, 0, 0, 0]])), rtol=1e-06, atol=1e-06,
        )

    def test_bagging_logistic_regression(self):
        X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
        clf = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=10, random_state=0)

        clf.fit(X, y)

        hb_model = hummingbird.ml.convert(clf, "torch")

        np.testing.assert_allclose(
            clf.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06,
        )

    def test_bagging_linear_svr_1(self):
        X, y = make_regression(n_samples=1000, n_features=8, n_informative=5, n_targets=1, random_state=0, shuffle=True)
        reg = BaggingRegressor(base_estimator=LinearSVR(), n_estimators=10, random_state=0)

        reg.fit(X, y)

        hb_model = hummingbird.ml.convert(reg, "torch")

        np.testing.assert_allclose(
            reg.predict([[0, 0, 0, 0, 0, 0, 0, 0]]),
            hb_model.predict(np.array([[0, 0, 0, 0, 0, 0, 0, 0]])),
            rtol=1e-05,
            atol=1e-05,
        )

    def test_bagging_linear_svr(self):
        X, y = make_regression(n_samples=1000, n_features=8, n_informative=5, n_targets=1, random_state=0, shuffle=True)
        reg = BaggingRegressor(base_estimator=LinearSVR(), n_estimators=10, random_state=0)

        reg.fit(X, y)

        hb_model = hummingbird.ml.convert(reg, "torch")

        np.testing.assert_allclose(
            reg.predict(X), hb_model.predict(X), rtol=1e-05, atol=1e-05,
        )

    def test_bagging_linear_regression(self):
        X, y = make_regression(n_samples=1000, n_features=8, n_informative=5, n_targets=1, random_state=0, shuffle=True)
        reg = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=10, random_state=0)

        reg.fit(X, y)

        hb_model = hummingbird.ml.convert(reg, "torch")

        np.testing.assert_allclose(
            reg.predict(X), hb_model.predict(X), rtol=1e-05, atol=1e-05,
        )


if __name__ == "__main__":
    unittest.main()
