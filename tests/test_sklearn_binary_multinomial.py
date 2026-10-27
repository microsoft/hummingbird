"""Binary multinomial logistic regression must retain sklearn probabilities."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from hummingbird.ml import convert


@pytest.mark.parametrize("estimator", [LogisticRegression, LogisticRegressionCV])
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_binary_multinomial_probabilities(estimator, solver, fit_intercept):
    x = np.array([[-3], [-2], [-1], [-0.5], [0.5], [1], [2], [3]], dtype=np.float32)
    y = np.array([-4, -4, -4, -4, 7, 7, 7, 7])
    kwargs = {"cv": 2, "Cs": [1.0]} if estimator is LogisticRegressionCV else {}
    model = estimator(solver=solver, multi_class="multinomial", fit_intercept=fit_intercept, **kwargs).fit(x, y)
    converted = convert(model, "torch", extra_config={"n_threads": 1})

    probabilities = converted.predict_proba(x)
    assert probabilities.shape == (len(x), 2)
    np.testing.assert_allclose(probabilities, model.predict_proba(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(converted.predict(x), model.predict(x))
