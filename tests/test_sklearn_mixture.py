"""
Tests sklearn mixture converters
"""
import unittest

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import make_blobs

import hummingbird.ml


class TestSklearnMixture(unittest.TestCase):
    def _test_BayesianGaussianMixture(self,
                                      n_components,
                                      weight_concentration_prior_type='dirichlet_process',
                                      covariance_type='full',
                                      backend='torch',
                                      extra_config={}):
        X, y = make_blobs(n_samples=100, random_state=0)

        model = BayesianGaussianMixture(n_components=n_components, verbose=1, max_iter=100,
                                        covariance_type=covariance_type,
                                        weight_concentration_prior_type=weight_concentration_prior_type,
                                        random_state=0)
        model.fit(X)

        torch_model = hummingbird.ml.convert(model, backend, X, extra_config=extra_config)
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # BayesianGaussianMixture n_components=1
    def test_BayesianGaussianMixture_comp1(self):
        self._test_BayesianGaussianMixture(1)

    # BayesianGaussianMixture n_components=2
    def test_BayesianGaussianMixture_comp2(self):
        self._test_BayesianGaussianMixture(2)

    # BayesianGaussianMixture n_components=3
    def test_BayesianGaussianMixture_comp3(self):
        self._test_BayesianGaussianMixture(3)

    # BayesianGaussianMixture n_components=7
    def test_BayesianGaussianMixture_comp7(self):
        self._test_BayesianGaussianMixture(7)


if __name__ == "__main__":
    unittest.main()
