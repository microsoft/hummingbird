"""
Base class for Bayesian Gaussian Mixture model implementation: (BayesianGaussianMixture).
"""

import torch
import torch.nn
from ._physical_operator import PhysicalOperator


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = torch.sum(torch.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1)
        return log_det_chol
    else:
        raise NotImplementedError(
            "Hummingbird does not currently support {} covariance type for BayesianGaussianMixture. The supported covariance type is full.".format(
                covariance_type
            )
        )


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)
    if covariance_type == "full":
        log_prob = torch.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = torch.mm(X, prec_chol) - torch.matmul(mu, prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), axis=1)
        log_gaussian_prob = (-0.5 * (n_features * torch.log(torch.FloatTensor([2 * torch.pi])) + log_prob)) + log_det
        return log_gaussian_prob
    else:
        raise NotImplementedError(
            "Hummingbird does not currently support {} covariance type for BayesianGaussianMixture. The supported covariance type is full.".format(
                covariance_type
            )
        )


def _compute_precision_cholesky(covariances, covariance_type):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = torch.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = torch.linalg.cholesky(torch.FloatTensor(covariance), upper=False)
            except torch.linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = torch.linalg.solve_triangular(cov_chol, torch.eye(n_features), upper=False).T
        return precisions_chol
    else:
        raise NotImplementedError(
            "Hummingbird does not currently support {} covariance type for BayesianGaussianMixture. The supported covariance type is full.".format(
                covariance_type
            )
        )


class BayesianGaussianMixture(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator,
                 weight_concentration_prior_type,
                 weight_concentration_,
                 means_,
                 covariances_,
                 covariance_type,
                 degrees_of_freedom_,
                 mean_precision_,
                 device):
        super(BayesianGaussianMixture, self).__init__(logical_operator, regression=True)

        if (weight_concentration_prior_type == 'dirichlet_process'):
            self.weight_concentration_prior_type = weight_concentration_prior_type
        else:
            raise RuntimeError("Unsupported weight_concentration_prior_type: {0}".format(weight_concentration_prior_type))

        if (covariance_type == 'full'):
            self.covariance_type = covariance_type
        else:
            raise RuntimeError("Unsupported covariance_type: {0}".format(covariance_type))

        self.weight_concentration_ = torch.nn.Parameter(torch.FloatTensor(weight_concentration_), requires_grad=False)
        self.means_ = torch.nn.Parameter(torch.FloatTensor(means_), requires_grad=False)
        precisions_cholesky_ = _compute_precision_cholesky(covariances_, covariance_type)
        self.precisions_cholesky_ = torch.nn.Parameter(torch.FloatTensor(precisions_cholesky_), requires_grad=False)
        self.degrees_of_freedom_ = torch.nn.Parameter(torch.FloatTensor(degrees_of_freedom_), requires_grad=False)
        self.mean_precision_ = torch.nn.Parameter(torch.FloatTensor(mean_precision_), requires_grad=False)

    def forward(self, X):
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == "dirichlet_process":
            digamma_sum = torch.digamma(self.weight_concentration_[0] + self.weight_concentration_[1])
            digamma_a = torch.digamma(self.weight_concentration_[0])
            digamma_b = torch.digamma(self.weight_concentration_[1])
            return (digamma_a - digamma_sum + torch.hstack((torch.Tensor([0]), torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1])))

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        log_gauss = _estimate_log_gaussian_prob(X, self.means_, self.precisions_cholesky_, self.covariance_type) - 0.5 * n_features * torch.log(self.degrees_of_freedom_)

        log_lambda = n_features * torch.log(torch.FloatTensor([2.0])) + torch.sum(
            torch.digamma(0.5 * (self.degrees_of_freedom_ - torch.arange(0, n_features)[:, None])),
            0,)

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)
