import numpy as np
from onnxconverter_common.registration import register_converter

from .._mixture_implementations import BayesianGaussianMixture


def convert_sklearn_BayesianGaussianMixture(operator, device, extra_config):
    """
    Converter for `sklearn.mixture.BayesianGaussianMixture`

    Args:
        operator: An operator wrapping a `sklearn.mixture.BayesianGaussianMixture` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    weight_concentration_prior_type = operator.raw_operator.weight_concentration_prior_type
    weight_concentration_ = np.array(operator.raw_operator.weight_concentration_)
    means_ = operator.raw_operator.means_
    covariances_ = operator.raw_operator.covariances_
    covariance_type = operator.raw_operator.covariance_type
    degrees_of_freedom_ = operator.raw_operator.degrees_of_freedom_
    mean_precision_ = operator.raw_operator.mean_precision_

    return BayesianGaussianMixture(operator,
                                   weight_concentration_prior_type,
                                   weight_concentration_,
                                   means_,
                                   covariances_,
                                   covariance_type,
                                   degrees_of_freedom_,
                                   mean_precision_,
                                   device)


register_converter("SklearnBayesianGaussianMixture", convert_sklearn_BayesianGaussianMixture)
