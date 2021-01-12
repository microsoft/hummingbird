# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn Naive Bayes models: BernoulliNB, GaussianNB, MultinomialNB
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._nb_implementations import BernoulliNBModel, GaussianNBModel


def convert_sklearn_bernouli_naive_bayes(operator, device, extra_config):
    """
    Converter for `sklearn.naive_bayes.BernoulliNB`

    Args:
        operator: An operator wrapping a `sklearn.naive_bayes.BernoulliNB` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    model = operator.raw_operator

    classes = model.classes_
    if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
        raise RuntimeError("Hummingbird supports only integer labels for class labels.")

    binarize = model.binarize

    neg_prob = np.log(1 - np.exp(model.feature_log_prob_))
    feature_log_prob_minus_neg_prob = (model.feature_log_prob_ - neg_prob).T
    jll_calc_bias = (model.class_log_prior_ + neg_prob.sum(1)).reshape(1, -1)

    return BernoulliNBModel(operator, classes, binarize, jll_calc_bias, feature_log_prob_minus_neg_prob, device)


def convert_sklearn_multinomial_naive_bayes(operator, device, extra_config):
    """
    Converter for `sklearn.naive_bayes.MultinomialNB`

    Args:
        operator: An operator wrapping a `sklearn.naive_bayes.MultinomialNB` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    model = operator.raw_operator

    classes = model.classes_
    if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
        raise RuntimeError("Hummingbird supports only integer labels for class labels.")

    feature_log_prob_minus_neg_prob = model.feature_log_prob_.T
    jll_calc_bias = model.class_log_prior_.reshape(1, -1)

    return BernoulliNBModel(operator, classes, None, jll_calc_bias, feature_log_prob_minus_neg_prob, device)


def convert_sklearn_gaussian_naive_bayes(operator, device, extra_config):
    """
    Converter for `sklearn.naive_bayes.GaussianNB`

    Args:
        operator: An operator wrapping a `sklearn.naive_bayes.GaussianNB` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    model = operator.raw_operator
    classes = model.classes_
    if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
        raise RuntimeError("Hummingbird supports only integer labels for class labels.")

    jll_calc_bias = np.log(model.class_prior_.reshape(-1, 1)) - 0.5 * np.sum(np.log(2.0 * np.pi * model.sigma_), 1).reshape(
        -1, 1
    )
    return GaussianNBModel(operator, classes, jll_calc_bias, model.theta_, model.sigma_, device)


register_converter("SklearnBernoulliNB", convert_sklearn_bernouli_naive_bayes)
register_converter("SklearnGaussianNB", convert_sklearn_gaussian_naive_bayes)
register_converter("SklearnMultinomialNB", convert_sklearn_multinomial_naive_bayes)
