# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn MLP models: MLPClassifier, MLPRegressor
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._mlp_implementations import MLPModel, MLPClassificationModel


def convert_sklearn_mlp_classifier(operator, device, extra_config):
    """
    Converter for `sklearn.neural_network.MLPClassifier`

    Args:
        operator: An operator wrapping a `sklearn.neural_network.MLPClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    classes = operator.raw_operator.classes_
    if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
        raise RuntimeError("Hummingbird supports only integer labels for class labels.")

    activation = operator.raw_operator.activation
    weights = operator.raw_operator.coefs_
    biases = operator.raw_operator.intercepts_

    return MLPClassificationModel(operator, weights, biases, activation, classes, device)


def convert_sklearn_mlp_regressor(operator, device, extra_config):
    """
    Converter for `sklearn.neural_network.MLPRegressor`

    Args:
        operator: An operator wrapping a `sklearn.neural_network.MLPRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    activation = operator.raw_operator.activation
    weights = operator.raw_operator.coefs_
    biases = operator.raw_operator.intercepts_

    return MLPModel(operator, weights, biases, activation, device)


register_converter("SklearnMLPClassifier", convert_sklearn_mlp_classifier)
register_converter("SklearnMLPRegressor", convert_sklearn_mlp_regressor)
