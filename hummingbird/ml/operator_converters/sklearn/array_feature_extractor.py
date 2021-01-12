# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn feature selectors: SelectKBest, SelectPercentile, VarianceThreshold.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._array_feature_extractor_implementations import ArrayFeatureExtractor


def convert_sklearn_select_k_best(operator, device, extra_config):
    """
    Converter for `sklearn.feature_selection.SelectKBest`.

    Args:
        operator: An operator wrapping a `sklearn.feature_selection.SelectKBest` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    indices = np.array([i for i, val in enumerate(operator.raw_operator.get_support()) if val])
    return ArrayFeatureExtractor(operator, np.ascontiguousarray(indices), device)


def convert_sklearn_variance_threshold(operator, device, extra_config):
    """
    Converter for `sklearn.feature_selection.VarianceThreshold`.

    Args:
        operator: An operator wrapping a `sklearn.feature_selection.VarianceThreshold` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    var = operator.raw_operator.variances_
    threshold = operator.raw_operator.threshold
    indices = np.array([i for i in range(len(var)) if var[i] > threshold])
    return ArrayFeatureExtractor(operator, np.ascontiguousarray(indices), device)


def convert_sklearn_select_percentile(operator, device, extra_config):
    """
    Converter for `sklearn.feature_selection.SelectPercentile`.

    Args:
        operator: An operator wrapping a `sklearn.feature_selection.SelectPercentile` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    indices = np.array([i for i, val in enumerate(operator.raw_operator.get_support()) if val])
    return ArrayFeatureExtractor(operator, np.ascontiguousarray(indices), device)


register_converter("SklearnSelectKBest", convert_sklearn_select_k_best)
register_converter("SklearnVarianceThreshold", convert_sklearn_variance_threshold)
register_converter("SklearnSelectPercentile", convert_sklearn_select_percentile)
