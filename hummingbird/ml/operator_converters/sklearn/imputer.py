# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn Imputers: SimpleImputer and MissingIndicator
"""
from .._physical_operator import PhysicalOperator
import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from .._imputer_implementations import SimpleImputer, MissingIndicator


def convert_sklearn_simple_imputer(operator, device, extra_config):
    """
    Converter for `sklearn.impute.SimpleImputer`

    Args:
        operator: An operator wrapping a `sklearn.impute.SimpleImputer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    return SimpleImputer(operator, device)


def convert_sklearn_missing_indicator(operator, device, extra_config):
    """
    Converter for `sklearn.impute.MissingIndicator`
    Args:
        operator: An operator wrapping a `sklearn.impute.MissingIndicator` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy
    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    return MissingIndicator(operator, device)


register_converter("SklearnImputer", convert_sklearn_simple_imputer)
register_converter("SklearnSimpleImputer", convert_sklearn_simple_imputer)
register_converter("SklearnMissingIndicator", convert_sklearn_missing_indicator)
