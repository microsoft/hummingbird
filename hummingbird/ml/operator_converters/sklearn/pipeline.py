# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for operators necessary for supporting scikit-learn Pipelines.
"""

import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from .. import constants
from .._array_feature_extractor_implementations import ArrayFeatureExtractor
from .._base_operator import BaseOperator


class Concat(BaseOperator, torch.nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *x):
        return torch.cat(x, dim=1)


class Multiply(BaseOperator, torch.nn.Module):
    def __init__(self, score):
        super(Multiply, self).__init__()

        self.score = score

    def forward(self, x):
        return x * self.score


def convert_sklearn_array_feature_extractor(operator, device, extra_config):
    """
    Converter for `sklearn.feature_selection.VarianceThreshold`.

    Args:
        operator: An operator wrapping a `sklearn.feature_selection.VarianceThreshold` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    indices = operator.column_indices
    return ArrayFeatureExtractor(np.ascontiguousarray(indices), device)


def convert_sklearn_concat(operator, device=None, extra_config={}):
    """
    Converter for concat operators injected when parsing Sklearn pipelines.

    Args:
        operator: An empty operator
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    return Concat()


def convert_sklearn_multiply(operator, device=None, extra_config={}):
    """
    Converter for multiply operators injected when parsing Sklearn pipelines.

    Args:
        operator: An empty operator
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None
    assert hasattr(operator, "operand")

    score = operator.operand

    # Generate the model.
    return Multiply(score)


register_converter("SklearnArrayFeatureExtractor", convert_sklearn_array_feature_extractor)
register_converter("SklearnConcat", convert_sklearn_concat)
register_converter("SklearnMultiply", convert_sklearn_multiply)
