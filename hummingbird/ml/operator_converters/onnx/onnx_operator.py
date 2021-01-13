# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for ONNX operators.
"""

import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from .. import constants
from .._physical_operator import PhysicalOperator
from .._pipeline_implementations import Concat


class Cast(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, to_type):
        super(Cast, self).__init__(logical_operator)

        assert to_type is not None

        self._to_type = to_type

    def forward(self, x):
        if self._to_type == 1:  # Cast to float
            return x.float()
        elif self._to_type == 7:  # Cast to long
            return x.long()
        elif self._to_type == 11:  # Cast to double
            return x.double()
        else:
            raise RuntimeError(
                "Cast to ONNX type {} not supported yet. Please fill an issue at https://github.com/microsoft/hummingbird".format(
                    self._to_type
                )
            )


class Reshape(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, shape):
        super(Reshape, self).__init__(logical_operator)

        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


def convert_onnx_cast(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Cast`.

    Args:
        operator: An operator wrapping a `ai.onnx.Cast` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    to_type = None

    for attr in operator.raw_operator.origin.attribute:
        if attr.name == "to":
            to_type = attr.i

    # Generate the model.
    return Cast(operator, to_type)


def convert_onnx_concat(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Concat`.

    Args:
        operator: An operator wrapping a `ai.onnx.Concat` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    # Generate the model.
    return Concat(operator)


def convert_onnx_reshape(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Reshape`.

    Args:
        operator: An operator wrapping a `ai.onnx.Reshape` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    shape = []
    initializers = extra_config[constants.ONNX_INITIALIZERS]
    shape = list(initializers[operator.raw_operator.origin.input[1]].int64_data)

    # Generate the model.
    return Reshape(operator, shape)


register_converter("ONNXMLCast", convert_onnx_cast)
register_converter("ONNXMLConcat", convert_onnx_concat)
register_converter("ONNXMLReshape", convert_onnx_reshape)
