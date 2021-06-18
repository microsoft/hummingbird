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


class ArgMax(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, axis):
        super(ArgMax, self).__init__(logical_operator)

        self.axis = axis

    def forward(self, x):
        return torch.argmax(x, dim=self.axis)


class Sum(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator):
        super(Sum, self).__init__(logical_operator)

    def forward(self, *x):
        if len(x) > 1:
            x = torch.cat(x, dim=1)
        return torch.sum(*x)


class Add(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, val):
        super(Add, self).__init__(logical_operator)

        if val is not None:
            assert len(self.inputs) == 1, "Unexpected input length for Add val"
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.add(*x, self.val)
        return torch.add(*x)


class Sub(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, val):
        super(Sub, self).__init__(logical_operator)

        if val is not None:
            assert len(self.inputs) == 1, "Unexpected input length for Sub val"
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.sub(*x, self.val)
        return torch.sub(*x)


class Less(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, val):
        super(Less, self).__init__(logical_operator)

        self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, x):
        return torch.lt(x, self.val)


class Neg(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator):
        super(Neg, self).__init__(logical_operator)

    def forward(self, *x):
        return torch.neg(*x)


class Abs(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator):
        super(Abs, self).__init__(logical_operator)

    def forward(self, x):
        return torch.abs(x)


class Mul(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, val):
        super(Mul, self).__init__(logical_operator)

        if val is not None:
            assert len(self.inputs) == 1, "Unexpected input length for Mul val"
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.mul(*x, self.val)
        return torch.mul(*x)


class MatMul(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, val):
        super(MatMul, self).__init__(logical_operator)

        self.val = val

    def forward(self, x):
        return torch.mm(x, self.val)


class Div(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, val):
        super(Div, self).__init__(logical_operator)

        if val is not None:
            assert len(self.inputs) == 1, "Unexpected input length for Div val"
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.div(*x, self.val)
        return torch.div(*x)


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
    assert operator is not None

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
    assert operator is not None

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
    assert operator is not None

    shape = []
    initializers = extra_config[constants.ONNX_INITIALIZERS]
    shape = list(initializers[operator.raw_operator.origin.input[1]].int64_data)

    # Generate the model.
    return Reshape(operator, shape)


def convert_onnx_argmax(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ArgMax`.

    Args:
        operator: An operator wrapping a `ai.onnx.ArgMax` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    axis = None
    for attribute in operator.raw_operator.origin.attribute:
        if attribute.name == "axis":
            axis = attribute.i

    assert axis is not None

    # Generate the model.
    return ArgMax(operator, axis)


def convert_onnx_sum(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Sum`.

    Args:
        operator: An operator wrapping a `ai.onnx.Sum` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Generate the model.
    return Sum(operator)


def convert_onnx_add(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Add`.

    Args:
        operator: An operator wrapping a `ai.onnx.Add` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    initializers = extra_config[constants.ONNX_INITIALIZERS]
    val = None
    if operator.raw_operator.origin.input[1] in initializers:
        val = list(initializers[operator.raw_operator.origin.input[1]].float_data)

    # Generate the model.
    return Add(operator, val)


def convert_onnx_sub(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Sub`.

    Args:
        operator: An operator wrapping a `ai.onnx.Sub` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    initializers = extra_config[constants.ONNX_INITIALIZERS]
    val = None
    if operator.raw_operator.origin.input[1] in initializers:
        init = initializers[operator.raw_operator.origin.input[1]]
        if init.data_type == 11:
            val = list(init.double_data)
        elif init.data_type == 1:
            val = list(init.float_data)
        else:
            raise TypeError("Data type %r not supported for initializer %r." % (init.data_type, init))

    # Generate the model.
    return Sub(operator, val)


def convert_onnx_neg(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Neg`.

    Args:
        operator: An operator wrapping a `ai.onnx.Neg` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Generate the model.
    return Neg(operator)


def convert_onnx_abs(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Abs`.

    Args:
        operator: An operator wrapping a `ai.onnx.Abs` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Generate the model.
    return Abs(operator)


def convert_onnx_mul(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Mul`.

    Args:
        operator: An operator wrapping a `ai.onnx.Mul` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    initializers = extra_config[constants.ONNX_INITIALIZERS]
    val = None
    if operator.raw_operator.origin.input[1] in initializers:
        val = list(initializers[operator.raw_operator.origin.input[1]].float_data)

    # Generate the model.
    return Mul(operator, val)


def convert_onnx_mat_mul(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.MatMul`.

    Args:
        operator: An operator wrapping a `ai.onnx.MatMul` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    initializers = extra_config[constants.ONNX_INITIALIZERS]
    val = list(initializers[operator.raw_operator.origin.input[1]].float_data)

    # Generate the model.
    return MatMul(operator, val)


def convert_onnx_div(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Div`.

    Args:
        operator: An operator wrapping a `ai.onnx.Div` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    initializers = extra_config[constants.ONNX_INITIALIZERS]
    val = None
    if operator.raw_operator.origin.input[1] in initializers:
        init = initializers[operator.raw_operator.origin.input[1]]
        if init.data_type == 11:
            val = list(init.double_data)
        elif init.data_type == 1:
            val = list(init.float_data)
        else:
            raise TypeError("Data type %r not supported for initializer %r." % (init.data_type, init))

    # Generate the model.
    return Div(operator, val)


def convert_onnx_less(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.Less`.

    Args:
        operator: An operator wrapping a `ai.onnx.Less` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    initializers = extra_config[constants.ONNX_INITIALIZERS]
    val = list(initializers[operator.raw_operator.origin.input[1]].float_data)

    # Generate the model.
    return Less(operator, val)


register_converter("ONNXMLArgMax", convert_onnx_argmax)
register_converter("ONNXMLAbs", convert_onnx_abs)
register_converter("ONNXMLAdd", convert_onnx_add)
register_converter("ONNXMLCast", convert_onnx_cast)
register_converter("ONNXMLConcat", convert_onnx_concat)
register_converter("ONNXMLDiv", convert_onnx_div)
register_converter("ONNXMLLess", convert_onnx_less)
register_converter("ONNXMLMatMul", convert_onnx_mat_mul)
register_converter("ONNXMLMul", convert_onnx_mul)
register_converter("ONNXMLNeg", convert_onnx_neg)
register_converter("ONNXMLReshape", convert_onnx_reshape)
register_converter("ONNXMLSub", convert_onnx_sub)
register_converter("ONNXMLSum", convert_onnx_sum)
