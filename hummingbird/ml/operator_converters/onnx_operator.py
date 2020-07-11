# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for ONNX operators.
"""

import numpy as np
import torch

from onnxconverter_common.registration import register_converter

from . import constants
from ._base_operator import BaseOperator


class Cast(BaseOperator, torch.nn.Module):
    def __init__(self, to_type):
        super(Cast, self).__init__()

        assert to_type is not None

        self.to_type = to_type

    def forward(self, x):
        if self.to_type == 7:  # Cast to long
            return x.long()


def convert_onnx_cast(operator, device=None, extra_config={}):
    """
    Converter for `onnxml.TreeEnsembleClassifier`.

    Args:
        operator: An operator wrapping a `onnxml.TreeEnsembleClassifier` model
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
    return Cast(to_type)


register_converter("ONNXMLCast", convert_onnx_cast)
