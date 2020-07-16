# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from ._base_operator import BaseOperator
from ._one_hot_encoder_implementations import OneHotEncoderString, OneHotEncoder


def convert_onnx_one_hot_encoder(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.OneHotEncoder
`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.OneHotEncoder` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    pass


register_converter("ONNXMLOneHotEncoder", convert_onnx_one_hot_encoder)
