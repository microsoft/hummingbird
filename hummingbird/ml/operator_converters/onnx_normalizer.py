# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import torch
from onnxconverter_common.registration import register_converter

from ._base_operator import BaseOperator
from ._normalizer_implementations import Normalizer


def convert_onnx_normalizer(operator, device, extra_config):
    """
    Converter for `ai.onnx.ml.Normalizer`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.Normalizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    return Normalizer(operator.raw_operator.norm, device)


register_converter("ONNXMLNormalizer", convert_onnx_normalizer)
