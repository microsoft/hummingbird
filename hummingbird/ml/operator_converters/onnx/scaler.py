# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for ONNX-ML Scaler.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._scaler_implementations import Scaler


def convert_onnx_scaler(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.Scaler`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.Scaler` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    offset = scale = None
    for attr in operator.raw_operator.origin.attribute:
        if attr.name == "offset":
            offset = np.array(attr.floats)
        if attr.name == "scale":
            scale = np.array(attr.floats)

    if any(v is None for v in [offset, scale]):
        raise RuntimeError("Error parsing Scalar, found unexpected None")

    return Scaler(operator, offset, scale, device)


register_converter("ONNXMLScaler", convert_onnx_scaler)
