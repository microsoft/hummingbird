# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from ._base_operator import BaseOperator
from ._scaler_implementations import Scaler


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

    operator = operator.raw_operator
    offset = scale = None

    for attr in operator.origin.attribute:
        if attr.name == "offset":
            offset = np.array(attr.floats).astype("float32")
        if attr.name == "scale":
            scale = np.array(attr.floats).astype("float32")

    if any(v is None for v in [offset, scale]):
        print("offset: {}, scale: {}".format(offset, scale))
        raise RuntimeError("Error parsing Scalar, found unexpected None")

    return Scaler(operator, device)


register_converter("ONNXMLScaler", convert_onnx_scaler)
