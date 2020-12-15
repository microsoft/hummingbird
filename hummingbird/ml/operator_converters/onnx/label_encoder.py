# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for ONNX-ML LabelEncoder.
"""

import numpy as np
from onnxconverter_common.registration import register_converter


from .._label_encoder_implementations import NumericLabelEncoder, StringLabelEncoder


def convert_onnx_label_encoder(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.LabelEncoder`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.LabelEncoder` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    is_strings = False
    keys = None
    for attr in operator.original_operator.origin.attribute:
        if attr.name == "keys_int64s":
            keys = np.array(attr.ints)
        elif attr.name == "keys_strings":
            is_strings = True
            keys = np.array([x.decode("UTF-8") for x in attr.strings])

    if keys is None:
        raise RuntimeError("Error parsing LabelEncoder, found unexpected None for keys")

    if is_strings:
        return StringLabelEncoder(keys, device)
    else:
        return NumericLabelEncoder(keys, device)


register_converter("ONNXMLLabelEncoder", convert_onnx_label_encoder)
