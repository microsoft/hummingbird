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
    assert operator is not None, "Cannot convert None operator"

    for attr in operator.original_operator.origin.attribute:
        if attr.name == "keys_int64s":
            return NumericLabelEncoder(operator, np.array(attr.ints), device)
        elif attr.name == "keys_strings":
            # Note that these lines will fail later on for pytorch < 1.8
            keys = np.array([x.decode("UTF-8") for x in attr.strings])
            return StringLabelEncoder(operator, keys, device, extra_config)

    # If we reach here, we have a parsing error.
    raise RuntimeError("Error parsing LabelEncoder, found unexpected None for keys")


register_converter("ONNXMLLabelEncoder", convert_onnx_label_encoder)
