# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxconverter_common.registration import register_converter

from .._discretizer_implementations import Binarizer

"""
Converter for ONNX-ML Binarizer.
"""


def convert_onnx_binarizer(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.Binarizer`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.Binarizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    threshold = None
    for attr in operator.raw_operator.origin.attribute:
        if attr.name == "threshold":
            threshold = attr.f
            break

    if threshold is None:
        raise RuntimeError("Error parsing Binarizer, found unexpected None")

    return Binarizer(operator, threshold, device)


register_converter("ONNXMLBinarizer", convert_onnx_binarizer)
