# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for ONNX-ML Array Feature Vectorizer.
"""

from onnxconverter_common.registration import register_converter

from .. import constants
from .._pipeline_implementations import Concat


def convert_onnx_feature_vectorizer(operator, device, extra_config):
    """
    Converter for `ai.onnx.ml.FeatureVectorizer.

    Args:
        operator: An operator wrapping a `ai.onnx.ml.FeatureVectorizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    return Concat(operator)


register_converter("ONNXMLFeatureVectorizer", convert_onnx_feature_vectorizer)
