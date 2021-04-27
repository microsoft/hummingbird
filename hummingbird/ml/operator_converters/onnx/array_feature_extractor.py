# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for ONNX-ML Array Feature Extractor.
"""

from onnxconverter_common.registration import register_converter

from .. import constants
from .._array_feature_extractor_implementations import ArrayFeatureExtractor


def convert_onnx_array_feature_extractor(operator, device, extra_config):
    """
    Converter for `ai.onnx.ml.ArrayFeatureExtractor`.

    Args:
        operator: An operator wrapping a `ai.onnx.ml.ArrayFeatureExtractor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    column_indices = []
    initializers = extra_config[constants.ONNX_INITIALIZERS]
    operator_inputs = operator.raw_operator.origin.input
    column_indices = None
    for input_ in operator_inputs:
        if input_ in initializers:
            assert column_indices is None, "More than one ArrayFeatureExtractor input matches with stored initializers."
            column_indices = list(initializers[input_].int64_data)
            if len(column_indices) == 0:
                # If we are here it means that the column indices were not int64.
                column_indices = list(initializers[input_].int32_data)
            assert len(column_indices) > 0, "Cannot convert ArrayFeatureExtractor with empty column indices."

    return ArrayFeatureExtractor(operator, column_indices, device)


register_converter("ONNXMLArrayFeatureExtractor", convert_onnx_array_feature_extractor)
