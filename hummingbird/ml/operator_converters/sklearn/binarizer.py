# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn Binarizer.
"""

from onnxconverter_common.registration import register_converter

from .._binarizer_implementations import Binarizer


def convert_sklearn_binarizer(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.Binarizer`

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.Binarizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    return Binarizer(operator.raw_operator.threshold, device)


register_converter("SklearnBinarizer", convert_sklearn_binarizer)
