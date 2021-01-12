# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn Normalizer.
"""

from onnxconverter_common.registration import register_converter

from .._normalizer_implementations import Normalizer


def convert_sklearn_normalizer(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.Normalizer`

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.Normalizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    return Normalizer(operator, operator.raw_operator.norm, device)


register_converter("SklearnNormalizer", convert_sklearn_normalizer)
