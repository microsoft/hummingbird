# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for ONNX-ML Imputer.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._imputer_implementations import SimpleImputer


def convert_onnx_imputer(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.Imputer`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.Imputer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    stats = missing = None
    for attr in operator.raw_operator.origin.attribute:
        if attr.name == "imputed_value_floats":
            stats = np.array(attr.floats).astype("float64")
        elif attr.name == "replaced_value_float":
            missing = attr.f

    if any(v is None for v in [stats, missing]):
        raise RuntimeError("Error parsing Imputer, found unexpected None. stats: {}, missing: {}", stats, missing)

    # ONNXML has no "strategy" field, but always behaves similar to SKL's constant: "replace missing values with fill_value"
    return SimpleImputer(operator, device, statistics=stats, missing=missing, strategy="constant")


register_converter("ONNXMLImputer", convert_onnx_imputer)
