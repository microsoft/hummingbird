# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn label encoder.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._label_encoder_implementations import NumericLabelEncoder, StringLabelEncoder


def convert_sklearn_label_encoder(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.LabelEncoder`

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.LabelEncoder` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    if all([type(x) in [int, np.int32, np.int64] for x in operator.raw_operator.classes_]):
        return NumericLabelEncoder(operator.raw_operator.classes_, device)
    else:
        return StringLabelEncoder(operator.raw_operator.classes_, device, extra_config)


register_converter("SklearnLabelEncoder", convert_sklearn_label_encoder)
