# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn one hot encoder.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .. import constants
from .._one_hot_encoder_implementations import OneHotEncoderString, OneHotEncoder


def convert_sklearn_one_hot_encoder(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.OneHotEncoder`

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.OneHotEncoder` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    if all(
        [
            np.array(c).dtype == object or np.array(c).dtype.kind in constants.SUPPORTED_STRING_TYPES
            for c in operator.raw_operator.categories_
        ]
    ):
        categories = [[str(x) for x in c.tolist()] for c in operator.raw_operator.categories_]
        return OneHotEncoderString(operator, categories, device, extra_config)
    else:
        return OneHotEncoder(operator, operator.raw_operator.categories_, device)


register_converter("SklearnOneHotEncoder", convert_sklearn_one_hot_encoder)
