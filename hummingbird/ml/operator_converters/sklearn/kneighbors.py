# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn k neighbor models: KNeighborsClassifier.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._kneighbors_implementations import KNeighborsClassifierModel


def convert_sklearn_kneighbors_classification_model(operator, device, extra_config):
    """
    Converter for `sklearn.neighbors.KNeighborsClassifier`

    Args:
        operator: An operator wrapping a `sklearn.neighbors.KNeighborsClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    return KNeighborsClassifierModel()

register_converter("SklearnKNeighborsClassifier", convert_sklearn_kneighbors_classification_model)