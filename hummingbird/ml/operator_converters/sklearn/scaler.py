# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn scalers: RobustScaler, MaxAbsScaler, MinMaxScaler, StandardScaler.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._scaler_implementations import Scaler


def convert_sklearn_robust_scaler(operator, device, extra_config):
    assert operator is not None, "Cannot convert None operator"

    scale = operator.raw_operator.scale_
    if scale is not None:
        scale = np.reciprocal(scale)
    return Scaler(operator, operator.raw_operator.center_, scale, device)


def convert_sklearn_max_abs_scaler(operator, device, extra_config):
    assert operator is not None, "Cannot convert None operator"

    scale = operator.raw_operator.scale_
    if scale is not None:
        scale = np.reciprocal(scale)
    return Scaler(operator, np.array([0]), scale, device)


def convert_sklearn_min_max_scaler(operator, device, extra_config):
    assert operator is not None, "Cannot convert None operator"

    scale = np.array([x for x in operator.raw_operator.scale_])
    offset = np.array([-1.0 / x * y for x, y in zip(operator.raw_operator.scale_, operator.raw_operator.min_)])
    return Scaler(operator, offset, scale, device)


def convert_sklearn_standard_scaler(operator, device, extra_config):
    assert operator is not None, "Cannot convert None operator"

    scale = operator.raw_operator.scale_
    if scale is not None:
        scale = np.reciprocal(scale)
    return Scaler(operator, operator.raw_operator.mean_, scale, device)


register_converter("SklearnRobustScaler", convert_sklearn_robust_scaler)
register_converter("SklearnMaxAbsScaler", convert_sklearn_max_abs_scaler)
register_converter("SklearnMinMaxScaler", convert_sklearn_min_max_scaler)
register_converter("SklearnStandardScaler", convert_sklearn_standard_scaler)
