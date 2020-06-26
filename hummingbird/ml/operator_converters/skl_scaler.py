# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch
import numpy as np
from onnxconverter_common.registration import register_converter

from ._base_operator import BaseOperator


class Scaler(BaseOperator, torch.nn.Module):
    def __init__(self, offset, scale, device):
        super(Scaler, self).__init__()
        self.transformer = True

        if offset is not None:
            self.offset = torch.nn.Parameter(torch.FloatTensor([offset]), requires_grad=False)
        else:
            self.offset = None

        if scale is not None:
            self.scale = torch.nn.Parameter(torch.FloatTensor([scale]), requires_grad=False)
        else:
            self.scale = None

    def forward(self, x):
        if self.offset is not None:
            x = x - self.offset

        if self.scale is not None:
            x = x * self.scale

        return x


def convert_sklearn_robust_scaler(operator, device, extra_config):
    scale = operator.raw_operator.scale_
    if scale is not None:
        scale = np.reciprocal(scale)
    return Scaler(operator.raw_operator.center_, scale, device)


def convert_sklearn_max_abs_scaler(operator, device, extra_config):
    scale = operator.raw_operator.scale_
    if scale is not None:
        scale = np.reciprocal(scale)
    return Scaler(0, scale, device)


def convert_sklearn_min_max_scaler(operator, device, extra_config):
    scale = [x for x in operator.raw_operator.scale_]
    offset = [-1.0 / x * y for x, y in zip(operator.raw_operator.scale_, operator.raw_operator.min_)]
    return Scaler(offset, scale, device)


def convert_sklearn_standard_scaler(operator, device, extra_config):
    scale = operator.raw_operator.scale_
    if scale is not None:
        scale = np.reciprocal(scale)
    return Scaler(operator.raw_operator.mean_, scale, device)


register_converter("SklearnRobustScaler", convert_sklearn_robust_scaler)
register_converter("SklearnMaxAbsScaler", convert_sklearn_max_abs_scaler)
register_converter("SklearnMinMaxScaler", convert_sklearn_min_max_scaler)
register_converter("SklearnStandardScaler", convert_sklearn_standard_scaler)
