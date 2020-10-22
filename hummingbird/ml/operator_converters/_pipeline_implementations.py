# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for operators necessary for supporting scikit-learn Pipelines.
"""

import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from ._base_operator import BaseOperator


class Concat(BaseOperator, torch.nn.Module):
    def __init__(self):
        super(Concat, self).__init__(transformer=True)

    def forward(self, *x):
        if len(x[0].shape) > 1:
            return torch.cat(x, dim=1)
        else:
            return torch.stack(x, dim=1)
