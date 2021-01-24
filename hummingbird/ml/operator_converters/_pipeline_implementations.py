# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for operators necessary for supporting scikit-learn Pipelines.
"""

from distutils.version import LooseVersion
import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from ._physical_operator import PhysicalOperator


class Concat(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator):
        super(Concat, self).__init__(logical_operator, transformer=True)

    def forward(self, *x):
        if len(x[0].shape) > 1:
            # We need to explictly cast the tensors if their types don't agree.
            dtypes = {t.dtype for t in x}
            if len(dtypes) > 1:
                if torch.float64 in dtypes:
                    x = [t.double() for t in x]
                elif torch.float32 in dtypes:
                    x = [t.float() for t in x]
                else:
                    raise RuntimeError(
                        "Combination of data types for Concat input tensors not supported. Please fill an issue at https://github.com/microsoft/hummingbird."
                    )
            return torch.cat(x, dim=1)
        else:
            return torch.stack(x, dim=1)
