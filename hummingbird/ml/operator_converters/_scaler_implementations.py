# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for scaler implementations.
"""

import torch

from ._base_operator import BaseOperator


class Scaler(BaseOperator, torch.nn.Module):
    """
    Class implementing Scaler operators in PyTorch. Supported normalizers are L1, L2 and Max.
    """

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
