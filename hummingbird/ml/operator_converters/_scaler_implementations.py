# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for scaler implementations.
"""

import torch

from ._physical_operator import PhysicalOperator


class Scaler(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Scaler operators in PyTorch. Supported normalizers are L1, L2 and Max.
    """

    def __init__(self, logical_operator, offset, scale, device):
        super(Scaler, self).__init__(logical_operator, transformer=True)

        self.offset = offset
        self.scale = scale

        if offset is not None:
            self.offset = torch.nn.Parameter(torch.from_numpy(offset), requires_grad=False)

        if scale is not None:
            self.scale = torch.nn.Parameter(torch.from_numpy(scale), requires_grad=False)

    def forward(self, x):
        if self.offset is not None:
            x = x - self.offset

        if self.scale is not None:
            x = x * self.scale

        return x.float()
