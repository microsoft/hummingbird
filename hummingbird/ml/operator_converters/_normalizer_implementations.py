# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for normalizer implementations.
"""

import torch

from ._physical_operator import PhysicalOperator


class Normalizer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Normalizer operators in PyTorch. Supported normalizers are L1, L2 and Max.
    """

    def __init__(self, logical_operator, norm, device):
        super(Normalizer, self).__init__(logical_operator)
        self.norm = norm
        self.transformer = True

    def forward(self, x):
        if self.norm == "l1":
            return x / torch.abs(x).sum(1, keepdim=True)
        elif self.norm == "l2":
            return x / torch.pow(torch.pow(x, 2).sum(1, keepdim=True), 0.5)
        elif self.norm == "max":
            return x / torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        else:
            raise RuntimeError("Unsupported norm: {0}".format(self.norm))
