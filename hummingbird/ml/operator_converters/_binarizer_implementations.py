# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for binarizer implementations.
"""

import torch

from ._base_operator import BaseOperator


class Binarizer(BaseOperator, torch.nn.Module):
    """
    Class implementing Binarizer operators in PyTorch.
    """

    def __init__(self, threshold, device):
        super(Binarizer, self).__init__()
        self.threshold = torch.nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)

    def forward(self, x):
        return torch.gt(x, self.threshold).float()
