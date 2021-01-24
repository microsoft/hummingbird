# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for sklearn discretizers: Binarizer, KBinsDiscretizer
"""
import torch

from ._physical_operator import PhysicalOperator
from ._one_hot_encoder_implementations import OneHotEncoder


class Binarizer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Binarizer operators in PyTorch.
    """

    def __init__(self, logical_operator, threshold, device):
        super(Binarizer, self).__init__(logical_operator)
        self.transformer = True
        self.threshold = torch.nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)

    def forward(self, x):
        return torch.gt(x, self.threshold).float()


class KBinsDiscretizer(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, encode, bin_edges, labels, device):
        super(KBinsDiscretizer, self).__init__(logical_operator)
        self.transformer = True
        self.encode = encode
        # We use DoubleTensors for better precision.
        # We use a small delta value of 1e-9.
        self.ge_tensor = torch.nn.Parameter(torch.FloatTensor(bin_edges[:, :-1]), requires_grad=False)
        self.lt_tensor = torch.nn.Parameter(torch.FloatTensor(bin_edges[:, 1:]), requires_grad=False)
        self.ohe = OneHotEncoder(logical_operator, labels, device)

    def forward(self, x):
        x = x.float()
        x = torch.unsqueeze(x, 2)
        x = torch.ge(x, self.ge_tensor) & torch.lt(x, self.lt_tensor)
        x = x.float()
        x = torch.argmax(x, dim=2, keepdim=False)

        if self.encode in ["onehot-dense", "onehot"]:
            x = self.ohe(x)

        return x
