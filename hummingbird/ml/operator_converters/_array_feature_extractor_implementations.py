# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base class for array feature extractor
"""

import torch

from ._physical_operator import PhysicalOperator


class ArrayFeatureExtractor(PhysicalOperator, torch.nn.Module):
    """
    Class implementing ArrayFeatureExtractor in PyTorch

    This is used by SelectKBest, VarianceThreshold operators in scikit-learn
    """

    def __init__(self, logical_operator, column_indices, device):
        super(ArrayFeatureExtractor, self).__init__(logical_operator, transformer=True)

        is_contiguous = False
        if max(column_indices) - min(column_indices) + 1 == len(column_indices):
            is_contiguous = True
            self.min = min(column_indices)
            self.max = max(column_indices) + 1
        self.column_indices = torch.nn.Parameter(torch.LongTensor(column_indices), requires_grad=False)
        self.is_contiguous = is_contiguous

    def forward(self, x):
        if self.is_contiguous:
            return x[:, self.min : self.max]
        else:
            return torch.index_select(x, 1, self.column_indices)
