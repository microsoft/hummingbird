# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base class for array feature extractor
"""

import torch

from ._base_operator import BaseOperator


class ArrayFeatureExtractor(BaseOperator, torch.nn.Module):
    """
    Class implementing ArrayFeatureExtractor in PyTorch

    This is used by SelectKBest, VarianceThreshold operators in scikit-learn
    """

    def __init__(self, column_indices, device):
        super(ArrayFeatureExtractor, self).__init__(transformer=True)

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


class VectorAssemblerModel(BaseOperator, torch.nn.Module):
    """
    Class implementing ArrayFeatureExtractor in PyTorch

    This is used by Spark0=-ML VectorAssembler
    """

    def __init__(self, input_indices=None, append_output=False):
        super(VectorAssemblerModel, self).__init__(input_indices=input_indices, append_output=append_output, transformer=True)

    def forward(self, *x):
        x = self.select_input_if_needed(x)
        x = torch.cat(x, 1)
        return self.get_appended_output_if_needed(x)
