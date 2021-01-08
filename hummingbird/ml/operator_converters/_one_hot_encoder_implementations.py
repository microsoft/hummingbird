# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for one hot encoder implementations.
"""

import numpy as np
import torch

from ._physical_operator import PhysicalOperator
from . import constants


class OneHotEncoderString(PhysicalOperator, torch.nn.Module):
    """
    Class implementing OneHotEncoder operators for strings in PyTorch.

    Because we are dealing with tensors, strings require additional length information for processing.
    """

    def __init__(self, logical_operator, categories, device, extra_config={}):
        super(OneHotEncoderString, self).__init__(logical_operator, transformer=True)

        self.num_columns = len(categories)
        self.max_word_length = max([max([len(c) for c in cat]) for cat in categories])

        # Strings are casted to int32, therefore we need to properly size the tensor to me dividable by 4.
        while self.max_word_length % 4 != 0:
            self.max_word_length += 1

        max_length = 0
        if constants.MAX_STRING_LENGTH in extra_config:
            max_length = extra_config[constants.MAX_STRING_LENGTH]
        extra_config[constants.MAX_STRING_LENGTH] = max(max_length, self.max_word_length)

        # We build condition tensors as a 2d tensor of integers.
        # The first dimension is of size num words, the second dimension is fixed to the max word length (// 4).
        condition_tensors = []
        categories_idx = [0]
        for arr in categories:
            cats = (
                np.array(arr, dtype="|S" + str(self.max_word_length))  # Encode objects into 4 byte strings.
                .view("int32")
                .reshape(-1, self.max_word_length // 4)
                .tolist()
            )
            # We merge all categories for all columns into a single tensor
            condition_tensors.extend(cats)
            # Since all categories are merged together, we need to track of indexes to retrieve them at inference time.
            categories_idx.append(categories_idx[-1] + len(cats))
        self.condition_tensors = torch.nn.Parameter(torch.IntTensor(condition_tensors), requires_grad=False)
        self.categories_idx = categories_idx

    def forward(self, x):
        encoded_tensors = []
        for i in range(self.num_columns):
            # First we fetch the condition for the particular column.
            conditions = self.condition_tensors[self.categories_idx[i] : self.categories_idx[i + 1], :].view(
                1, -1, self.max_word_length // 4
            )
            # Differently than the numeric case where eq is enough, here we need to aggregate per object (dim = 2)
            # because objects can span multiple integers. We use product here since all ints must match to get encoding of 1.
            encoded_tensors.append(torch.prod(torch.eq(x[:, i : i + 1, :], conditions), dim=2))

        return torch.cat(encoded_tensors, dim=1).float()


class OneHotEncoder(PhysicalOperator, torch.nn.Module):
    """
    Class implementing OneHotEncoder operators for ints in PyTorch.
    """

    def __init__(self, logical_operator, categories, device):
        super(OneHotEncoder, self).__init__(logical_operator, transformer=True)

        self.num_columns = len(categories)

        condition_tensors = []
        for arr in categories:
            condition_tensors.append(torch.nn.Parameter(torch.LongTensor(arr), requires_grad=False))
        self.condition_tensors = torch.nn.ParameterList(condition_tensors)

    def forward(self, *x):
        encoded_tensors = []

        if len(x) > 1:
            assert len(x) == self.num_columns

            for i in range(self.num_columns):
                input = x[i]
                if input.dtype != torch.int64:
                    input = input.long()

                encoded_tensors.append(torch.eq(input, self.condition_tensors[i]))
        else:
            # This is already a tensor.
            x = x[0]
            if x.dtype != torch.int64:
                x = x.long()

            for i in range(self.num_columns):
                encoded_tensors.append(torch.eq(x[:, i : i + 1], self.condition_tensors[i]))

        return torch.cat(encoded_tensors, dim=1).float()
