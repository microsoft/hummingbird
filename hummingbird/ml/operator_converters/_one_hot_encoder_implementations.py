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

    def __init__(self, logical_operator, categories, handle_unknown, device, infrequent=None, extra_config={}):
        super(OneHotEncoderString, self).__init__(logical_operator, transformer=True)

        self.num_columns = len(categories)
        self.max_word_length = max([max([len(c) for c in cat]) for cat in categories])
        self.handle_unknown = handle_unknown
        self.mask = None

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

        if infrequent is not None:
            infrequent_tensors = []
            categories_idx = [0]
            for arr in infrequent:
                cats = (
                    np.array(arr, dtype="|S" + str(self.max_word_length))  # Encode objects into 4 byte strings.
                    .view("int32")
                    .reshape(-1, self.max_word_length // 4)
                    .tolist()
                )
                # We merge all categories for all columns into a single tensor
                infrequent_tensors.extend(cats)
                # Since all categories are merged together, we need to track of indexes to retrieve them at inference time.
                categories_idx.append(categories_idx[-1] + len(cats))
            self.infrequent_tensors = torch.nn.Parameter(torch.IntTensor(infrequent_tensors), requires_grad=False)

            # We need to create a mask to filter out infrequent categories.
            self.mask = torch.nn.ParameterList([])
            for i in range(len(self.condition_tensors[0])):
                if self.condition_tensors[0][i] not in self.infrequent_tensors[0]:
                    self.mask.append(torch.nn.Parameter(self.condition_tensors[0][i], requires_grad=False))
        else:
            self.infrequent_tensors = None

    def forward(self, x):
        encoded_tensors = []

        # TODO: implement 'error' case separately
        if self.handle_unknown == "ignore" or self.handle_unknown == "error":
            compare_tensors = self.condition_tensors
        elif self.handle_unknown == "infrequent_if_exist":
            compare_tensors = self.mask if self.mask is not None else self.condition_tensors
        else:
            raise RuntimeError("Unsupported handle_unknown setting: {0}".format(self.handle_unknown))

        for i in range(self.num_columns):
            # First we fetch the condition for the particular column.
            conditions = compare_tensors[self.categories_idx[i] : self.categories_idx[i + 1], :].view(
                1, -1, self.max_word_length // 4
            )
            # Differently than the numeric case where eq is enough, here we need to aggregate per object (dim = 2)
            # because objects can span multiple integers. We use product here since all ints must match to get encoding of 1.
            encoded_tensors.append(torch.prod(torch.eq(x[:, i : i + 1, :], conditions), dim=2))

        # if self.infrequent_tensors is not None, then append another tensor that is the "not" of the sum of the encoded tensors.
        if self.infrequent_tensors is not None:
            encoded_tensors.append(torch.logical_not(torch.sum(torch.stack(encoded_tensors), dim=0)))

        return torch.cat(encoded_tensors, dim=1).float()


class OneHotEncoder(PhysicalOperator, torch.nn.Module):
    """
    Class implementing OneHotEncoder operators for ints in PyTorch.
    """

    def __init__(self, logical_operator, categories, handle_unknown, device, infrequent=None):
        super(OneHotEncoder, self).__init__(logical_operator, transformer=True)

        self.num_columns = len(categories)
        self.handle_unknown = handle_unknown
        self.mask = None

        condition_tensors = []
        for arr in categories:
            condition_tensors.append(torch.nn.Parameter(torch.LongTensor(arr).detach().clone(), requires_grad=False))
        self.condition_tensors = torch.nn.ParameterList(condition_tensors)

        if infrequent is not None:
            infrequent_tensors = []
            for arr in infrequent:
                infrequent_tensors.append(torch.nn.Parameter(torch.LongTensor(arr).detach().clone(), requires_grad=False))
            self.infrequent_tensors = torch.nn.ParameterList(infrequent_tensors)

            # We need to create a mask to filter out infrequent categories.
            self.mask = torch.nn.ParameterList([])
            for i in range(len(self.condition_tensors[0])):
                if self.condition_tensors[0][i] not in self.infrequent_tensors[0]:
                    self.mask.append(torch.nn.Parameter(self.condition_tensors[0][i], requires_grad=False))

        else:
            self.infrequent_tensors = None

    def forward(self, *x):
        encoded_tensors = []

        if self.handle_unknown == "ignore" or self.handle_unknown == "error":  # TODO: error
            compare_tensors = self.condition_tensors
        elif self.handle_unknown == "infrequent_if_exist":
            compare_tensors = self.mask if self.mask is not None else self.condition_tensors
        else:
            raise RuntimeError("Unsupported handle_unknown setting: {0}".format(self.handle_unknown))

        if len(x) > 1:
            assert len(x) == self.num_columns

            for i in range(self.num_columns):
                input = x[i]
                if input.dtype != torch.int64:
                    input = input.long()

                encoded_tensors.append(torch.eq(input, compare_tensors[i]))
        else:
            # This is already a tensor.
            x = x[0]
            if x.dtype != torch.int64:
                x = x.long()

            for i in range(self.num_columns):
                encoded_tensors.append(torch.eq(x[:, i : i + 1], compare_tensors[i]))

        # if self.infrequent_tensors is not None, then append another tensor that is the "not" of the sum of the encoded tensors.
        if self.infrequent_tensors is not None:
            encoded_tensors.append(torch.logical_not(torch.sum(torch.stack(encoded_tensors), dim=0)))

        return torch.cat(encoded_tensors, dim=1).float()
