# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import torch
import numpy as np
from onnxconverter_common.registration import register_converter


class OneHotEncoderString(torch.nn.Module):
    def __init__(self, categories, device):
        super(OneHotEncoderString, self).__init__()

        self.num_columns = len(categories)
        self.max_word_length = max([max([len(c) for c in cat]) for cat in categories])

        while self.max_word_length % 4 != 0:
            self.max_word_length += 1

        condition_tensors = []
        num_categories = [0]
        for arr in categories:
            cats = (
                np.array(arr, dtype="|S" + str(self.max_word_length))
                .view("int32")
                .reshape(-1, self.max_word_length // 4)
                .tolist()
            )
            condition_tensors.extend(cats)
            num_categories.append(num_categories[-1] + len(cats))
        self.condition_tensors = torch.nn.Parameter(torch.IntTensor(condition_tensors), requires_grad=False)
        self.num_categories = num_categories
        self.regression = False

    def forward(self, x):
        encoded_tensors = []
        for i in range(self.num_columns):
            conditions = self.condition_tensors[self.num_categories[i] : self.num_categories[i + 1], :].view(
                1, -1, self.max_word_length // 4
            )
            encoded_tensors.append(torch.prod(torch.eq(x[:, i : i + 1, :], conditions), dim=2))

        return torch.cat(encoded_tensors, dim=1).float()


class OneHotEncoder(torch.nn.Module):
    def __init__(self, indices, retain_order, categories, device):
        super(OneHotEncoder, self).__init__()

        self.num_columns = len(categories)
        self.retain_order = retain_order
        if indices is None:
            n_features = self.num_columns
            self.n_selected = n_features
        else:
            n_features = len(indices)
            ind = np.arange(n_features)
            sel = np.zeros(n_features, dtype=bool)
            sel[np.asarray(indices)] = True
            not_sel = np.logical_not(sel)
            self.n_selected = np.sum(sel)
            self.ind_sel = torch.nn.Parameter(torch.LongTensor(ind[sel]), requires_grad=False)
            self.ind_not_sel = torch.nn.Parameter(torch.LongTensor(ind[not_sel]), requires_grad=False)

        self.all = n_features == self.num_columns

        condition_tensors = []
        for arr in categories:
            condition_tensors.append(torch.nn.Parameter(torch.LongTensor(arr), requires_grad=False))
        self.condition_tensors = torch.nn.ParameterList(condition_tensors)

    def forward(self, x):
        if x.dtype != torch.int64:
            x = x.long()

        if self.n_selected == 0:
            # No features selected.
            return x
        elif self.all:
            # All features selected.
            return self.transform(x)
        else:
            x_sel = self.transform(x[:, self.ind_sel])

        if self.retain_order:
            pass
            # This is not yet implemented and future work
            # (As-s: This does not require impl at this time!!)
            #
            ## For the moment it never retain order
            ## x[:, ind[sel]] = x_sel
            ## return x

        x_not_sel = x[:, self.ind_not_sel].float()
        return torch.cat([x_sel, x_not_sel], dim=1)

    def transform(self, x):
        encoded_tensors = []
        for i in range(self.num_columns):
            encoded_tensors.append(torch.eq(x[:, i : i + 1], self.condition_tensors[i]))
        return torch.cat(encoded_tensors, dim=1).float()


def convert_sklearn_one_hot_encoder(operator, device, extra_config):
    if all([np.array(c).dtype == object for c in operator.raw_operator.categories_]):
        categories = [[str(x) for x in c.tolist()] for c in operator.raw_operator.categories_]
        return OneHotEncoderString(categories, device)
    else:
        return OneHotEncoder(operator.raw_operator.categorical_features, False, operator.raw_operator.categories_, device)


register_converter("SklearnOneHotEncoder", convert_sklearn_one_hot_encoder)
