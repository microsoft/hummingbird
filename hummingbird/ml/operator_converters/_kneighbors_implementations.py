# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for KNeighbors model implementations: (KNeighborsClassifier).
"""

import torch
from ._base_operator import BaseOperator


class KNeighborsClassifierModel(BaseOperator, torch.nn.Module):
    def __init__(self, train_data, train_labels, n_neighbors, weights, classes, p):
        super(KNeighborsClassifierModel, self).__init__()
        self.classification = True
        self.train_data = torch.nn.Parameter(torch.from_numpy(train_data.astype("float32")), requires_grad=False)
        self.train_data_t = torch.nn.Parameter(torch.transpose(self.train_data, 0, 1), requires_grad=False)
        self.train_data_norm = torch.nn.Parameter((self.train_data ** 2).sum(1).view(1, -1), requires_grad=False)
        self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype("int64")), requires_grad=False)
        self.n_neighbors = n_neighbors
        self.p = p
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.n_classes = len(classes)
        self.perform_class_select = False

        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.weights = weights

    def forward(self, x):
        k = torch.cdist(x, self.train_data, p=float(self.p), compute_mode="donot_use_mm_for_euclid_dist")
        d, k = torch.topk(k, self.n_neighbors, dim=1, largest=False)
        output = torch.index_select(self.train_labels, 0, k.view(-1)).view(-1, self.n_neighbors)

        if self.weights == "distance":
            d = torch.pow(d, -1)
            inf_mask = torch.isinf(d)
            inf_row = torch.any(inf_mask, axis=1)
            d[inf_row] = inf_mask[inf_row].float()
        else:
            d = torch.ones_like(k, dtype=torch.float32)

        output = torch.scatter_add(torch.zeros((x.size()[0], self.n_classes), dtype=torch.float32), 1, output, d)
        proba_sum = output.sum(1, keepdim=True)
        proba_sum = torch.where(proba_sum == 0, torch.FloatTensor([1.0]), proba_sum)
        output = torch.pow(proba_sum, -1) * output

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output
