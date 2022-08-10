# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base class for SV implementation.
"""

import numpy as np
import scipy
import torch

from ._physical_operator import PhysicalOperator


class SVC(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, kernel, degree, sv, nv, a, b, gamma, coef0, classes, device):
        super(SVC, self).__init__(logical_operator, classification=True)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.regression = False
        sv = sv.toarray() if type(sv) == scipy.sparse.csr_matrix else sv
        self.sv = torch.nn.Parameter(torch.from_numpy(sv).double(), requires_grad=False)
        self.sv_t = torch.nn.Parameter(torch.transpose(self.sv, 0, 1), requires_grad=False)
        self.sv_norm = torch.nn.Parameter(-self.gamma * (self.sv ** 2).sum(1).view(1, -1), requires_grad=False)
        self.coef0 = coef0
        self.n_features = sv.shape[1]
        self.a = a
        self.b = torch.nn.Parameter(torch.from_numpy(b.reshape(1, -1)).double(), requires_grad=False)
        self.start = [sum(nv[:i]) for i in range(len(nv))]
        self.end = [self.start[i] + nv[i] for i in range(len(nv))]
        self.len_nv = len(nv)
        true_classes, false_classes = zip(*[(i, j) for i in range(self.len_nv) for j in range(i + 1, self.len_nv)])
        self.true_classes = torch.nn.Parameter(torch.IntTensor([true_classes]), requires_grad=False)
        self.false_classes = torch.nn.Parameter(torch.IntTensor([false_classes]), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.n_classes = len(classes)

    def forward(self, x):
        x = x.double()

        if self.kernel == "linear":
            k = torch.mm(x, self.sv_t)
        elif self.kernel == "rbf":
            # using quadratic expansion--susseptible to rounding-off errors
            # http://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
            x_norm = -self.gamma * (x ** 2).sum(1).view(-1, 1)
            k = torch.exp(x_norm + self.sv_norm + 2.0 * self.gamma * torch.mm(x, self.sv_t).double())
        elif self.kernel == "sigmoid":
            k = torch.sigmoid(self.gamma * torch.mm(x, self.sv_t) + self.coef0)
        else:  # poly kernel
            k = torch.pow(self.gamma * torch.mm(x, self.sv_t) + self.coef0, self.degree)

        c = [
            sum(self.a[i, p] * k[:, p : p + 1] for p in range(self.start[j], self.end[j]))
            + sum(self.a[j - 1, p] * k[:, p : p + 1] for p in range(self.start[i], self.end[i]))
            for i in range(self.len_nv)
            for j in range(i + 1, self.len_nv)
        ]
        c = torch.cat(c, dim=1) + self.b
        if self.n_classes == 2:
            class_ids = torch.gt(c, 0.0).int().flatten()
        else:
            votes = torch.where(c > 0, self.true_classes, self.false_classes)
            # TODO mode is still not implemented for GPU backend.
            votes = votes.data.cpu()
            class_ids, _ = torch.mode(votes, dim=1)
        # No class probabilities in SVC.
        if self.perform_class_select:
            temp = torch.index_select(self.classes, 0, class_ids.long())
            return temp, temp
        else:
            return class_ids, class_ids
