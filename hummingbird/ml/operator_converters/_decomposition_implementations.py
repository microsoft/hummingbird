# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for matrix decomposition algorithm implementations.
"""

import torch

from ._base_operator import BaseOperator


class Decomposition(BaseOperator, torch.nn.Module):
    def __init__(self, mean, transform_matrix, device):
        super(Decomposition, self).__init__()
        self.transformer = True
        if mean is not None:
            self.mean = torch.nn.Parameter(torch.from_numpy(mean), requires_grad=False)
        else:
            self.mean = None
        self.transform_matrix = torch.nn.Parameter(torch.from_numpy(transform_matrix), requires_grad=False)

    def forward(self, x):
        if self.mean is not None:
            x = x - self.mean
        return torch.mm(x, self.transform_matrix)


class KernelPCA(BaseOperator, torch.nn.Module):
    def __init__(self, kernel, degree, sv, scaled_alphas, gamma, coef0, k_fit_rows, k_fit_all, device):
        super(KernelPCA, self).__init__()
        self.transformer = True
        self.kernel = kernel
        self.degree = degree
        self.n_samples = sv.shape[0]
        self.sv = torch.from_numpy(sv).float()
        self.n_features = sv.shape[1]
        self.k_fit_rows = torch.from_numpy(k_fit_rows).float()
        self.k_fit_all = k_fit_all
        if gamma is None:
            gamma = 1.0 / self.n_features
        self.gamma = gamma
        self.coef0 = coef0
        self.scaled_alphas = torch.from_numpy(scaled_alphas).float()

    def forward(self, x):
        x = x.view(-1, 1, self.n_features)
        if self.kernel == "linear":
            k = self.sv * x
        elif self.kernel == "rbf":
            k = torch.exp(-self.gamma * torch.pow(self.sv - x, 2))
        else:  # poly kernel
            k = torch.pow(self.gamma * self.sv * x + self.coef0, self.degree)

        k = k.sum(2)

        k_pred_cols = (torch.sum(k, 1) / self.n_samples).view(-1, 1)
        k -= self.k_fit_rows
        k -= k_pred_cols
        k += self.k_fit_all

        return torch.mm(k, self.scaled_alphas)
