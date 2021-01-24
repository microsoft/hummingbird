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

from ._physical_operator import PhysicalOperator


class Decomposition(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, mean, transform_matrix, device):
        super(Decomposition, self).__init__(logical_operator)
        self.transformer = True
        if mean is not None:
            self.mean = torch.nn.Parameter(torch.from_numpy(mean), requires_grad=False)
        else:
            self.mean = None
        self.transform_matrix = torch.nn.Parameter(torch.from_numpy(transform_matrix), requires_grad=False)

    def forward(self, x):
        if self.mean is not None:
            x = x - self.mean
        return torch.mm(x, self.transform_matrix).float()


class KernelPCA(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, kernel, degree, sv, scaled_alphas, gamma, coef0, k_fit_rows, k_fit_all, device):
        super(KernelPCA, self).__init__(logical_operator)
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
        if self.kernel == "linear":
            x = x.view(-1, 1, self.n_features)
            k = self.sv * x
            k = k.sum(2)
        elif self.kernel == "rbf":
            x = x.view(-1, 1, self.n_features)
            k = torch.pow(self.sv - x, 2)
            k = k.sum(2)
            k = torch.exp(-self.gamma * k)
        elif self.kernel == "poly":
            k = torch.pow(self.gamma * torch.mm(x, self.sv.t()) + self.coef0, self.degree)
        elif self.kernel == "sigmoid":
            k = torch.tanh(self.gamma * torch.mm(x, self.sv.t()) + self.coef0)
        elif self.kernel == "cosine":
            norm_x = torch.norm(x, keepdim=True, dim=1)
            norm_sv = torch.norm(self.sv, keepdim=True, dim=1)
            norm = torch.mm(norm_x, norm_sv.t())
            k = torch.mm(x, self.sv.t())
            k = torch.div(k, norm)
        elif self.kernel == "precomputed":
            k = x
        else:
            raise NotImplementedError(
                "Hummingbird does not currently support {} kernel for KernelPCA. The supported kernels are linear, poly, rbf, sigmoid, cosine, and precomputed.".format(
                    self.kernel
                )
            )

        k_pred_cols = (torch.sum(k, 1) / self.n_samples).view(-1, 1)
        k -= self.k_fit_rows
        k -= k_pred_cols
        k += self.k_fit_all

        return torch.mm(k, self.scaled_alphas)
