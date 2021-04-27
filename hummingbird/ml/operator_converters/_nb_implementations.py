# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for Naive Bayes model implementation: (BernoulliNB, GaussianNB).
"""

import torch

from ._physical_operator import PhysicalOperator


class BernoulliNBModel(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, classes, binarize, jll_calc_bias, feature_log_prob_minus_neg_prob, device):
        super(BernoulliNBModel, self).__init__(logical_operator)
        self.classification = True
        self.binarize = binarize
        self.jll_calc_bias = torch.nn.Parameter(
            torch.from_numpy(jll_calc_bias.astype("float64")).view(-1), requires_grad=False
        )
        self.feature_log_prob_minus_neg_prob = torch.nn.Parameter(
            torch.from_numpy(feature_log_prob_minus_neg_prob.astype("float64")), requires_grad=False
        )
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

    def forward(self, x):
        x = x.double()
        if self.binarize is not None:
            x = torch.gt(x, self.binarize).double()

        jll = torch.addmm(self.jll_calc_bias, x, self.feature_log_prob_minus_neg_prob)
        log_prob_x = torch.logsumexp(jll, dim=1)
        log_prob_x = jll - log_prob_x.view(-1, 1)
        prob_x = torch.exp(log_prob_x).float()

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(jll, dim=1)), prob_x
        else:
            return torch.argmax(jll, dim=1), prob_x


class GaussianNBModel(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, classes, jll_calc_bias, theta, sigma, device):
        super(GaussianNBModel, self).__init__(logical_operator)
        self.classification = True
        self.jll_calc_bias = torch.nn.Parameter(torch.from_numpy(jll_calc_bias.astype("float32")), requires_grad=False)
        self.theta = torch.nn.Parameter(
            torch.from_numpy(theta.astype("float32")).view((len(classes), 1, -1)), requires_grad=False
        )
        self.sigma = torch.nn.Parameter(
            torch.from_numpy(sigma.astype("float32")).view((len(classes), 1, -1)), requires_grad=False
        )
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

    def forward(self, x):
        jll = self.jll_calc_bias - 0.5 * torch.sum(torch.div(torch.pow(x - self.theta, 2), self.sigma), 2)
        jll = torch.transpose(jll, 0, 1)
        log_prob_x = torch.logsumexp(jll, dim=1)
        log_prob_x = jll - log_prob_x.view(-1, 1)
        prob_x = torch.exp(log_prob_x)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(jll, dim=1)), prob_x
        else:
            return torch.argmax(jll, dim=1), prob_x
