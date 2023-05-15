# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base class for linear algorithm implementations.
"""

import torch

from ._physical_operator import PhysicalOperator


class LinearModel(PhysicalOperator, torch.nn.Module):
    def __init__(
        self,
        logical_operator,
        coefficients,
        intercepts,
        device,
        classes=[0],
        multi_class=None,
        loss=None,
        is_linear_regression=False,
    ):
        super(LinearModel, self).__init__(logical_operator)
        self.coefficients = torch.nn.Parameter(torch.from_numpy(coefficients).detach().clone(), requires_grad=False)
        self.intercepts = torch.nn.Parameter(torch.from_numpy(intercepts).view(-1).detach().clone(), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.multi_class = multi_class
        self.regression = is_linear_regression
        self.classification = not is_linear_regression
        self.loss = loss
        if self.loss is None and self.classification:
            self.loss = "log"

        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, x):
        x = x.float()
        output = torch.addmm(self.intercepts, x, self.coefficients)

        if self.regression:
            if self.loss == "log":
                return torch.exp(output)
            return output

        if self.binary_classification:
            indices = (output > 0).squeeze().int()
        else:
            indices = torch.argmax(output, dim=1)
        predict_res = torch.index_select(self.classes, 0, indices)

        if self.multi_class == "multinomial":
            output = torch.softmax(output, dim=1)
        else:
            if self.loss == "modified_huber":
                output = torch.clip(output, -1, 1)
                output += 1
                output /= 2
            else:
                output = torch.sigmoid(output)
            if not self.binary_classification:
                if self.loss == "modified_huber":
                    # This loss might assign zero to all classes, which doesn't
                    # normalize neatly; work around this to produce uniform
                    # probabilities.
                    prob_sum = torch.sum(output, dim=1, keepdim=False)
                    all_zero = prob_sum == 0
                    if torch.any(all_zero):
                        output[all_zero, :] = 1
                        prob_sum[all_zero] = len(self.classes)
                    output /= prob_sum.view((output.shape[0], -1))
                else:
                    output /= torch.sum(output, dim=1, keepdim=True)

        if self.binary_classification:
            output = torch.cat([1 - output, output], dim=1)

        return predict_res, output
