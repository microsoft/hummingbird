# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base class for multi-layer perceptrons (MLP) implementation.
"""

import torch

from ._physical_operator import PhysicalOperator


class MLPModel(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, weights, biases, activation, device):
        super(MLPModel, self).__init__(logical_operator)
        self.regression = True
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(weight.astype("float32")), requires_grad=False) for weight in weights]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(bias.astype("float32")), requires_grad=False) for bias in biases]
        )
        self.activation = activation

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = torch.addmm(self.biases[i], x, self.weights[i])

            if self.activation == "relu":
                x = torch.relu(x)
            elif self.activation == "logistic":
                x = torch.sigmoid(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
            elif self.activation != "identity":
                raise RuntimeError("Unsupported activation {0}".format(self.activation))

        return torch.addmm(self.biases[-1], x, self.weights[-1])


class MLPClassificationModel(MLPModel):
    def __init__(self, logical_operator, weights, biases, activation, classes, device):
        super(MLPClassificationModel, self).__init__(logical_operator, weights, biases, activation, device)
        self.regression = False
        self.classification = True
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False

        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, x):
        x = super().forward(x)
        if self.binary_classification:
            output = torch.sigmoid(x)
            output = torch.cat([1 - output, output], dim=1)
        else:
            output = torch.softmax(x, dim=1)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output
