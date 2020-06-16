# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch
import numpy as np
from onnxconverter_common.registration import register_converter

"""
Converters for scikit-learn linear models: LinearRegression, LogisticRegression, LinearSVC, SGDClassifier, LogisticRegressionCV
"""


class SklearnLinearModel(torch.nn.Module):
    def __init__(self, coefficients, intercepts, classes, multi_class, device, is_linear_regression=False):
        super(SklearnLinearModel, self).__init__()
        self.coefficients = torch.nn.Parameter(torch.from_numpy(coefficients), requires_grad=False)
        self.intercepts = torch.nn.Parameter(torch.from_numpy(intercepts), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.multi_class = multi_class
        self.regression = is_linear_regression

        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, x):
        if self.multi_class == "multinomial":
            output = torch.softmax(torch.addmm(self.intercepts, x, self.coefficients), dim=1)
        elif self.regression:
            output = torch.addmm(self.intercepts, x, self.coefficients)
            if not self.binary_classification:
                return output
        else:
            output = torch.sigmoid(torch.addmm(self.intercepts, x, self.coefficients))
            if not self.binary_classification:
                output /= torch.sum(output, dim=1, keepdim=True)

        if self.binary_classification:
            output = torch.cat([1 - output, output], dim=1)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


def convert_sklearn_linear_model(operator, device, extra_config):
    classes = [0] if not hasattr(operator.raw_operator, "classes_") else operator.raw_operator.classes_

    if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
        raise RuntimeError("hummingbird supports only integer labels for class labels.")
    if operator.type == "SklearnLinearRegression":
        is_linear_regression = True
        coefficients = operator.raw_operator.coef_.transpose().reshape(-1, len(classes)).astype("float32")
    else:
        is_linear_regression = False
        coefficients = operator.raw_operator.coef_.transpose().astype("float32")
    intercepts = operator.raw_operator.intercept_.reshape(1, -1).astype("float32")

    if hasattr(operator.raw_operator, "multi_class"):
        if operator.raw_operator.multi_class == "ovr" or operator.raw_operator.solver in ["warn", "liblinear"]:
            multi_class = "ovr"
        else:
            multi_class = "multinomial"
    else:
        multi_class = None

    return SklearnLinearModel(coefficients, intercepts, classes, multi_class, device, is_linear_regression)


register_converter("SklearnLinearRegression", convert_sklearn_linear_model)
register_converter("SklearnLogisticRegression", convert_sklearn_linear_model)
register_converter("SklearnLinearSVC", convert_sklearn_linear_model)
register_converter("SklearnSGDClassifier", convert_sklearn_linear_model)
register_converter("SklearnLogisticRegressionCV", convert_sklearn_linear_model)
