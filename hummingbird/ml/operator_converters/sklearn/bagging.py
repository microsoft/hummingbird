# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for Bagging operators.
"""

from distutils.version import LooseVersion
import numpy as np
from onnxconverter_common.registration import register_converter
import torch

from .._physical_operator import PhysicalOperator


class Bagging(PhysicalOperator, torch.nn.Module):
    def __init__(self, logical_operator, is_classifier, n_estimators, classes):
        super(Bagging, self).__init__(logical_operator, transformer=True)

        self.is_classifier = is_classifier
        self.n_estimators = float(n_estimators)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, *x):
        if self.is_classifier:
            x = [t[1].view(-1, 1) if len(t[1].shape) == 1 else t[1][:, 1].view(-1, 1) for t in x]

        output = torch.cat(x, dim=1)
        output = torch.sum(output, dim=1) / self.n_estimators

        if not self.is_classifier:
            return output
        if self.binary_classification:
            output = torch.stack([1 - output, output], dim=1)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


def convert_sklearn_baggind_model(operator, device, extra_config):
    assert operator is not None, "Cannot convert None operator"

    n_executors = operator.raw_operator.n_estimators
    is_classifier = operator.raw_operator._estimator_type == "classifier"
    classes = [0]
    if is_classifier:
        classes = operator.raw_operator.classes_

    return Bagging(operator, is_classifier, n_executors, classes)


register_converter("SklearnBagging", convert_sklearn_baggind_model)
