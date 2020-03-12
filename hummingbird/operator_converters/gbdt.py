# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import warnings

import torch
import numpy as np
from ._tree_commons import get_parameters_for_batch, get_parameters_for_beam
from ._tree_commons import BatchedTreeEnsemble, BeamTreeEnsemble, BeamPPTreeEnsemble
from ..common._registration import register_converter


class BatchGBDTClassifier(BatchedTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, learning_rate=None, alpha=None, device=None):
        super(BatchGBDTClassifier, self).__init__(
            net_parameters, n_features, 1)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.classes = torch.nn.Parameter(
                torch.IntTensor(classes), requires_grad=False)
            self.perform_class_select = True

        self.n_classes = 1
        self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

        self.binary_classification = False
        if self.n_gbdt_classes == 1:
            self.binary_classification = True

        self.learning_rate = learning_rate
        if alpha is not None:
            self.alpha = torch.nn.Parameter(
                torch.FloatTensor(alpha), requires_grad=False)
        else:
            self.alpha = None

    def forward(self, x):
        output = super().forward(x)
        output = torch.squeeze(output).t(
        ).view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)
        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha

        if self.binary_classification:
            output = torch.sigmoid(output)
            output = torch.cat([1 - output, output], dim=1)
        else:
            output = torch.softmax(output, dim=1)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BatchGBDTRegressor(BatchedTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, learning_rate=None, alpha=None, device=None):
        super(BeamGBDTRegressor, self).__init__(net_parameters, n_features, 1)

        self.n_classes = 1
        self.n_gbdt_classes = 1
        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

        self.learning_rate = learning_rate
        if alpha is not None:
            self.alpha = torch.nn.Parameter(
                torch.FloatTensor(alpha), requires_grad=False)
        else:
            self.alpha = None

    def forward(self, x):
        output = super().forward(x)
        output = torch.squeeze(output).t(
        ).view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)
        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha

        return output


class BeamGBDTClassifier(BeamTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, learning_rate=None, alpha=None, device=None):
        super(BeamGBDTClassifier, self).__init__(net_parameters, n_features, 1)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.classes = torch.nn.Parameter(
                torch.IntTensor(classes), requires_grad=False)
            self.perform_class_select = True

        self.n_classes = 1
        self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

        self.binary_classification = False
        if self.n_gbdt_classes == 1:
            self.binary_classification = True

        self.learning_rate = learning_rate
        if alpha is not None:
            self.alpha = torch.nn.Parameter(
                torch.FloatTensor(alpha), requires_grad=False)
        else:
            self.alpha = None

    def forward(self, x):
        output = super().forward(x)
        output = output.view(-1, self.n_gbdt_classes,
                             self.n_trees_per_class).sum(2)
        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha

        if self.binary_classification:
            output = torch.sigmoid(output)
            output = torch.cat([1 - output, output], dim=1)
        else:
            output = torch.softmax(output, dim=1)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BeamGBDTRegressor(BeamTreeEnsemble):

    def __init__(self, net_parameters, n_features, learning_rate=None, alpha=None, device=None):
        super(BeamGBDTRegressor, self).__init__(net_parameters, n_features, 1)

        self.n_classes = 1
        self.n_gbdt_classes = 1
        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

        self.learning_rate = learning_rate
        if alpha is not None:
            self.alpha = torch.nn.Parameter(
                torch.FloatTensor([alpha]), requires_grad=False)
        else:
            self.alpha = None

    def forward(self, x):
        output = super().forward(x)
        output = output.view(-1, self.n_gbdt_classes,
                             self.n_trees_per_class).sum(2)
        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha

        return output


class BeamPPGBDTClassifier(BeamPPTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, learning_rate=None, alpha=None, device=None):
        super(BeamPPGBDTClassifier, self).__init__(
            net_parameters, n_features, 1)

        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.classes = torch.nn.Parameter(
                torch.IntTensor(classes), requires_grad=False)
            self.perform_class_select = True

        self.n_classes = 1
        self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

        self.binary_classification = False
        if self.n_gbdt_classes == 1:
            self.binary_classification = True

        self.learning_rate = learning_rate
        if alpha is not None:
            self.alpha = torch.nn.Parameter(
                torch.FloatTensor(alpha), requires_grad=False)
        else:
            self.alpha = None

    def forward(self, x):
        output = super().forward(x)
        output = output.view(-1, self.n_gbdt_classes,
                             self.n_trees_per_class).sum(2)
        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha

        if self.binary_classification:
            output = torch.sigmoid(output)
            output = torch.cat([1 - output, output], dim=1)
        else:
            output = torch.softmax(output, dim=1)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BeamPPGBDTRegressor(BeamPPTreeEnsemble):

    def __init__(self, net_parameters, n_features, learning_rate=None, alpha=None, device=None):
        super(BeamPPGBDTRegressor, self).__init__(
            net_parameters, n_features, 1)

        self.n_classes = 1
        self.n_gbdt_classes = 1
        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

        self.learning_rate = learning_rate
        if alpha is not None:
            self.alpha = torch.nn.Parameter(
                torch.FloatTensor([alpha]), requires_grad=False)
        else:
            self.alpha = None

    def forward(self, x):
        output = super().forward(x)
        output = output.view(-1, self.n_gbdt_classes,
                             self.n_trees_per_class).sum(2)
        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha

        return output


def convert_sklearn_gbdt_classifier(operator, device, extra_config):
    sklearn_rf_classifier = operator.raw_operator

    if not all(isinstance(c, int) for c in operator.raw_operator.classes_.tolist()):
        raise RuntimeError(
            'GBDT Classifier translation only supports integer class labels')

    n_classes = len(operator.raw_operator.classes_)
    if n_classes == 2:
        n_classes -= 1
    sklearn_rf_classifier.estimators_ = [sklearn_rf_classifier.estimators_[i][j]
                                         for j in range(n_classes) for i in
                                         range(len(sklearn_rf_classifier.estimators_))]

    learning_rate = operator.raw_operator.learning_rate
    if operator.raw_operator.init == 'zero':
        alpha = [[0.0]]
    elif operator.raw_operator.init is None:
        if n_classes == 1:
            alpha = [[np.log(
                operator.raw_operator.init_.class_prior_[1] / (1 - operator.raw_operator.init_.class_prior_[1]))]]
        else:
            alpha = [[np.log(
                operator.raw_operator.init_.class_prior_[i])
                for i in range(n_classes)]]
    else:
        raise RuntimeError(
            'Custom initializers for GBDT are not yet supported in hummingbird')

    # TODO: automatically find the max tree depth by traversing the trees without relying on user input.
    if sklearn_rf_classifier.max_depth is not None and sklearn_rf_classifier.max_depth <= 10:
        if sklearn_rf_classifier.max_depth <= 4:
            net_parameters = [get_parameters_for_batch(
                e) for e in sklearn_rf_classifier.estimators_]
            return BatchGBDTClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                       operator.raw_operator.classes_.tolist(), learning_rate, alpha, device)
        else:
            net_parameters = [get_parameters_for_beam(
                e) for e in sklearn_rf_classifier.estimators_]
            return BeamPPGBDTClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                        operator.raw_operator.classes_.tolist(), learning_rate, alpha, device)
    else:
        if sklearn_rf_classifier.max_depth is None:
            warnings.warn("GBDT model does not have a defined max_depth value. Consider setting one as it "
                          "will help the translator to pick a better translation method")
        elif sklearn_rf_classifier.max_depth > 10:
            warnings.warn("GBDT model max_depth value is {0}. Consider setting a smaller value as it improves"
                          " translated tree scoring performance.".format(sklearn_rf_classifier.max_depth))
        net_parameters = [get_parameters_for_beam(
            e) for e in sklearn_rf_classifier.estimators_]
        return BeamGBDTClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                  operator.raw_operator.classes_.tolist(), learning_rate, alpha, device)


register_converter('SklearnGradientBoostingClassifier',
                   convert_sklearn_gbdt_classifier)
