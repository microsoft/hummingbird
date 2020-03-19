# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import warnings
import copy

import torch

from ._tree_commons import get_parameters_for_batch, get_parameters_for_beam, find_depth, Node
from ._tree_commons import BatchedTreeEnsemble, BeamTreeEnsemble, BeamPPTreeEnsemble
from ..common._registration import register_converter


class BatchRandomForestClassifier(BatchedTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, device):
        super(BatchRandomForestClassifier, self).__init__(
            net_parameters, n_features, len(classes))
        self.final_probability_divider = len(net_parameters)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.classes = torch.nn.Parameter(
                torch.IntTensor(classes), requires_grad=False)
            self.perform_class_select = True

    def forward(self, x):
        output = super().forward(x)
        output = output.sum(0)
        if self.final_probability_divider > 1:
            output = output.t() / self.final_probability_divider
        else:
            output = output.t()

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BatchRandomForestRegressor(BatchedTreeEnsemble):

    def __init__(self, net_parameters, n_features, device):
        super(BatchRandomForestRegressor, self).__init__(
            net_parameters, n_features, 1)
        self.final_divider = len(net_parameters)

    def forward(self, x):
        output = super().forward(x)
        output = output.sum(0)
        if self.final_divider > 1:
            output = output.t() / self.final_divider
        else:
            output = output.t()

        return output


class BeamRandomForestClassifier(BeamTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, device):
        super(BeamRandomForestClassifier, self).__init__(
            net_parameters, n_features, len(classes))
        self.final_probability_divider = len(net_parameters)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
            self.classes = torch.nn.Parameter(
                torch.IntTensor(classes), requires_grad=False)

    def forward(self, x):
        output = super().forward(x)
        output = output.sum(1) / self.final_probability_divider

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BeamRandomForestRegressor(BeamTreeEnsemble):

    def __init__(self, net_parameters, n_features, device):
        super(BeamRandomForestRegressor, self).__init__(
            net_parameters, n_features, 1)
        self.final_divider = len(net_parameters)

    def forward(self, x):
        output = super().forward(x)
        output = output.sum(1) / self.final_divider

        return output


class BeamPPRandomForestClassifier(BeamPPTreeEnsemble):

    def __init__(self, net_parameters, n_features, classes, device):
        super(BeamPPRandomForestClassifier, self).__init__(
            net_parameters, n_features, len(classes))
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
            self.classes = torch.nn.Parameter(
                torch.IntTensor(classes), requires_grad=False)
        self.final_probability_divider = len(net_parameters)

    def forward(self, x):
        output = super().forward(x)
        output = output.sum(1) / self.final_probability_divider

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BeamPPRandomForestRegressor(BeamPPTreeEnsemble):

    def __init__(self, net_parameters, n_features, device):
        super(BeamPPRandomForestRegressor, self).__init__(
            net_parameters, n_features, 1)
        self.final_divider = len(net_parameters)

    def forward(self, x):
        output = super().forward(x)
        output = output.sum(1) / self.final_divider

        return output


def find_max_depth(operator):
    depth = 0

    for tree in operator.raw_operator.estimators_:
        tree = copy.deepcopy(tree)

        lefts = tree.tree_.children_left
        rights = tree.tree_.children_right

        ids = [i for i in range(len(lefts))]
        nodes = list(zip(ids, lefts, rights))

        nodes_map = {0: Node(0)}
        current_node = 0
        for i, node in enumerate(nodes):
            id, left, right = node

            if left != -1:
                l_node = Node(left)
                nodes_map[left] = l_node
            else:
                lefts[i] = id
                l_node = -1

            if right != -1:
                r_node = Node(right)
                nodes_map[right] = r_node
            else:
                rights[i] = id
                r_node = -1

            nodes_map[current_node].left = l_node
            nodes_map[current_node].right = r_node

            current_node += 1

        depth = max(depth, find_depth(nodes_map[0], -1))

    return depth


def convert_sklearn_random_forest_classifier(operator, device, extra_config):
    sklearn_rf_classifier = operator.raw_operator

    if not all(isinstance(c, int) for c in operator.raw_operator.classes_.tolist()):
        raise RuntimeError(
            'Random Forest Classifier translation only supports integer class labels')

    if "tree_implementation" not in extra_config:  # use heurstics to get the tree implementation
        max_depth = sklearn_rf_classifier.max_depth
        if max_depth is None:
            max_depth = max_depth = find_max_depth(operator)

        if max_depth <= 10:
            if max_depth <= 4:
                net_parameters = [get_parameters_for_batch(
                    e) for e in sklearn_rf_classifier.estimators_]
                return BatchRandomForestClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                                   operator.raw_operator.classes_.tolist(), device)
            else:
                net_parameters = [get_parameters_for_beam(
                    e) for e in sklearn_rf_classifier.estimators_]
                return BeamPPRandomForestClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                                    operator.raw_operator.classes_.tolist(), device)
        else:
            net_parameters = [get_parameters_for_beam(
                e) for e in sklearn_rf_classifier.estimators_]
            return BeamRandomForestClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                              operator.raw_operator.classes_.tolist(), device)
    else:  # manually set tree implementation
        if 'tree_implementation' in extra_config and extra_config['tree_implementation'] == 'batch':
            net_parameters = [get_parameters_for_batch(
                e) for e in sklearn_rf_classifier.estimators_]
            return BatchRandomForestClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                               operator.raw_operator.classes_.tolist(), device)
        elif 'tree_implementation' in extra_config and extra_config['tree_implementation'] == 'beam':
            net_parameters = [get_parameters_for_beam(
                e) for e in sklearn_rf_classifier.estimators_]
            return BeamRandomForestClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                              operator.raw_operator.classes_.tolist(), device)
        elif 'tree_implementation' in extra_config and extra_config['tree_implementation'] == 'beam++':
            net_parameters = [get_parameters_for_beam(
                e) for e in sklearn_rf_classifier.estimators_]
            return BeamPPRandomForestClassifier(net_parameters, sklearn_rf_classifier.n_features_,
                                                operator.raw_operator.classes_.tolist(), device)
        else:
            raise ValueError("Tree implementation {} not found".format(extra_config))


def convert_sklearn_random_forest_regressor(operator, device, extra_config):

    # TODO: extraconfig
    sklearn_rf_regressor = operator.raw_operator

    # TODO: automatically find the max tree depth by traversing the trees without relying on user input.
    if sklearn_rf_regressor.max_depth is not None and sklearn_rf_regressor.max_depth <= 10:
        if sklearn_rf_regressor.max_depth <= 4:
            net_parameters = [get_parameters_for_batch(
                e) for e in sklearn_rf_regressor.estimators_]
            return BatchRandomForestRegressor(net_parameters, sklearn_rf_regressor.n_features_, device)
        else:
            net_parameters = [get_parameters_for_beam(
                e) for e in sklearn_rf_regressor.estimators_]
            return BeamPPRandomForestRegressor(net_parameters, sklearn_rf_regressor.n_features_, device)
    else:
        if sklearn_rf_regressor.max_depth is None:
            warnings.warn("RandomForest model does not have a defined max_depth value. Consider setting one as it "
                          "will help the translator to pick a better translation method")
        elif sklearn_rf_regressor.max_depth > 10:
            warnings.warn("RandomForest model max_depth value is {0}. Consider setting a smaller value as it improves"
                          " translated tree scoring performance.".format(sklearn_rf_regressor.max_depth))
        net_parameters = [get_parameters_for_beam(
            e) for e in sklearn_rf_regressor.estimators_]
        return BeamRandomForestRegressor(net_parameters, sklearn_rf_regressor.n_features_, device)


def convert_sklearn_decision_tree_classifier(operator, device, extra_config):
    operator.raw_operator.estimators_ = [operator.raw_operator]
    dt_config = extra_config.copy()

    max_depth = operator.raw_operator.max_depth
    if max_depth is None:
        max_depth = max_depth = find_max_depth(operator)
    if 'tree_implementation' not in dt_config:
        if max_depth <= 10:
            dt_config['tree_implementation'] = 'beam++'
        else:
            dt_config['tree_implementation'] = 'beam++'

    return convert_sklearn_random_forest_classifier(operator, device, dt_config)


register_converter('SklearnRandomForestClassifier',
                   convert_sklearn_random_forest_classifier)
register_converter('SklearnRandomForestRegressor',
                   convert_sklearn_random_forest_regressor)
register_converter('SklearnDecisionTreeClassifier',
                   convert_sklearn_decision_tree_classifier)
register_converter('SklearnExtraTreesClassifier',
                   convert_sklearn_random_forest_classifier)
