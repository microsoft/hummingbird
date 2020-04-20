# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import warnings
import copy

import torch
from onnxconverter_common.registration import register_converter

from ._tree_commons import get_parameters_sklearn_common, get_parameters_for_tree_trav_sklearn
from ._tree_commons import get_tree_params_and_type, get_parameters_for_gemm_common
from ._tree_commons import GEMMTreeEnsemble, TreeTraversalTreeEnsemble, PerfectTreeTraversalTreeEnsemble, TreeImpl


class GEMMDecisionTree(GEMMTreeEnsemble):
    """
    Class implementing the GEMM strategy in PyTorch for decision tree models.
    """

    def __init__(self, net_parameters, n_features, classes=None, device=None):
        super(GEMMDecisionTree, self).__init__(net_parameters, n_features, classes)
        self.final_probability_divider = len(net_parameters)

    def aggregation(self, x):
        output = x.sum(0).t()

        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider

        return output


class TreeTraversalDecisionTree(TreeTraversalTreeEnsemble):
    """
    Class implementing the Tree Traversal strategy in PyTorch for decision tree models.
    """

    def __init__(self, net_parameters, max_depth, n_features, classes=None, device=None):
        super(TreeTraversalDecisionTree, self).__init__(net_parameters, max_depth, n_features, classes)
        self.final_probability_divider = len(net_parameters)

    def aggregation(self, x):
        output = x.sum(1)

        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider

        return output


class PerfectTreeTraversalDecisionTree(PerfectTreeTraversalTreeEnsemble):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch for decision tree models.
    """

    def __init__(self, net_parameters, max_depth, n_features, classes=None, device=None):
        super(PerfectTreeTraversalDecisionTree, self).__init__(net_parameters, max_depth, n_features, classes)
        self.final_probability_divider = len(net_parameters)

    def aggregation(self, x):
        output = x.sum(1)

        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider

        return output


def convert_sklearn_random_forest_classifier(model, device, extra_config):
    """
    Converter for Sklearn's Random Forest, DecisionTree, ExtraTrees classifiers.
    """
    assert model is not None

    # Get tree information out of the model.
    tree_infos = model.raw_operator.estimators_
    n_features = model.raw_operator.n_features_
    classes = model.raw_operator.classes_.tolist()

    # Analyze classes.
    if not all(isinstance(c, int) for c in classes):
        raise RuntimeError("Random Forest Classifier translation only supports integer class labels")

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(tree_infos, get_parameters_sklearn_common, extra_config)

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, n_features
            )
            for tree_param in tree_parameters
        ]
        return GEMMDecisionTree(net_parameters, n_features, classes, device)

    net_parameters = [
        get_parameters_for_tree_trav_sklearn(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.tree_trav:
        return TreeTraversalDecisionTree(net_parameters, max_depth, n_features, classes, device)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return PerfectTreeTraversalDecisionTree(net_parameters, max_depth, n_features, classes, device)


def convert_sklearn_random_forest_regressor(model, device, extra_config):
    """
    Converter for Sklearn's Random Forest, DecisionTree, ExtraTrees regressors.
    """
    assert model is not None

    # Get tree information out of the model.
    tree_infos = model.raw_operator.estimators_
    n_features = model.raw_operator.n_features_

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(tree_infos, get_parameters_sklearn_common, extra_config)

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, n_features
            )
            for tree_param in tree_parameters
        ]
        return GEMMDecisionTree(net_parameters, n_features, device=device)

    net_parameters = [
        get_parameters_for_tree_trav_sklearn(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.perf_tree_trav:
        return PerfectTreeTraversalDecisionTree(net_parameters, max_depth, n_features, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.tree_trav
        return TreeTraversalDecisionTree(net_parameters, max_depth, n_features, device=device)


def convert_sklearn_decision_tree_classifier(model, device, extra_config):
    """
    Converter for Sklearn Decision Tree classifier.
    """
    model.raw_operator.estimators_ = [model.raw_operator]
    return convert_sklearn_random_forest_classifier(model, device, extra_config)


# Register the converters.
register_converter("SklearnRandomForestClassifier", convert_sklearn_random_forest_classifier)
register_converter("SklearnRandomForestRegressor", convert_sklearn_random_forest_regressor)
register_converter("SklearnDecisionTreeClassifier", convert_sklearn_decision_tree_classifier)
register_converter("SklearnExtraTreesClassifier", convert_sklearn_random_forest_classifier)
