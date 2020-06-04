# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn decision-tree-based models: DecisionTree, RandomForest and ExtraTrees
"""

import warnings
import copy

import torch
from onnxconverter_common.registration import register_converter

from ._tree_commons import get_parameters_for_sklearn_common, get_parameters_for_tree_trav_sklearn
from ._tree_commons import get_tree_params_and_type, get_parameters_for_gemm_common
from ._tree_implementations import GEMMTreeImpl, TreeTraversalTreeImpl, PerfectTreeTraversalTreeImpl, TreeImpl


class GEMMDecisionTreeImpl(GEMMTreeImpl):
    """
    Class implementing the GEMM strategy in PyTorch for decision tree models.

    """

    def __init__(self, net_parameters, n_features, classes=None):
        """
        Args:
            net_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
        """
        super(GEMMDecisionTreeImpl, self).__init__(net_parameters, n_features, classes)
        self.final_probability_divider = len(net_parameters)

    def aggregation(self, x):
        output = x.sum(0).t()

        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider

        return output


class TreeTraversalDecisionTreeImpl(TreeTraversalTreeImpl):
    """
    Class implementing the Tree Traversal strategy in PyTorch for decision tree models.
    """

    def __init__(self, net_parameters, max_depth, n_features, classes=None):
        """
        Args:
            net_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
        """
        super(TreeTraversalDecisionTreeImpl, self).__init__(net_parameters, max_depth, n_features, classes)
        self.final_probability_divider = len(net_parameters)

    def aggregation(self, x):
        output = x.sum(1)

        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider

        return output


class PerfectTreeTraversalDecisionTreeImpl(PerfectTreeTraversalTreeImpl):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch for decision tree models.
    """

    def __init__(self, net_parameters, max_depth, n_features, classes=None):
        """
        Args:
            net_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
        """
        super(PerfectTreeTraversalDecisionTreeImpl, self).__init__(net_parameters, max_depth, n_features, classes)
        self.final_probability_divider = len(net_parameters)

    def aggregation(self, x):
        output = x.sum(1)

        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider

        return output


def convert_sklearn_random_forest_classifier(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.RandomForestClassifier` and `sklearn.ensemble.ExtraTreesClassifier`.

    Args:
        operator: An operator wrapping a tree (ensemble) classifier model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Get tree information out of the model.
    tree_infos = operator.raw_operator.estimators_
    n_features = operator.raw_operator.n_features_
    classes = operator.raw_operator.classes_.tolist()

    # Analyze classes.
    if not all(isinstance(c, int) for c in classes):
        raise RuntimeError("Random Forest Classifier translation only supports integer class labels")

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(
        tree_infos, get_parameters_for_sklearn_common, extra_config
    )

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, n_features
            )
            for tree_param in tree_parameters
        ]
        return GEMMDecisionTreeImpl(net_parameters, n_features, classes)

    net_parameters = [
        get_parameters_for_tree_trav_sklearn(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.tree_trav:
        return TreeTraversalDecisionTreeImpl(net_parameters, max_depth, n_features, classes)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return PerfectTreeTraversalDecisionTreeImpl(net_parameters, max_depth, n_features, classes)


def convert_sklearn_random_forest_regressor(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.RandomForestRegressor` and `sklearn.ensemble.ExtraTreesRegressor`

    Args:
        operator: An operator wrapping the RandomForestRegressor and ExtraTreesRegressor model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator.estimators_
    n_features = operator.raw_operator.n_features_

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(
        tree_infos, get_parameters_for_sklearn_common, extra_config
    )

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, n_features
            )
            for tree_param in tree_parameters
        ]
        return GEMMDecisionTreeImpl(net_parameters, n_features)

    net_parameters = [
        get_parameters_for_tree_trav_sklearn(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.perf_tree_trav:
        return PerfectTreeTraversalDecisionTreeImpl(net_parameters, max_depth, n_features)
    else:  # Remaining possible case: tree_type == TreeImpl.tree_trav
        return TreeTraversalDecisionTreeImpl(net_parameters, max_depth, n_features)


def convert_sklearn_decision_tree_classifier(operator, device, extra_config):
    """
    Converter for `sklearn.tree.DecisionTreeClassifier`.

    Args:
        operator: An operator wrapping a `sklearn.tree.DecisionTreeClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    operator.raw_operator.estimators_ = [operator.raw_operator]
    return convert_sklearn_random_forest_classifier(operator, device, extra_config)


def convert_sklearn_decision_tree_regressor(operator, device, extra_config):
    """
    Converter for `sklearn.tree.DecisionTreeRegressor`.

    Args:
        operator: An operator wrapping a `sklearn.tree.DecisionTreeRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    operator.raw_operator.estimators_ = [operator.raw_operator]
    return convert_sklearn_random_forest_regressor(operator, device, extra_config)


# Register the converters.
register_converter("SklearnDecisionTreeClassifier", convert_sklearn_decision_tree_classifier)
register_converter("SklearnDecisionTreeRegressor", convert_sklearn_decision_tree_regressor)
register_converter("SklearnExtraTreesClassifier", convert_sklearn_random_forest_classifier)
register_converter("SklearnExtraTreesRegressor", convert_sklearn_random_forest_regressor)
register_converter("SklearnRandomForestClassifier", convert_sklearn_random_forest_classifier)
register_converter("SklearnRandomForestRegressor", convert_sklearn_random_forest_regressor)
