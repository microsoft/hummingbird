# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn decision-tree-based models: DecisionTree, RandomForest and ExtraTrees.
"""

import copy
from onnxconverter_common.registration import register_converter

from .. import constants
from .._tree_commons import get_parameters_for_sklearn_common, get_parameters_for_tree_trav_sklearn
from .._tree_commons import convert_decision_ensemble_tree_common


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
    assert operator is not None, "Cannot convert None operator"

    # Get tree information out of the model.
    tree_infos = operator.raw_operator.estimators_
    n_features = operator.raw_operator.n_features_
    classes = operator.raw_operator.classes_.tolist()

    # For Sklearn Trees we need to know how many trees are there for normalization.
    extra_config[constants.NUM_TREES] = len(tree_infos)

    # Analyze classes.
    if not all(isinstance(c, int) for c in classes):
        raise RuntimeError("Random Forest Classifier translation only supports integer class labels")

    def get_parameters_for_tree_trav(lefts, rights, features, thresholds, values, missings=None, extra_config={}):
        return get_parameters_for_tree_trav_sklearn(lefts, rights, features, thresholds, values, missings, classes, extra_config)

    return convert_decision_ensemble_tree_common(
        operator,
        tree_infos,
        get_parameters_for_sklearn_common,
        get_parameters_for_tree_trav,
        n_features,
        classes,
        extra_config,
    )


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
    assert operator is not None, "Cannot convert None operator"

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator.estimators_
    n_features = operator.raw_operator.n_features_

    # For Sklearn Trees we need to know how many trees are there for normalization.
    extra_config[constants.NUM_TREES] = len(tree_infos)

    def get_parameters_for_tree_trav(lefts, rights, features, thresholds, values, missings=None, extra_config={}):
        return get_parameters_for_tree_trav_sklearn(lefts, rights, features, thresholds, values, missings, None, extra_config)

    return convert_decision_ensemble_tree_common(
        operator,
        tree_infos,
        get_parameters_for_sklearn_common,
        get_parameters_for_tree_trav,
        n_features,
        extra_config=extra_config,
    )


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
    assert operator is not None, "Cannot convert None operator"

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
    assert operator is not None, "Cannot convert None operator"

    operator.raw_operator.estimators_ = [operator.raw_operator]
    return convert_sklearn_random_forest_regressor(operator, device, extra_config)


# Register the converters.
register_converter("SklearnDecisionTreeClassifier", convert_sklearn_decision_tree_classifier)
register_converter("SklearnDecisionTreeRegressor", convert_sklearn_decision_tree_regressor)
register_converter("SklearnExtraTreesClassifier", convert_sklearn_random_forest_classifier)
register_converter("SklearnExtraTreesRegressor", convert_sklearn_random_forest_regressor)
register_converter("SklearnRandomForestClassifier", convert_sklearn_random_forest_classifier)
register_converter("SklearnRandomForestRegressor", convert_sklearn_random_forest_regressor)
