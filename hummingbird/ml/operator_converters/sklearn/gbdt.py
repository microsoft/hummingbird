# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for Sklearn's GradientBoosting models.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .. import constants
from .._gbdt_commons import convert_gbdt_common, convert_gbdt_classifier_common
from .._tree_commons import get_parameters_for_sklearn_common, get_parameters_for_tree_trav_sklearn, TreeParameters


def _get_parameters_hist_gbdt(trees):
    """
    Extract the tree parameters from SklearnHistGradientBoostingClassifier trees
    Args:
        trees: The information representing a tree (ensemble)
        Returns: The tree parameters wrapped into an instance of `operator_converters._tree_commons_TreeParameters`
    """
    features = [n["feature_idx"] for n in trees.nodes]
    thresholds = [n["threshold"] if n["threshold"] != 0 else -1 for n in trees.nodes]
    lefts = [n["left"] if n["left"] != 0 else -1 for n in trees.nodes]
    rights = [n["right"] if n["right"] != 0 else -1 for n in trees.nodes]
    values = [[n["value"]] if n["value"] != 0 else [-1] for n in trees.nodes]

    return TreeParameters(lefts, rights, features, thresholds, values)


def convert_sklearn_gbdt_classifier(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.GradientBoostingClassifier`

    Args:
        operator: An operator wrapping a `sklearn.ensemble.GradientBoostingClassifier`
                  or `sklearn.ensemble.HistGradientBoostingClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator.estimators_
    # GBDT does not scale the value using the learning rate upfront, we have to do it.
    extra_config[constants.LEARNING_RATE] = operator.raw_operator.learning_rate
    # GBDT does not normalize values upfront, we have to do it.
    extra_config[constants.GET_PARAMETERS_FOR_TREE_TRAVERSAL] = get_parameters_for_tree_trav_sklearn

    n_features = operator.raw_operator.n_features_
    classes = operator.raw_operator.classes_.tolist()
    n_classes = len(classes)

    # Analyze classes.
    if not all(isinstance(c, int) for c in classes):
        raise RuntimeError("GBDT Classifier translation only supports integer class labels.")
    if n_classes == 2:
        n_classes -= 1

    # Reshape the tree_infos into hummingbird gbdt internal format.
    tree_infos = [tree_infos[i][j] for j in range(n_classes) for i in range(len(tree_infos))]

    # Get the value for Alpha.
    if hasattr(operator.raw_operator, "init"):
        if operator.raw_operator.init == "zero":
            base_prediction = [[0.0]]
        elif operator.raw_operator.init is None:
            if n_classes == 1:
                base_prediction = [
                    [np.log(operator.raw_operator.init_.class_prior_[1] / (1 - operator.raw_operator.init_.class_prior_[1]))]
                ]
            else:
                base_prediction = [[np.log(operator.raw_operator.init_.class_prior_[i]) for i in range(n_classes)]]
        else:
            raise RuntimeError("Custom initializers for GBDT are not yet supported in Hummingbird.")
    elif hasattr(operator.raw_operator, "_baseline_prediction"):
        if n_classes == 1:
            base_prediction = [[operator.raw_operator._baseline_prediction]]
        else:
            base_prediction = np.array([operator.raw_operator._baseline_prediction.flatten().tolist()])

    extra_config[constants.BASE_PREDICTION] = base_prediction
    extra_config[constants.REORDER_TREES] = False

    return convert_gbdt_classifier_common(
        operator, tree_infos, get_parameters_for_sklearn_common, n_features, n_classes, classes, missing_val=None, extra_config=extra_config
    )


def convert_sklearn_gbdt_regressor(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.GradientBoostingRegressor`.

    Args:
        operator: An operator wrapping a `sklearn.ensemble.GradientBoostingRegressor` or
        `sklearn.ensemble.HistGradientBoostingRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator.estimators_.ravel().tolist()
    n_features = operator.raw_operator.n_features_
    extra_config[constants.LEARNING_RATE] = operator.raw_operator.learning_rate
    # For sklearn models we need to massage the parameters a bit before generating the parameters for tree_trav.
    extra_config[constants.GET_PARAMETERS_FOR_TREE_TRAVERSAL] = get_parameters_for_tree_trav_sklearn

    # Get the value for Alpha.
    if operator.raw_operator.init == "zero":
        base_prediction = [[0.0]]
    elif operator.raw_operator.init is None:
        base_prediction = operator.raw_operator.init_.constant_.tolist()
    else:
        raise RuntimeError("Custom initializers for GBDT are not yet supported in Hummingbird.")

    extra_config[constants.BASE_PREDICTION] = base_prediction

    return convert_gbdt_common(operator, tree_infos, get_parameters_for_sklearn_common, n_features, missing_val=None, extra_config=extra_config)


def convert_sklearn_hist_gbdt_classifier(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.HistGradientBoostingClassifier`

    Args:
        operator: An operator wrapping a `sklearn.ensemble.HistGradientBoostingClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator._predictors
    n_features = operator.raw_operator.n_features_
    classes = operator.raw_operator.classes_.tolist()
    n_classes = len(classes)

    # Analyze classes.
    if not all(isinstance(c, int) for c in classes):
        raise RuntimeError("GBDT Classifier translation only supports integer class labels.")
    if n_classes == 2:
        n_classes -= 1

    # Reshape the tree_infos to a more generic format.
    tree_infos = [tree_infos[i][j] for j in range(n_classes) for i in range(len(tree_infos))]

    # Get the value for Alpha.
    if n_classes == 1:
        base_prediction = [[operator.raw_operator._baseline_prediction]]
    else:
        base_prediction = np.array([operator.raw_operator._baseline_prediction.flatten().tolist()])

    extra_config[constants.BASE_PREDICTION] = base_prediction
    extra_config[constants.REORDER_TREES] = False

    return convert_gbdt_classifier_common(operator, tree_infos, _get_parameters_hist_gbdt, n_features, n_classes, classes, missing_val=None, extra_config=extra_config)


def convert_sklearn_hist_gbdt_regressor(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.HistGradientBoostingRegressor`

    Args:
        operator: An operator wrapping a `sklearn.ensemble.HistGradientBoostingRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator._predictors
    tree_infos = [tree_infos[i][0] for i in range(len(tree_infos))]
    n_features = operator.raw_operator.n_features_
    extra_config[constants.BASE_PREDICTION] = [[operator.raw_operator._baseline_prediction]]

    return convert_gbdt_common(operator, tree_infos, _get_parameters_hist_gbdt, n_features, missing_val=None, extra_config=extra_config)


# Register the converters.
register_converter("SklearnGradientBoostingClassifier", convert_sklearn_gbdt_classifier)
register_converter("SklearnGradientBoostingRegressor", convert_sklearn_gbdt_regressor)
register_converter("SklearnHistGradientBoostingClassifier", convert_sklearn_hist_gbdt_classifier)
register_converter("SklearnHistGradientBoostingRegressor", convert_sklearn_hist_gbdt_regressor)
