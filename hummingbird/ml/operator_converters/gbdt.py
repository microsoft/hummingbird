# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for Sklearn's GradientBoosting models.
"""

import warnings

import numpy as np
from onnxconverter_common.registration import register_converter

from . import constants
from ._gbdt_commons import convert_gbdt_common, convert_gbdt_classifier_common
from ._tree_commons import get_parameters_for_sklearn_common, get_parameters_for_tree_trav_sklearn


def convert_sklearn_gbdt_classifier(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.GradientBoostingClassifier` or `sklearn.ensemble.HistGradientBoostingClassifier`

    Args:
        operator: An operator wrapping a `sklearn.ensemble.GradientBoostingClassifier` or `sklearn.ensemble.HistGradientBoostingClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Get tree information out of the operator.
    if hasattr(operator.raw_operator, "estimators_"):
        # SklearnGradientBoostingClassifier
        tree_infos = operator.raw_operator.estimators_
    elif hasattr(operator.raw_operator, "_predictors"):
        # SklearnHistGradientBoostingClassifier
        tree_infos = operator.raw_operator._predictors
    n_features = operator.raw_operator.n_features_
    learning_rate = operator.raw_operator.learning_rate
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
    if operator.raw_operator.init == "zero":
        alpha = [[0.0]]
    elif operator.raw_operator.init is None:
        if n_classes == 1:
            alpha = [[np.log(operator.raw_operator.init_.class_prior_[1] / (1 - operator.raw_operator.init_.class_prior_[1]))]]
        else:
            alpha = [[np.log(operator.raw_operator.init_.class_prior_[i]) for i in range(n_classes)]]
    else:
        raise RuntimeError("Custom initializers for GBDT are not yet supported in Hummingbird.")

    extra_config[constants.ALPHA] = alpha
    extra_config[constants.LEARNING_RATE] = learning_rate
    # For sklearn models we need to massage the parameters a bit before generating the parameters for tree_trav.
    extra_config[constants.GET_PARAMETERS_FOR_TREE_TRAVERSAL] = get_parameters_for_tree_trav_sklearn

    return convert_gbdt_classifier_common(
        tree_infos, get_parameters_for_sklearn_common, n_features, n_classes, classes, extra_config
    )


def convert_sklearn_gbdt_regressor(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.GradientBoostingRegressor`.

    Args:
        operator: An operator wrapping a `sklearn.ensemble.GradientBoostingRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Get tree information out of the operator.
    tree_infos = operator.raw_operator.estimators_.ravel().tolist()
    n_features = operator.raw_operator.n_features_
    learning_rate = operator.raw_operator.learning_rate

    # Get the value for Alpha.
    if operator.raw_operator.init == "zero":
        alpha = [[0.0]]
    elif operator.raw_operator.init is None:
        alpha = operator.raw_operator.init_.constant_.tolist()
    else:
        raise RuntimeError("Custom initializers for GBDT are not yet supported in Hummingbird.")

    extra_config[constants.ALPHA] = alpha
    extra_config[constants.LEARNING_RATE] = learning_rate
    # For sklearn models we need to massage the parameters a bit before generating the parameters for tree_trav.
    extra_config[constants.GET_PARAMETERS_FOR_TREE_TRAVERSAL] = get_parameters_for_tree_trav_sklearn

    return convert_gbdt_common(tree_infos, get_parameters_for_sklearn_common, n_features, None, extra_config)


# Register the converters.
register_converter("SklearnGradientBoostingClassifier", convert_sklearn_gbdt_classifier)
register_converter("SklearnGradientBoostingRegressor", convert_sklearn_gbdt_regressor)
