# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for LightGBM models.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from . import constants
from ._gbdt_commons import convert_gbdt_classifier_common, convert_gbdt_common
from ._tree_commons import TreeParameters


def _tree_traversal(node, lefts, rights, features, thresholds, values, missings, count):
    """
    Recursive function for parsing a tree and filling the input data structures.
    """
    if "left_child" in node:
        features.append(node["split_feature"])
        thresholds.append(node["threshold"])
        values.append([-1])
        lefts.append(count + 1)
        rights.append(-1)
        missings.append(-1)
        pos = len(rights) - 1
        count = _tree_traversal(node["left_child"], lefts, rights, features, thresholds, values, missings, count + 1)
        rights[pos] = count + 1
        if node['missing_type'] == 'None':
            # Missing values not present in training data are treated as zeros during inference.
            missings[pos] = lefts[pos] if 0 < node["threshold"] else rights[pos]
        else:
            missings[pos] = lefts[pos] if node["default_left"] else rights[pos]
        return _tree_traversal(node["right_child"], lefts, rights, features, thresholds, values, missings, count + 1)
    else:
        features.append(0)
        thresholds.append(0)
        values.append([node["leaf_value"]])
        lefts.append(-1)
        rights.append(-1)
        missings.append(-1)
        return count


def _get_tree_parameters(tree_info):
    """
    Parse the tree and returns an in-memory friendly representation of its structure.
    """
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    missings = []
    _tree_traversal(tree_info["tree_structure"], lefts, rights, features, thresholds, values, missings, 0)

    return TreeParameters(lefts, rights, features, thresholds, values, missings)


def convert_sklearn_lgbm_classifier(operator, device, extra_config):
    """
    Converter for `lightgbm.LGBMClassifier` (trained using the Sklearn API).

    Args:
        operator: An operator wrapping a `lightgbm.LGBMClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"
    assert not hasattr(operator.raw_operator, "use_missing") or operator.raw_operator.use_missing
    assert not hasattr(operator.raw_operator, "zero_as_missing") or not operator.raw_operator.zero_as_missing

    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()["tree_info"]
    n_classes = operator.raw_operator._n_classes

    return convert_gbdt_classifier_common(operator, tree_infos, _get_tree_parameters, n_features, n_classes, missing_val=None, extra_config=extra_config)


def convert_sklearn_lgbm_regressor(operator, device, extra_config):
    """
    Converter for `lightgbm.LGBMRegressor` and `lightgbm.LGBMRanker` (trained using the Sklearn API).

    Args:
        operator: An operator wrapping a `lightgbm.LGBMRegressor` or `lightgbm.LGBMRanker` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"
    assert not hasattr(operator.raw_operator, "use_missing") or operator.raw_operator.use_missing
    assert not hasattr(operator.raw_operator, "zero_as_missing") or not operator.raw_operator.zero_as_missing

    # Get tree information out of the model.
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()["tree_info"]
    if operator.raw_operator._objective == "tweedie":
        extra_config[constants.POST_TRANSFORM] = constants.TWEEDIE

    return convert_gbdt_common(operator, tree_infos, _get_tree_parameters, n_features, missing_val=None, extra_config=extra_config)


# Register the converters.
register_converter("SklearnLGBMClassifier", convert_sklearn_lgbm_classifier)
register_converter("SklearnLGBMRanker", convert_sklearn_lgbm_regressor)
register_converter("SklearnLGBMRegressor", convert_sklearn_lgbm_regressor)
