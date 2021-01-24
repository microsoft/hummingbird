# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for XGBoost models.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from . import constants
from ._gbdt_commons import convert_gbdt_classifier_common, convert_gbdt_common
from ._tree_commons import TreeParameters


def _tree_traversal(tree_info, lefts, rights, missing, features, thresholds, values):
    """
    Recursive function for parsing a tree and filling the input data structures.
    """
    count = 0
    while count < len(tree_info):
        if "leaf" in tree_info[count]:
            features.append(0)
            thresholds.append(0)
            values.append([float(tree_info[count].split("=")[1])])
            lefts.append(-1)
            rights.append(-1)
            missing.append(-1)
            count += 1
        else:
            features.append(int(tree_info[count].split(":")[1].split("<")[0].replace("[f", "")))
            thresholds.append(float(tree_info[count].split(":")[1].split("<")[1].replace("]", "")))
            values.append([-1])
            count += 1
            l_wrong_id = tree_info[count].split(",")[0].replace("yes=", "")
            l_correct_id = 0
            temp = 0
            while not tree_info[temp].startswith(str(l_wrong_id + ":")):
                if "leaf" in tree_info[temp]:
                    temp += 1
                else:
                    temp += 2
                l_correct_id += 1
            lefts.append(l_correct_id)

            r_wrong_id = tree_info[count].split(",")[1].replace("no=", "")
            r_correct_id = 0
            temp = 0
            while not tree_info[temp].startswith(str(r_wrong_id + ":")):
                if "leaf" in tree_info[temp]:
                    temp += 1
                else:
                    temp += 2
                r_correct_id += 1
            rights.append(r_correct_id)

            missing_wrong_id = tree_info[count].split(",")[2].replace("missing=", "")
            missing.append(l_correct_id if l_wrong_id == missing_wrong_id else r_correct_id)
            count += 1


def _get_tree_parameters(tree_info):
    """
    Parse the tree and returns an in-memory friendly representation of its structure.
    """
    lefts = []
    rights = []
    missing = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(
        tree_info.replace("[f", "").replace("[", "").replace("]", "").split(), lefts, rights, missing, features, thresholds, values
    )

    return TreeParameters(lefts, rights, features, thresholds, values, missing)


def convert_sklearn_xgb_classifier(operator, device, extra_config):
    """
    Converter for `xgboost.XGBClassifier` (trained using the Sklearn API).

    Args:
        operator: An operator wrapping a `xgboost.XGBClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"
    if "n_features" in extra_config:
        n_features = extra_config["n_features"]
    else:
        raise RuntimeError(
            'XGBoost converter is not able to infer the number of input features.\
             Please pass "n_features:N" as extra configuration to the converter or fill a bug report.'
        )
    tree_infos = operator.raw_operator.get_booster().get_dump()
    n_classes = operator.raw_operator.n_classes_
    missing_val = operator.raw_operator.missing

    return convert_gbdt_classifier_common(operator, tree_infos, _get_tree_parameters, n_features, n_classes, missing_val=missing_val, extra_config=extra_config)


def convert_sklearn_xgb_regressor(operator, device, extra_config):
    """
    Converter for `xgboost.XGBRegressor` (trained using the Sklearn API).

    Args:
        operator: An operator wrapping a `xgboost.XGBRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"
    if "n_features" in extra_config:
        n_features = extra_config["n_features"]
    else:
        raise RuntimeError(
            'XGBoost converter is not able to infer the number of input features.\
             Please pass "n_features:N" as extra configuration to the converter or fill a bug report.'
        )

    tree_infos = operator.raw_operator.get_booster().get_dump()
    base_prediction = operator.raw_operator.base_score
    if base_prediction is None:
        base_prediction = [0.5]
    if type(base_prediction) is float:
        base_prediction = [base_prediction]

    extra_config[constants.BASE_PREDICTION] = base_prediction
    missing_val = operator.raw_operator.missing

    return convert_gbdt_common(operator, tree_infos, _get_tree_parameters, n_features, missing_val=missing_val, extra_config=extra_config)


# Register the converters.
register_converter("SklearnXGBClassifier", convert_sklearn_xgb_classifier)
register_converter("SklearnXGBRanker", convert_sklearn_xgb_regressor)
register_converter("SklearnXGBRegressor", convert_sklearn_xgb_regressor)
