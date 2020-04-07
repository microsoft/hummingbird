# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from hummingbird.operator_converters.gbdt import BatchGBDTClassifier, BatchGBDTRegressor, BeamPPGBDTClassifier
from hummingbird.operator_converters.gbdt import BeamPPGBDTRegressor, BeamGBDTClassifier, BeamGBDTRegressor

from ._tree_commons import get_gbdt_by_config_or_depth, TreeImpl
from ._tree_commons import get_parameters_for_tree_trav_generic, get_parameters_for_gemm_generic
from ..common._registration import register_converter


def _tree_traversal(tree_info, ls, rs, fs, ts, vs):
    count = 0
    while count < len(tree_info):
        if "leaf" in tree_info[count]:
            fs.append(0)
            ts.append(0)
            vs.append([float(tree_info[count].split("=")[1])])
            ls.append(-1)
            rs.append(-1)
            count += 1
        else:
            fs.append(int(tree_info[count].split(":")[1].split("<")[0].replace("[f", "")))
            ts.append(float(tree_info[count].split(":")[1].split("<")[1].replace("]", "")))
            vs.append([-1])
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
            ls.append(l_correct_id)

            r_wrong_id = tree_info[count].split(",")[1].replace("no=", "")
            r_correct_id = 0
            temp = 0
            while not tree_info[temp].startswith(str(r_wrong_id + ":")):
                if "leaf" in tree_info[temp]:
                    temp += 1
                else:
                    temp += 2
                r_correct_id += 1
            rs.append(r_correct_id)

            count += 1


def _get_tree_parameters_for_gemm(tree_info, n_features):
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(
        tree_info.replace("[f", "").replace("[", "").replace("]", "").split(), lefts, rights, features, thresholds, values
    )

    if len(lefts) == 1:
        # XGB model creating tree with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        values = [np.array([0.0]), values[0], values[0]]

    weights = []
    biases = []

    values = np.array(values)

    # first hidden layer has all inequalities
    hidden_weights = []
    hidden_biases = []
    for left, feature, thresh in zip(lefts, features, thresholds):
        if left != -1:
            hidden_weights.append([1 if i == feature else 0 for i in range(n_features)])
            hidden_biases.append(thresh)
    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))
    n_splits = len(hidden_weights)

    # second hidden layer has ANDs for each leaf of the decision tree.
    # depth first enumeration of the tree in order to determine the AND by the path.
    return get_parameters_for_gemm_generic(lefts, rights, features, thresholds, values, weights, biases, n_splits)


def _get_tree_parameters_for_tree_trav(tree_info):
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(
        tree_info.replace("[f", "").replace("[", "").replace("]", "").split(), lefts, rights, features, thresholds, values
    )

    if len(lefts) == 1:
        # XGB model creating tree with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        values = [np.array([0.0]), values[0], values[0]]

    return get_parameters_for_tree_trav_generic(lefts, rights, features, thresholds, values)


def convert_sklearn_xgb_classifier(operator, device, extra_config):
    if "n_features" in extra_config:
        n_features = extra_config["n_features"]
    else:
        raise RuntimeError(
            'XGBoost converter is not able to infer the number of input features.\
             Please pass "n_features:N" as extra configuration to the converter or fill a bug report.'
        )
    tree_infos = operator.raw_operator.get_booster().get_dump()

    n_classes = operator.raw_operator.n_classes_
    tree_infos = [tree_infos[i * n_classes + j] for j in range(n_classes) for i in range(len(tree_infos) // n_classes)]
    if n_classes == 2:
        n_classes -= 1
    classes = [i for i in range(n_classes)]
    max_depth = operator.raw_operator.max_depth  # TODO this should be a call to max_depth() and NOT fall through!
    tree_type = get_gbdt_by_config_or_depth(extra_config, max_depth)

    if tree_type == TreeImpl.gemm:
        net_parameters = [_get_tree_parameters_for_gemm(tree_info, n_features) for tree_info in tree_infos]
        return BatchGBDTClassifier(net_parameters, n_features, classes, device=device)

    net_parameters = [_get_tree_parameters_for_tree_trav(tree_info) for tree_info in tree_infos]
    if tree_type == TreeImpl.tree_trav:
        return BeamGBDTClassifier(net_parameters, n_features, classes, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return BeamPPGBDTClassifier(net_parameters, n_features, classes, device=device)


def convert_sklearn_xgb_regressor(operator, device, extra_config):
    if "n_features" in extra_config:
        n_features = extra_config["n_features"]
    else:
        raise RuntimeError(
            'XGBoost converter is not able to infer the number of input features.\
             Please pass "n_features:N" as extra configuration to the converter or fill a bug report.'
        )
    tree_infos = operator.raw_operator.get_booster().get_dump()

    # TODO: in xgboost 1.0.2 (not yet supported), we will need to handle the None case for max_depth
    max_depth = operator.raw_operator.max_depth
    alpha = operator.raw_operator.base_score
    if type(alpha) is float:
        alpha = [alpha]
    tree_type = get_gbdt_by_config_or_depth(extra_config, max_depth)

    if tree_type == TreeImpl.gemm:
        net_parameters = [_get_tree_parameters_for_gemm(tree_info, n_features) for tree_info in tree_infos]
        return BatchGBDTRegressor(net_parameters, n_features, alpha=alpha, device=device)

    net_parameters = [_get_tree_parameters_for_tree_trav(tree_info) for tree_info in tree_infos]
    if tree_type == TreeImpl.tree_trav:
        return BeamGBDTRegressor(net_parameters, n_features, alpha=alpha, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return BeamPPGBDTRegressor(net_parameters, n_features, alpha=alpha, device=device)


register_converter("SklearnXGBClassifier", convert_sklearn_xgb_classifier)
register_converter("SklearnXGBRegressor", convert_sklearn_xgb_regressor)
