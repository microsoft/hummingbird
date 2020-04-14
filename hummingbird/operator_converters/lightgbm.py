# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from hummingbird.operator_converters.gbdt import BatchGBDTClassifier, BatchGBDTRegressor, BeamPPGBDTClassifier
from hummingbird.operator_converters.gbdt import BeamPPGBDTRegressor, BeamGBDTClassifier, BeamGBDTRegressor

from ._tree_commons import get_gbdt_by_config_or_depth, TreeImpl
from ._tree_commons import get_parameters_for_tree_trav_common, get_parameters_for_gemm_common
from .._registration import register_converter


def _tree_traversal(node, lefts, rights, features, thresholds, values, count):
    """
    Recursive function for parsing a tree and filling the input data structures.
    """
    if "left_child" in node:
        features.append(node["split_feature"])
        thresholds.append(node["threshold"])
        values.append([-1])
        lefts.append(count + 1)
        rights.append(-1)
        pos = len(rights) - 1
        count = _tree_traversal(node["left_child"], lefts, rights, features, thresholds, values, count + 1)
        rights[pos] = count + 1
        return _tree_traversal(node["right_child"], lefts, rights, features, thresholds, values, count + 1)
    else:
        features.append(0)
        thresholds.append(0)
        values.append([node["leaf_value"]])
        lefts.append(-1)
        rights.append(-1)
        return count


def _get_tree_parameters_for_gemm(tree_info, n_features):
    """
    Parse the tree and prepare it according to the GEMM strategy.
    """
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(tree_info["tree_structure"], lefts, rights, features, thresholds, values, 0)

    return get_parameters_for_gemm_common(lefts, rights, features, thresholds, values, n_features)


def _get_tree_parameters_for_tree_trav(tree_info):
    """
    Parse the tree and prepare it according to the tree traversal strategies.
    """
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(tree_info["tree_structure"], lefts, rights, features, thresholds, values, 0)

    return get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values)


def convert_sklearn_lgbm_classifier(operator, device, extra_config):
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()["tree_info"]

    n_classes = operator.raw_operator._n_classes
    tree_infos = [tree_infos[i * n_classes + j] for j in range(n_classes) for i in range(len(tree_infos) // n_classes)]
    if n_classes == 2:
        n_classes -= 1
    classes = [i for i in range(n_classes)]
    max_depth = operator.raw_operator.max_depth  # TODO FIXME this should be a call to max_depth and NOT fall through!
    tree_type = get_gbdt_by_config_or_depth(extra_config, max_depth)

    if tree_type == TreeImpl.gemm:
        net_parameters = [_get_tree_parameters_for_gemm(tree_info, n_features) for tree_info in tree_infos]
        return BatchGBDTClassifier(net_parameters, n_features, classes, device=device)

    net_parameters = [_get_tree_parameters_for_tree_trav(tree_info) for tree_info in tree_infos]
    if tree_type == TreeImpl.tree_trav:
        return BeamGBDTClassifier(net_parameters, n_features, classes, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return BeamPPGBDTClassifier(net_parameters, n_features, classes, device=device)


def convert_sklearn_lgbm_regressor(operator, device, extra_config):
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()["tree_info"]
    max_depth = operator.raw_operator.max_depth  # TODO FIXME this should be a call to max_depth and NOT fall through!
    tree_type = get_gbdt_by_config_or_depth(extra_config, max_depth)

    if tree_type == TreeImpl.gemm:
        net_parameters = [_get_tree_parameters_for_gemm(tree_info, n_features) for tree_info in tree_infos]
        return BatchGBDTRegressor(net_parameters, n_features, device=device)

    net_parameters = [_get_tree_parameters_for_tree_trav(tree_info) for tree_info in tree_infos]
    if tree_type == TreeImpl.tree_trav:
        return BeamGBDTRegressor(net_parameters, n_features, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return BeamPPGBDTRegressor(net_parameters, n_features, device=device)


register_converter("SklearnLGBMClassifier", convert_sklearn_lgbm_classifier)
register_converter("SklearnLGBMRegressor", convert_sklearn_lgbm_regressor)
