# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from hummingbird.operator_converters.gbdt import BatchGBDTClassifier, BatchGBDTRegressor, BeamPPGBDTClassifier
from hummingbird.operator_converters.gbdt import BeamPPGBDTRegressor, BeamGBDTClassifier, BeamGBDTRegressor

from ._tree_commons import get_gbdt_by_config_or_depth, TreeImpl
from ._tree_commons import get_parameters_for_beam_generic, get_parameters_for_batch_generic
from ..common._registration import register_converter


def _tree_traversal(node, ls, rs, fs, ts, vs, count):
    if 'left_child' in node:
        fs.append(node['split_feature'])
        ts.append(node['threshold'])
        vs.append([-1])
        ls.append(count + 1)
        rs.append(-1)
        pos = len(rs) - 1
        count = _tree_traversal(
            node['left_child'], ls, rs, fs, ts, vs, count + 1)
        rs[pos] = count + 1
        return _tree_traversal(node['right_child'], ls, rs, fs, ts, vs, count + 1)
    else:
        fs.append(0)
        ts.append(0)
        vs.append([node['leaf_value']])
        ls.append(-1)
        rs.append(-1)
        return count


def _get_tree_parameters_for_batch(tree_info, n_features):
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(tree_info['tree_structure'], lefts, rights, features, thresholds, values, 0)
    weights = []
    biases = []

    values = np.array(values)

    # first hidden layer has all inequalities
    hidden_weights = []
    hidden_biases = []
    for left, feature, thresh in zip(lefts, features, thresholds):
        if left != -1:
            hidden_weights.append(
                [1 if i == feature else 0 for i in range(n_features)])
            hidden_biases.append(thresh)

    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))
    n_splits = len(hidden_weights)

    # second hidden layer has ANDs for each leaf of the decision tree.
    # depth first enumeration of the tree in order to determine the AND by the path.
    return get_parameters_for_batch_generic(lefts, rights, features, thresholds, values, weights, biases, n_splits)


def _get_tree_parameters_for_beam(tree_info):
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(tree_info['tree_structure'], lefts,
                    rights, features, thresholds, values, 0)

    return get_parameters_for_beam_generic(lefts, rights, features, thresholds, values)


def convert_sklearn_lgbm_classifier(operator, device, extra_config):
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()['tree_info']

    n_classes = operator.raw_operator._n_classes
    tree_infos = [tree_infos[i * n_classes + j]
                  for j in range(n_classes) for i in range(len(tree_infos)//n_classes)]
    if n_classes == 2:
        n_classes -= 1
    classes = [i for i in range(n_classes)]
    max_depth = operator.raw_operator.max_depth  # TODO FIXME this should be a call to max_depth and NOT fall through!
    tree_type = get_gbdt_by_config_or_depth(extra_config, max_depth)

    if tree_type == TreeImpl.batch:
        net_parameters = [_get_tree_parameters_for_batch(tree_info, n_features) for tree_info in tree_infos]
        return BatchGBDTClassifier(net_parameters, n_features, classes, device=device)

    net_parameters = [_get_tree_parameters_for_beam(tree_info) for tree_info in tree_infos]
    if tree_type == TreeImpl.beam:
        return BeamGBDTClassifier(net_parameters, n_features, classes, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.beampp
        return BeamPPGBDTClassifier(net_parameters, n_features, classes, device=device)


def convert_sklearn_lgbm_regressor(operator, device, extra_config):
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()['tree_info']
    max_depth = operator.raw_operator.max_depth  # TODO FIXME this should be a call to max_depth and NOT fall through!
    tree_type = get_gbdt_by_config_or_depth(extra_config, max_depth)

    if tree_type == TreeImpl.batch:
        net_parameters = [_get_tree_parameters_for_batch(tree_info, n_features) for tree_info in tree_infos]
        return BatchGBDTRegressor(net_parameters, n_features, device=device)

    net_parameters = [_get_tree_parameters_for_beam(tree_info) for tree_info in tree_infos]
    if tree_type == TreeImpl.beam:
        return BeamGBDTRegressor(net_parameters, n_features, device=device)
    else:  # Remaining possible case: tree_type == TreeImpl.beampp
        return BeamPPGBDTRegressor(net_parameters, n_features, device=device)


register_converter('SklearnLGBMClassifier', convert_sklearn_lgbm_classifier)
register_converter('SklearnLGBMRegressor', convert_sklearn_lgbm_regressor)
