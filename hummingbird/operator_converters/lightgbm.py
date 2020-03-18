# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from hummingbird.operator_converters.gbdt import BatchGBDTClassifier, BatchGBDTRegressor, BeamPPGBDTClassifier
from hummingbird.operator_converters.gbdt import BeamPPGBDTRegressor, BeamGBDTClassifier, BeamGBDTRegressor

from ._tree_commons import Node, find_depth
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


# TODO: redundant code with tree_commons
def get_tree_parameters_for_batch(tree_info, n_features):
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(tree_info['tree_structure'], lefts, rights, features, thresholds, values, 0)
    n_nodes = len(lefts)
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
    hidden_weights = []
    hidden_biases = []

    path = [0]
    visited = [False for _ in range(n_nodes)]
    # save classes for later ORing
    class_proba = []
    nodes = list(zip(lefts, rights, features, thresholds, values))

    while True:
        i = path[-1]
        visited[i] = True
        left, right, feature, thresh, value = nodes[i]
        if left == -1 and right == -1:
            vec = [0 for _ in range(n_splits)]
            # keep track of positive weights for calculating bias.
            num_positive = 0
            for j, p in enumerate(path[:-1]):
                num_leaves_before_p = list(lefts[:p]).count(-1)
                if path[j + 1] in lefts:
                    vec[p - num_leaves_before_p] = 1
                    num_positive += 1
                elif path[j + 1] in rights:
                    vec[p - num_leaves_before_p] = -1
                else:
                    raise Exception(
                        "Warning: Inconsistent tree translation encountered")
            if values.shape[-1] > 1:
                class_proba.append((values[i] / np.sum(values[i])).flatten())
            else:
                # we have only a single value. e.g., GBDT
                class_proba.append(values[i].flatten())

            hidden_weights.append(vec)
            hidden_biases.append(num_positive)
            path.pop()
        elif not visited[left]:
            path.append(left)
        elif not visited[right]:
            path.append(right)
        else:
            path.pop()
            if len(path) == 0:
                break
    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))

    # OR neurons from the preceding layer in order to get final classes.
    weights.append(np.transpose(np.array(class_proba).astype("float32")))
    biases.append(None)
    return weights, biases


# TODO: redundant code with tree_commons
def get_tree_parameters_for_beam(tree_info):
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []
    _tree_traversal(tree_info['tree_structure'], lefts,
                    rights, features, thresholds, values, 0)

    ids = [i for i in range(len(lefts))]
    nodes = list(zip(ids, lefts, rights, features, thresholds, values))

    nodes_map = {0: Node(0)}
    current_node = 0

    for i, node in enumerate(nodes):
        id, left, right, feature, thresh, value = node

        if left != -1:
            l_node = Node(left)
            nodes_map[left] = l_node
        else:
            lefts[i] = id
            l_node = -1
            feature = -1

        if right != -1:
            r_node = Node(right)
            nodes_map[right] = r_node
        else:
            rights[i] = id
            r_node = -1
            feature = -1

        nodes_map[current_node].left = l_node
        nodes_map[current_node].right = r_node
        nodes_map[current_node].feature = feature
        nodes_map[current_node].threshold = thresh
        nodes_map[current_node].value = value

        current_node += 1

    depth = find_depth(nodes_map[0], -1)
    lefts = np.array(lefts)
    rights = np.array(rights)
    features = np.array(features)
    thresholds = np.array(thresholds)
    values = np.array(values)
    return [depth, nodes_map, ids, lefts, rights, features, thresholds, values]


def convert_sklearn_lgbm_classifier(operator, device, extra_config):
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()['tree_info']

    n_classes = operator.raw_operator._n_classes
    tree_infos = [tree_infos[i * n_classes + j]
                  for j in range(n_classes) for i in range(len(tree_infos)//n_classes)]
    if n_classes == 2:
        n_classes -= 1
    classes = [i for i in range(n_classes)]

    if "tree_implementation" not in extra_config:  # use heurstics to get the tree implementation
        if operator.raw_operator.max_depth <= 3:
            net_parameters = [get_tree_parameters_for_batch(tree_info, n_features) for tree_info in
                              tree_infos]
            return BatchGBDTClassifier(net_parameters, n_features, classes, device=device)
        elif operator.raw_operator.max_depth <= 10:
            net_parameters = [get_tree_parameters_for_beam(
                tree_info) for tree_info in tree_infos]
            return BeamPPGBDTClassifier(net_parameters, n_features, classes, device=device)
        else:
            net_parameters = [get_tree_parameters_for_beam(tree_info) for tree_info in
                              tree_infos]
            return BeamGBDTClassifier(net_parameters, n_features, classes, device=device)
    else:  # manually set tree implementation
        if 'tree_implementation' in extra_config and extra_config['tree_implementation'] == 'batch':
            net_parameters = [get_tree_parameters_for_batch(tree_info, n_features) for tree_info in
                              tree_infos]
            return BatchGBDTClassifier(net_parameters, n_features, classes, device=device)
        elif 'tree_implementation' in extra_config and extra_config['tree_implementation'] == 'beam':
            net_parameters = [get_tree_parameters_for_beam(tree_info) for tree_info in
                              tree_infos]
            return BeamGBDTClassifier(net_parameters, n_features, classes, device=device)
        elif 'tree_implementation' in extra_config and extra_config['tree_implementation'] == 'beam++':
            net_parameters = [get_tree_parameters_for_beam(tree_info) for tree_info in
                              tree_infos]
            return BeamGBDTClassifier(net_parameters, n_features, classes, device=device)
        else:
            raise Exception("Tree implementation {} not found".format(
                extra_config['tree_implementation']))


def convert_sklearn_lgbm_regressor(operator, device, extra_config):
    n_features = operator.raw_operator._n_features
    tree_infos = operator.raw_operator.booster_.dump_model()['tree_info']

    if operator.raw_operator.max_depth <= 3:
        net_parameters = [get_tree_parameters_for_batch(tree_info, n_features) for tree_info in
                          tree_infos]
        return BatchGBDTRegressor(net_parameters, n_features, device=device)
    elif operator.raw_operator.max_depth <= 10:
        net_parameters = [get_tree_parameters_for_beam(
            tree_info) for tree_info in tree_infos]
        return BeamPPGBDTRegressor(net_parameters, n_features, device=device)
    else:
        net_parameters = [get_tree_parameters_for_beam(tree_info) for tree_info in
                          tree_infos]
        return BeamGBDTRegressor(net_parameters, n_features, device=device)


register_converter('SklearnLGBMClassifier', convert_sklearn_lgbm_classifier)
register_converter('SklearnLGBMRegressor', convert_sklearn_lgbm_regressor)
