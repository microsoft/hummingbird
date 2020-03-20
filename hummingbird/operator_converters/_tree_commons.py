# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import torch
import numpy as np
from enum import Enum


class Node:

    def __init__(self, id):
        self.id = id
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None


def find_depth(node, current_depth):
    if node.left == -1 and node.right == -1:
        return current_depth + 1
    elif node.left != -1 and node.right == -1:
        return find_depth(node.l, current_depth + 1)
    elif node.right != -1 and node.left == -1:
        return find_depth(node.r, current_depth + 1)
    elif node.right != -1 and node.left != -1:
        return max(find_depth(node.left, current_depth + 1),
                   find_depth(node.right, current_depth + 1))


class TreeImpl(Enum):
    batch = 1
    beam = 2
    beampp = 3


# TODO move this to gbdt_gree_commons.py? (create new file)
# TODO: consider reanming this to get_tree_implementation_by_config_or_depth if I
#     can generalize this to the RF tree as well
def get_gbdt_by_config_or_depth(extra_config, max_depth, low=3, high=10):
    if 'tree_implementation' not in extra_config:
        if max_depth is not None and max_depth <= low:
            return TreeImpl.batch
        elif max_depth is not None and max_depth <= high:
            return TreeImpl.beam
        else:
            return TreeImpl.beampp

    if extra_config['tree_implementation'] == 'batch':
        return TreeImpl.batch
    elif extra_config['tree_implementation'] == 'beam':
        return TreeImpl.beam
    elif extra_config['tree_implementation'] == 'beam++':
        return TreeImpl.beampp
    else:
        raise ValueError("Tree implementation {} not found".format(extra_config))


def get_parameters_for_beam(tree):
    tree = copy.deepcopy(tree)

    lefts = tree.tree_.children_left
    rights = tree.tree_.children_right
    features = [max(x, 0) for x in tree.tree_.feature]
    thresholds = tree.tree_.threshold

    values = np.array(tree.tree_.value)
    if len(values.shape) == 3:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] > 1:
        values /= np.sum(values, axis=1, keepdims=True)

    ids = [i for i in range(len(lefts))]
    nodes = list(zip(ids, lefts, rights, features, thresholds, values))

    nodes_map = {0: Node(0)}
    current_node = 0
    for i, node in enumerate(nodes):
        id, left, right, feature, threshold, value = node

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
        nodes_map[current_node].threshold = threshold
        nodes_map[current_node].value = value

        current_node += 1

    depth = find_depth(nodes_map[0], -1)

    return [depth, nodes_map, ids, lefts, rights, features, thresholds, values]


def get_parameters_for_batch(tree):
    lefts = tree.tree_.children_left
    rights = tree.tree_.children_right
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    values = tree.tree_.value
    n_features = tree.tree_.n_features
    n_nodes = len(lefts)

    weights = []
    biases = []

    hidden_weights = []
    hidden_biases = []
    for feature, threshold in zip(features, thresholds):
        if feature >= 0:
            hidden_weights.append(
                [1 if i == feature else 0 for i in range(n_features)])
            hidden_biases.append(threshold)

    if len(hidden_weights) == 0:
        weights.append(np.ones((1, n_features)).astype("float32"))
        weights.append(np.ones((1, 1)).astype("float32"))
        weights.append(np.transpose(
            np.array(values).astype("float32")).reshape(-1, 1))
        biases.append(np.array([0]).astype("float32"))
        biases.append(np.array([0]).astype("float32"))
        biases.append(None)
        return weights, biases

    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))
    n_splits = len(hidden_weights)

    hidden_weights = []
    hidden_biases = []

    path = [0]
    visited = [False for _ in range(n_nodes)]

    class_proba = []
    nodes = list(zip(lefts, rights, features, thresholds, values))

    while True and len(path) > 0:
        i = path[-1]
        visited[i] = True
        left, right, feature, threshold, value = nodes[i]
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
                    raise RuntimeError(
                        "Inconsistent state encountered while tree translation.")

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

    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))

    weights.append(np.transpose(np.array(class_proba).astype("float32")))
    biases.append(None)

    return weights, biases


class BatchedTreeEnsemble(torch.nn.Module):

    def __init__(self, net_parameters, n_features, n_classes):
        super(BatchedTreeEnsemble, self).__init__()
        hidden_one_size = 0
        hidden_two_size = 0
        hidden_three_size = n_classes

        for weight, bias in net_parameters:
            hidden_one_size = max(hidden_one_size, weight[0].shape[0])
            hidden_two_size = max(hidden_two_size, weight[1].shape[0])

        n_trees = len(net_parameters)
        weight_1 = np.zeros((n_trees, hidden_one_size, n_features))
        bias_1 = np.zeros((n_trees, hidden_one_size))
        weight_2 = np.zeros((n_trees, hidden_two_size, hidden_one_size))
        bias_2 = np.zeros((n_trees, hidden_two_size))
        weight_3 = np.zeros((n_trees, hidden_three_size, hidden_two_size))

        for i, (weight, bias) in enumerate(net_parameters):
            if (len(weight[0]) > 0):
                weight_1[i, 0:weight[0].shape[0],
                         0:weight[0].shape[1]] = weight[0]
                bias_1[i, 0:bias[0].shape[0]] = bias[0]
                weight_2[i, 0:weight[1].shape[0],
                         0:weight[1].shape[1]] = weight[1]
                bias_2[i, 0:bias[1].shape[0]] = bias[1]
                weight_3[i, 0:weight[2].shape[0],
                         0:weight[2].shape[1]] = weight[2]

        self.n_trees = n_trees
        self.n_features = n_features
        self.hidden_one_size = hidden_one_size
        self.hidden_two_size = hidden_two_size
        self.hidden_three_size = hidden_three_size

        self.weight_1 = torch.nn.Parameter(torch.from_numpy(
            weight_1.reshape(-1, self.n_features).astype('float32')))
        self.bias_1 = torch.nn.Parameter(torch.from_numpy(
            bias_1.reshape(-1, 1).astype('float32')))

        self.weight_2 = torch.nn.Parameter(
            torch.from_numpy(weight_2.astype('float32')))
        self.bias_2 = torch.nn.Parameter(torch.from_numpy(
            bias_2.reshape(-1, 1).astype('float32')))

        self.weight_3 = torch.nn.Parameter(
            torch.from_numpy(weight_3.astype('float32')))

    def forward(self, x):

        x = x.t()
        x = torch.mm(self.weight_1, x) < self.bias_1
        x = x.view(self.n_trees, self.hidden_one_size, -1)
        x = x.float()

        x = torch.matmul(self.weight_2, x)

        x = x.view(self.n_trees * self.hidden_two_size, -1) == self.bias_2
        x = x.view(self.n_trees, self.hidden_two_size, -1)
        x = x.float()

        x = torch.matmul(self.weight_3, x)
        x = x.view(self.n_trees, self.hidden_three_size, -1)
        return x


class BeamTreeEnsemble(torch.nn.Module):

    def __init__(self, tree_parameters, n_features, n_classes):
        super(BeamTreeEnsemble, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        tree_depths = [tree_parameter[0] for tree_parameter in tree_parameters]
        self.max_tree_depth = max(tree_depths)
        self.num_trees = len(tree_depths)
        self.num_nodes = max([len(tree_parameter[2])
                              for tree_parameter in tree_parameters])

        lefts = np.zeros((self.num_trees, self.num_nodes), dtype=np.float32)
        rights = np.zeros((self.num_trees, self.num_nodes), dtype=np.float32)

        features = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
        thresholds = np.zeros(
            (self.num_trees, self.num_nodes), dtype=np.float32)
        values = np.zeros((self.num_trees, self.num_nodes,
                           self.n_classes), dtype=np.float32)

        for i in range(self.num_trees):
            lefts[i][:len(tree_parameters[i][1])] = tree_parameters[i][3]
            rights[i][:len(tree_parameters[i][1])] = tree_parameters[i][4]
            features[i][:len(tree_parameters[i][1])] = tree_parameters[i][5]
            thresholds[i][:len(tree_parameters[i][1])] = tree_parameters[i][6]
            values[i][:len(tree_parameters[i][1])][:] = tree_parameters[i][7]

        self.lefts = torch.nn.Parameter(
            torch.from_numpy(lefts).view(-1), requires_grad=False)
        self.rights = torch.nn.Parameter(
            torch.from_numpy(rights).view(-1), requires_grad=False)

        self.features = torch.nn.Parameter(
            torch.from_numpy(features).view(-1), requires_grad=False)
        self.thresholds = torch.nn.Parameter(
            torch.from_numpy(thresholds).view(-1))
        self.values = torch.nn.Parameter(
            torch.from_numpy(values).view(-1, self.n_classes))

        nodes_offset = [[i * self.num_nodes for i in range(self.num_trees)]]
        self.nodes_offset = torch.nn.Parameter(
            torch.LongTensor(nodes_offset), requires_grad=False)

    def forward(self, x):
        indexes = self.nodes_offset
        indexes = indexes.expand(x.size()[0], self.num_trees)
        indexes = indexes.reshape(-1)

        for _ in range(self.max_tree_depth):
            tree_nodes = indexes
            feature_nodes = torch.index_select(
                self.features, 0, tree_nodes).view(-1, self.num_trees)
            feature_values = torch.gather(x, 1, feature_nodes)

            thresholds = torch.index_select(
                self.thresholds, 0, indexes).view(-1, self.num_trees)
            lefts = torch.index_select(
                self.lefts, 0, indexes).view(-1, self.num_trees)
            rights = torch.index_select(
                self.rights, 0, indexes).view(-1, self.num_trees)

            indexes = torch.where(
                torch.ge(feature_values, thresholds), rights, lefts).long()
            indexes = indexes + self.nodes_offset
            indexes = indexes.view(-1)

        output = torch.index_select(
            self.values, 0, indexes).view(-1, self.num_trees, self.n_classes)
        return output


class BeamPPTreeEnsemble(torch.nn.Module):

    def __init__(self, tree_parameters, n_features, n_classes):
        super(BeamPPTreeEnsemble, self).__init__()

        tree_depths = [tree_parameter[0] for tree_parameter in tree_parameters]
        max_tree_depth = max(tree_depths)
        self.max_tree_depth = max_tree_depth
        self.num_trees = len(tree_depths)
        self.n_features = n_features
        self.n_classes = n_classes

        node_maps = [tp[1] for tp in tree_parameters]

        weight_0 = np.zeros((self.num_trees, 2 ** max_tree_depth - 1))
        bias_0 = np.zeros((self.num_trees, 2 ** max_tree_depth - 1))
        weight_1 = np.zeros((self.num_trees, 2 ** max_tree_depth, n_classes))

        for i, node_map in enumerate(node_maps):
            self._get_weights_and_biases(
                node_map, max_tree_depth, weight_0[i], weight_1[i], bias_0[i])

        node_by_levels = [set() for _ in range(max_tree_depth)]
        self._traverse_by_level(node_by_levels, 0, -1, max_tree_depth)

        self.root_nodes = torch.nn.Parameter(torch.from_numpy(weight_0[:, 0].flatten().astype('int64')),
                                             requires_grad=False)
        self.root_biases = torch.nn.Parameter(-1 * torch.from_numpy(
            bias_0[:, 0].astype('float32')), requires_grad=False)

        tree_indices = np.array(
            [i for i in range(0, 2 * self.num_trees, 2)]).astype('int64')
        self.tree_indices = torch.nn.Parameter(
            torch.from_numpy(tree_indices), requires_grad=False)

        self.nodes = []
        self.biases = []
        for i in range(1, max_tree_depth):
            nodes = torch.nn.Parameter(torch.from_numpy(weight_0[:, list(sorted(node_by_levels[i]))]
                                                        .flatten().astype('int64')), requires_grad=False)
            biases = torch.nn.Parameter(torch.from_numpy(-1 * bias_0[:, list(sorted(node_by_levels[i]))]
                                                         .flatten().astype('float32')), requires_grad=False)
            self.nodes.append(nodes)
            self.biases.append(biases)

        self.nodes = torch.nn.ParameterList(self.nodes)
        self.biases = torch.nn.ParameterList(self.biases)

        self.leaf_nodes = torch.nn.Parameter(torch.from_numpy(weight_1.reshape((-1, n_classes)).astype('float32')),
                                             requires_grad=False)

    def forward(self, x):
        prev_indices = (torch.ge(torch.index_select(
            x, 1, self.root_nodes), self.root_biases)).long()
        prev_indices = prev_indices + self.tree_indices
        prev_indices = prev_indices.view(-1)

        factor = 2
        for nodes, biases in zip(self.nodes, self.biases):
            gather_indices = torch.index_select(
                nodes, 0, prev_indices).view(-1, self.num_trees)
            features = torch.gather(x, 1, gather_indices).view(-1)
            prev_indices = factor * prev_indices + torch.ge(
                features, torch.index_select(biases, 0, prev_indices)).long().view(-1)

        output = torch.index_select(
            self.leaf_nodes, 0, prev_indices.view(-1)).view(-1, self.num_trees, self.n_classes)
        return output

    def _traverse_by_level(self, node_by_levels, node_id,
                           current_level, max_level):
        current_level += 1
        if current_level == max_level:
            return node_id
        node_by_levels[current_level].add(node_id)
        node_id += 1
        node_id = self._traverse_by_level(
            node_by_levels, node_id, current_level, max_level)
        node_id = self._traverse_by_level(
            node_by_levels, node_id, current_level, max_level)
        return node_id

    def _get_weights_and_biases(
            self, nodes_map, tree_depth, weight_0, weight_1, bias_0):

        def depth_f_traversal(node, current_depth, node_id, leaf_start_id):
            weight_0[node_id] = node.feature
            bias_0[node_id] = -node.threshold
            current_depth += 1
            node_id += 1

            if node.left.feature == -1:
                node_id += 2 ** (tree_depth - current_depth - 1) - 1
                v = node.left.value
                weight_1[leaf_start_id:leaf_start_id + 2 ** (tree_depth - current_depth - 1)] = np.ones((
                    2 ** (tree_depth - current_depth - 1), self.n_classes)) * v
                leaf_start_id += 2 ** (tree_depth - current_depth - 1)
            else:
                node_id, leaf_start_id = depth_f_traversal(
                    node.left, current_depth, node_id, leaf_start_id)

            if node.right.feature == -1:
                node_id += 2 ** (tree_depth - current_depth - 1) - 1
                v = node.right.value
                weight_1[leaf_start_id:leaf_start_id + 2 ** (tree_depth - current_depth - 1)] = np.ones((
                    2 ** (tree_depth - current_depth - 1), self.n_classes)) * v
                leaf_start_id += 2 ** (tree_depth - current_depth - 1)
            else:
                node_id, leaf_start_id = depth_f_traversal(
                    node.right, current_depth, node_id, leaf_start_id)

            return node_id, leaf_start_id

        if(nodes_map[0].feature == -1):
            # Model creating tree with just a single leaf node. We transform it
            # to a model with one internal node.
            nodes_map[0].feature = 0
            left = Node(1)
            right = Node(2)
            nodes_map[0].left = left
            nodes_map[0].right = right

            left.left = -1
            left.right = -1
            left.feature = -1
            left.threshold = -1
            left.value = nodes_map[0].value
            nodes_map[1] = left

            right.left = -1
            right.right = -1
            right.feature = -1
            right.threshold = -1
            right.value = nodes_map[0].value
            nodes_map[2] = right

        depth_f_traversal(nodes_map[0], -1, 0, 0)
