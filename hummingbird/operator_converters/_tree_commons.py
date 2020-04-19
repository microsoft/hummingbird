# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import torch
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


class Node:
    """
    Class defininig a tree node.
    """

    def __init__(self, id=None):
        """
        :param id: A unique ID for the node
        :param left: The id of the left node
        :param right: The id of the right node
        :param feature: The feature used to make a decision (if not leaf node, ignored otherwise)
        :param threshold: The threshold used in the decision (if not leaf node, ignored otherwise)
        :param value: The value stored in the leaf (ignored if not leaf node).
        """
        self.id = id
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None


class TreeParameters:
    """
    Class containing a convenient in-memory representation of a decision tree.
    """

    def __init__(self, lefts, rights, features, thresholds, values):
        """
        :param lefts: The id of the left nodes
        :param rights: The id of the right nodes
        :param feature: The features used to make decisions
        :param thresholds: The thresholds used in the decisions
        :param values: The value stored in the leaves
        """
        self.lefts = lefts
        self.rights = rights
        self.features = features
        self.thresholds = thresholds
        self.values = values


class TreeImpl(Enum):
    """
    Enum defininig the available implementations for tree scoring.
    """

    gemm = 1
    tree_trav = 2
    perf_tree_trav = 3


class AbstractTreeEnsemble(ABC):
    """
    Abstract class defininig the basic structure for tree ensemble models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def aggregation(self, x):
        """
        Method defining the aggregation operation to execute after the model is evaluated.
        """
        pass

    @abstractmethod
    def calibation(self, x):
        """
        Method implementating the calibration operation for classifiers.
        """
        pass


class AbstractPyTorchTreeEnsemble(AbstractTreeEnsemble, torch.nn.Module):
    """
    Abstract class defininig the basic structure for tree ensemble models implemented in PyTorch.
    """

    def __init__(self, net_parameters, n_features, classes, n_classes):
        super(AbstractPyTorchTreeEnsemble, self).__init__()

        # Set up the variables for the subclasses.
        # Each subclass will trigger different behaviours by properly setting these.
        self.perform_class_select = False
        self.binary_classification = False
        self.classes = classes
        self.learning_rate = None
        self.regression = False
        self.alpha = None

        # Are we doing regression or classification?
        if classes is None:
            self.regression = True
            self.n_classes = 1
        else:
            self.n_classes = len(classes) if n_classes is None else n_classes
            if min(classes) != 0 or max(classes) != len(classes) - 1:
                self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
                self.perform_class_select = True


class GEMMTreeEnsemble(AbstractPyTorchTreeEnsemble):
    """
    Class implementing the GEMM strategy in PyTorch for ensemble tree models.
    """

    def __init__(self, net_parameters, n_features, classes, n_classes=None):
        super(GEMMTreeEnsemble, self).__init__(net_parameters, n_features, classes, n_classes)

        # Initialize the actual model.
        hidden_one_size = 0
        hidden_two_size = 0
        hidden_three_size = self.n_classes

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
            if len(weight[0]) > 0:
                weight_1[i, 0 : weight[0].shape[0], 0 : weight[0].shape[1]] = weight[0]
                bias_1[i, 0 : bias[0].shape[0]] = bias[0]
                weight_2[i, 0 : weight[1].shape[0], 0 : weight[1].shape[1]] = weight[1]
                bias_2[i, 0 : bias[1].shape[0]] = bias[1]
                weight_3[i, 0 : weight[2].shape[0], 0 : weight[2].shape[1]] = weight[2]

        self.n_trees = n_trees
        self.n_features = n_features
        self.hidden_one_size = hidden_one_size
        self.hidden_two_size = hidden_two_size
        self.hidden_three_size = hidden_three_size

        self.weight_1 = torch.nn.Parameter(torch.from_numpy(weight_1.reshape(-1, self.n_features).astype("float32")))
        self.bias_1 = torch.nn.Parameter(torch.from_numpy(bias_1.reshape(-1, 1).astype("float32")))

        self.weight_2 = torch.nn.Parameter(torch.from_numpy(weight_2.astype("float32")))
        self.bias_2 = torch.nn.Parameter(torch.from_numpy(bias_2.reshape(-1, 1).astype("float32")))

        self.weight_3 = torch.nn.Parameter(torch.from_numpy(weight_3.astype("float32")))

    def aggregation(self, x):
        return x

    def calibation(self, x):
        return x

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

        x = self.aggregation(x)

        if self.learning_rate is not None:
            x = x * self.learning_rate
        if self.alpha is not None:
            x += self.alpha
        if self.regression:
            return x

        x = self.calibation(x)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(x, dim=1)), x
        else:
            return torch.argmax(x, dim=1), x


class TreeTraversalTreeEnsemble(AbstractPyTorchTreeEnsemble):
    """
    Class implementing the Tree Traversal strategy in PyTorch for ensemble tree models.
    """

    def __init__(self, tree_parameters, max_depth, n_features, classes, n_classes=None):
        super(TreeTraversalTreeEnsemble, self).__init__(tree_parameters, n_features, classes, n_classes)

        # Initialize the actual model.
        self.n_features = n_features
        self.max_tree_depth = max_depth
        self.num_trees = len(tree_parameters)
        self.num_nodes = max([len(tree_parameter[1]) for tree_parameter in tree_parameters])

        lefts = np.zeros((self.num_trees, self.num_nodes), dtype=np.float32)
        rights = np.zeros((self.num_trees, self.num_nodes), dtype=np.float32)

        features = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
        thresholds = np.zeros((self.num_trees, self.num_nodes), dtype=np.float32)
        values = np.zeros((self.num_trees, self.num_nodes, self.n_classes), dtype=np.float32)

        for i in range(self.num_trees):
            lefts[i][: len(tree_parameters[i][0])] = tree_parameters[i][2]
            rights[i][: len(tree_parameters[i][0])] = tree_parameters[i][3]
            features[i][: len(tree_parameters[i][0])] = tree_parameters[i][4]
            thresholds[i][: len(tree_parameters[i][0])] = tree_parameters[i][5]
            values[i][: len(tree_parameters[i][0])][:] = tree_parameters[i][6]

        self.lefts = torch.nn.Parameter(torch.from_numpy(lefts).view(-1), requires_grad=False)
        self.rights = torch.nn.Parameter(torch.from_numpy(rights).view(-1), requires_grad=False)

        self.features = torch.nn.Parameter(torch.from_numpy(features).view(-1), requires_grad=False)
        self.thresholds = torch.nn.Parameter(torch.from_numpy(thresholds).view(-1))
        self.values = torch.nn.Parameter(torch.from_numpy(values).view(-1, self.n_classes))

        nodes_offset = [[i * self.num_nodes for i in range(self.num_trees)]]
        self.nodes_offset = torch.nn.Parameter(torch.LongTensor(nodes_offset), requires_grad=False)

    def aggregation(self, x):
        return x

    def calibation(self, x):
        return x

    def forward(self, x):
        indexes = self.nodes_offset
        indexes = indexes.expand(x.size()[0], self.num_trees)
        indexes = indexes.reshape(-1)

        for _ in range(self.max_tree_depth):
            tree_nodes = indexes
            feature_nodes = torch.index_select(self.features, 0, tree_nodes).view(-1, self.num_trees)
            feature_values = torch.gather(x, 1, feature_nodes)

            thresholds = torch.index_select(self.thresholds, 0, indexes).view(-1, self.num_trees)
            lefts = torch.index_select(self.lefts, 0, indexes).view(-1, self.num_trees)
            rights = torch.index_select(self.rights, 0, indexes).view(-1, self.num_trees)

            indexes = torch.where(torch.ge(feature_values, thresholds), rights, lefts).long()
            indexes = indexes + self.nodes_offset
            indexes = indexes.view(-1)

        output = torch.index_select(self.values, 0, indexes).view(-1, self.num_trees, self.n_classes)

        output = self.aggregation(output)

        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha
        if self.regression:
            return output

        output = self.calibation(output)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class PerfectTreeTraversalTreeEnsemble(AbstractPyTorchTreeEnsemble):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch for ensemble tree models.
    """

    def __init__(self, tree_parameters, max_depth, n_features, classes, n_classes=None):
        super(PerfectTreeTraversalTreeEnsemble, self).__init__(tree_parameters, n_features, classes, n_classes)

        # Initialize the actual model.
        self.max_tree_depth = max_depth
        self.num_trees = len(tree_parameters)
        self.n_features = n_features

        node_maps = [tp[0] for tp in tree_parameters]

        weight_0 = np.zeros((self.num_trees, 2 ** max_depth - 1))
        bias_0 = np.zeros((self.num_trees, 2 ** max_depth - 1))
        weight_1 = np.zeros((self.num_trees, 2 ** max_depth, self.n_classes))

        for i, node_map in enumerate(node_maps):
            self._get_weights_and_biases(node_map, max_depth, weight_0[i], weight_1[i], bias_0[i])

        node_by_levels = [set() for _ in range(max_depth)]
        self._traverse_by_level(node_by_levels, 0, -1, max_depth)

        self.root_nodes = torch.nn.Parameter(torch.from_numpy(weight_0[:, 0].flatten().astype("int64")), requires_grad=False)
        self.root_biases = torch.nn.Parameter(-1 * torch.from_numpy(bias_0[:, 0].astype("float32")), requires_grad=False)

        tree_indices = np.array([i for i in range(0, 2 * self.num_trees, 2)]).astype("int64")
        self.tree_indices = torch.nn.Parameter(torch.from_numpy(tree_indices), requires_grad=False)

        self.nodes = []
        self.biases = []
        for i in range(1, max_depth):
            nodes = torch.nn.Parameter(
                torch.from_numpy(weight_0[:, list(sorted(node_by_levels[i]))].flatten().astype("int64")), requires_grad=False
            )
            biases = torch.nn.Parameter(
                torch.from_numpy(-1 * bias_0[:, list(sorted(node_by_levels[i]))].flatten().astype("float32")),
                requires_grad=False,
            )
            self.nodes.append(nodes)
            self.biases.append(biases)

        self.nodes = torch.nn.ParameterList(self.nodes)
        self.biases = torch.nn.ParameterList(self.biases)

        self.leaf_nodes = torch.nn.Parameter(
            torch.from_numpy(weight_1.reshape((-1, self.n_classes)).astype("float32")), requires_grad=False
        )

    def aggregation(self, x):
        return x

    def calibation(self, x):
        return x

    def forward(self, x):
        prev_indices = (torch.ge(torch.index_select(x, 1, self.root_nodes), self.root_biases)).long()
        prev_indices = prev_indices + self.tree_indices
        prev_indices = prev_indices.view(-1)

        factor = 2
        for nodes, biases in zip(self.nodes, self.biases):
            gather_indices = torch.index_select(nodes, 0, prev_indices).view(-1, self.num_trees)
            features = torch.gather(x, 1, gather_indices).view(-1)
            prev_indices = factor * prev_indices + torch.ge(features, torch.index_select(biases, 0, prev_indices)).long().view(
                -1
            )

        output = torch.index_select(self.leaf_nodes, 0, prev_indices.view(-1)).view(-1, self.num_trees, self.n_classes)

        output = self.aggregation(output)

        if self.learning_rate is not None:
            output = output * self.learning_rate
        if self.alpha is not None:
            output += self.alpha
        if self.regression:
            return output

        output = self.calibation(output)

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output

    def _traverse_by_level(self, node_by_levels, node_id, current_level, max_level):
        current_level += 1
        if current_level == max_level:
            return node_id
        node_by_levels[current_level].add(node_id)
        node_id += 1
        node_id = self._traverse_by_level(node_by_levels, node_id, current_level, max_level)
        node_id = self._traverse_by_level(node_by_levels, node_id, current_level, max_level)
        return node_id

    def _get_weights_and_biases(self, nodes_map, tree_depth, weight_0, weight_1, bias_0):
        def depth_f_traversal(node, current_depth, node_id, leaf_start_id):
            weight_0[node_id] = node.feature
            bias_0[node_id] = -node.threshold
            current_depth += 1
            node_id += 1

            if node.left.feature == -1:
                node_id += 2 ** (tree_depth - current_depth - 1) - 1
                v = node.left.value
                weight_1[leaf_start_id : leaf_start_id + 2 ** (tree_depth - current_depth - 1)] = (
                    np.ones((2 ** (tree_depth - current_depth - 1), self.n_classes)) * v
                )
                leaf_start_id += 2 ** (tree_depth - current_depth - 1)
            else:
                node_id, leaf_start_id = depth_f_traversal(node.left, current_depth, node_id, leaf_start_id)

            if node.right.feature == -1:
                node_id += 2 ** (tree_depth - current_depth - 1) - 1
                v = node.right.value
                weight_1[leaf_start_id : leaf_start_id + 2 ** (tree_depth - current_depth - 1)] = (
                    np.ones((2 ** (tree_depth - current_depth - 1), self.n_classes)) * v
                )
                leaf_start_id += 2 ** (tree_depth - current_depth - 1)
            else:
                node_id, leaf_start_id = depth_f_traversal(node.right, current_depth, node_id, leaf_start_id)

            return node_id, leaf_start_id

        if nodes_map[0].feature == -1:
            # Model creating tree with just a single leaf node. We transform it to a model with one internal node.
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


def _find_max_depth(tree_parameters):
    """
    Function traversing all trees in sequence and returning the maximum depth.
    """
    depth = 0

    for tree in tree_parameters:
        tree = copy.deepcopy(tree)

        lefts = tree.lefts
        rights = tree.rights

        ids = [i for i in range(len(lefts))]
        nodes = list(zip(ids, lefts, rights))

        nodes_map = {0: Node(0)}
        current_node = 0
        for i, node in enumerate(nodes):
            id, left, right = node

            if left != -1:
                l_node = Node(left)
                nodes_map[left] = l_node
            else:
                lefts[i] = id
                l_node = -1

            if right != -1:
                r_node = Node(right)
                nodes_map[right] = r_node
            else:
                rights[i] = id
                r_node = -1

            nodes_map[current_node].left = l_node
            nodes_map[current_node].right = r_node

            current_node += 1

        depth = max(depth, _find_depth(nodes_map[0], -1))

    return depth


def _find_depth(node, current_depth):
    """
    Recursive function traversing a tree and returning the maximum depth.
    """
    if node.left == -1 and node.right == -1:
        return current_depth + 1
    elif node.left != -1 and node.right == -1:
        return _find_depth(node.l, current_depth + 1)
    elif node.right != -1 and node.left == -1:
        return _find_depth(node.r, current_depth + 1)
    elif node.right != -1 and node.left != -1:
        return max(_find_depth(node.left, current_depth + 1), _find_depth(node.right, current_depth + 1))


def get_tree_implementation_by_config_or_depth(extra_config, max_depth, low=3, high=10):
    """
    Utility function used to pick the tree implementation based on input parameters and heurstics.
    The current heuristic is such that GEMM <= low < PerfTreeTrav <= high < TreeTrav
    :param max_depth: The maximum tree-depth found in the tree model.
    :param low: the maximum depth below which GEMM strategy is used
    :param high: the maximum depth for which PerfTreeTrav strategy is used
    """
    if "tree_implementation" not in extra_config:
        if max_depth is not None and max_depth <= low:
            return TreeImpl.gemm
        elif max_depth is not None and max_depth <= high:
            return TreeImpl.tree_trav
        else:
            return TreeImpl.perf_tree_trav

    if extra_config["tree_implementation"] == "gemm":
        return TreeImpl.gemm
    elif extra_config["tree_implementation"] == "tree_trav":
        return TreeImpl.tree_trav
    elif extra_config["tree_implementation"] == "perf_tree_trav":
        return TreeImpl.perf_tree_trav
    else:
        raise ValueError("Tree implementation {} not found".format(extra_config))


def get_tree_params_and_type(tree_infos, get_tree_parameters, extra_config):
    """
    Populate the parameters from the trees and pick the tree implementation strategy.
    """
    tree_parameters = [get_tree_parameters(tree_info) for tree_info in tree_infos]
    max_depth = max(1, _find_max_depth(tree_parameters))
    tree_type = get_tree_implementation_by_config_or_depth(extra_config, max_depth)

    return tree_parameters, max_depth, tree_type


def get_parameters_sklearn_common(tree_info):
    """
    Parse sklearn-based trees, including
    SklearnRandomForestClassifier/Regressor and SklearnGradientBoostingClassifier
    """
    tree = tree_info
    lefts = tree.tree_.children_left
    rights = tree.tree_.children_right
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    values = tree.tree_.value

    return TreeParameters(lefts, rights, features, thresholds, values)


def get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values):
    """
    Common functions used by all tree algorithms to generate the parameters according to the tree_trav strategies.
    """
    if len(lefts) == 1:
        # Model creating tree with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        values = [np.array([0.0]), values[0], values[0]]

    ids = [i for i in range(len(lefts))]
    nodes = list(zip(ids, lefts, rights, features, thresholds, values))

    # Refactor the tree parameters in the proper format.
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

    lefts = np.array(lefts)
    rights = np.array(rights)
    features = np.array(features)
    thresholds = np.array(thresholds)
    values = np.array(values)

    return [nodes_map, ids, lefts, rights, features, thresholds, values]


def get_parameters_for_tree_trav_sklearn(lefts, rights, features, thresholds, values):
    """
    This function is used to generate tree parameters for sklearn trees accordingy to the tree_trav strategy.
    Includes SklearnRandomForestClassifier/Regressor and SklearnGradientBoostingClassifier
    """
    features = [max(x, 0) for x in features]
    values = np.array(values)
    if len(values.shape) == 3:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] > 1:
        values /= np.sum(values, axis=1, keepdims=True)

    return get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values)


def get_parameters_for_gemm_common(lefts, rights, features, thresholds, values, n_features):
    """
    Common functions used by all tree algorithms to generate the parameters according to the GEMM strategy.
    """
    if len(lefts) == 1:
        # Model creating trees with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        values = [np.array([0.0]), values[0], values[0]]

    values = np.array(values)
    weights = []
    biases = []

    # First hidden layer has all inequalities.
    hidden_weights = []
    hidden_biases = []
    for left, feature, thresh in zip(lefts, features, thresholds):
        if left != -1:
            hidden_weights.append([1 if i == feature else 0 for i in range(n_features)])
            hidden_biases.append(thresh)
    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))

    n_splits = len(hidden_weights)

    # Second hidden layer has ANDs for each leaf of the decision tree.
    # Depth first enumeration of the tree in order to determine the AND by the path.
    hidden_weights = []
    hidden_biases = []

    path = [0]
    n_nodes = len(lefts)
    visited = [False for _ in range(n_nodes)]

    class_proba = []
    nodes = list(zip(lefts, rights, features, thresholds, values))

    while True and len(path) > 0:
        i = path[-1]
        visited[i] = True
        left, right, feature, threshold, value = nodes[i]
        if left == -1 and right == -1:
            vec = [0 for _ in range(n_splits)]
            # Keep track of positive weights for calculating bias.
            num_positive = 0
            for j, p in enumerate(path[:-1]):
                num_leaves_before_p = list(lefts[:p]).count(-1)
                if path[j + 1] in lefts:
                    vec[p - num_leaves_before_p] = 1
                    num_positive += 1
                elif path[j + 1] in rights:
                    vec[p - num_leaves_before_p] = -1
                else:
                    raise RuntimeError("Inconsistent state encountered while tree translation.")

            if values.shape[-1] > 1:
                class_proba.append((values[i] / np.sum(values[i])).flatten())
            else:
                # We have only a single value. e.g., GBDT
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

    # OR neurons from the preceding layer in order to get final classes.
    weights.append(np.transpose(np.array(class_proba).astype("float32")))
    biases.append(None)

    return weights, biases
