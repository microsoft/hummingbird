# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Collections of classes and functions shared among all tree converters.
"""

import copy
import numpy as np

from ._tree_implementations import TreeImpl
from ._tree_implementations import GEMMDecisionTreeImpl, TreeTraversalDecisionTreeImpl, PerfectTreeTraversalDecisionTreeImpl
from . import constants
from hummingbird.ml.exceptions import MissingConverter


class Node:
    """
    Class defining a tree node.
    """

    def __init__(self, id=None):
        """
        Args:
            id: A unique ID for the node
            left: The id of the left node
            right: The id of the right node
            feature: The feature used to make a decision (if not leaf node, ignored otherwise)
            threshold: The threshold used in the decision (if not leaf node, ignored otherwise)
            value: The value stored in the leaf (ignored if not leaf node).
            missing: In the case of a missingle value chosen child node id
        """
        self.id = id
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None
        self.missing = None


class TreeParameters:
    """
    Class containing a convenient in-memory representation of a decision tree.
    """

    def __init__(self, lefts, rights, features, thresholds, values, missings=None):
        """
        Args:
            lefts: The id of the left nodes
            rights: The id of the right nodes
            feature: The features used to make decisions
            thresholds: The thresholds used in the decisions
            values: The value stored in the leaves
            missings: In the case of a missing value which child node to select
        """
        self.lefts = lefts
        self.rights = rights
        self.features = features
        self.thresholds = thresholds
        self.values = values
        self.missings = missings


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
    Args:
        max_depth: The maximum tree-depth found in the tree model.
        low: The maximum depth below which GEMM strategy is used
        high: The maximum depth for which PerfTreeTrav strategy is used

    Returns: A tree implementation
    """
    if constants.TREE_IMPLEMENTATION not in extra_config:
        if max_depth is not None and max_depth <= low:
            return TreeImpl.gemm
        elif max_depth is not None and max_depth <= high:
            return TreeImpl.perf_tree_trav
        else:
            return TreeImpl.tree_trav

    if extra_config[constants.TREE_IMPLEMENTATION] == TreeImpl.gemm.name:
        return TreeImpl.gemm
    elif extra_config[constants.TREE_IMPLEMENTATION] == TreeImpl.tree_trav.name:
        return TreeImpl.tree_trav
    elif extra_config[constants.TREE_IMPLEMENTATION] == TreeImpl.perf_tree_trav.name:
        return TreeImpl.perf_tree_trav
    else:
        raise MissingConverter("Tree implementation {} not found".format(extra_config))


def get_tree_params_and_type(tree_infos, get_tree_parameters, extra_config):
    """
    Populate the parameters from the trees and pick the tree implementation strategy.

    Args:
        tree_infos: The information representaing a tree (ensemble)
        get_tree_parameters: A function specifying how to parse the tree_infos into a
                             `operator_converters._tree_commons_TreeParameters` object
        extra_config: param extra_config: Extra configuration used also to select the best conversion strategy

    Returns:
        The tree parameters, the maximum tree-depth and the tre implementation to use
    """
    tree_parameters = [get_tree_parameters(tree_info) for tree_info in tree_infos]
    max_depth = max(1, _find_max_depth(tree_parameters))
    tree_type = get_tree_implementation_by_config_or_depth(extra_config, max_depth)

    return tree_parameters, max_depth, tree_type


def get_parameters_for_sklearn_common(tree_infos):
    """
    Parse sklearn-based trees, including SklearnRandomForestClassifier/Regressor and SklearnGradientBoostingClassifier/Regressor
    Args:
        tree_infos: The information representing a tree (ensemble)
        Returns: The tree parameters wrapped into an instance of `operator_converters._tree_commons_TreeParameters`
    """
    lefts = tree_infos.tree_.children_left
    rights = tree_infos.tree_.children_right
    features = tree_infos.tree_.feature
    thresholds = tree_infos.tree_.threshold
    values = tree_infos.tree_.value

    return TreeParameters(lefts, rights, features, thresholds, values)


def get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values, missings=None, extra_config={}):
    """
    Common functions used by all tree algorithms to generate the parameters according to the tree_trav strategies.

    Args:
        left: The left nodes
        right: The right nodes
        features: The features used in the decision nodes
        thresholds: The thresholds used in the decision nodes
        values: The values stored in the leaf nodes
        missings: In the case of a missing value which child node to select
    Returns:
        An array containing the extracted parameters
    """
    if len(lefts) == 1:
        # Model creating tree with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        if missings is not None:
            missings = [2, -1, -1]
        n_classes = values.shape[1] if type(values) is np.ndarray else 1
        values = np.array([np.zeros(n_classes), values[0], values[0]])
        values.reshape(3, n_classes)

    ids = [i for i in range(len(lefts))]
    if missings is not None:
        nodes = list(zip(ids, lefts, rights, features, thresholds, values, missings))
    else:
        nodes = list(zip(ids, lefts, rights, features, thresholds, values))

    # Refactor the tree parameters in the proper format.
    nodes_map = {0: Node(0)}
    current_node = 0
    for i, node in enumerate(nodes):
        if missings is not None:
            id, left, right, feature, threshold, value, missing = node
        else:
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

        if missings is not None:
            m_node = l_node if missing == left else r_node
            nodes_map[current_node].missing = m_node

            if missings[i] == -1:
                missings[i] = id

        current_node += 1

    lefts = np.array(lefts)
    rights = np.array(rights)
    features = np.array(features)
    thresholds = np.array(thresholds)
    values = np.array(values)
    if missings is not None:
        missings = np.array(missings)

    return [nodes_map, ids, lefts, rights, features, thresholds, values, missings]


def get_parameters_for_tree_trav_sklearn(lefts, rights, features, thresholds, values, missings=None, classes=None, extra_config={}):
    """
    This function is used to generate tree parameters for sklearn trees.
    Includes SklearnRandomForestClassifier/Regressor, and SklearnGradientBoostingClassifier.

    Args:
        left: The left nodes
        right: The right nodes
        features: The features used in the decision nodes
        thresholds: The thresholds used in the decision nodes
        values: The values stored in the leaf nodes
        missings: In the case of a missing value which child node to select
        classes: The list of class labels. None if regression model
    Returns:
        An array containing the extracted parameters
    """
    features = [max(x, 0) for x in features]
    values = np.array(values)
    if len(values.shape) == 3:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] > 1 and classes is not None and len(classes) > 0:
        # Triggers only for classification.
        values /= np.sum(values, axis=1, keepdims=True)
    if constants.NUM_TREES in extra_config:
        values /= extra_config[constants.NUM_TREES]

    return get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values, missings)


def get_parameters_for_gemm_common(lefts, rights, features, thresholds, values, n_features, missings=None, extra_config={}):
    """
    Common functions used by all tree algorithms to generate the parameters according to the GEMM strategy.

    Args:
        left: The left nodes
        right: The right nodes
        features: The features used in the decision nodes
        thresholds: The thresholds used in the decision nodes
        values: The values stored in the leaf nodes
        n_features: The number of expected input features
        missings: In the case of a missing value which child node to select
    Returns:
        The weights and bias for the GEMM implementation
    """
    values = np.array(values)
    weights = []
    biases = []

    if len(lefts) == 1:
        # Model creating trees with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        if missings is not None:
            missings = [2, -1, -1]
        n_classes = values.shape[1]
        values = np.array([np.zeros(n_classes), values[0], values[0]])
        values.reshape(3, n_classes)

    if missings is None:
        missings = rights

    # First hidden layer has all inequalities.
    hidden_weights = []
    hidden_biases = []
    hidden_missing_biases = []
    for left, right, missing, feature, thresh in zip(lefts, rights, missings, features, thresholds):
        if left != -1 or right != -1:
            hidden_weights.append([1 if i == feature else 0 for i in range(n_features)])
            hidden_biases.append(thresh)

            if missing == right:
                hidden_missing_biases.append(1)
            else:
                hidden_missing_biases.append(0)
    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))

    # Missing value handling biases.
    weights.append(None)
    biases.append(np.array(hidden_missing_biases).astype("float32"))

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
                    vec[p - num_leaves_before_p] = -1
                elif path[j + 1] in rights:
                    num_positive += 1
                    vec[p - num_leaves_before_p] = 1
                else:
                    raise RuntimeError("Inconsistent state encountered while tree translation.")

            if values.shape[-1] > 1:
                proba = (values[i] / np.sum(values[i])).flatten()
            else:
                # We have only a single value. e.g., GBDT
                proba = values[i].flatten()
            # Some Sklearn tree implementations require normalization.
            if constants.NUM_TREES in extra_config:
                proba /= extra_config[constants.NUM_TREES]
            class_proba.append(proba)

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


def convert_decision_ensemble_tree_common(
    operator, tree_infos, get_parameters, get_parameters_for_tree_trav, n_features, classes=None, extra_config={}
):
    tree_parameters, max_depth, tree_type = get_tree_params_and_type(tree_infos, get_parameters, extra_config)

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts,
                tree_param.rights,
                tree_param.features,
                tree_param.thresholds,
                tree_param.values,
                n_features,
                tree_param.missings,
                extra_config,
            )
            for tree_param in tree_parameters
        ]
        return GEMMDecisionTreeImpl(operator, net_parameters, n_features, classes)

    net_parameters = [
        get_parameters_for_tree_trav(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, tree_param.missings, extra_config,
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.tree_trav:
        return TreeTraversalDecisionTreeImpl(operator, net_parameters, max_depth, n_features, classes, extra_config)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return PerfectTreeTraversalDecisionTreeImpl(operator, net_parameters, max_depth, n_features, classes)
