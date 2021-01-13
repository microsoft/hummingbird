# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn isolation forest.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .. import constants
from .._tree_commons import (
    get_parameters_for_sklearn_common,
    get_parameters_for_tree_trav_sklearn,
    get_tree_params_and_type,
    get_parameters_for_gemm_common,
)
from .._tree_implementations import TreeImpl, GEMMTreeImpl, TreeTraversalTreeImpl, PerfectTreeTraversalTreeImpl


def _average_path_length(n_samples_leaf):
    """
    Taken from sklearn implementation of isolation forest:
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/ensemble/_iforest.py#L480
    For each given number of samples in the array n_samples_leaf, this calculates average path length of unsucceesful
    BST search.

    Args:
        n_samples_leaf: array of number of samples (in leaf)

    Returns:
        array of average path lengths
    """
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


def _get_iforest_anomaly_score_per_node(children_left, children_right, n_node_samples):
    """
    Get anomaly score per node in isolation forest, which is node depth + _average_path_length(n_node_samples). Will
    be used to replace "value" in each tree.

    Args:
        children_left: left children
        children_right: right children
        n_node_samples: number of samples per node
    """
    # Get depth per node.
    node_depth = np.zeros(shape=n_node_samples.shape, dtype=np.int64)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
    return _average_path_length(n_node_samples) + node_depth


def _get_parameters_for_sklearn_iforest(tree_infos):
    """
    Parse sklearn-based isolation forest, replace existing values of node with anomaly score calculated in
    _get_iforest_anomaly_score_per_node

    Args:
        tree_infos: The information representing a tree (ensemble)

    Returns:
        The tree parameters wrapped into an instance of `operator_converters._tree_commons_TreeParameters`
    """
    tree_parameters = get_parameters_for_sklearn_common(tree_infos)
    tree_parameters.values = _get_iforest_anomaly_score_per_node(
        tree_parameters.lefts, tree_parameters.rights, tree_infos.tree_.n_node_samples
    ).reshape(tree_parameters.values.shape)
    return tree_parameters


# Isolation Forest implementations.
class GEMMIsolationForestImpl(GEMMTreeImpl):
    """
    Class implementing the GEMM strategy (in PyTorch) for isolation forest model.
    """

    def __init__(self, logical_operator, tree_parameters, n_features, classes=None, extra_config={}):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(GEMMIsolationForestImpl, self).__init__(
            logical_operator, tree_parameters, n_features, classes, None, anomaly_detection=True
        )

        # Assign the required constants.
        if constants.OFFSET in extra_config:
            self.offset = extra_config[constants.OFFSET]
        if constants.MAX_SAMPLES in extra_config:
            self.max_samples = extra_config[constants.MAX_SAMPLES]
        # Backward compatibility for sklearn <= 0.21
        if constants.IFOREST_THRESHOLD in extra_config:
            self.offset += extra_config[constants.IFOREST_THRESHOLD]
        self.final_probability_divider = len(tree_parameters)
        self.average_path_length = _average_path_length(np.array([self.max_samples]))[0]

    def aggregation(self, x):
        output = x.sum(0).t()
        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider
        # Further normalize to match "decision_function" in sklearn implementation.
        output = -1.0 * 2 ** (-output / self.average_path_length) - self.offset
        return output


class TreeTraversalIsolationForestImpl(TreeTraversalTreeImpl):
    """
    Class implementing the Tree Traversal strategy in PyTorch for isolation forest model.
    """

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes=None, extra_config={}):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(TreeTraversalIsolationForestImpl, self).__init__(
            logical_operator, tree_parameters, max_depth, n_features, classes, n_classes=None, anomaly_detection=True
        )

        # Assign the required constants.
        if constants.OFFSET in extra_config:
            self.offset = extra_config[constants.OFFSET]
        if constants.MAX_SAMPLES in extra_config:
            self.max_samples = extra_config[constants.MAX_SAMPLES]
        # Backward compatibility for sklearn <= 0.21
        if constants.IFOREST_THRESHOLD in extra_config:
            self.offset += extra_config[constants.IFOREST_THRESHOLD]
        self.final_probability_divider = len(tree_parameters)
        self.average_path_length = _average_path_length(np.array([self.max_samples]))[0]

    def aggregation(self, x):
        output = x.sum(1)
        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider
        # Further normalize to match "decision_function" in sklearn implementation.
        output = -1.0 * 2 ** (-output / self.average_path_length) - self.offset
        return output


class PerfectTreeTraversalIsolationForestImpl(PerfectTreeTraversalTreeImpl):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch for isolation forest model.
    """

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes=None, extra_config={}):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(PerfectTreeTraversalIsolationForestImpl, self).__init__(
            logical_operator, tree_parameters, max_depth, n_features, classes, None, anomaly_detection=True
        )

        # Assign the required constants.
        if constants.OFFSET in extra_config:
            self.offset = extra_config[constants.OFFSET]
        if constants.MAX_SAMPLES in extra_config:
            self.max_samples = extra_config[constants.MAX_SAMPLES]
        # Backward compatibility for sklearn <= 0.21
        if constants.IFOREST_THRESHOLD in extra_config:
            self.offset += extra_config[constants.IFOREST_THRESHOLD]
        self.final_probability_divider = len(tree_parameters)
        self.average_path_length = _average_path_length(np.array([self.max_samples]))[0]

    def aggregation(self, x):
        output = x.sum(1)
        if self.final_probability_divider > 1:
            output = output / self.final_probability_divider
        # Further normalize to match "decision_function" in sklearn implementation.
        output = -1.0 * 2 ** (-output / self.average_path_length) - self.offset
        return output


def convert_sklearn_isolation_forest(operator, device, extra_config):
    """
    Converter for `sklearn.ensemble.IsolationForest`.

    Args:
        operator: An operator wrapping a tree (ensemble) isolation forest model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    tree_infos = operator.raw_operator.estimators_
    n_features = operator.raw_operator.n_features_
    # Following constants will be passed in the tree implementation to normalize the anomaly score.
    extra_config[constants.OFFSET] = operator.raw_operator.offset_
    if hasattr(operator.raw_operator, "threshold_"):
        extra_config[constants.IFOREST_THRESHOLD] = operator.raw_operator.threshold_
    extra_config[constants.MAX_SAMPLES] = operator.raw_operator.max_samples_

    # Predict in isolation forest sklearn implementation produce 2 classes: -1 (normal) & 1 (anomaly).
    classes = [-1, 1]

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(
        tree_infos, _get_parameters_for_sklearn_iforest, extra_config
    )

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, n_features, tree_param.missings
            )
            for tree_param in tree_parameters
        ]
        return GEMMIsolationForestImpl(operator, net_parameters, n_features, classes, extra_config=extra_config)

    net_parameters = [
        get_parameters_for_tree_trav_sklearn(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, tree_param.missings
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.tree_trav:
        return TreeTraversalIsolationForestImpl(
            operator, net_parameters, max_depth, n_features, classes, extra_config=extra_config
        )
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav
        return PerfectTreeTraversalIsolationForestImpl(
            operator, net_parameters, max_depth, n_features, classes, extra_config=extra_config
        )


# Register the converters.
register_converter("SklearnIsolationForest", convert_sklearn_isolation_forest)
