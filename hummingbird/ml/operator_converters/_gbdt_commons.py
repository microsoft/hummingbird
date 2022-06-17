# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Collections of functions shared among GBDT converters.
"""
import torch
import numpy as np

from . import constants
from ._tree_commons import get_tree_params_and_type, get_parameters_for_tree_trav_common, get_parameters_for_gemm_common
from ._tree_commons import (
    ApplySigmoidPostTransform,
    ApplySoftmaxPostTransform,
    ApplyTweediePostTransform,
    ApplyBasePredictionPostTransform,
    ApplySigmoidBasePredictionPostTransform,
    ApplySoftmaxBasePredictionPostTransform,
    ApplyTweedieBasePredictionPostTransform,
)
from ._tree_implementations import GEMMGBDTImpl, TreeTraversalGBDTImpl, PerfectTreeTraversalGBDTImpl, TreeImpl


def convert_gbdt_classifier_common(
    operator, tree_infos, get_tree_parameters, n_features, n_classes, classes=None, extra_config={}, decision_cond="<="
):
    """
    Common converter for GBDT classifiers.

    Args:
        tree_infos: The information representaing a tree (ensemble)
        get_tree_parameters: A function specifying how to parse the tree_infos into parameters
        n_features: The number of features input to the model
        n_classes: How many classes are expected. 1 for regression tasks
        classes: The classes used for classification. None if implementing a regression model
        extra_config: Extra configuration used to properly implement the source tree
        decision_cond: The condition of the decision nodes in the x <cond> threshold order. Default '<='. Values can be <=, <, >=, >

    Returns:
        A tree implementation in PyTorch
    """
    assert tree_infos is not None
    assert get_tree_parameters is not None
    assert n_features is not None
    assert n_classes is not None

    # Rearrange classes and tree information.
    if n_classes == 2:
        n_classes -= 1
    if classes is None:
        classes = [i for i in range(n_classes)]

    reorder_trees = True
    if constants.REORDER_TREES in extra_config:
        reorder_trees = extra_config[constants.REORDER_TREES]
    if reorder_trees and n_classes > 1:
        tree_infos = [tree_infos[i * n_classes + j] for j in range(n_classes) for i in range(len(tree_infos) // n_classes)]

    return convert_gbdt_common(
        operator, tree_infos, get_tree_parameters, n_features, classes, extra_config=extra_config, decision_cond=decision_cond
    )


def convert_gbdt_common(
    operator, tree_infos, get_tree_parameters, n_features, classes=None, extra_config={}, decision_cond="<="
):
    """
    Common converter for GBDT models.

    Args:
        tree_infos: The information representaing a tree (ensemble)
        get_tree_parameters: A function specifying how to parse the tree_infos into parameters
        n_features: The number of features input to the model
        classes: The classes used for classification. None if implementing a regression model
        extra_config: Extra configuration used to properly implement the source tree
        decision_cond: The condition of the decision nodes in the x <cond> threshold order. Default '<='. Values can be <=, <, >=, >

    Returns:
        A tree implementation in PyTorch
    """
    assert tree_infos is not None
    assert get_tree_parameters is not None
    assert n_features is not None

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(tree_infos, get_tree_parameters, extra_config)

    # Apply learning rate directly on the values rather then at runtime.
    if constants.LEARNING_RATE in extra_config:
        for parameter in tree_parameters:
            parameter.values = parameter.values * extra_config[constants.LEARNING_RATE]

    # Generate the model parameters based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts,
                tree_param.rights,
                tree_param.features,
                tree_param.thresholds,
                tree_param.values,
                n_features,
                extra_config,
            )
            for tree_param in tree_parameters
        ]
    else:
        # Some models require some additional massaging of the parameters before generating the tree_trav implementation.
        get_parameters_for_tree_trav = get_parameters_for_tree_trav_common

        if constants.GET_PARAMETERS_FOR_TREE_TRAVERSAL in extra_config:
            get_parameters_for_tree_trav = extra_config[constants.GET_PARAMETERS_FOR_TREE_TRAVERSAL]
        net_parameters = [
            get_parameters_for_tree_trav(
                tree_param.lefts,
                tree_param.rights,
                tree_param.features,
                tree_param.thresholds,
                tree_param.values,
                extra_config,
            )
            for tree_param in tree_parameters
        ]

    # Define the post transform.
    if constants.BASE_PREDICTION in extra_config:

        base_pred = torch.FloatTensor(extra_config[constants.BASE_PREDICTION])

        # For newer versions of scikit-learn (>1.1.1),
        if len(base_pred.shape) == 4:
            base_pred = base_pred[0][0]

        base_prediction = torch.nn.Parameter(base_pred, requires_grad=False)
        extra_config[constants.BASE_PREDICTION] = base_prediction

    # For models following the Sklearn API we need to build the post transform ourselves.
    if classes is not None and constants.POST_TRANSFORM not in extra_config:
        if len(classes) <= 2:
            extra_config[constants.POST_TRANSFORM] = constants.SIGMOID
        else:
            extra_config[constants.POST_TRANSFORM] = constants.SOFTMAX

    # Set the post transform.
    if constants.POST_TRANSFORM in extra_config:
        if extra_config[constants.POST_TRANSFORM] == constants.SIGMOID:
            if constants.BASE_PREDICTION in extra_config:
                extra_config[constants.POST_TRANSFORM] = ApplySigmoidBasePredictionPostTransform(base_prediction)
            else:
                extra_config[constants.POST_TRANSFORM] = ApplySigmoidPostTransform()
        elif extra_config[constants.POST_TRANSFORM] == constants.SOFTMAX:
            if constants.BASE_PREDICTION in extra_config:
                extra_config[constants.POST_TRANSFORM] = ApplySoftmaxBasePredictionPostTransform(base_prediction)
            else:
                extra_config[constants.POST_TRANSFORM] = ApplySoftmaxPostTransform()
        elif extra_config[constants.POST_TRANSFORM] == constants.TWEEDIE:
            if constants.BASE_PREDICTION in extra_config:
                extra_config[constants.POST_TRANSFORM] = ApplyTweedieBasePredictionPostTransform(base_prediction)
            else:
                extra_config[constants.POST_TRANSFORM] = ApplyTweediePostTransform()
        else:
            raise NotImplementedError("Post transform {} not implemeneted yet".format(extra_config[constants.POST_TRANSFORM]))
    elif constants.BASE_PREDICTION in extra_config:
        extra_config[constants.POST_TRANSFORM] = ApplyBasePredictionPostTransform(base_prediction)

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        return GEMMGBDTImpl(
            operator, net_parameters, n_features, classes, extra_config=extra_config, decision_cond=decision_cond
        )
    if tree_type == TreeImpl.tree_trav:
        return TreeTraversalGBDTImpl(
            operator, net_parameters, max_depth, n_features, classes, extra_config=extra_config, decision_cond=decision_cond
        )
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav.
        return PerfectTreeTraversalGBDTImpl(
            operator, net_parameters, max_depth, n_features, classes, extra_config=extra_config, decision_cond=decision_cond
        )
