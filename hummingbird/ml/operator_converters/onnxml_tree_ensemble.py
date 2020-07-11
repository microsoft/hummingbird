# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for ONNX-ML tree-ensemble models.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from . import constants
from ._gbdt_commons import convert_gbdt_classifier_common, convert_gbdt_common
from ._tree_commons import TreeParameters


def _get_tree_infos_from_onnx_ml_operator(model):
    """
    Function used to extract the parameters from a ONNXML TreeEnsemble model.
    """
    tree_infos = []
    left = right = features = values = threshold = None
    tree_ids = target_node_ids = target_tree_ids = modes = None
    classes = post_transform = None

    # The list of attributes is a merge between the classifier and regression operators.
    # The operators descriptions can be found here
    # https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#aionnxmltreeensembleclassifier and
    # here https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#aionnxmltreeensembleregressor
    for attr in model.origin.attribute:
        if attr.name == "nodes_falsenodeids":
            right = attr.ints
        elif attr.name == "nodes_truenodeids":
            left = attr.ints
        elif attr.name == "nodes_featureids":
            features = attr.ints
        elif attr.name == "nodes_values":
            threshold = attr.floats
        elif attr.name == "class_weights" or attr.name == "target_weights":
            values = attr.floats
        elif attr.name == "class_nodeids" or attr.name == "target_nodeids":
            target_node_ids = attr.ints
        elif attr.name == "class_treeids" or attr.name == "target_treeids":
            target_tree_ids = attr.ints
        elif attr.name == "nodes_treeids":
            tree_ids = attr.ints
        elif attr.name == "classlabels_int64s":
            classes = attr.ints
        elif attr.name == "classlabels_strings ":
            if len(attr.strings) > 0:
                raise AssertionError("String class labels not supported yet.")
        elif attr.name == "post_transform_transform":
            post_transform = attr.s
            if post_transform not in [None, b"LOGISTIC", b"SOFTMAX"]:
                raise AssertionError("Post transform {} not supported".format(post_transform))
        elif attr.name == "nodes_modes":
            modes = attr.strings
            for mode in modes:
                if (not mode == b"BRANCH_LEQ") and (not mode == b"LEAF"):
                    raise AssertionError("Modality {} not supported".format(mode))

    # Order values based on target node and tree ids.
    new_values = []
    j = 0
    for i in range(max(target_tree_ids) + 1):
        k = j
        while k < len(target_tree_ids) and target_tree_ids[k] == i:
            k += 1
        target_ids = target_node_ids[j:k]
        target_ids_zipped = dict(zip(target_ids, range(len(target_ids))))
        for key in sorted(target_ids_zipped):
            new_values.append(values[j + target_ids_zipped[key]])
        j = k

    values = new_values
    i = 0
    prev_id = 0
    count = 0
    l_count = 0
    for n, id in enumerate(tree_ids):
        if id == i:
            if modes[n] == b"LEAF":
                left[n] = -1
                right[n] = -1
                threshold[n] = -1
        else:
            t_left = left[prev_id:count]
            t_right = right[prev_id:count]
            t_features = features[prev_id:count]
            t_threshold = threshold[prev_id:count]
            t_values = [0] * len(t_left)
            for j in range(len(t_left)):
                if t_threshold[j] == -1 and l_count < len(values):
                    t_values[j] = values[l_count]
                    l_count += 1
            tree_infos.append(TreeParameters(t_left, t_right, t_features, t_threshold, np.array(t_values).reshape(-1, 1)))
            prev_id = count
            i += 1
        count += 1

    t_left = left[prev_id:count]
    t_right = right[prev_id:count]
    t_features = features[prev_id:count]
    t_threshold = threshold[prev_id:count]
    t_values = [0] * len(t_left)
    for i in range(len(t_left)):
        if t_threshold[i] == -1 and l_count < len(values):
            t_values[i] = values[l_count]
            l_count += 1
        else:
            t_values[i] = 0
    tree_infos.append(TreeParameters(t_left, t_right, t_features, t_threshold, np.array(t_values).reshape(-1, 1)))
    return tree_infos, classes, post_transform


def _dummy_get_parameter(tree_info):
    """
    Dummy function used to return parameters (TreeEnsemble converters already have parameters in the right format)
    """
    return tree_info


def _get_tree_infos_from_tree_ensemble(operator, device=None, extra_config={}):
    """
    Base method for extracting parameters from `ai.onnx.ml.TreeEnsemble`s.
    """
    assert constants.ONNX_INPUTS in extra_config or constants.N_FEATURES in extra_config

    # Get the number of features.
    if constants.ONNX_INPUTS in extra_config:
        inputs = extra_config[constants.ONNX_INPUTS]

        assert operator.origin.input[0] in inputs

        n_features = inputs[operator.origin.input[0]].type.tensor_type.shape.dim[1].dim_value
    else:
        n_features = extra_config[constants.N_FEATURES]

    tree_infos, classes, post_transform = _get_tree_infos_from_onnx_ml_operator(operator)

    # Get tree informations from the operator.
    return n_features, tree_infos, classes, post_transform


def convert_onnx_tree_ensemble_classifier(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.TreeEnsembleClassifier`.

    Args:
        operator: An operator wrapping a `ai.onnx.ml.TreeEnsembleClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Get tree informations from the operator.
    n_features, tree_infos, classes, post_transform = _get_tree_infos_from_tree_ensemble(
        operator.raw_operator, device, extra_config
    )

    # Generate the model.
    return convert_gbdt_classifier_common(tree_infos, _dummy_get_parameter, n_features, len(classes), classes, extra_config)


def convert_onnx_tree_ensemble_regressor(operator, device=None, extra_config={}):
    """
    Converter for `ai.onnx.ml.TreeEnsembleRegressor`.

    Args:
        operator: An operator wrapping a `ai.onnx.ml.TreeEnsembleRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None

    # Get tree informations from the operator.
    n_features, tree_infos, _, _ = _get_tree_infos_from_tree_ensemble(operator.raw_operator, device, extra_config)

    # Generate the model.
    return convert_gbdt_common(tree_infos, _dummy_get_parameter, n_features, extra_config=extra_config)


register_converter("ONNXMLTreeEnsembleClassifier", convert_onnx_tree_ensemble_classifier)
register_converter("ONNXMLTreeEnsembleRegressor", convert_onnx_tree_ensemble_regressor)
