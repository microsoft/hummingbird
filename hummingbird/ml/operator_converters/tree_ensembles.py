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
from ._gbdt_commons import convert_gbdt_classifier_common
from ._tree_commons import TreeParameters, get_parameters_for_tree_trav, convert_decision_ensemble_tree_common


def _get_tree_infos_from_onnx_ml_operator(model):
    tree_infos = []
    left = []
    right = []
    features = []
    values = []
    threshold = []
    tree_ids = []
    classes = []
    modes = []
    target_node_ids = []
    target_tree_ids = []
    post_transform = ""

    for attr in model.origin.attribute:
        if attr.name == "nodes_falsenodeids":
            right = attr.ints
        elif attr.name == "nodes_truenodeids":
            left = attr.ints
        elif attr.name == "nodes_featureids":
            features = attr.ints
        elif attr.name == "nodes_values":
            threshold = attr.floats
        elif attr.name == "class_weights":
            values = attr.floats
        elif attr.name == "class_nodeids" or "target_nodeids":
            target_node_ids = attr.ints
        elif attr.name == "class_treeids" or "target_treeids":
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
                features[n] = -1
                threshold[n] = -1
        else:
            t_left = left[prev_id:count]
            t_right = right[prev_id:count]
            t_features = features[prev_id:count]
            t_threshold = threshold[prev_id:count]
            t_values = [0] * len(t_left)
            for j in range(len(t_left)):
                if t_features[j] == -1 and l_count < len(values):
                    t_values[j] = values[l_count]
                    l_count += 1
            tree_infos.append(TreeParameters(t_left, t_right, t_features, t_threshold, np.array(t_values)))
            prev_id = count
            i += 1
        count += 1

    t_left = left[prev_id:count]
    t_right = right[prev_id:count]
    t_features = features[prev_id:count]
    t_threshold = threshold[prev_id:count]
    t_values = [0] * len(t_left)
    for i in range(len(t_left)):
        if t_features[i] == -1 and l_count < len(values):
            t_values[i] = values[l_count]
            l_count += 1
        else:
            t_values[i] = 0
    tree_infos.append(TreeParameters(t_left, t_right, t_features, t_threshold, np.array(t_values)))
    return tree_infos, classes, post_transform


def _dummy_get_parameter(tree_info):
    return tree_info


def convert_onnx_tree_enseble_classifier(operator, device, extra_config):
    assert operator is not None

    inputs = extra_config["inputs"]
    n_features = classes = None

    assert operator.get_input_by_idx(0) in inputs

    # Get the number of features.
    n_features = inputs[operator.get_input_by_idx(0)].type.tensor_type.shape.dim[1].dim_value

    # Get tree informations from the operator
    tree_infos, classes, post_transform = _get_tree_infos_from_onnx_ml_operator(operator)
    # if(n_classes > 2 and post_transform is not None):
    #     trees = [trees[i * n_classes + j] for j in range(n_classes) for i in range(len(trees)//n_classes)]

    # extra_config['compute_proba'] = False

    # Generate the model.
    if post_transform is None:
        return convert_decision_ensemble_tree_common(
            tree_infos, _dummy_get_parameter, get_parameters_for_tree_trav, n_features, classes, extra_config=extra_config
        )
    else:
        return convert_gbdt_classifier_common(
            tree_infos, _dummy_get_parameter, n_features, len(classes), classes, extra_config
        )


def convert_onnx_tree_enseble_regressor(operator, device, extra_config):
    assert operator is not None

    n_features = classes = None
    inputs = extra_config["inputs"]

    assert operator.get_input_by_idx(0) in inputs

    # Get the number of features.
    n_features = inputs[operator.get_input_by_idx(0)].type.tensor_type.shape.dim[1].dim_value

    # Get tree informations from the operator.
    tree_infos, classes, post_transform = _get_tree_infos_from_onnx_ml_operator(operator)

    # Generate the model.
    return convert_decision_ensemble_tree_common(
        tree_infos, _dummy_get_parameter, get_parameters_for_tree_trav, n_features, classes, extra_config=extra_config
    )


register_converter("ONNXMLTreeEnsembleClassifier", convert_onnx_tree_enseble_classifier)
register_converter("ONNXMLTreeEnsembleRegressor", convert_onnx_tree_enseble_regressor)
