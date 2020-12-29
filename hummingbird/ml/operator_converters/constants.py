# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Constants used in the Hummingbird converters are defined here.
"""

BASE_PREDICTION = "base_prediction"
"""base prediction for ensemble models requiring it."""

LEARNING_RATE = "learning_rate"
"""Learning Rate."""

POST_TRANSFORM = "post_transform"
"""Post transform for tree inference."""

SIGMOID = "LOGISTIC"
"""Sigmoid post transform."""

SOFTMAX = "SOFTMAX"
"""Softmax post transform."""

TWEEDIE = "TWEEDIE"
"""Tweedie post transform."""

GET_PARAMETERS_FOR_TREE_TRAVERSAL = "get_parameters_for_tree_trav"
"""Which function to use to extract the parameters for the tree traversal strategy."""

REORDER_TREES = "reorder_trees"
"""Whether to reorder trees in multiclass tasks."""

ONNX_INITIALIZERS = "onnx_initializers"
"""The initializers of the onnx model."""

TVM_CONTEXT = "tvm_context"
"""The context for TVM containing information on the target."""

TVM_GRAPH = "tvm_graph"
"""The graph defining the TVM model. This parameter is used for saving and loading a TVM model."""

TVM_LIB = "tvm_lib"
"""The lib for the TVM model. This parameter is used for saving and loading a TVM model."""

TVM_PARAMS = "tvm_params"
"""The params for the TVM model. This parameter is used for saving and loading a TVM model."""

TVM_INPUT_NAMES = "tvm_input_names"
"""TVM expects named inputs. This is used to set the names for the inputs."""

SAVE_LOAD_MODEL_TYPE_PATH = "model_type.txt"
"""Path where to find the model type when saving or loading."""

SAVE_LOAD_CONTAINER_PATH = "container.pkl"
"""Path where to find the container when saving or loading."""

SAVE_LOAD_TVM_LIB_PATH = "deploy_lib.tar"
"""Path where to find the TVM lib when saving or loading."""

SAVE_LOAD_TVM_GRAPH_PATH = "deploy_graph.json"
"""Path where to find the TVM graph when saving or loading."""

SAVE_LOAD_TVM_PARAMS_PATH = "deploy_param.params"
"""Path where to find the TVM params when saving or loading."""

SAVE_LOAD_TORCH_JIT_PATH = "deploy_model.zip"
"""Path where to find the torchscript model when saving or loading."""

SAVE_LOAD_ONNX_PATH = "deploy_model.onnx"
"""Path where to find the onnx model when saving or loading."""

TEST_INPUT = "test_input"
"""The test input data for models that need to be traced."""

N_INPUTS = "n_inputs"
"""Number of inputs expected by the model."""

NUM_TREES = "n_trees"
"""Number of trees composing an ensemble."""

OFFSET = "offset"
"""offset of the sklearn anomaly detection implementation."""

IFOREST_THRESHOLD = "iforest_threshold"
"""threshold of the sklearn isolation forest implementation, backward compatibility for sklearn <= 0.21."""

MAX_SAMPLES = "max_samples"
"""max_samples of sklearn isolation forest implementation."""

N_FEATURES = "n_features"
"""Number of features expected in the input data."""

SUPPORTED_STRING_TYPES = {"S", "U"}
"""Numpy string types suppoted by Humingbird."""
