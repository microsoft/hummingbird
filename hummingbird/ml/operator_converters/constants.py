# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Constants used in the Hummingbird converters are defined here.
"""

BASE_PREDICTION = "base_prediction"
"""Alpha"""

LEARNING_RATE = "learning_rate"
"""Learning Rate"""

POST_TRANSFORM = "post_transform"
"""Post transform for tree inference"""

SIGMOID = "LOGISTIC"
"""Sigmoid post transform"""

SOFTMAX = "SOFTMAX"
"""Softmax post transform"""

TWEEDIE = "TWEEDIE"
"""Tweedie post transform"""

GET_PARAMETERS_FOR_TREE_TRAVERSAL = "get_parameters_for_tree_trav"
"""Which function to use to extract the parameters for the tree traversal strategy"""

REORDER_TREES = "reorder_trees"
"""Whether to reorder trees in multiclass tasks"""

ONNX_INITIALIZERS = "onnx_initializers"
"""The initializers of the onnx model"""

ONNX_INPUTS = "onnx_inputs"
"""The input of the onnx model"""

TEST_INPUT = "test_input"
"""The test input data for models that need to be traced"""

NUM_TREES = "n_trees"
"""Number of trees composing an ensemble"""

OFFSET = "offset"
"""offset of the sklearn anomaly detection implementation"""

IFOREST_THRESHOLD = "iforest_threshold"
"""threshold of the sklearn isolation forest implementation, backward compatibility for sklearn <= 0.21"""

MAX_SAMPLES = "max_samples"
"""max_samples of sklearn isolation forest implementation"""
