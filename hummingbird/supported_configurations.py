# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Supported configurations settings accepted by Hummingbird are defined here.
"""

N_FEATURES = "n_features"
"""Number of features expected in the input data."""

TREE_IMPLEMENTATION = "tree_implementation"
"""Which tree implementation to use. Values can be: gemm, tree-trav, perf_tree_trav."""
