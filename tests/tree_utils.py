"""
Collection of utils for testing tree converters.
"""
gbdt_implementation_map = {
    "tree_trav": "<class 'hummingbird.ml.operator_converters._tree_implementations.TreeTraversalGBDTImpl'>",
    "perf_tree_trav": "<class 'hummingbird.ml.operator_converters._tree_implementations.PerfectTreeTraversalGBDTImpl'>",
    "gemm": "<class 'hummingbird.ml.operator_converters._tree_implementations.GEMMGBDTImpl'>",
}

dt_implementation_map = {
    "tree_trav": "<class 'hummingbird.ml.operator_converters._tree_implementations.TreeTraversalDecisionTreeImpl'>",
    "perf_tree_trav": "<class 'hummingbird.ml.operator_converters._tree_implementations.PerfectTreeTraversalDecisionTreeImpl'>",
    "gemm": "<class 'hummingbird.ml.operator_converters._tree_implementations.GEMMDecisionTreeImpl'>",
}
