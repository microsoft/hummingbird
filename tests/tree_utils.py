"""
Collection of utils for testing tree converters.
"""
gbdt_implementation_map = {
    "tree_trav": "<class 'hummingbird.operator_converters._gbdt_commons.TreeTraversalGBDTImpl'>",
    "perf_tree_trav": "<class 'hummingbird.operator_converters._gbdt_commons.PerfectTreeTraversalGBDTImpl'>",
    "gemm": "<class 'hummingbird.operator_converters._gbdt_commons.GEMMGBDTImpl'>",
}

dt_implementation_map = {
    "tree_trav": "<class 'hummingbird.operator_converters.decision_tree.TreeTraversalDecisionTreeImpl'>",
    "perf_tree_trav": "<class 'hummingbird.operator_converters.decision_tree.PerfectTreeTraversalDecisionTreeImpl'>",
    "gemm": "<class 'hummingbird.operator_converters.decision_tree.GEMMDecisionTreeImpl'>",
}
