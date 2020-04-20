# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch
import numpy as np

from ._tree_commons import get_tree_params_and_type, get_parameters_for_tree_trav_common, get_parameters_for_gemm_common
from ._tree_commons import GEMMTreeEnsemble, TreeTraversalTreeEnsemble, PerfectTreeTraversalTreeEnsemble, TreeImpl


class GEMMGBDT(GEMMTreeEnsemble):
    """
    Class implementing the GEMM strategy (in PyTorch) for GBDT models.
    """

    def __init__(self, net_parameters, n_features, classes=None, learning_rate=None, alpha=None, device=None):
        super(GEMMGBDT, self).__init__(net_parameters, n_features, classes, 1)
        self.n_gbdt_classes = 1
        self.learning_rate = learning_rate

        if alpha is not None:
            self.alpha = torch.nn.Parameter(torch.FloatTensor(alpha), requires_grad=False)

        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
            if self.n_gbdt_classes == 1:
                self.binary_classification = True

        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        return torch.squeeze(x).t().view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)

    def calibration(self, x):
        if self.binary_classification:
            output = torch.sigmoid(x)
            return torch.cat([1 - output, output], dim=1)
        else:
            return torch.softmax(x, dim=1)


class TreeTraversalGBDT(TreeTraversalTreeEnsemble):
    """
    Class implementing the Tree Traversal strategy in PyTorch.
    """

    def __init__(self, net_parameters, max_detph, n_features, classes=None, learning_rate=None, alpha=None, device=None):
        super(TreeTraversalGBDT, self).__init__(net_parameters, max_detph, n_features, classes, 1)
        self.n_gbdt_classes = 1
        self.learning_rate = learning_rate

        if alpha is not None:
            self.alpha = torch.nn.Parameter(torch.FloatTensor(alpha), requires_grad=False)

        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
            if self.n_gbdt_classes == 1:
                self.binary_classification = True

        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        return x.view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)

    def calibration(self, x):
        if self.binary_classification:
            output = torch.sigmoid(x)
            return torch.cat([1 - output, output], dim=1)
        else:
            return torch.softmax(x, dim=1)


class PerfectTreeTraversalGBDT(PerfectTreeTraversalTreeEnsemble):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch.
    """

    def __init__(self, net_parameters, max_depth, n_features, classes=None, learning_rate=None, alpha=None, device=None):
        super(PerfectTreeTraversalGBDT, self).__init__(net_parameters, max_depth, n_features, classes, 1)
        self.n_gbdt_classes = 1
        self.learning_rate = learning_rate

        if alpha is not None:
            self.alpha = torch.nn.Parameter(torch.FloatTensor(alpha), requires_grad=False)

        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
            if self.n_gbdt_classes == 1:
                self.binary_classification = True

        self.n_trees_per_class = len(net_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        return x.view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)

    def calibration(self, x):
        if self.binary_classification:
            output = torch.sigmoid(x)
            return torch.cat([1 - output, output], dim=1)
        else:
            return torch.softmax(x, dim=1)


def convert_gbdt_classifier_common(
    tree_infos,
    get_tree_parameters,
    n_features,
    n_classes,
    classes=None,
    learning_rate=None,
    alpha=None,
    device=None,
    extra_config={},
):
    """
    Common converter for GBDT classifiers.
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
        tree_infos = [tree_infos[i * n_classes + j] for j in range(n_classes) for i in range(len(tree_infos) // n_classes)]

    return convert_gbdt_common(
        tree_infos, get_tree_parameters, n_features, classes, learning_rate, alpha, device, extra_config
    )


def convert_gbdt_common(
    tree_infos, get_tree_parameters, n_features, classes=None, learning_rate=None, alpha=None, device=None, extra_config={}
):
    """
    Common converter for GBDT models.
    """
    assert tree_infos is not None
    assert get_tree_parameters is not None
    assert n_features is not None

    tree_parameters, max_depth, tree_type = get_tree_params_and_type(tree_infos, get_tree_parameters, extra_config)

    # Generate the tree implementation based on the selected strategy.
    if tree_type == TreeImpl.gemm:
        net_parameters = [
            get_parameters_for_gemm_common(
                tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values, n_features
            )
            for tree_param in tree_parameters
        ]
        return GEMMGBDT(net_parameters, n_features, classes, learning_rate, alpha, device)

    # Some models require some additional massagging of the parameters before generating the tree_trav implementation.
    get_parameters_for_tree_trav = get_parameters_for_tree_trav_common
    if "get_parameters_for_tree_trav" in extra_config:
        get_parameters_for_tree_trav = extra_config["get_parameters_for_tree_trav"]
    net_parameters = [
        get_parameters_for_tree_trav(
            tree_param.lefts, tree_param.rights, tree_param.features, tree_param.thresholds, tree_param.values
        )
        for tree_param in tree_parameters
    ]
    if tree_type == TreeImpl.tree_trav:
        return TreeTraversalGBDT(net_parameters, max_depth, n_features, classes, learning_rate, alpha, device)
    else:  # Remaining possible case: tree_type == TreeImpl.perf_tree_trav.
        return PerfectTreeTraversalGBDT(net_parameters, max_depth, n_features, classes, learning_rate, alpha, device)
