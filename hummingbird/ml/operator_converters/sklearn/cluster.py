# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn KMeans and MeanShift models.
"""

import torch

from .._physical_operator import PhysicalOperator
from onnxconverter_common.registration import register_converter


class KMeans(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Kmeans in PyTorch
    """

    def __init__(self, logical_operator, centroids, device):
        super(KMeans, self).__init__(logical_operator, regression=True)

        self.centroids = torch.nn.Parameter(torch.FloatTensor(centroids), requires_grad=False)

    def forward(self, x):
        # Compute the Euclidean distance
        dist = torch.cdist(x, self.centroids, compute_mode="donot_use_mm_for_euclid_dist")
        label = torch.argmin(dist, dim=1)
        return label


def convert_sklearn_kmeans_model(operator, device, extra_config):
    assert operator is not None, "Cannot convert None operator"

    centroids = operator.raw_operator.cluster_centers_
    return KMeans(operator, centroids, device)


register_converter("SklearnKMeans", convert_sklearn_kmeans_model)
register_converter("SklearnMeanShift", convert_sklearn_kmeans_model)
