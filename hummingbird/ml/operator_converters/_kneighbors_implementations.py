# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for KNeighbors model implementations: (KNeighborsClassifier, KNeighborsRegressor).
"""

from enum import Enum
import torch
import numpy as np
from ._physical_operator import PhysicalOperator


class MetricType(Enum):
    minkowski = 1
    wminkowski = 2
    seuclidean = 3
    mahalanobis = 4


class KNeighborsModel(PhysicalOperator, torch.nn.Module):
    def __init__(
        self,
        logical_operator,
        train_data,
        train_labels,
        n_neighbors,
        weights,
        classes,
        batch_size,
        is_classifier,
        metric_type,
        metric_params,
    ):
        super(KNeighborsModel, self).__init__(logical_operator)
        self.classification = is_classifier
        self.regression = not is_classifier
        self.train_data = torch.nn.Parameter(torch.from_numpy(train_data.astype("float32")), requires_grad=False)
        self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype("int64")), requires_grad=False)
        self.n_neighbors = n_neighbors
        self.metric_type = metric_type

        if self.metric_type == MetricType.minkowski:
            self.p = float(metric_params["p"])
        elif self.metric_type == MetricType.wminkowski:
            self.p = float(metric_params["p"])
            self.w = torch.nn.Parameter(
                torch.from_numpy(metric_params["w"].astype("float32").reshape(1, -1)), requires_grad=False
            )
            self.train_data = torch.nn.Parameter(
                torch.from_numpy(metric_params["w"].astype("float32").reshape(1, -1) * train_data.astype("float32")),
                requires_grad=False,
            )
        elif self.metric_type == MetricType.seuclidean:
            self.V = torch.nn.Parameter(
                torch.from_numpy(np.power(metric_params["V"].astype("float32").reshape(1, -1), -0.5)), requires_grad=False
            )
            self.train_data = torch.nn.Parameter(
                torch.from_numpy(
                    np.power(metric_params["V"].astype("float32").reshape(1, -1), -0.5) * train_data.astype("float32")
                ),
                requires_grad=False,
            )
        elif self.metric_type == MetricType.mahalanobis:
            cholesky_l = np.linalg.cholesky(metric_params["VI"]).astype("float32")
            self.L = torch.nn.Parameter(torch.from_numpy(cholesky_l), requires_grad=False)
            self.train_data = torch.nn.Parameter(
                torch.from_numpy(np.matmul(train_data.astype("float32"), cholesky_l)), requires_grad=False
            )

        if is_classifier:
            # classification
            self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype("int64")), requires_grad=False)
            self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
            self.n_classes = len(classes)
            self.perform_class_select = False

            if min(classes) != 0 or max(classes) != len(classes) - 1:
                self.perform_class_select = True
            self.one_tensor = torch.FloatTensor([1.0])
            self.proba_tensor = torch.zeros((batch_size, self.n_classes), dtype=torch.float32)
        else:
            # regression
            self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype("float32")), requires_grad=False)
            self.n_targets = 1
            if len(self.train_labels.shape) == 2:
                self.n_targets = self.train_labels.shape[1]

        self.weights = weights

    def forward(self, x):
        if self.metric_type == MetricType.minkowski:
            k = torch.cdist(x, self.train_data, p=self.p, compute_mode="donot_use_mm_for_euclid_dist")
        elif self.metric_type == MetricType.wminkowski:
            k = torch.cdist(self.w * x, self.train_data, p=self.p, compute_mode="donot_use_mm_for_euclid_dist")
        elif self.metric_type == MetricType.seuclidean:
            k = torch.cdist(self.V * x, self.train_data, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        elif self.metric_type == MetricType.mahalanobis:
            # We use the Cholesky decomposition to calculate the Mahalanobis distance
            # Mahalanobis distance d^2(x, x') = (x - x')T VI (x - x')
            # using Cholesky decomposition we have VI = LT L
            # then:
            #                      d^2(x, x') = (x - x')T (LT L) (x - x')
            #                                 = (Lx - Lx')T (Lx - Lx')
            k = torch.cdist(torch.mm(x, self.L), self.train_data, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        d, k = torch.topk(k, self.n_neighbors, dim=1, largest=False)
        output = torch.index_select(self.train_labels, 0, k.view(-1))

        if self.weights == "distance":
            d = torch.pow(d, -1)
            inf_mask = torch.isinf(d)
            inf_row = torch.any(inf_mask, axis=1)
            d[inf_row] = inf_mask[inf_row].float()
        else:
            d = torch.ones_like(k, dtype=torch.float32)

        if self.classification:
            # classification
            output = output.view(-1, self.n_neighbors)
            output = torch.scatter_add(self.proba_tensor, 1, output, d)
            proba_sum = output.sum(1, keepdim=True)
            proba_sum = torch.where(proba_sum == 0, self.one_tensor, proba_sum)
            output = torch.pow(proba_sum, -1) * output

            if self.perform_class_select:
                return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
            else:
                return torch.argmax(output, dim=1), output
        else:
            # regression
            if self.n_targets > 1:
                output = output.view(-1, self.n_neighbors, self.n_targets)
                d = d.view(-1, self.n_neighbors, 1)
            else:
                output = output.view(-1, self.n_neighbors)
            output = d * output
            if self.weights != "distance":
                output = output.sum(1) / self.n_neighbors
            else:
                denom = d.sum(1)
                output = output.sum(1) / denom
            return output
