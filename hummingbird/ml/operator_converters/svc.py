# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch
import scipy
import numpy as np

from onnxconverter_common.registration import register_converter

"""
Converters for scikit-learn linear models: SVC, NuSVC.  (LinearSVC is covered by linear_classifier.py)
"""


class SVC(torch.nn.Module):
    def __init__(self, kernel, degree, sv, nv, a, b, gamma, coef0, classes, device):
        super(SVC, self).__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.regression = False
        sv = sv.toarray() if type(sv) == scipy.sparse.csr.csr_matrix else sv
        # if type(sv) == scipy.sparse.csr.csr_matrix and self.kernel == 'poly':
        #     dense = sv.toarray()
        #     coo = scipy.sparse.coo_matrix(dense, shape=dense.shape)
        #     values = coo.data
        #     indices = np.vstack((coo.row, coo.col))
        #     i = torch.LongTensor(indices)
        #     v = torch.FloatTensor(values)
        #     shape = coo.shape
        #     self.sv = torch.nn.Parameter(torch.sparse.FloatTensor(i, v, shape), requires_grad=False)
        # else:
        self.sv = torch.nn.Parameter(torch.from_numpy(sv).double(), requires_grad=False)
        self.sv_t = torch.nn.Parameter(torch.transpose(self.sv, 0, 1), requires_grad=False)
        self.sv_norm = torch.nn.Parameter(-self.gamma * (self.sv ** 2).sum(1).view(1, -1), requires_grad=False)
        self.coef0 = coef0
        self.n_features = sv.shape[1]
        self.a = a
        self.b = torch.nn.Parameter(torch.nn.Parameter(torch.from_numpy(b.reshape(1, -1)).double()), requires_grad=False)
        self.start = [sum(nv[:i]) for i in range(len(nv))]
        self.end = [self.start[i] + nv[i] for i in range(len(nv))]
        self.len_nv = len(nv)
        true_classes, false_classes = zip(*[(i, j) for i in range(self.len_nv) for j in range(i + 1, self.len_nv)])
        self.true_classes = torch.nn.Parameter(torch.IntTensor([true_classes]), requires_grad=False)
        self.false_classes = torch.nn.Parameter(torch.IntTensor([false_classes]), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.n_classes = len(classes)

        # if self.n_classes is not 2:
        #     self.perform_class_select = True
        # if self.n_classes == 2:
        #     tmp_classes = self.true_classes
        #     self.true_classes = self.false_classes
        #     self.false_classes = tmp_classes

    def forward(self, x):
        x = x.double()
        # x = x.view(-1, 1, self.n_features)
        if self.kernel == "linear":
            # k = self.sv*x
            k = torch.mm(x, self.sv_t)
        elif self.kernel == "rbf":
            # k = torch.exp(-self.gamma*torch.pow(self.sv-x, 2))
            # using quadratic expansion--susseptible to rounding-off errors
            x_norm = -self.gamma * (x ** 2).sum(1).view(-1, 1)
            k = torch.exp(x_norm + self.sv_norm + 2.0 * self.gamma * torch.mm(x, self.sv_t))
        elif self.kernel == "sigmoid":
            # k = torch.sigmoid(self.gamma*self.sv*x + self.coef0)
            k = torch.sigmoid(self.gamma * torch.mm(x, self.sv_t) + self.coef0)
        else:  # poly kernel
            # k = torch.pow(self.gamma*self.sv*x + self.coef0, self.degree)
            k = torch.pow(self.gamma * torch.mm(x, self.sv_t) + self.coef0, self.degree)

        # c = [sum(torch.sum(self.a[i][p] * k[:,p,:], dim=1, keepdim=True) for p in range(self.start[j], self.end[j]))
        #      + sum(torch.sum(self.a[j - 1][p] * k[:,p,:], dim=1, keepdim=True) for p in range(self.start[i], self.end[i]))

        c = [
            sum(self.a[i, p] * k[:, p : p + 1] for p in range(self.start[j], self.end[j]))
            + sum(self.a[j - 1, p] * k[:, p : p + 1] for p in range(self.start[i], self.end[i]))
            for i in range(self.len_nv)
            for j in range(i + 1, self.len_nv)
        ]
        c = torch.cat(c, dim=1) + self.b
        if self.n_classes == 2:
            class_ids = torch.gt(c, 0.0).int().flatten()
        else:
            votes = torch.where(c > 0, self.true_classes, self.false_classes)
            # TODO mode is still not implemented for GPU backend
            votes = votes.data.cpu()
            class_ids, _ = torch.mode(votes, dim=1)
        # no class probabilities in SVC
        if self.perform_class_select:
            temp = torch.index_select(self.classes, 0, class_ids.long())
            return temp, temp
        else:
            return class_ids, class_ids


def convert_sklearn_svc_model(operator, device, extra_config):
    if operator.raw_operator.kernel in ["linear", "poly", "rbf", "sigmoid"]:
        # https://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict
        kernel = operator.raw_operator.kernel
        degree = operator.raw_operator.degree
        classes = operator.raw_operator.classes_
        sv = operator.raw_operator.support_vectors_
        nv = operator.raw_operator.n_support_
        a = operator.raw_operator.dual_coef_
        b = operator.raw_operator.intercept_
        coef0 = operator.raw_operator.coef0

        if hasattr(operator.raw_operator, "_gamma"):
            gamma = operator.raw_operator._gamma
        else:
            gamma = operator.raw_operator.gamma

        return SVC(kernel, degree, sv, nv, a, b, gamma, coef0, classes, device)
    else:
        raise RuntimeError("Unsupported kernel for SVC: {}".format(operator.raw_operator.kernel))


register_converter("SklearnSVC", convert_sklearn_svc_model)
register_converter("SklearnNuSVC", convert_sklearn_svc_model)
