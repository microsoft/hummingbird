# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn SV models: SVC, NuSVC.  (LinearSVC is covered by linear_classifier.py).
"""

from onnxconverter_common.registration import register_converter
from .._sv_implementations import SVC


def convert_sklearn_svc_model(operator, device, extra_config):
    """
    Converter for `sklearn.svm.SVC` and `sklearn.svm.NuSVC`

    Args:
        operator: An operator wrapping a `sklearn.svm.SVC` or `sklearn.svm.NuSVC` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
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
            # TODO: which versions is this case for, and how to test?
            gamma = operator.raw_operator.gamma

        return SVC(operator, kernel, degree, sv, nv, a, b, gamma, coef0, classes, device)
    else:
        raise RuntimeError("Unsupported kernel for SVC: {}".format(operator.raw_operator.kernel))


register_converter("SklearnSVC", convert_sklearn_svc_model)
register_converter("SklearnNuSVC", convert_sklearn_svc_model)
