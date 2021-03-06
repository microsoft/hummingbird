# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for ONNX-ML SV models.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._sv_implementations import SVC


def convert_onnx_svm_classifier_model(operator, device, extra_config):
    """
    Converter for `ai.onnx.ml.SVMClassifier`

    Args:
        operator: An operator wrapping a `ai.onnx.ml.SVMClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    # These are passed as params to SVC()
    kernel = degree = sv = nv = a = b = gamma = coef0 = classes = None

    # These are stored for reshaping after parsing is done
    sv_vals = coeffis = None

    for attr in operator.raw_operator.origin.attribute:

        if attr.name == "kernel_type":
            # ex: Convert b'RBF' to 'rbf' for consistency
            kernel = attr.s.lower().decode("UTF-8")
            if kernel not in ["linear", "poly", "rbf"]:  # from svc.py ln 58
                raise RuntimeError("Unsupported kernel for SVC: {}".format(kernel))

        elif attr.name == "coefficients":
            coeffis = np.array(attr.floats)

        elif attr.name == "vectors_per_class":
            nv = np.array(attr.ints).astype("int32")

        elif attr.name == "support_vectors":
            sv_vals = np.array(attr.floats)

        elif attr.name == "rho":
            b = np.array(attr.floats)

        elif attr.name == "kernel_params":
            # See
            # https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/operator_converters/support_vector_machines.py
            # for details on [op._gamma, op.coef0, op.degree]
            kp_arr = np.array(attr.floats)
            gamma = kp_arr[0]
            coef0 = kp_arr[1]
            degree = int(kp_arr[2])

        elif attr.name == "classlabels_ints":
            classes = np.array(attr.ints)

    if any(v is None for v in [sv_vals, coeffis]):
        raise RuntimeError("Error parsing SVC arrays, found unexpected None")

    # Now that we have parsed the degree and lengths, reshape 'a' and 'sv'
    # For 'a', these are in 'dual' shape, so resize into 2:
    # https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/operator_converters/support_vector_machines.py#L41
    #
    # Except for when they're not...
    # https://stackoverflow.com/questions/22816646/the-dimension-of-dual-coef-in-sklearn-svc
    if len(classes) > 2:
        a = coeffis.reshape(2, len(coeffis) // 2)
    else:  # if not in "dual" form with classes > 3 (binary), 'a' and 'b' are the inverse. Don't ask why.
        a = np.negative([coeffis])
        b = np.negative(b)

    sv = sv_vals.reshape(len(a[0]), len(sv_vals) // len(a[0]))

    if any(v is None for v in [kernel, degree, sv, nv, a, b, gamma, coef0, classes]):
        raise RuntimeError(
            "Error parsing SVC, found unexpected None. kernel{} degree{} sv{} nv{} a{} b{} gamma{} coef0{} classes{}".format(
                kernel, degree, sv, nv, a, b, gamma, coef0, classes
            )
        )

    return SVC(operator, kernel, degree, sv, nv, a, b, gamma, coef0, classes, device)


register_converter("ONNXMLSVMClassifier", convert_onnx_svm_classifier_model)
