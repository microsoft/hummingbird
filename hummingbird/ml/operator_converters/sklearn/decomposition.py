# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn matrix decomposition transformers: PCA, KernelPCA, TruncatedSVD, FastICA.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._decomposition_implementations import KernelPCA, Decomposition


def convert_sklearn_pca(operator, device, extra_config):
    """
    Converter for `sklearn.decomposition.PCA`

    Args:
        operator: An operator wrapping a `sklearn.decomposition.PCA` transformer
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    transform_matrix = operator.raw_operator.components_.transpose()
    mean = operator.raw_operator.mean_.reshape(1, -1)
    if operator.raw_operator.whiten:
        transform_matrix = transform_matrix / np.sqrt(operator.raw_operator.explained_variance_)

    return Decomposition(operator, mean, transform_matrix, device)


def convert_sklearn_kernel_pca(operator, device, extra_config):
    """
    Converter for `sklearn.decomposition.KernelPCA`

    Args:
        operator: An operator wrapping a `sklearn.decomposition.KernelPCA` transformer
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    if operator.raw_operator.kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]:
        kernel = operator.raw_operator.kernel
        degree = operator.raw_operator.degree
        sv = operator.raw_operator.X_fit_
        non_zeros = np.flatnonzero(operator.raw_operator.lambdas_)
        scaled_alphas = np.zeros_like(operator.raw_operator.alphas_)
        scaled_alphas[:, non_zeros] = operator.raw_operator.alphas_[:, non_zeros] / np.sqrt(
            operator.raw_operator.lambdas_[non_zeros]
        )
        return KernelPCA(
            operator,
            kernel,
            degree,
            sv,
            scaled_alphas,
            operator.raw_operator.gamma,
            operator.raw_operator.coef0,
            operator.raw_operator._centerer.K_fit_rows_,
            operator.raw_operator._centerer.K_fit_all_,
            device,
        )
    else:
        raise NotImplementedError(
            "Hummingbird does not currently support {} kernel for KernelPCA. The supported kernels are linear, poly, rbf, sigmoid, cosine, and precomputed.".format(
                operator.raw_operator.kernel
            )
        )


def convert_sklearn_fast_ica(operator, device, extra_config):
    """
    Converter for `sklearn.decomposition.FastICA`

    Args:
        operator: An operator wrapping a `sklearn.decomposition.FastICA` transformer
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    transform_matrix = operator.raw_operator.components_.transpose()
    if hasattr(operator.raw_operator, "mean_"):
        mean = operator.raw_operator.mean_.reshape(1, -1).astype("float32")
    else:
        mean = None

    return Decomposition(operator, mean, transform_matrix.astype("float32"), device)


def convert_sklearn_truncated_svd(operator, device, extra_config):
    """
    Converter for `sklearn.decomposition.TruncatedSVD`

    Args:
        operator: An operator wrapping a `sklearn.decomposition.TruncatedSVD` transformer
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    transform_matrix = operator.raw_operator.components_.transpose()
    return Decomposition(operator, None, transform_matrix.astype("float32"), device)


register_converter("SklearnPCA", convert_sklearn_pca)
register_converter("SklearnKernelPCA", convert_sklearn_kernel_pca)
register_converter("SklearnFastICA", convert_sklearn_fast_ica)
register_converter("SklearnTruncatedSVD", convert_sklearn_truncated_svd)
