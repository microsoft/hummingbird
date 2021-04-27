# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn k neighbor models: KNeighborsClassifier, KNeighborsRegressor.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from hummingbird.ml.operator_converters._kneighbors_implementations import KNeighborsModel, MetricType
from hummingbird.ml.operator_converters import constants


def convert_sklearn_kneighbors_regression_model(operator, device, extra_config):
    """
    Converter for `sklearn.neighbors.KNeighborsRegressor`

    Args:
        operator: An operator wrapping a `sklearn.neighbors.KNeighborsRegressor` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    return _convert_kneighbors_model(operator, device, extra_config, False)


def convert_sklearn_kneighbors_classification_model(operator, device, extra_config):
    """
    Converter for `sklearn.neighbors.KNeighborsClassifier`

    Args:
        operator: An operator wrapping a `sklearn.neighbors.KNeighborsClassifier` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    return _convert_kneighbors_model(operator, device, extra_config, True)


def _convert_kneighbors_model(operator, device, extra_config, is_classifier):
    if constants.BATCH_SIZE not in extra_config:
        raise RuntimeError(
            "Hummingbird requires explicit specification of "
            + constants.BATCH_SIZE
            + " parameter when compiling KNeighborsClassifier"
        )

    classes = None
    if is_classifier:
        classes = operator.raw_operator.classes_.tolist()
        if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
            raise RuntimeError("Hummingbird supports only integer labels for class labels.")

    metric = operator.raw_operator.metric
    params = operator.raw_operator.metric_params

    if metric not in ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"]:
        raise NotImplementedError(
            "Hummingbird currently supports only the metric type 'minkowski', 'wminkowski', 'manhattan', 'chebyshev', 'mahalanobis', 'euclidean', and 'seuclidean' for KNeighbors"
            + "Classifier"
            if is_classifier
            else "Regressor"
        )

    metric_type = None
    metric_params = None
    if metric in ["minkowski", "euclidean", "manhattan", "chebyshev"]:
        metric_type = MetricType.minkowski
        p = 2
        if metric == "minkowski" and params is not None and "p" in params:
            p = params["p"]
        elif metric == "manhattan":
            p = 1
        elif metric == "chebyshev":
            p = float("inf")
        metric_params = {"p": p}
    elif metric == "wminkowski":
        metric_type = MetricType.wminkowski
        p = 2
        if params is not None and "p" in params:
            p = params["p"]
        w = params["w"]
        metric_params = {"p": p, "w": w}
    elif metric == "seuclidean":
        metric_type = MetricType.seuclidean
        V = params["V"]
        metric_params = {"V": V}
    elif metric == "mahalanobis":
        metric_type = MetricType.mahalanobis
        if "VI" in params:
            VI = params["VI"]
        else:
            VI = np.linalg.inv(params["V"])

        metric_params = {"VI": VI}

    weights = operator.raw_operator.weights
    if weights not in ["uniform", "distance"]:
        raise NotImplementedError(
            "Hummingbird currently supports only the weights type 'uniform' and 'distance' for KNeighbors" + "Classifier"
            if is_classifier
            else "Regressor"
        )

    train_data = operator.raw_operator._fit_X
    train_labels = operator.raw_operator._y
    n_neighbors = operator.raw_operator.n_neighbors

    return KNeighborsModel(
        operator,
        train_data,
        train_labels,
        n_neighbors,
        weights,
        classes,
        extra_config[constants.BATCH_SIZE],
        is_classifier,
        metric_type,
        metric_params,
    )


register_converter("SklearnKNeighborsClassifier", convert_sklearn_kneighbors_classification_model)
register_converter("SklearnKNeighborsRegressor", convert_sklearn_kneighbors_regression_model)
