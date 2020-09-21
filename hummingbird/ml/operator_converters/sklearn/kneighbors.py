# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn k neighbor models: KNeighborsClassifier.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._kneighbors_implementations import KNeighborsClassifierModel
from ..constants import BATCH_SIZE

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

    if BATCH_SIZE not in extra_config:
        raise RuntimeError("Hummingbird requires explicit specification of "+BATCH_SIZE+" parameter when compiling KNeighborsClassifier")

    classes = operator.raw_operator.classes_.tolist()
    if not all([type(x) in [int, np.int32, np.int64] for x in classes]):
        raise RuntimeError("Hummingbird supports only integer labels for class labels.")

    metric = operator.raw_operator.metric
    metric_params = operator.raw_operator.metric_params

    if metric not in ["minkowski", "euclidean"]:
        raise NotImplementedError(
            "Hummingbird currently supports only the metric type 'minkowski' and 'euclidean' for KNeighborsClassifier"
        )

    p = 2
    if metric == "minkowski" and metric_params is not None and "p" in metric_params:
        p = metric_params["p"]

    weights = operator.raw_operator.weights
    if weights not in ["uniform", "distance"]:
        raise NotImplementedError(
            "Hummingbird currently supports only the weights type 'uniform' and 'distance' for KNeighborsClassifier"
        )

    train_data = operator.raw_operator._fit_X
    train_labels = operator.raw_operator._y
    n_neighbors = operator.raw_operator.n_neighbors

    return KNeighborsClassifierModel(train_data, train_labels, n_neighbors, weights, classes, p, extra_config[BATCH_SIZE])


register_converter("SklearnKNeighborsClassifier", convert_sklearn_kneighbors_classification_model)
