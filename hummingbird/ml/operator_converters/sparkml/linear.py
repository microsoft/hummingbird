# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for SparkML linear models: LogisticRegressionModel
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._linear_implementations import LinearModel


def convert_sparkml_linear_model(operator, device, extra_config):
    """
    Converter for `pyspark.ml.classification.LogisticRegressionModel`

    Args:
        operator: An operator wrapping a `pyspark.ml.classification.LogisticRegressionModel`
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    num_classes = operator.raw_operator.numClasses
    # Spark ML assumes the label column is encoded such that labels start from zero.
    classes = [i for i in range(num_classes)]

    coefficients = operator.raw_operator.coefficientMatrix.toArray().transpose().astype("float32")
    intercepts = operator.raw_operator.interceptVector.reshape(1, -1).astype("float32")

    if num_classes > 2:
        multi_class = "multinomial"
    else:
        multi_class = None

    return LinearModel(operator, coefficients, intercepts, device, classes=classes, multi_class=multi_class)


register_converter("SparkMLLogisticRegressionModel", convert_sparkml_linear_model)
