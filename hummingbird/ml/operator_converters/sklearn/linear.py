# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for scikit-learn linear models: LinearRegression, LogisticRegression, LinearSVC, SGDClassifier, LogisticRegressionCV.
"""

import numpy as np
from onnxconverter_common.registration import register_converter

from .._linear_implementations import LinearModel


def convert_sklearn_linear_model(operator, device, extra_config):
    """
    Converter for `sklearn.svm.LinearSVC`, `sklearn.linear_model.LogisticRegression`,
    `sklearn.linear_model.SGDClassifier`, and `sklearn.linear_model.LogisticRegressionCV`

    Args:
        operator: An operator wrapping a `sklearn.svm.LinearSVC`, `sklearn.linear_model.LogisticRegression`,
            `sklearn.linear_model.SGDClassifier`, or `sklearn.linear_model.LogisticRegressionCV` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    supported_loss = {"log", "modified_huber", "squared_hinge"}
    classes = [0] if not hasattr(operator.raw_operator, "classes_") else operator.raw_operator.classes_

    if not all(["int" in str(type(x)) for x in classes]):
        raise RuntimeError(
            "Hummingbird currently supports only integer labels for class labels. Please file an issue at https://github.com/microsoft/hummingbird."
        )

    coefficients = operator.raw_operator.coef_.transpose().astype("float32")
    intercepts = operator.raw_operator.intercept_.reshape(1, -1).astype("float32")

    multi_class = None
    loss = None
    if hasattr(operator.raw_operator, "multi_class"):
        if operator.raw_operator.multi_class == "ovr" or operator.raw_operator.solver in ["warn", "liblinear"]:
            multi_class = "ovr"
        elif operator.raw_operator.multi_class == "auto" and len(classes) == 2:
            multi_class = "ovr"
        else:
            multi_class = "multinomial"
    if hasattr(operator.raw_operator, "loss"):
        loss = operator.raw_operator.loss
        assert (
            loss in supported_loss
        ), "predict_proba for linear models currently only support {}. (Given {}). Please fill an issue at https://github.com/microsoft/hummingbird".format(
            supported_loss, loss
        )

    return LinearModel(operator, coefficients, intercepts, device, classes=classes, multi_class=multi_class, loss=loss)


def convert_sklearn_linear_regression_model(operator, device, extra_config):
    """
    Converter for `sklearn.linear_model.LinearRegression`

    Args:
        operator: An operator wrapping a `sklearn.linear_model.LinearRegression` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    coefficients = operator.raw_operator.coef_.transpose().astype("float32")
    if len(coefficients.shape) == 1:
        coefficients = coefficients.reshape(-1, 1)
    intercepts = operator.raw_operator.intercept_.reshape(1, -1).astype("float32")

    return LinearModel(operator, coefficients, intercepts, device, is_linear_regression=True)


register_converter("SklearnLinearRegression", convert_sklearn_linear_regression_model)
register_converter("SklearnLogisticRegression", convert_sklearn_linear_model)
register_converter("SklearnLinearSVC", convert_sklearn_linear_model)
register_converter("SklearnSGDClassifier", convert_sklearn_linear_model)
register_converter("SklearnLogisticRegressionCV", convert_sklearn_linear_model)
