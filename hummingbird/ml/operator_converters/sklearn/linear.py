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
from sklearn._loss.link import LogLink

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

    supported_loss = {"log_loss", "modified_huber", "squared_hinge"}
    classes = [0] if not hasattr(operator.raw_operator, "classes_") else operator.raw_operator.classes_

    if not all(["int" in str(type(x)) for x in classes]):
        raise RuntimeError(
            "Hummingbird currently supports only integer labels for class labels. Please file an issue at https://github.com/microsoft/hummingbird."
        )

    coefficients = operator.raw_operator.coef_.transpose().astype("float32")

    intercepts = operator.raw_operator.intercept_
    if np.ndim(intercepts) == 0:
        intercepts = np.array(intercepts, dtype="float32")
    else:
        intercepts = intercepts.reshape(1, -1).astype("float32")

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
    Converter for `sklearn.linear_model.LinearRegression`, `sklearn.linear_model.Lasso`, `sklearn.linear_model.ElasticNet`, `sklearn.linear_model.Ridge`, `sklearn.svm.LinearSVR` and `sklearn.linear_model.RidgeCV`

    Args:
        operator: An operator wrapping a `sklearn.linear_model.LinearRegression`, `sklearn.svm.LinearSVR`
            or `sklearn.linear_model.RidgeCV` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    loss = None
    coefficients = operator.raw_operator.coef_.transpose().astype("float32")
    if len(coefficients.shape) == 1:
        coefficients = coefficients.reshape(-1, 1)

    intercepts = operator.raw_operator.intercept_
    if np.ndim(intercepts) == 0:
        intercepts = np.array(intercepts, dtype="float32")
    else:
        intercepts = intercepts.reshape(1, -1).astype("float32")

    if hasattr(operator.raw_operator, "_base_loss") and type(operator.raw_operator._base_loss.link) == LogLink:
        loss = "log"

    return LinearModel(operator, coefficients, intercepts, device, loss=loss, is_linear_regression=True)


register_converter("SklearnLinearRegression", convert_sklearn_linear_regression_model)
register_converter("SklearnLasso", convert_sklearn_linear_regression_model)
register_converter("SklearnElasticNet", convert_sklearn_linear_regression_model)
register_converter("SklearnRidge", convert_sklearn_linear_regression_model)
register_converter("SklearnLogisticRegression", convert_sklearn_linear_model)
register_converter("SklearnLinearSVC", convert_sklearn_linear_model)
register_converter("SklearnLinearSVR", convert_sklearn_linear_regression_model)
register_converter("SklearnSGDClassifier", convert_sklearn_linear_model)
register_converter("SklearnLogisticRegressionCV", convert_sklearn_linear_model)
register_converter("SklearnRidgeCV", convert_sklearn_linear_regression_model)
register_converter("SklearnTweedieRegressor", convert_sklearn_linear_regression_model)
register_converter("SklearnPoissonRegressor", convert_sklearn_linear_regression_model)
register_converter("SklearnGammaRegressor", convert_sklearn_linear_regression_model)
