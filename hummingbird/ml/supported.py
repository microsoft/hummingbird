# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All operators, backends, and configurations settings supported in Hummingbird are registered here.

**Supported Backends**
PyTorch,
ONNX

**Supported Operators**
DecisionTreeClassifier,
RandomForestClassifier,
RandomForestRegressor,
GradientBoostingClassifier,
GradientBoostingRegressor,
ExtraTreesClassifier,
ExtraTreesRegressor,
LGBMClassifier,
LGBMRegressor,
XGBClassifier,
XGBRegressor,
TreeEnsembleClassifier,
TreeEnsembleRegressor

"""
from .exceptions import MissingConverter
from ._utils import torch_installed, sklearn_installed, lightgbm_installed, xgboost_installed, onnx_installed


def _build_sklearn_operator_list():
    """
    Put all suported Sklearn operators on a list.
    """
    if sklearn_installed():
        # Tree-based models.
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
        )
        from sklearn.tree import DecisionTreeClassifier

        return [
            # Tree-methods
            DecisionTreeClassifier,
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
        ]

    return None


def _build_xgboost_operator_list():
    """
    List all suported XGBoost (Sklearn API) operators.
    """
    if xgboost_installed():
        from xgboost import XGBClassifier, XGBRegressor

        return [XGBClassifier, XGBRegressor]

    return None


def _build_lightgbm_operator_list():
    """
    List all suported LightGBM (Sklearn API) operators.
    """
    if lightgbm_installed():
        from lightgbm import LGBMClassifier, LGBMRegressor

        return [LGBMClassifier, LGBMRegressor]

    return None


# Associate onnxml types with our operator names.
def _build_onnxml_operator_list():
    """
    List all suported ONNXML operators.
    """
    if onnx_installed():
        return [
            # Tree-based models.
            "TreeEnsembleClassifier",
            "TreeEnsembleRegressor",
        ]
    return None


def _build_backend_map():
    """
    The set of supported backends is defined here.
    """
    backends = set()

    if torch_installed():
        import torch

        backends.add(torch.__name__)

    if onnx_installed():
        import onnx

        backends.add(onnx.__name__)

    return backends


def _build_sklearn_api_operator_name_map():
    """
    Associate Sklearn with the operator class names.
    If two scikit-learn (API) models share a single name, it means they are equivalent in terms of conversion.
    """
    return {k: "Sklearn" + k.__name__ for k in sklearn_operator_list + xgb_operator_list + lgbm_operator_list if k is not None}


def _build_onnxml_api_operator_name_map():
    """
    Associate ONNXML with the operator class names.
    If two ONNXML models share a single name, it means they are equivalent in terms of conversion.
    """
    return {k: "ONNXML" + k for k in onnxml_operator_list if k is not None}


def get_sklearn_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *scikit-learn API* format.

    Args:
        model_type: A scikit-learn model object (e.g., RandomForestClassifier)
                    or an object with scikit-learn API (e.g., LightGBM)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework
    """
    if model_type not in sklearn_api_operator_name_map:
        raise MissingConverter("Unable to find converter for model type {}.".format(model_type))
    return sklearn_api_operator_name_map[model_type]


def get_onnxml_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *ONNX-ML API* format.

    Args:
        model_type: A ONNX-ML model object (e.g., TreeEnsembleClassifier)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework.
        None if the model_type is not supported
    """
    if model_type not in onnxml_api_operator_name_map:
        return None
    return onnxml_api_operator_name_map[model_type]


# Supported operators.
sklearn_operator_list = _build_sklearn_operator_list()
xgb_operator_list = _build_xgboost_operator_list()
lgbm_operator_list = _build_lightgbm_operator_list()
onnxml_operator_list = _build_onnxml_operator_list()

sklearn_api_operator_name_map = _build_sklearn_api_operator_name_map()
onnxml_api_operator_name_map = _build_onnxml_api_operator_name_map()


# Supported backends.
backends = _build_backend_map()


# Supported configurations settings accepted by Hummingbird are defined below.
N_FEATURES = "n_features"
"""Number of features expected in the input data."""

TREE_IMPLEMENTATION = "tree_implementation"
"""Which tree implementation to use. Values can be: gemm, tree-trav, perf_tree_trav."""
