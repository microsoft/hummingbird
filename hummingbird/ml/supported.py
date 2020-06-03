# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All operators, backends, and configurations settings supported in Hummingbird are registered here.

**Supported Backends**
PyTorch

**Supported Operators**
DecisionTreeClassifier,
DecisionTreeRegressor,
RandomForestClassifier,
RandomForestRegressor,
GradientBoostingClassifier,
GradientBoostingRegressor,
ExtraTreesClassifier,
ExtraTreesRegressor,
LGBMClassifier,
LGBMRegressor,
XGBClassifier,
XGBRegressor

"""
from .exceptions import MissingConverter
from ._utils import sklearn_installed, lightgbm_installed, xgboost_installed

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def _build_sklearn_operator_list():
    """
    Put all suported Sklearn operators on a list.
    """
    if sklearn_installed():
        # Tree-based models
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
        )
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        return [
            # Tree-methods
            DecisionTreeClassifier,
            DecisionTreeRegressor,
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
        return [XGBClassifier, XGBRegressor]

    return None


def _build_lightgbm_operator_list():
    """
    List all suported LightGBM (Sklearn API) operators.
    """
    if lightgbm_installed:
        return [LGBMClassifier, LGBMRegressor]

    return None


def _build_backend_map():
    """
    The set of supported backends is defined here.
    """
    return {"pytorch"}


def _build_sklearn_api_operator_name_map():
    """
    Associate Sklearn with the operator class names.
    If two scikit-learn (API) models share a single name, it means they are equivalent in terms of conversion.
    """
    return {k: "Sklearn" + k.__name__ for k in sklearn_operator_list + xgb_operator_list + lgbm_operator_list if k is not None}


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


# Supported operators.
sklearn_operator_list = _build_sklearn_operator_list()
xgb_operator_list = _build_xgboost_operator_list()
lgbm_operator_list = _build_lightgbm_operator_list()
sklearn_api_operator_name_map = _build_sklearn_api_operator_name_map()


# Supported backends.
backend_map = _build_backend_map()


# Supported configurations settings accepted by Hummingbird are defined below.
N_FEATURES = "n_features"
"""Number of features expected in the input data."""

TREE_IMPLEMENTATION = "tree_implementation"
"""Which tree implementation to use. Values can be: gemm, tree-trav, perf_tree_trav."""
