# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All operators supported in Hummingbird are registred here.
"""
from .exceptions import MissingConverter

# Tree-based models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def _build_sklearn_api_operator_name_map():
    """
    Associate Sklearn with the operator class names.
    If two scikit-learn (API) models share a single name, it means they are equivalent in terms of conversion.
    """
    res = {
        k: "Sklearn" + k.__name__
        for k in [
            # Tree-methods
            DecisionTreeClassifier,
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            ExtraTreesClassifier,
            XGBClassifier,
            XGBRegressor,
            LGBMClassifier,
            LGBMRegressor,
        ]
        if k is not None
    }

    return res


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


# Registered converters.
sklearn_api_operator_name_map = _build_sklearn_api_operator_name_map()
