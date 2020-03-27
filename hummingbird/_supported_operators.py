# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


# Tree-based models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier

# Operators for preprocessing and feature engineering
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# Associate scikit-learn types with our operator names. If two
# scikit-learn models share a single name, it means their are
# equivalent in terms of conversion.
def build_sklearn_operator_name_map():
    res = {
        k: "Sklearn" + k.__name__
        for k in [
            # Classifiers: Trees
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


def _get_sklearn_operator_name(model_type):
    """
    Get operator name of the input argument

    :param model_type:  A scikit-learn object (e.g., SGDClassifier
                        and Binarizer)
    :return: A string which stands for the type of the input model in
             our conversion framework
    """
    if model_type not in sklearn_operator_name_map:
        # "No proper operator name found, it means a local operator.
        return None
    return sklearn_operator_name_map[model_type]


# registered converters
sklearn_operator_name_map = build_sklearn_operator_name_map()
