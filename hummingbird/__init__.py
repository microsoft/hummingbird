# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Hummingbird is a compiler for translating traditional ML operators (e.g., tree-based models) and featurizers
(e.g., one-hot encoding) into tensor operations.
Through Hummingbird, DNN frameworks can be used for both optimizing and enabling seamless hardware acceleration of traditional ML.
"""

__version__ = "0.0.1"
__author__ = "Microsoft"
__producer__ = "hummingbird"
__producer_version__ = __version__
__domain__ = "microsoft.gsl"
__model_version__ = 0

# Register constants used for Hummingbird extra configs.
from . import supported_configurations as hummingbird_constants
from ._utils import _Constants

# Add constants in scope.
constants = _Constants(hummingbird_constants)

# Add the converters in the Hummingbird scope.
from .convert import to_pytorch_sklearn  # noqa: F401
from .convert import to_pytorch_lightgbm  # noqa: F401
from .convert import to_pytorch_xgboost  # noqa: F401

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)  # noqa: F401
from sklearn.tree import DecisionTreeClassifier  # noqa: F401

RandomForestClassifier.to_pytorch = to_pytorch_sklearn
RandomForestRegressor.to_pytorch = to_pytorch_sklearn
GradientBoostingClassifier.to_pytorch = to_pytorch_sklearn
ExtraTreesClassifier.to_pytorch = to_pytorch_sklearn
DecisionTreeClassifier.to_pytorch = to_pytorch_sklearn

# Pdoc stuff.
__pdoc__ = {}
__pdoc__["hummingbird._container"] = True
__pdoc__["hummingbird._parse"] = True
__pdoc__["hummingbird._supported_operators"] = True
__pdoc__["hummingbird._utils"] = True
