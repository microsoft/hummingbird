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
from . import supported as hummingbird_constants
from ._utils import _Constants

# Add constants in scope.
constants = _Constants(hummingbird_constants)


# Add the converters in the Hummingbird scope.
from .convert import _to_pytorch_sklearn  # noqa: F401
from .convert import _to_pytorch_lightgbm  # noqa: F401
from .convert import _to_pytorch_xgboost  # noqa: F401


# Set up the converter dispatcher. 
from .supported import sklearn_operator_list  # noqa: F401
from .supported import xgb_operator_list  # noqa: F401
from .supported import lgbm_operator_list  # noqa: F401


for operator in sklearn_operator_list:
    if operator is not None:
        operator.to = _to_pytorch_sklearn

for operator in xgb_operator_list:
    if operator is not None:
        operator.to = _to_pytorch_xgboost

for operator in lgbm_operator_list:
    if operator is not None:
        operator.to = _to_pytorch_lightgbm


# Pdoc stuff.
__pdoc__ = {}
__pdoc__["hummingbird._container"] = True
__pdoc__["hummingbird._parse"] = True
__pdoc__["hummingbird._supported_operators"] = True
__pdoc__["hummingbird._utils"] = True
