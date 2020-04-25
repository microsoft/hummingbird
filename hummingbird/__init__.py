# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Main entry point to the Hummingbird converters.
"""

__version__ = "0.0.1"
__author__ = "Microsoft"
__producer__ = "hummingbird"
__producer_version__ = __version__
__domain__ = "microsoft.gsl"
__model_version__ = 0

# Register constants used for Hummingbird extra configs.
from . import supported_configurations as hummingbird_constants
from .utils import _Constants

# Add constants in scope.
constants = _Constants(hummingbird_constants)

# Add the converters in the Hummingbird scope.
from .convert import convert_sklearn  # noqa: F401
from .convert import convert_lightgbm  # noqa: F401
from .convert import convert_xgboost  # noqa: F401

__pdoc__ = {}
__pdoc__['hummingbird._container'] = True
__pdoc__['hummingbird._parse'] = True
__pdoc__['hummingbird._supported_operators'] = True
