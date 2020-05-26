# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Hummingbird.ml is a compiler for translating traditional ML operators (e.g., tree-based models) and featurizers
(e.g., one-hot encoding) into tensor operations.
Through Hummingbird, DNN frameworks can be used for both optimizing and enabling seamless hardware acceleration of traditional ML.
"""


# Register constants used for Hummingbird extra configs.
from . import supported as hummingbird_constants
from ._utils import _Constants

# Add constants in scope.
constants = _Constants(hummingbird_constants)


# Add the converters in the Hummingbird scope.
from .convert import convert  # noqa: F401, E402


# Pdoc stuff.
__pdoc__ = {}
__pdoc__["hummingbird._container"] = True
__pdoc__["hummingbird._parse"] = True
__pdoc__["hummingbird._supported_operators"] = True
__pdoc__["hummingbird._utils"] = True
