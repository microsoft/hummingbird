# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Main entry point to the converter from the *scikit-learn* to *PyTorch*.
"""

__version__ = "0.0.1"
__author__ = "Microsoft"
__producer__ = "hummingbird"
__producer_version__ = __version__
__domain__ = "microsoft.gsl"
__model_version__ = 0

from .convert import convert_sklearn  # noqa: F401
