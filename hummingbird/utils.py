# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Collection of utility functions used throughout Hummingbird.
"""
from distutils.version import LooseVersion
import warnings


def torch_installed():
    """
    Checks that *PyTorch* is available.
    """
    try:
        import torch

        return True
    except ImportError:
        return False


def lightgbm_installed():
    """
    Checks that *LightGBM* is available.
    """
    try:
        import lightgbm

        return True
    except ImportError:
        return False


def xgboost_installed():
    """
    Checks that *XGBoost* is available.
    """
    try:
        import xgboost
    except ImportError:
        return False
    from xgboost.core import _LIB

    try:
        _LIB.XGBoosterDumpModelEx
    except AttributeError:
        # The version is not recent enough even though it is version 0.6.
        # You need to install xgboost from github and not from pypi.
        return False
    from xgboost import __version__

    vers = LooseVersion(__version__)
    allowed_min = LooseVersion("0.70")
    allowed_max = LooseVersion("0.90")
    if vers < allowed_min or vers > allowed_max:
        warnings.warn("The converter works for xgboost >= 0.7 and <= 0.9. Different versions might not.")
    return True
