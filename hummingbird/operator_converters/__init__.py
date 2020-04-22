# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register a converter for scikit-learn API operators, import associated modules here.
from . import gbdt
from . import lightgbm
from . import decision_tree
from . import xgb

# Register constants used within Hummingbird converters.
import sys
from . import constants as converters_constants
from ..exceptions import ConstantError


class _Constants(object):
    """
    Class enabling the proper definition on constants.
    """

    def __init__(self):
        for constant in dir(converters_constants):
            if constant.isupper():
                setattr(self, constant, getattr(converters_constants, constant))

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ConstantError("Overwriting a constant is not allowed {}".format(name))
        self.__dict__[name] = value


# Add constants in scope.
constants = _Constants()
