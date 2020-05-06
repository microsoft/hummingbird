# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All operators converters are stored under this package.
"""

# Register constants used within Hummingbird converters.
from . import constants as converter_constants
from .. import supported as hummingbird_constants
from .._utils import _Constants

# Add constants in scope.
constants = _Constants(converter_constants, hummingbird_constants)

# To register a converter for scikit-learn API operators, import associated modules here.
from . import gbdt  # noqa: E402
from . import lightgbm  # noqa: E402
from . import decision_tree  # noqa: E402
from . import xgb  # noqa: E402

__pdoc__ = {}
__pdoc__["hummingbird.operator_converters._gbdt_commons"] = True
__pdoc__["hummingbird.operator_converters._tree_commons"] = True
__pdoc__["hummingbird.operator_converters._tree_implementations"] = True
