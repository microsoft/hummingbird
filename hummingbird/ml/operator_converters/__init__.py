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
from .onnx import onnx_operator  # noqa: E402
from .onnx import array_feature_extractor as onnx_afe  # noqa: E402, F811
from .onnx import linear as onnx_linear  # noqa: E402, F811
from .onnx import normalizer as onnx_normalizer  # noqa: E402, F811
from .onnx import one_hot_encoder as onnx_ohe  # noqa: E402, F811
from .onnx import scaler as onnx_scaler  # noqa: E402, F811
from .onnx import tree_ensemble  # noqa: E402
from .sklearn import array_feature_extractor as sklearn_afe  # noqa: E402
from .sklearn import binarizer  # noqa: E402
from .sklearn import decision_tree  # noqa: E402
from .sklearn import gbdt  # noqa: E402
from .sklearn import iforest  # noqa: E402
from .sklearn import linear as sklearn_linear  # noqa: E402
from .sklearn import normalizer as sklearn_normalizer  # noqa: E402
from .sklearn import one_hot_encoder as sklearn_ohe  # noqa: E402
from .sklearn import scaler as sklearn_scaler  # noqa: E402
from .sklearn import sv  # noqa: E402
from . import lightgbm  # noqa: E402
from . import xgb  # noqa: E402


__pdoc__ = {}
__pdoc__["hummingbird.operator_converters._array_feature_extractor_implementations"] = True
__pdoc__["hummingbird.operator_converters._gbdt_commons"] = True
__pdoc__["hummingbird.operator_converters._linear_implementations"] = True
__pdoc__["hummingbird.operator_converters._normalizer_implementations"] = True
__pdoc__["hummingbird.operator_converters._one_hot_encoder_implementations"] = True
__pdoc__["hummingbird.operator_converters._scaler_implementations"] = True
__pdoc__["hummingbird.operator_converters._tree_commons"] = True
__pdoc__["hummingbird.operator_converters._tree_implementations"] = True
