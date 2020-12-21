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
from .onnx import binarizer as onnx_binarizer  # noqa: E402, F811
from .onnx import feature_vectorizer  # noqa: E402
from .onnx import label_encoder as onnx_label_encoder  # noqa: E402, F811
from .onnx import linear as onnx_linear  # noqa: E402, F811
from .onnx import normalizer as onnx_normalizer  # noqa: E402, F811
from .onnx import one_hot_encoder as onnx_ohe  # noqa: E402, F811
from .onnx import scaler as onnx_scaler  # noqa: E402, F811
from .onnx import tree_ensemble  # noqa: E402
from .sklearn import array_feature_extractor as sklearn_afe  # noqa: E402
from .sklearn import decision_tree  # noqa: E402
from .sklearn import decomposition  # noqa: E402
from .sklearn import discretizer as sklearn_discretizer  # noqa: E402
from .sklearn import gbdt  # noqa: E402
from .sklearn import iforest  # noqa: E402
from .sklearn import imputer  # noqa: E402
from .sklearn import kneighbors  # noqa: E402
from .sklearn import label_encoder  # noqa: E402
from .sklearn import linear as sklearn_linear  # noqa: E402
from .sklearn import mlp as sklearn_mlp  # noqa: E402
from .sklearn import nb as sklearn_nb  # noqa: E402
from .sklearn import normalizer as sklearn_normalizer  # noqa: E402
from .sklearn import one_hot_encoder as sklearn_ohe  # noqa: E402
from .sklearn import pipeline  # noqa: E402
from .sklearn import poly_features  # noqa: E402
from .sklearn import scaler as sklearn_scaler  # noqa: E402
from .sklearn import sv  # noqa: E402
from . import lightgbm  # noqa: E402
from . import xgb  # noqa: E402
from .sparkml import discretizer  # noqa: E402
from .sparkml import linear  # noqa: E402
from .sparkml import vector_assembler  # noqa: E402
