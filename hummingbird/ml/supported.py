# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All operators, backends, and configurations settings supported in Hummingbird are registered here.

**Supported Backends**
PyTorch,
TorchScript,
ONNX,
TVM

**Supported Operators (scikit-learn)**
BernoulliNB,
Binarizer,
DecisionTreeClassifier,
DecisionTreeRegressor,
ExtraTreesClassifier,
ExtraTreesRegressor,
FastICA,
GaussianNB,
GradientBoostingClassifier,
GradientBoostingRegressor,
HistGradientBoostingClassifier,
HistGradientBoostingRegressor,
IsolationForest,
KernelPCA,
KBinsDiscretizer,
KNeighborsClassifier,
KNeighborsRegressor,
LabelEncoder,
LinearRegression,
LinearSVC,
LogisticRegression,
LogisticRegressionCV,
MaxAbsScaler,
MinMaxScaler,
MissingIndicator,
MLPClassifier,
MLPRegressor,
MultinomialNB,
Normalizer,
OneHotEncoder,
PCA,
PolynomialFeatures,
RandomForestClassifier,
RandomForestRegressor,
RobustScaler,
SelectKBest,
SelectPercentile,
SimpleImputer,
SGDClassifier,
StandardScaler,
TreeEnsembleClassifier,
TreeEnsembleRegressor,
TruncatedSVD,
VarianceThreshold,

**Supported Operators (LGBM)**
LGBMClassifier,
LGBMRanker,
LGBMRegressor,

**Supported Operators (XGB)**
XGBClassifier,
XGBRanker,
XGBRegressor,

**Supported Operators (ONNX-ML)**
"ArrayFeatureExtractor",
"Binarizer"
"Cast",
"Concat",
"FeatureVectorizer"
"LabelEncoder",
"LinearClassifier",
"LinearRegressor",
"OneHotEncoder",
"Normalizer",
"Reshape",
"Scaler",
"TreeEnsembleClassifier",
"TreeEnsembleRegressor",
"""
from collections import defaultdict

from .exceptions import MissingConverter
from ._utils import (
    torch_installed,
    sklearn_installed,
    lightgbm_installed,
    xgboost_installed,
    onnx_runtime_installed,
    tvm_installed,
    sparkml_installed,
)


def _build_sklearn_operator_list():
    """
    Put all suported Sklearn operators on a list.
    """
    if sklearn_installed():
        # Enable experimental to import HistGradientBoosting*
        from sklearn.experimental import enable_hist_gradient_boosting

        # Tree-based models
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            IsolationForest,
            RandomForestClassifier,
            RandomForestRegressor,
        )

        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Linear-based models
        from sklearn.linear_model import (
            LinearRegression,
            LogisticRegression,
            LogisticRegressionCV,
            SGDClassifier,
        )

        # SVM-based models
        from sklearn.svm import LinearSVC, SVC, NuSVC

        # Imputers
        from sklearn.impute import MissingIndicator, SimpleImputer

        # MLP Models
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        # Naive Bayes Models
        from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

        # Matrix decomposition transformers
        from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD

        # KNeighbors models
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neighbors import KNeighborsRegressor

        # Preprocessing
        from sklearn.preprocessing import (
            Binarizer,
            KBinsDiscretizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            OneHotEncoder,
            PolynomialFeatures,
            RobustScaler,
            StandardScaler,
        )

        try:
            from sklearn.preprocessing import Imputer
        except ImportError:
            # Imputer was deprecate in sklearn >= 0.22
            Imputer = None

        # Features
        from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold

        supported_ops = [
            # Trees
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            IsolationForest,
            OneHotEncoder,
            RandomForestClassifier,
            RandomForestRegressor,
            # Linear-methods
            LinearRegression,
            LinearSVC,
            LogisticRegression,
            LogisticRegressionCV,
            SGDClassifier,
            # Other models
            BernoulliNB,
            GaussianNB,
            KNeighborsClassifier,
            KNeighborsRegressor,
            MLPClassifier,
            MLPRegressor,
            MultinomialNB,
            # SVM
            NuSVC,
            SVC,
            # Imputers
            Imputer,
            MissingIndicator,
            SimpleImputer,
            # Preprocessing
            Binarizer,
            KBinsDiscretizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            PolynomialFeatures,
            RobustScaler,
            StandardScaler,
            # Matrix Decomposition
            FastICA,
            KernelPCA,
            PCA,
            TruncatedSVD,
            # Feature selection
            SelectKBest,
            SelectPercentile,
            VarianceThreshold,
        ]

        # Remove all deprecated operators given the sklearn version. E.g., Imputer for sklearn > 0.21.3.
        return [x for x in supported_ops if x is not None]

    return []


def _build_sparkml_operator_list():
    """
    List all suported SparkML operators.
    """
    if sparkml_installed():
        from pyspark.ml.classification import LogisticRegressionModel
        from pyspark.ml.feature import Bucketizer, VectorAssembler

        supported_ops = [
            # Featurizers
            Bucketizer,
            VectorAssembler,
            # Linear Models
            LogisticRegressionModel,
        ]

        return supported_ops

    return []


def _build_xgboost_operator_list():
    """
    List all suported XGBoost (Sklearn API) operators.
    """
    if xgboost_installed():
        from xgboost import XGBClassifier, XGBRanker, XGBRegressor

        return [XGBClassifier, XGBRanker, XGBRegressor]

    return []


def _build_lightgbm_operator_list():
    """
    List all suported LightGBM (Sklearn API) operators.
    """
    if lightgbm_installed():
        from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

        return [LGBMClassifier, LGBMRanker, LGBMRegressor]

    return []


# Associate onnxml types with our operator names.
def _build_onnxml_operator_list():
    """
    List all suported ONNXML operators.
    """
    if onnx_runtime_installed():
        return [
            # Linear-based models
            "LinearClassifier",
            "LinearRegressor",
            # ONNX operators.
            "Cast",
            "Concat",
            "Reshape",
            # Preprocessing
            "ArrayFeatureExtractor",
            "Binarizer",
            "FeatureVectorizer",
            "LabelEncoder",
            "OneHotEncoder",
            "Normalizer",
            "Scaler",
            # Tree-based models
            "TreeEnsembleClassifier",
            "TreeEnsembleRegressor",
        ]
    return []


def _build_backend_map():
    """
    The set of supported backends is defined here.
    """
    backends = defaultdict(lambda: None)

    if torch_installed():
        import torch

        backends[torch.__name__] = torch.__name__
        backends["py" + torch.__name__] = torch.__name__  # For compatibility with earlier versions.

        backends[torch.jit.__name__] = torch.jit.__name__
        backends["torchscript"] = torch.jit.__name__  # For reference outside Hummingbird.

    if onnx_runtime_installed():
        import onnx

        backends[onnx.__name__] = onnx.__name__

    if tvm_installed():
        import tvm

        backends[tvm.__name__] = tvm.__name__

    return backends


def _build_sklearn_api_operator_name_map():
    """
    Associate Sklearn with the operator class names.
    If two scikit-learn (API) models share a single name, it means they are equivalent in terms of conversion.
    """
    # Pipeline ops. These are ops injected by the parser not "real" sklearn operators.
    pipeline_operator_list = [
        "ArrayFeatureExtractor",
        "Concat",
        "Multiply",
    ]

    return {
        k: "Sklearn" + k.__name__ if hasattr(k, "__name__") else k
        for k in sklearn_operator_list + pipeline_operator_list + xgb_operator_list + lgbm_operator_list
    }


def _build_onnxml_api_operator_name_map():
    """
    Associate ONNXML with the operator class names.
    If two ONNXML models share a single name, it means they are equivalent in terms of conversion.
    """
    return {k: "ONNXML" + k for k in onnxml_operator_list if k is not None}


def _build_sparkml_api_operator_name_map():
    """
    Associate Spark-ML with the operator class names.
    If two Spark-ML models share a single name, it means they are equivalent in terms of conversion.
    """
    return {k: "SparkML" + k.__name__ if hasattr(k, "__name__") else k for k in sparkml_operator_list if k is not None}


def get_sklearn_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *scikit-learn API* format.

    Args:
        model_type: A scikit-learn model object (e.g., RandomForestClassifier)
                    or an object with scikit-learn API (e.g., LightGBM)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework
    """
    if model_type not in sklearn_api_operator_name_map:
        raise MissingConverter("Unable to find converter for model type {}.".format(model_type))
    return sklearn_api_operator_name_map[model_type]


def get_onnxml_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *ONNX-ML API* format.

    Args:
        model_type: A ONNX-ML model object (e.g., TreeEnsembleClassifier)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework.
        None if the model_type is not supported
    """
    if model_type not in onnxml_api_operator_name_map:
        return None
    return onnxml_api_operator_name_map[model_type]


def get_sparkml_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *Spark-ML API* format.

    Args:
        model_type: A Spark-ML model object (e.g., LogisticRegression)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework.
        None if the model_type is not supported
    """
    if model_type not in sparkml_api_operator_name_map:
        return None
    return sparkml_api_operator_name_map[model_type]


# Supported operators.
sklearn_operator_list = _build_sklearn_operator_list()
xgb_operator_list = _build_xgboost_operator_list()
lgbm_operator_list = _build_lightgbm_operator_list()
onnxml_operator_list = _build_onnxml_operator_list()
sparkml_operator_list = _build_sparkml_operator_list()

sklearn_api_operator_name_map = _build_sklearn_api_operator_name_map()
onnxml_api_operator_name_map = _build_onnxml_api_operator_name_map()
sparkml_api_operator_name_map = _build_sparkml_api_operator_name_map()

# Supported backends.
backends = _build_backend_map()

# Supported configurations settings accepted by Hummingbird are defined below.
# Please check `test.test_extra_conf.py` for examples on how to use these.
TREE_IMPLEMENTATION = "tree_implementation"
"""Which tree implementation to use. Values can be: gemm, tree-trav, perf_tree_trav."""

ONNX_OUTPUT_MODEL_NAME = "onnx_model_name"
"""For ONNX models we can set the name of the output model."""

ONNX_TARGET_OPSET = "onnx_target_opset"
"""For ONNX models we can set the target opset to use. 11 by default."""

TVM_MAX_FUSE_DEPTH = "tvm_max_fuse_depth"
"""For TVM we can fix the number of operations that will be fused.
If not set, compilation may take forever (https://github.com/microsoft/hummingbird/issues/232).
By default Hummingbird uses a max_fuse_depth of 50, but this can be override using this parameter."""

TVM_PAD_INPUT = "tvm_pad_prediction_inputs"
"""TVM statically compiles models, therefore each input shape is fixed.
However, at prediction time, we can have inputs with different batch size.
This option allows to pad the inputs on the batch dimension with zeros. Note that enabling this option may considerably hurt performance"""

INPUT_NAMES = "input_names"
"""Set the names of the inputs. Assume that the numbers of inputs_names is equal to the number of inputs."""

OUTPUT_NAMES = "output_names"
"""Set the names of the outputs."""

CONTAINER = "container"
"""Boolean used to chose whether to return the container for Sklearn API or just the model."""

N_THREADS = "n_threads"
"""Select how many threads to use for scoring. This paremeter will set the number of intra-op threads.
Inter-op threads are by default set to 1 in Hummingbird. Check `tests.test_extra_conf.py` for usage examples."""

BATCH_SIZE = "batch_size"
"""Select whether to partition the input dataset at inference time in N batch_size partitions."""

REMAINDER_SIZE = "remainder_size"
"""Determines the number of rows that an auxiliary remainder model can accept."""

MAX_STRING_LENGTH = "max_string_length"
"""Maximum expected length for string features. By deafult this value is set using the training information."""
