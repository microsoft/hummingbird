# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Hummingbird main (converters) API.
"""
from copy import deepcopy
import numpy as np

from .operator_converters import constants
from ._parse import parse_sklearn_api_model, parse_onnx_api_model
from ._topology import convert as topology_converter
from ._utils import torch_installed, lightgbm_installed, xgboost_installed, onnx_runtime_installed
from .exceptions import MissingConverter, MissingBackend
from .supported import backends

# Invoke the registration of all our converters.
from . import operator_converters  # noqa

# Set up the converter dispatcher.
from .supported import xgb_operator_list  # noqa
from .supported import lgbm_operator_list  # noqa


def _is_onnx_model(model):
    """
    Function returning whether the input model is an ONNX model or not.
    """
    return type(model).__name__ == "ModelProto"


def _supported_backend_check(backend):
    """
    Function used to check whether the specified backend is supported or not.
    """
    if backend not in backends:
        raise MissingBackend("Backend: {}".format(backend))


def _supported_model_format_backend_mapping_check(model, backend):
    """
    Function used to check whether the specified backend/input model format is supported or not.
    """
    if _is_onnx_model(model):
        assert onnx_runtime_installed()
        import onnx

        if not backend == onnx.__name__:
            raise RuntimeError("Hummingbird currently support conversion of ONNX(-ML) models only into ONNX.")
    else:
        assert torch_installed()
        import torch

        if not backend == torch.__name__ and not backend == "py" + torch.__name__:
            raise RuntimeError(
                "Hummingbird currently support conversion of XGBoost / LightGBM / Sklearn models only into PyTorch."
            )


def _convert_sklearn(model, test_input=None, extra_config={}):
    """
    This function converts the specified *scikit-learn* (API) model into its [PyTorch] counterpart.
    The supported operators can be found at `hummingbird.ml.supported`.
    [PyTorch]: https://pytorch.org/

    Args:
        model: A scikit-learn model
        test_input: some input data used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> pytorch_model = _convert_sklearn(sklearn_model)

    Returns:
        A model implemented in *PyTorch*, which is equivalent to the input *scikit-learn* model
    """
    assert model is not None
    assert torch_installed(), "To use Hummingbird you need to install torch."

    import torch

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    # We modify the scikit learn model during translation.
    model = deepcopy(model)
    topology = parse_sklearn_api_model(model)

    # Convert the Topology object into a PyTorch model.
    hb_model = topology_converter(topology, torch.__name__, extra_config=extra_config)
    return hb_model


def _convert_lightgbm(model, test_input=None, extra_config={}):
    """
    This function is used to generate a [PyTorch] model from a given input [LightGBM] model.
    [LightGBM]: https://lightgbm.readthedocs.io/
    [PyTorch]: https://pytorch.org/

    Args:
        model: A LightGBM model (trained using the scikit-learn API)
        test_input: Some input data that will be used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> pytorch_model = _convert_lightgbm(lgbm_model)

    Returns:
        A *PyTorch* model which is equivalent to the input *LightGBM* model
    """
    assert (
        lightgbm_installed()
    ), "To convert LightGBM models you need to install LightGBM (or `pip install hummingbird-ml[extra]`)."

    return _convert_sklearn(model, test_input, extra_config)


def _convert_xgboost(model, test_input, extra_config={}):
    """
    This function is used to generate a [PyTorch] model from a given input [XGBoost] model.
    [PyTorch]: https://pytorch.org/
    [XGBoost]: https://xgboost.readthedocs.io/

    Args:
        model: A XGBoost model (trained using the scikit-learn API)
        test_input: Some input data used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> pytorch_model = _convert_xgboost(xgb_model, [], extra_config={"n_features":200})

    Returns:
        A *PyTorch* model which is equivalent to the input *XGBoost* model
    """
    assert (
        xgboost_installed()
    ), "To convert XGboost models you need to instal XGBoost (or `pip install hummingbird-ml[extra]`)."

    # XGBoostRegressor and Classifier have different APIs for extracting the number of features.
    # In the former case we need to infer them from the test_input.
    if constants.N_FEATURES not in extra_config:
        if "_features_count" in dir(model):
            extra_config[constants.N_FEATURES] = model._features_count
        elif test_input is not None:
            if type(test_input) is np.ndarray and len(test_input.shape) == 2:
                extra_config[constants.N_FEATURES] = test_input.shape[1]
            else:
                raise RuntimeError(
                    "XGBoost converter is not able to infer the number of input features.\
                        Apparently test_input is not an ndarray. \
                        Please fill an issue at https://github.com/microsoft/hummingbird/."
                )
        else:
            raise RuntimeError(
                "XGBoost converter is not able to infer the number of input features.\
                    Please pass some test_input to the converter."
            )
    return _convert_sklearn(model, test_input, extra_config)


def _convert_onnxml(model, test_input=None, extra_config={}):
    """
    This function converts the specified [ONNX-ML] model into its [ONNX] counterpart.
    The supported operators can be found at `hummingbird.ml.supported`.
    The ONNX-ML converter requires either a test_input of a the initial types set through the exta_config parameter.
    [ONNX-ML]: https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md
    [ONNX]: https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Args:
        model: A model containing ONNX-ML operators
        test_input: Some input data used to trace the model execution.
                    For the ONNX backend the test_input size is supposed to be as large as the expected batch size.
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        extra_config = {}
        extra_config[constans.ONNX_INITIAL_TYPES] =[('input', FloatTensorType([1, 20])]
        >>> onnx_model = _convert_onnxml(onnx_ml_model, None, extra_config)

    Returns:
        A model containing only *ONNX* operators. The mode is equivalent to the input *ONNX-ML* model
    """
    assert model is not None
    assert torch_installed(), "To use Hummingbird you need to install torch."
    assert (
        onnx_runtime_installed()
    ), "To use the onnxml converter you need to install onnxruntime (or `pip install hummingbird-ml[onnx]`)."

    import onnx

    # The conversion requires some test input for tracing.
    # Test inputs can be either provided or generate from the inital types.
    # Get the initial types if any.
    initial_types = None
    if constants.ONNX_INITIAL_TYPES in extra_config:
        initial_types = extra_config[constants.ONNX_INITIAL_TYPES]

    assert (
        test_input is not None and len(test_input) > 0
    ) or initial_types is not None, "Cannot generate test input data. Either pass some input data or the initial_types"

    # Generate some test input if necessary.
    if test_input is None:
        assert (
            not initial_types[0][1].shape is None
        ), "Cannot generate test input data. Initial_types do not contain shape information."
        assert len(initial_types[0][1].shape) == 2, "Hummingbird currently support only inputs with len(shape) == 2."

        from onnxconverter_common.data_types import FloatTensorType, Int32TensorType

        test_input = np.random.rand(initial_types[0][1].shape[0], initial_types[0][1].shape[1])
        extra_config[constants.N_FEATURES] = initial_types[0][1].shape[1]
        if type(initial_types[0][1]) is FloatTensorType:
            test_input = np.array(test_input, dtype=np.float32)
        elif type(initial_types[0][1]) is Int32TensorType:
            test_input = np.array(test_input, dtype=np.int32)
        else:
            raise RuntimeError(
                "Type {} not supported. Please fill an issue on https://github.com/microsoft/hummingbird/.".format(
                    type(initial_types[0][1])
                )
            )
    else:
        extra_config[constants.N_FEATURES] = np.array(test_input).shape[1]
    extra_config[constants.ONNX_TEST_INPUT] = test_input

    # Parse ONNX model as our internal data structure (i.e., Topology).
    topology = parse_onnx_api_model(model)

    # Convert the Topology object into a PyTorch model.
    hb_model = topology_converter(topology, onnx.__name__, extra_config=extra_config)
    return hb_model


def convert(model, backend, test_input=None, extra_config={}):
    """
    This function converts the specified input *model* into an implementation targeting *backend*.
    *Convert* supports [Sklearn], [LightGBM], [XGBoost] and [ONNX] models.
    For *LightGBM* and *XGBoost* currently only the Sklearn API is supported.
    For *Sklearn*, *LightGBM* and *XGBoost* currently only the *torch* backend is supported.
    For *ONNX* currently only the *onnx* backend is supported. For ONNX models, Hummingbird behave as a model
    rewriter converting [ONNX-ML] into [ONNX operators].
    The detailed list of models and backends can be found at `hummingbird.ml.supported`.
    [Sklearn]: https://scikit-learn.org/
    [LightGBM]: https://lightgbm.readthedocs.io/
    [XGBoost]: https://xgboost.readthedocs.io/
    [ONNX]: https://onnx.ai/
    [ONNX-ML]: https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md
    [ONNX operators]: https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Args:
        model: An input model
        backend: The target for the conversion
        test_input: some input data used to trace the model execution.
                    For the ONNX backend the test_input size is supposed to be as large as the expected batch size.
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> pytorch_model = convert(sklearn_model,`torch`)

    Returns:
        A model implemented in *backend*, which is equivalent to the input model
    """
    assert model is not None

    backend = backend.lower()
    _supported_backend_check(backend)
    _supported_model_format_backend_mapping_check(model, backend)

    if type(model) in xgb_operator_list:
        return _convert_xgboost(model, test_input, extra_config)

    if type(model) in lgbm_operator_list:
        return _convert_lightgbm(model, test_input, extra_config)

    if _is_onnx_model(model):
        return _convert_onnxml(model, test_input, extra_config)

    return _convert_sklearn(model, test_input, extra_config)
