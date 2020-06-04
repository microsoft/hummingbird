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

from onnxconverter_common.registration import get_converter

from ._container import PyTorchBackendModel
from .exceptions import MissingConverter, MissingBackend
from ._parse import parse_sklearn_api_model
from .supported import backend_map
from ._utils import torch_installed, lightgbm_installed, xgboost_installed
from . import constants

# Invoke the registration of all our converters.
from . import operator_converters  # noqa

# Set up the converter dispatcher.
from .supported import xgb_operator_list  # noqa
from .supported import lgbm_operator_list  # noqa


def _supported_backend_check(backend):
    """
    Function used to check whether the specified backend is supported or not.
    """
    if not backend.lower() in backend_map:
        raise MissingBackend("Backend: {}".format(backend))


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

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    # We modify the scikit learn model during optimizations.
    model = deepcopy(model)
    topology = parse_sklearn_api_model(model)

    # Convert the Topology object into a PyTorch model.
    hb_model = _convert_topology(topology, extra_config=extra_config)
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
    assert lightgbm_installed(), "To convert LightGBM models you need to instal LightGBM."

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
    assert xgboost_installed(), "To convert XGboost models you need to instal XGBoost."

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


def _convert_topology(topology, device=None, extra_config={}):
    """
    This function is used to convert a `onnxconverter_common.topology.Topology` object into a *PyTorch* model.

    Args:
        topology: The `onnxconverter_common.topology.Topology` object that will be converted into Pytorch
        device: Which device the translated model will be run on
        extra_config: Extra configurations to be used by individual operator converters

    Returns:
        A *PyTorch* model
    """
    assert topology is not None, "Cannot convert a Topology object of type None."

    operator_map = {}

    for operator in topology.topological_operator_iterator():
        try:
            converter = get_converter(operator.type)
            operator_map[operator.full_name] = converter(operator, device, extra_config)
        except ValueError:
            raise MissingConverter(
                "Unable to find converter for {} type {} with extra config: {}.".format(
                    operator.type, type(getattr(operator, "raw_model", None)), extra_config
                )
            )
        except Exception as e:
            raise e

    pytorch_model = PyTorchBackendModel(
        topology.raw_model.input_names, topology.raw_model.output_names, operator_map, topology, extra_config
    ).eval()

    if device is not None:
        pytorch_model = pytorch_model.to(device)
    return pytorch_model


def convert(model, backend, test_input=None, extra_config={}):
    """
    This function converts the specified input *model* into an implementation targeting *backend*.
    *Convert* supports [Sklearn], [LightGBM] and [XGBoost] models.
    For *LightGBM* and *XGBoost* currently only the Sklearn API is supported.
    The detailed list of models and backends can be found at `hummingbird.ml.supported`.
    [Sklearn]: https://scikit-learn.org/
    [LightGBM]: https://lightgbm.readthedocs.io/
    [XGBoost]: https://xgboost.readthedocs.io/

    Args:
        model: An input model
        backend: The target for the conversion
        test_input: some input data used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> pytorch_model = convert(sklearn_model,`pytorch`)

    Returns:
        A model implemented in *backend*, which is equivalent to the input model
    """
    _supported_backend_check(backend)

    if type(model) in xgb_operator_list:
        return _convert_xgboost(model, test_input, extra_config)

    if type(model) in lgbm_operator_list:
        return _convert_lightgbm(model, test_input, extra_config)

    return _convert_sklearn(model, test_input, extra_config)
