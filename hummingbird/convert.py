# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from copy import deepcopy
import numpy as np

from onnxconverter_common.registration import get_converter

from ._container import PyTorchBackendModel
from .exceptions import MissingConverter
from ._parse import parse_sklearn_api_model
from .utils import torch_installed, lightgbm_installed, xgboost_installed

# Invoke the registration of all our converters.
from . import operator_converters  # noqa


def convert_sklearn(model, test_input=None, extra_config={}):
    """
    This function converts the specified *scikit-learn* model into its *PyTorch* counterpart.
    The supported operators can be found at :func:`supported_operators <hummingbird._supported_opeartors>`.

    :param model: A scikit-learn model
    :param test_input: some input data used to trace the model execution
    :param extra_config: Extra configurations to be used by the individual operator converters

    :return: A model implemented in PyTorch, which is equivalent to the input scikit-learn model
    """
    assert torch_installed(), "To use Hummingbird you need to install torch."

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    # We modify the scikit learn model during optimizations.
    model = deepcopy(model)
    topology = parse_sklearn_api_model(model)

    # Convert the Topology object into a PyTorch model.
    hb_model = _convert_topology(topology, extra_config=extra_config)
    return hb_model


def convert_lightgbm(model, test_input=None, extra_config={}):
    """
    This function is used to generate a *PyTorch* model from a given input *LightGBM* model

    :param model: A LightGBM model (trained using the scikit-learn API)
    :param test_input: Some input data that will be used to trace the model execution
    :param extra_config: Extra configurations to be used by the individual operator converters

    :return: A PyTorch model which is equivalent to the input LightGBM model
    """
    assert lightgbm_installed(), "To convert LightGBM models you need to instal LightGBM."

    return convert_sklearn(model, test_input, extra_config)


def convert_xgboost(model, test_input, extra_config={}):
    """
    This function is used to generate a *PyTorch* model from a given input *XGBoost* model

    :param model: A XGBoost model (trained using the scikit-learn API)
    :param test_input: Some input data used to trace the model execution
    :param extra_config: Extra configurations to be used by the individual operator converters

    :return: A PyTorch model which is equivalent to the input XGBoost model
    """
    assert xgboost_installed(), "To convert XGboost models you need to instal XGBoost."

    # XGBoostRegressor and Classifier have different APIs for extracting the number of features.
    # In the former case we need to infer them from the test_input.
    if "_features_count" in dir(model):
        extra_config["n_features"] = model._features_count
    elif test_input is not None:
        if type(test_input) is np.ndarray and len(test_input.shape) == 2:
            extra_config["n_features"] = test_input.shape[1]
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
    return convert_sklearn(model, test_input, extra_config)


def _convert_topology(topology, device=None, extra_config={}):
    """
    This function is used to convert a ONNX *Topology* object into a *PyTorch* model.

    :param topology: The Topology object that will be converted into Pytorch
    :param device: Which device the translated model will be run on
    :param extra_config: Extra configurations to be used by individual operator converters

    :return: a PyTorch model
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
