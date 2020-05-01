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
from onnxconverter_common.optimizer import LinkedNode

from ._container import PyTorchBackendModel
from .exceptions import MissingConverter
from ._parse import parse_sklearn_api_model
from ._utils import torch_installed, lightgbm_installed, xgboost_installed
from . import constants

# Invoke the registration of all our converters.
from . import operator_converters  # noqa


def convert_sklearn(model, test_input=None, extra_config={}):
    """
    This function converts the specified [scikit-learn] model into its [PyTorch] counterpart.
    The supported operators can be found at `hummingbird._supported_operators`.
    [scikit-learn]: https://scikit-learn.org/
    [PyTorch]: https://pytorch.org/

    Args:
        model: A scikit-learn model
        test_input: some input data used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.supported_configurations`

    Examples:
        >>> pytorch_model = convert_sklearn(sklearn_model)

    Returns:
        A model implemented in *PyTorch*, which is equivalent to the input *scikit-learn* model
    """
    assert torch_installed(), "To use Hummingbird you need to install torch."

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    # We modify the scikit learn model during optimizations.
    model = deepcopy(model)
    topology = parse_sklearn_api_model(model)

    # Convert the Topology object into a PyTorch model.
    hb_model = _convert_topology(topology, extra_config=extra_config)
    return hb_model


def convert_onnxml(model, initial_types=None, input_names=None, output_names=None, test_data=None, extra_config={}):
    """
    This function converts the specified [ONNX-ML] model into its [ONNX] counterpart.
    The supported operators can be found at `hummingbird._supported_operators`.
    [ONNX-ML]: https://scikit-learn.org/
    [ONNX]: https://pytorch.org/

    Args:
        model: A model containing ONNX-ML operators
        initial_types: a python list where each element is a tuple of a input name and a `onnxmltools.convert.common.data_types`
        input_names: a python list containig input names. Should be a subset of the input variables in the input ONNX-ML model.
        output_names: a python list containing the names output expected from the translated model.
                      Should be a subset of the output variables in the input ONNX-ML model.
        test_data: some input data used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.supported_configurations`

    Examples:
        >>> pytorch_model = convert_onnxml(onnx_ml_model)

    Returns:
        A model containing only *ONNX* operators. The mode is equivalent to the input *ONNX-ML* model
    """
    assert model is not None
    assert torch_installed(), "To use Hummingbird you need to install torch."
    assert (
        test_data is not None or initial_types is not None
    ), "Cannot generate test input data. Either pass some input data or the initial_types"

    # Parse an ONNX-ML model into our internal data structure (i.e., LinkedNode)
    input_names = input_names if input_names is not None else [in_.name for in_ in model.input]
    inputs = [in_ for in_ in model.input if in_.name in input_names]

    assert len(inputs) > 0, "Provided input name does not match with any model's input."
    assert len(inputs) == 1, "Hummingbird currently do not support models with more than 1 input."
    assert initial_types is None or len(initial_types) == 1, "len(initial_types) {} differs from len(inputs) {}.".format(
        len(initial_types), len(inputs)
    )

    if output_names is None:
        output_names = [] if model.output is None else [o_.name for o_ in model.output]

    if test_data is None:
        assert (
            not initial_types[0][1].shape is None
        ), "Cannot generate test input data. Initial_types do not contain shape information."
        assert len(initial_types[0][1].shape) == 2, "Hummingbird currently support only inputs with len(shape) == 2."

        from onnxmltools.convert.common.data_types import FloatTensorType, Int32TensorType

        test_data = np.random.rand(initial_types[0][1].shape[0], initial_types[0][1].shape[1])
        if type(initial_types[0][1]) is FloatTensorType:
            test_data = np.array(test_data, dtype=np.float32)
        elif type(initial_types[0][1]) is Int32TensorType:
            test_data = np.array(test_data, dtype=np.int32)
        else:
            raise RuntimeError(
                "Type {} not supported. Please fill an issue on https://github.com/microsoft/hummingbird/.".format(
                    type(initial_types[0][1])
                )
            )

    ir_model = LinkedNode.build_from_onnx(
        model.node, [], [in_.name for in_ in model.input], output_names, [init_ for init_ in model.initializer]
    )

    # Convert the input model object into ONNX. The outcome is an ONNX model.
    # onnx_model = convert_ir_model(ir_model, inputs, model.initializer, output_names, test_data, extra_config)
    # return onnx_model
    return ir_model


def convert_lightgbm(model, test_input=None, extra_config={}):
    """
    This function is used to generate a [PyTorch] model from a given input [LightGBM] model.
    [LightGBM]: https://lightgbm.readthedocs.io/
    [PyTorch]: https://pytorch.org/

    Args:
        model: A LightGBM model (trained using the scikit-learn API)
        test_input: Some input data that will be used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.supported_configurations`

    Examples:
        >>> pytorch_model = convert_lightgbm(lgbm_model)

    Returns:
        A *PyTorch* model which is equivalent to the input *LightGBM* model
    """
    assert lightgbm_installed(), "To convert LightGBM models you need to instal LightGBM."

    return convert_sklearn(model, test_input, extra_config)


def convert_xgboost(model, test_input, extra_config={}):
    """
    This function is used to generate a [PyTorch] model from a given input [XGBoost] model.
    [PyTorch]: https://pytorch.org/
    [XGBoost]: https://xgboost.readthedocs.io/

    Args:
        model: A XGBoost model (trained using the scikit-learn API)
        test_input: Some input data used to trace the model execution
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.supported_configurations`

    Examples:
        >>> pytorch_model = convert_xgboost(xgb_model, [], extra_config={"n_features":200})

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
    return convert_sklearn(model, test_input, extra_config)


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
