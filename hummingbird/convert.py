# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from copy import deepcopy
import numpy as np
import pandas
from .common._topology import convert_topology
from ._parse import parse_sklearn_model

# Invoke the registration of all our converters and shape calculators.
from . import operator_converters  # noqa


def convert_sklearn(model, test_input=None, device=None, extra_config={}):
    """
    This function produces an equivalent PyTorch model of the given scikit-learn model.
    The supported converters is returned by function
    :func:`supported_converters <hummingbird.supported_converters>`.

    For pipeline conversion, user needs to make sure each component
    is one of our supported items.

    This function converts the specified *scikit-learn* model into its *PyTorch* counterpart.
    Note that for all conversions, initial types are required.
    *TorchScript* model file name can also be specified.

    :param model: A scikit-learn model
    :param test_input: some input data used to trace the model execution
    :param device: torch.device Which device to translate them model
    :param extra_config: Extra configurations to be used by the individual operator converters
    :return: A PyTorch model which is equivalent to the input scikit-learn model

    .. note::
        If a pipeline includes an instance of
        `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_,
        *scikit-learn* allow the user to specify columns by names. This option is not supported
        by *sklearn-pytorch* as features names could be different in input data and the PyTorch model.

    """  # noqa

    # Parse scikit-learn model as our internal data structure
    # (i.e., Topology)
    # we modify the scikit learn model during optimizations
    model = deepcopy(model)
    topology = parse_sklearn_model(model)

    # Convert our Topology object into PyTorch. The outcome is a PyTorch model.
    pytorch_model = convert_topology(topology, device, extra_config).eval()
    if device is not None:
        pytorch_model = pytorch_model.to(device)
    return pytorch_model


"""
This function is used to generate a PyTorch model from a given LighGBM model
:param model: A LightGBM model (trained using the scikit-learn API)
:param test_input: some input data used to trace the model execution
:param device: torch.device Which device to translate them model
:param extra_config: Extra configurations to be used by the individual operator converters
:return: A PyTorch model which is equivalent to the input LightGBM model
"""


def convert_lightgbm(model, test_input=None, device=None, extra_config={}):
    return convert_sklearn(model, test_input, device, extra_config)


"""
This function is used to generate a PyTorch model from a given XGBoost model
:param model: A XGBoost model (trained using the scikit-learn API)
:param test_input: some input data used to trace the model execution
:param device: torch.device Which device to translate them model
:param extra_config: Extra configurations to be used by the individual operator converters
:return: A PyTorch model which is equivalent to the input XGBoost model
"""


def convert_xgboost(model, test_input, device=None, extra_config={}):
    if "_features_count" in dir(model):
        extra_config["n_features"] = model._features_count
    elif test_input is not None:
        if type(test_input) is np.ndarray and len(test_input.shape) == 2:
            extra_config["n_features"] = test_input.shape[1]
        elif type(test_input) is pandas.DataFrame:
            extra_config["n_features"] = len(test_input.columns)
        else:
            raise RuntimeError(
                "XGBoost converter is not able to infer the number of input features.\
            Please fill an issue on https://github.com/microsoft/hummingbird/"
            )
    else:
        raise RuntimeError(
            "XGBoost converter is not able to infer the number of input features.\
            Please pass some test_input to the converter."
        )
    return convert_sklearn(model, test_input, device, extra_config)
