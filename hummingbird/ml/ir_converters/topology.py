# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for topology IR are stored in this file.
"""
from onnxconverter_common.registration import get_converter

from ..exceptions import MissingConverter
from .._container import PyTorchBackendModelRegression, PyTorchBackendModelClassification, PyTorchBackendModelTransformer


def convert(topology, device=None, extra_config={}):
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

    operators = list(topology.topological_operator_iterator())
    if operator_map[operators[-1].full_name].regression:
        # We are doing a regression task.
        pytorch_container = PyTorchBackendModelRegression
    elif operator_map[operators[-1].full_name].transformer:
        # We are just transforming the input data.
        pytorch_container = PyTorchBackendModelTransformer
    else:
        # We are doing a classification task.
        pytorch_container = PyTorchBackendModelClassification

    pytorch_model = pytorch_container(
        topology.raw_model.input_names, topology.raw_model.output_names, operator_map, operators, extra_config
    ).eval()

    if device is not None:
        pytorch_model = pytorch_model.to(device)
    return pytorch_model
