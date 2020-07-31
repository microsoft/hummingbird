# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for topology IR are stored in this file.
"""
from distutils.version import LooseVersion
import os
import torch
from uuid import uuid4

from onnxconverter_common.registration import get_converter

from ._container import (
    PyTorchBackendModelRegression,
    PyTorchBackendModelClassification,
    PyTorchBackendModelTransformer,
    PyTorchBackendModelAnomalyDetection,
)
from ._utils import onnx_runtime_installed
from .exceptions import MissingConverter
from .operator_converters import constants


def convert(topology, backend, device=None, extra_config={}):
    """
    This function is used to convert a `onnxconverter_common.topology.Topology` object into a *backend* model.

    Args:
        topology: The `onnxconverter_common.topology.Topology` object that will be converted into a backend model
        backend: Which backend the model should be run on
        device: Which device the translated model will be run on
        extra_config: Extra configurations to be used by individual operator converters

    Returns:
        A model implemented in the selected backend
    """
    assert topology is not None, "Cannot convert a Topology object of type None."
    assert backend is not None, "Cannot convert a Topology object into backend None."

    operator_map = {}
    onnx_backend = None

    if onnx_runtime_installed():
        import onnx

        onnx_backend = onnx.__name__

    for operator in topology.topological_operator_iterator():
        try:
            converter = get_converter(operator.type)

            if backend == onnx_backend:
                vers = LooseVersion(torch.__version__)
                allowed_min = LooseVersion("1.6")
                # Pytorch <= 1.6.0 has a bug with exporting GEMM into ONNX.
                # For the moment only tree_trav is enabled for pytorch <= 1.6.0
                if vers <= allowed_min:
                    extra_config[constants.TREE_IMPLEMENTATION] = "tree_trav"

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
    elif operator_map[operators[-1].full_name].anomaly_detection:
        # We are doing anomaly detection.
        pytorch_container = PyTorchBackendModelAnomalyDetection
    elif operator_map[operators[-1].full_name].transformer:
        # We are just transforming the input data.
        pytorch_container = PyTorchBackendModelTransformer
    else:
        # We are doing a classification task.
        pytorch_container = PyTorchBackendModelClassification

    pytorch_model = pytorch_container(
        topology.raw_model.input_names, topology.raw_model.output_names, operator_map, operators, extra_config
    ).eval()

    if backend == onnx_backend:
        onnx_model_name = output_model_name = None
        target_opset = 11

        # Set optional configuration options for ONNX if any.
        if constants.ONNX_OUTPUT_MODEL_NAME in extra_config:
            onnx_model_name = extra_config[constants.ONNX_OUTPUT_MODEL_NAME]
            output_model_name = onnx_model_name + ".onnx"
        if constants.ONNX_TARGET_OPSET in extra_config:
            target_opset = extra_config[constants.ONNX_TARGET_OPSET]
        if output_model_name is None:
            output_model_name = str(uuid4().hex) + ".onnx"

        # Generate the ONNX models
        torch.onnx.export(
            pytorch_model,
            torch.from_numpy(extra_config[constants.ONNX_TEST_INPUT]),
            output_model_name,
            input_names=topology.raw_model.input_names,
            output_names=topology.raw_model.output_names,
            keep_initializers_as_inputs=False,
            opset_version=target_opset,
            do_constant_folding=True,
        )
        onnx_model = onnx.load(output_model_name)
        os.remove(output_model_name)

        # Set the ONNX model name if any.
        if onnx_model_name is not None:
            onnx_model.graph.name = onnx_model_name

        return onnx_model

    if device is not None:
        pytorch_model = pytorch_model.to(device)
    return pytorch_model
