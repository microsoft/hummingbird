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
import onnx
import timeit

from ._container import (
    PyTorchBackendModel,
    PyTorchSklearnContainerRegression,
    PyTorchSklearnContainerClassification,
    PyTorchSklearnContainerTransformer,
    PyTorchSklearnContainerAnomalyDetection,
    TorchScriptSklearnContainerRegression,
    TorchScriptSklearnContainerClassification,
    TorchScriptSklearnContainerTransformer,
    TorchScriptSklearnContainerAnomalyDetection,
    ONNXSklearnContainerRegression,
    ONNXSklearnContainerClassification,
    ONNXSklearnContainerTransformer,
    ONNXSklearnContainerAnomalyDetection,
    TVMSklearnContainerRegression,
    TVMSklearnContainerClassification,
    TVMSklearnContainerTransformer,
    TVMSklearnContainerAnomalyDetection,
)
from ._utils import tvm_installed
from .exceptions import MissingConverter
from .operator_converters import constants


def _jit_model(torch_model, device, extra_config):
    """
    Function used to convert an input pytorch model into torchscript.
    """
    test_data = torch.from_numpy(extra_config[constants.TEST_INPUT])
    if device != "cpu":
        test_data.to(device)
    return torch.jit.trace(torch_model, test_data).eval()


def convert(topology, backend, device, extra_config={}):
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
    assert device is not None, "Cannot convert a Topology object into device None."

    tvm_backend = None
    operator_map = {}

    for operator in topology.topological_operator_iterator():
        try:
            converter = get_converter(operator.type)

            if backend == onnx.__name__:
                # vers = LooseVersion(torch.__version__)
                # allowed_min = LooseVersion("1.6.1")
                # Pytorch <= 1.6.0 has a bug with exporting GEMM into ONNX.
                # For the moment only tree_trav is enabled for pytorch <= 1.6.0
                # if vers < allowed_min:
                extra_config[constants.TREE_IMPLEMENTATION] = "tree_trav"
            elif backend == tvm_backend:
                # The TVM frontend for PyTorch currently don't support index_select
                # https://github.com/apache/incubator-tvm/issues/6282
                extra_config[constants.TREE_IMPLEMENTATION] = "gemm"

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
    torch_model = PyTorchBackendModel(
        topology.raw_model.input_names, topology.raw_model.output_names, operator_map, operators, extra_config
    ).eval()

    if tvm_installed():
        import tvm
        from tvm import relay
        from tvm.contrib import graph_runtime

        tvm_backend = tvm.__name__

    if backend == onnx.__name__:
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
            torch_model,
            torch.from_numpy(extra_config[constants.TEST_INPUT]),
            output_model_name,
            input_names=topology.raw_model.input_names,
            output_names=topology.raw_model.output_names,
            keep_initializers_as_inputs=False,
            opset_version=target_opset,
            do_constant_folding=True,
        )
        hb_model = onnx.load(output_model_name)
        os.remove(output_model_name)

        # Set the ONNX model name if any.
        if onnx_model_name is not None:
            hb_model.graph.name = onnx_model_name
    elif backend == tvm_backend:
        # First we need to generate the torchscript model.
        ts_model = _jit_model(torch_model, device, extra_config)

        # Generate the test input in the TVM format.
        test_input = [
            (topology.raw_model.input_names[i], extra_config[constants.TEST_INPUT].shape)
            for i in range(len(topology.raw_model.input_names))
        ]

        # Create the relay version of the model.
        model, params = relay.frontend.from_pytorch(ts_model, test_input)

        # Pick the proper target.
        if device == "cuda":
            target = tvm.target.cuda()
            ctx = tvm.gpu()
        elif device == "cpu":
            target = "llvm"
            ctx = tvm.cpu()
        elif "llvm" in device:
            target = device
            ctx = tvm.cpu()
        else:
            raise RuntimeError("Device {} not recognized".format(device))

        # Generate the model.
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(model, target=target, params=params)
        tvm_model = graph_runtime.create(graph, lib, ctx)
        tvm_model.set_input(**params)

        # In the container we will be using the context to properly configure the input tensors.
        extra_config[constants.TVM_CONTEXT] = ctx

        hb_model = tvm_model
    else:
        # Set the device for the model.
        if device != "cpu":
            if backend == torch.__name__ or torch.jit.__name__:
                torch_model = torch_model.to(device)

        # If the backend is tochscript, jit the model.
        if backend == torch.jit.__name__:
            torch_model = _jit_model(torch_model, device, extra_config)

        hb_model = torch_model

    # We scan the operators backwards until we find an operator with a defined type.
    # This is necessary because ONNX models can have arbitrary operators doing casting, reshaping etc.
    idx = len(operators) - 1
    while (
        idx >= 0
        and not operator_map[operators[idx].full_name].regression
        and not operator_map[operators[idx].full_name].classification
        and not operator_map[operators[idx].full_name].anomaly_detection
        and not operator_map[operators[idx].full_name].transformer
    ):
        idx -= 1

    assert idx >= 0, "Cannot detect container type. Please fill an issue at https://github.com/microsoft/hummingbird."

    # If the operator is a transformer, we need to check wheter there is another operator type before.
    # E.g., normalization after classification.
    tmp_idx = idx
    if operator_map[operators[idx].full_name].transformer:
        while (
            idx > 0
            and not operator_map[operators[idx].full_name].regression
            and not operator_map[operators[idx].full_name].classification
            and not operator_map[operators[idx].full_name].anomaly_detection
        ):
            idx -= 1
        if idx < 0:
            idx = tmp_idx

    # Get the proper container type.
    if operator_map[operators[idx].full_name].regression:
        # We are doing a regression task.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerRegression
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerRegression
        elif backend == tvm_backend:
            container = TVMSklearnContainerRegression
        else:
            container = PyTorchSklearnContainerRegression
    elif operator_map[operators[idx].full_name].anomaly_detection:
        # We are doing anomaly detection.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerAnomalyDetection
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerAnomalyDetection
        elif backend == tvm_backend:
            container = TVMSklearnContainerAnomalyDetection
        else:
            container = PyTorchSklearnContainerAnomalyDetection
    elif operator_map[operators[idx].full_name].transformer:
        # We are just transforming the input data.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerTransformer
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerTransformer
        elif backend == tvm_backend:
            container = TVMSklearnContainerTransformer
        else:
            container = PyTorchSklearnContainerTransformer
    else:
        # We are doing a classification task.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerClassification
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerClassification
        elif backend == tvm_backend:
            container = TVMSklearnContainerClassification
        else:
            container = PyTorchSklearnContainerClassification

    hb_model = container(hb_model, extra_config)

    return hb_model
