# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for topology IR are stored in this file.
"""
from distutils.version import LooseVersion
import numpy as np
import os
import torch
from uuid import uuid4

from onnxconverter_common.registration import get_converter
import onnx

from hummingbird.ml._container import (
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
)
from hummingbird.ml._utils import pandas_installed, _get_device
from hummingbird.ml.exceptions import MissingConverter
from hummingbird.ml.operator_converters import constants

if pandas_installed():
    from pandas import DataFrame
else:
    DataFrame = None


def _get_trace_input_from_test_input(input, batch_size):
    """
    Utility function used to properly put the inputs into a format understandable by torch.
    """
    if type(input) is tuple:
        if batch_size is not None:
            trace_input = tuple([torch.from_numpy(i)[0:batch_size, :] for i in input])
        else:
            trace_input = tuple([torch.from_numpy(i) for i in input])
    else:
        trace_input = torch.from_numpy(input)
        if batch_size is not None:
            trace_input = trace_input[0:batch_size, :]
    return trace_input


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

    operator_map = {}

    for operator in topology.topological_operator_iterator():
        try:
            converter = get_converter(operator.type)

            if backend == onnx.__name__:
                # vers = LooseVersion(torch.__version__)
                # allowed_min = LooseVersion("1.6.0")
                # Pytorch <= 1.6.0 has a bug with exporting GEMM into ONNX.
                # For the moment only tree_trav is enabled for pytorch <= 1.6.0
                # if vers < allowed_min:
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

    # Set the parameters for the model / container
    n_threads = None if constants.N_THREADS not in extra_config else extra_config[constants.N_THREADS]
    batch_size = None if constants.BATCH_SIZE not in extra_config else extra_config[constants.BATCH_SIZE]

    # We set the number of threads for torch here to avoid errors in case we JIT.
    # We set intra op concurrency while we force operators to run sequentially.
    # We can revise this later, but in general we don't have graphs requireing inter-op parallelism.
    if n_threads is not None:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.set_num_threads(n_threads)

    operators = list(topology.topological_operator_iterator())
    torch_model = _PyTorchBackendModel(
        topology.raw_model.input_names, topology.raw_model.output_names, operator_map, operators, extra_config
    ).eval()

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

        # Put the tracing test input into the right format.
        trace_input = _get_trace_input_from_test_input(extra_config[constants.TEST_INPUT], batch_size)

        # Generate the ONNX models
        torch.onnx.export(
            torch_model,
            trace_input,
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

        # Fix the model to use arbitrary batch dimensions
        def fix_dim(dim):
            updated = False
            if dim.HasField("dim_value"):
                dim.Clear()
                updated = True
                dim.dim_param = "sym"

            return updated

        def fix_value_info(value):
            num_fixed = 0
            if value.type.HasField("tensor_type"):
                shape = value.type.tensor_type.shape
                if shape:
                    dim = shape.dim[0]
                    if fix_dim(dim):
                        num_fixed += 1

            return num_fixed

        def fix_graph(graph):
            num_fixed = 0
            for input in graph.input:
                num_fixed += fix_value_info(input)

            for output in graph.output:
                num_fixed += fix_value_info(output)

            for node in graph.node:
                for attr in node.attribute:
                    if attr.HasField("g"):
                        num_fixed += fix_graph(attr.g)

            return num_fixed

        fix_graph(hb_model.graph)
    else:
        # Set the device for the model.
        if device != "cpu":
            if backend == torch.__name__ or torch.jit.__name__:
                torch_model = torch_model.to(device)

        # If the backend is tochscript, jit the model.
        if backend == torch.jit.__name__:
            trace_input = _get_trace_input_from_test_input(extra_config[constants.TEST_INPUT], batch_size)
            if device != "cpu":
                trace_input.to(device)
            torch_model = torch.jit.trace(torch_model, trace_input).eval()
            torch.jit.optimized_execution(torch_model)
        hb_model = torch_model

    # Return if the container is not needed.
    if constants.CONTAINER in extra_config and not extra_config[constants.CONTAINER]:
        return hb_model

    # We scan the operators backwards until we find an operator with a defined type
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

    # If is a transformer, we need to check whether there is another operator type before.
    # E.g., normalization after classification.
    tmp_idx = idx
    if operator_map[operators[idx].full_name].transformer:
        while (
            idx >= 0
            and not operator_map[operators[idx].full_name].regression
            and not operator_map[operators[idx].full_name].classification
            and not operator_map[operators[idx].full_name].anomaly_detection
        ):
            idx -= 1
        if idx < 0:
            idx = tmp_idx

    if operator_map[operators[idx].full_name].regression:
        # We are doing a regression task.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerRegression
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerRegression
        else:
            container = PyTorchSklearnContainerRegression
    elif operator_map[operators[idx].full_name].anomaly_detection:
        # We are doing anomaly detection.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerAnomalyDetection
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerAnomalyDetection
        else:
            container = PyTorchSklearnContainerAnomalyDetection
    elif operator_map[operators[idx].full_name].transformer:
        # We are just transforming the input data.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerTransformer
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerTransformer
        else:
            container = PyTorchSklearnContainerTransformer
    else:
        # We are doing a classification task.
        if backend == torch.jit.__name__:
            container = TorchScriptSklearnContainerClassification
        elif backend == onnx.__name__:
            container = ONNXSklearnContainerClassification
        else:
            container = PyTorchSklearnContainerClassification

    n_threads = None if constants.N_THREADS not in extra_config else extra_config[constants.N_THREADS]
    batch_size = None if constants.BATCH_SIZE not in extra_config else extra_config[constants.BATCH_SIZE]
    hb_model = container(hb_model, n_threads, batch_size, extra_config=extra_config)

    return hb_model


class _PyTorchBackendModel(torch.nn.Module, object):
    """
    Hummingbird model internal representation of a converted pipeline.
    """

    def __init__(self, input_names, output_names, operator_map, operators, extra_config):
        """
        Args:
            input_names: The names of the input `onnxconverter_common.topology.Variable`s for this model
            output_names: The names of the output `onnxconverter_common.topology.Variable`s generated by this model
            operator_map: A dictionary of operator aliases and related PyTorch implementations
            operators: The list of operators (in a topological order) that will be executed by the model (in order)
            extra_config: Some additional custom configuration parameter
        """
        super(_PyTorchBackendModel, self).__init__()

        # Define input \ output names.
        # This is required because the internal variable names may differ from the original (raw) one.
        # This may happen, for instance, because we force our internal naming to be unique.
        def _fix_var_naming(operators, names, mod="input"):
            new_names = []
            map = {}

            for op in operators:
                if mod == "input":
                    iter = op.inputs
                else:
                    iter = op.outputs
                for i in iter:
                    for name in names:
                        if i.raw_name == name and name not in map:
                            map[i.raw_name] = i.full_name
                if len(map) == len(names):
                    break
            for name in names:
                new_names.append(map[name])
            return new_names

        self._input_names = _fix_var_naming(operators, input_names)
        self._output_names = _fix_var_naming(reversed(operators), output_names, "output")
        self._operator_map = torch.nn.ModuleDict(operator_map)
        self._operators = operators

    def forward(self, *inputs):
        with torch.no_grad():
            assert len(self._input_names) == len(inputs) or (
                type(inputs[0]) == DataFrame and DataFrame is not None and len(self._input_names) == len(inputs[0].columns)
            ), "number of inputs or number of columns in the dataframe do not match with the expected number of inputs {}".format(
                self._input_names
            )

            if type(inputs[0]) == DataFrame and DataFrame is not None:
                # Split the dataframe into column ndarrays
                inputs = inputs[0]
                input_names = list(inputs.columns)
                splits = [inputs[input_names[idx]] for idx in range(len(input_names))]
                splits = [df.to_numpy().reshape(-1, 1) for df in splits]
                inputs = tuple(splits)
            inputs = [*inputs]
            variable_map = {}
            device = _get_device(self)

            # Maps data inputs to the expected variables.
            for i, input_name in enumerate(self._input_names):
                if type(inputs[i]) is list:
                    inputs[i] = np.array(inputs[i])
                if type(inputs[i]) is np.ndarray:
                    inputs[i] = torch.from_numpy(inputs[i])
                    if inputs[i].dtype == torch.float64:
                        # We convert double precision arrays into single precision. Sklearn does the same.
                        inputs[i] = inputs[i].float()
                elif type(inputs[i]) is not torch.Tensor:
                    raise RuntimeError("Inputer tensor {} of not supported type {}".format(input_name, type(inputs[i])))
                if device is not None and device.type != "cpu":
                    inputs[i] = inputs[i].to(device)
                variable_map[input_name] = inputs[i]

            # Evaluate all the operators in the topology by properly wiring inputs \ outputs
            for operator in self._operators:
                pytorch_op = self._operator_map[operator.full_name]
                pytorch_outputs = pytorch_op(*(variable_map[input] for input in operator.input_full_names))

                if len(operator.output_full_names) == 1:
                    variable_map[operator.output_full_names[0]] = pytorch_outputs
                else:
                    for i, output in enumerate(operator.output_full_names):
                        variable_map[output] = pytorch_outputs[i]

            # Prepare and return the output.
            if len(self._output_names) == 1:
                return variable_map[self._output_names[0]]
            else:
                return list(variable_map[output_name] for output_name in self._output_names)
