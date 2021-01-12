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
from onnxconverter_common.topology import Topology as ONNXTopology
import onnx

from hummingbird.ml.containers import (
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
    BatchContainer,
)
from hummingbird.ml._utils import pandas_installed, tvm_installed, get_device, from_strings_to_ints
from hummingbird.ml._executor import Executor
from hummingbird.ml.exceptions import MissingConverter
from hummingbird.ml.operator_converters import constants

if pandas_installed():
    from pandas import DataFrame
else:
    DataFrame = None


# We wrap the onnxconverter topology and do some renamining for consistency with Hummingbird design.
class Topology:
    def __init__(self, input_container):
        self.onnxconverter_topology = ONNXTopology(input_container)

        # Declare an object to provide variables' and operators' naming mechanism.
        # One global scope is enough for parsing Hummingbird's supported input models.
        self.scope = self.onnxconverter_topology.declare_scope("__root__")

    @property
    def input_container(self):
        """
        Returns the input container wrapping the original input model.
        """
        return self.onnxconverter_topology.raw_model

    @property
    def variables(self):
        """
        Returns all the logical variables of the topology.
        """
        return self.scope.variables

    def declare_logical_variable(self, original_input_name, type=None):
        """
        This function creates a new logical variable within the topology.
        If original_input_name has been used to create other variables,
        the new variable will hide all other variables created using original_input_name.
        """
        return self.scope.declare_local_variable(original_input_name, type=type)

    def declare_logical_operator(self, alias, model=None):
        """
        This function is used to declare new logical operator.
        """
        return self.scope.declare_local_operator(alias, model)

    def topological_operator_iterator(self):
        """
        This is an iterator of all operators in the Topology object.
        Operators are returned in a topological order.
        """
        return self.onnxconverter_topology.topological_operator_iterator()


def _jit_trace(executor, trace_input, device, extra_config):
    """
    Function used to convert an input pytorch model into torchscript.
    """
    if device != "cpu":
        trace_input.to(device)
    return torch.jit.trace(executor, trace_input).eval()


def _get_trace_input_from_test_input(input, remainder_size=None, extra_config={}):
    """
    Utility function used to properly put the inputs into a format understandable by torch.
    If `remainder_size` is provided, also return inputs for a remainder model (see below).
    """
    remainder = None
    if isinstance(input, tuple):
        trace_input = []
        for input_ in input:
            # Convert string arrays into int32.
            if input_.dtype.kind in constants.SUPPORTED_STRING_TYPES:
                assert constants.MAX_STRING_LENGTH in extra_config
                max_string_length = extra_config[constants.MAX_STRING_LENGTH]

                input_ = from_strings_to_ints(input_, max_string_length)
            trace_input.append(torch.from_numpy(input_))
        trace_input = tuple(trace_input)
        if remainder_size is not None and remainder_size != 0:
            remainder = tuple([inp[0:remainder_size, :] for inp in trace_input])
    else:
        # Convert string arrays into int32.
        if input.dtype.kind in constants.SUPPORTED_STRING_TYPES:
            assert constants.MAX_STRING_LENGTH in extra_config
            max_string_length = extra_config[constants.MAX_STRING_LENGTH]

            input = from_strings_to_ints(input, max_string_length)
        trace_input = torch.from_numpy(input)
        if remainder_size is not None and remainder_size != 0:
            remainder = trace_input[0:remainder_size, :]

    return (trace_input, remainder)


def _get_batch_size(batch):
    if isinstance(batch, tuple):
        return batch[0].shape[0]

    assert isinstance(batch, np.ndarray)
    return batch.shape[0]


def _compile_to_tvm(topology, executor, trace_input, target, ctx, config, extra_config):
    import tvm
    from tvm import relay
    from tvm.contrib import graph_runtime

    ts_model = _jit_trace(executor, trace_input, "cpu", extra_config)
    test_input = [
        (topology.input_container.input_names[i], trace_input[i].shape if type(trace_input) is tuple else trace_input.shape,)
        for i in range(len(topology.input_container.input_names))
    ]

    model, params = relay.frontend.from_pytorch(ts_model, test_input)

    with tvm.transform.PassContext(opt_level=3, config=config):
        graph, lib, params = relay.build(model, target=target, params=params)

    tvm_model = graph_runtime.create(graph, lib, ctx)
    tvm_model.set_input(**params)

    extra_config[constants.TVM_GRAPH] = graph
    extra_config[constants.TVM_LIB] = lib
    extra_config[constants.TVM_PARAMS] = params

    return tvm_model


def convert(topology, backend, test_input, device, extra_config={}):
    """
    This function is used to convert a `Topology` object into a *backend* model.

    Args:
        topology: The `Topology` object that will be converted into a backend model
        backend: Which backend the model should be run on
        test_input: Inputs for PyTorch model tracing
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

    if tvm_installed():
        import tvm

        tvm_backend = tvm.__name__

    for operator in topology.topological_operator_iterator():
        converter = get_converter(operator.type)
        if convert is None:
            raise MissingConverter(
                "Unable to find converter for {} type {} with extra config: {}.".format(
                    operator.type, type(getattr(operator, "raw_model", None)), extra_config
                )
            )

        if backend == onnx.__name__:
            # vers = LooseVersion(torch.__version__)
            # allowed_min = LooseVersion("1.6.0")
            # Pytorch <= 1.6.0 has a bug with exporting GEMM into ONNX.
            # For the moment only tree_trav is enabled for pytorch <= 1.6.0
            # if vers < allowed_min:
            extra_config[constants.TREE_IMPLEMENTATION] = "tree_trav"
        operator_map[operator.full_name] = converter(operator, device, extra_config)

    # Set the parameters for the model / container
    n_threads = None if constants.N_THREADS not in extra_config else extra_config[constants.N_THREADS]

    # We set the number of threads for torch here to avoid errors in case we JIT.
    # We set intra op concurrency while we force operators to run sequentially.
    # We can revise this later, but in general we don't have graphs requireing inter-op parallelism.
    if n_threads is not None:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.set_num_threads(n_threads)

    operators = list(topology.topological_operator_iterator())
    executor = Executor(
        topology.input_container.input_names, topology.input_container.output_names, operator_map, operators, extra_config
    ).eval()

    # if constants.REMAINDER_SIZE is present in extra_config, we are in the convert_batch mode.
    remainder_model = None
    remainder_size = None if constants.REMAINDER_SIZE not in extra_config else extra_config[constants.REMAINDER_SIZE]

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
        batch_trace_input, _ = _get_trace_input_from_test_input(test_input, remainder_size, extra_config)

        # Generate the ONNX models
        torch.onnx.export(
            executor,
            batch_trace_input,
            output_model_name,
            input_names=topology.input_container.input_names,
            output_names=topology.input_container.output_names,
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
    elif backend == tvm_backend:
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

        # Get configuration parameters.
        # 50 is a good depth for operator fusion. More than that will probably hurt performance.
        # https://github.com/microsoft/hummingbird/issues/232#issuecomment-697979508
        config = {"relay.FuseOps.max_depth": 50}

        if constants.TVM_MAX_FUSE_DEPTH in extra_config:
            config["relay.FuseOps.max_depth"] = extra_config[constants.TVM_MAX_FUSE_DEPTH]

        # First we need to generate the torchscript model.
        batch_trace_input, remainder_trace_input = _get_trace_input_from_test_input(test_input, remainder_size, extra_config)

        tvm_model = _compile_to_tvm(topology, executor, batch_trace_input, target, ctx, config, extra_config)

        if remainder_trace_input is not None:
            remainder_model = _compile_to_tvm(topology, executor, remainder_trace_input, target, ctx, config, extra_config)

        # In the container we will be using the context to properly configure the input tensors.
        extra_config[constants.TVM_CONTEXT] = ctx
        extra_config[constants.TVM_INPUT_NAMES] = topology.input_container.input_names

        hb_model = tvm_model
    else:
        # Set the device for the model.
        if device != "cpu":
            if backend == torch.__name__ or torch.jit.__name__:
                executor = executor.to(device)

        # If the backend is tochscript, jit the model.
        if backend == torch.jit.__name__:
            trace_input, _ = _get_trace_input_from_test_input(test_input, remainder_size, extra_config)
            executor = _jit_trace(executor, trace_input, device, extra_config)
            torch.jit.optimized_execution(executor)

        hb_model = executor

    # Return if the container is not needed.
    if constants.CONTAINER in extra_config and not extra_config[constants.CONTAINER]:
        return hb_model

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

    n_threads = None if constants.N_THREADS not in extra_config else extra_config[constants.N_THREADS]
    batch_size = None if constants.TEST_INPUT not in extra_config else _get_batch_size(test_input)
    hb_container = container(hb_model, n_threads, batch_size, extra_config=extra_config)

    if remainder_model:
        aux_container = container(remainder_model, n_threads, remainder_size, extra_config=extra_config)
        return BatchContainer(hb_container, aux_container)
    elif remainder_size is not None and remainder_size > 0:
        # remainder_size is non zero but remainder_model is not created
        # -> torch backend case
        aux_container = container(hb_model, n_threads, remainder_size, extra_config=extra_config)
        return BatchContainer(hb_container, aux_container)
    elif remainder_size is not None:
        # remainder_size is not None but remainder_model is not created
        # -> remainder_size must be zero (no need to create remainder_model)
        assert remainder_size == 0, "remainder_size is non zero but no remainder_model has been created"
        # remainder_size is not None only if called by convert_batch(...), so we return BatchContainer
        # for this code path, even though there is no remainder_model created.
        return BatchContainer(hb_container)

    return hb_container
