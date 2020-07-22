# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All functions used for parsing input models are listed here.
"""
from copy import deepcopy
from uuid import uuid4

from onnxconverter_common.container import CommonSklearnModelContainer
from onnxconverter_common.optimizer import LinkedNode, _topological_sort
from onnxconverter_common.topology import Topology

from . import constants
from ._container import CommonONNXModelContainer
from ._utils import sklearn_installed
from .supported import get_sklearn_api_operator_name, get_onnxml_api_operator_name


def parse_sklearn_api_model(model):
    """
    Puts *scikit-learn* object into an abstract representation so that our framework can work seamlessly on models created
    with different machine learning tools.

    Args:
        model: A model object in scikit-learn format

    Returns:
        A `onnxconverter_common.topology.Topology` object representing the input model
    """
    assert model is not None, "Cannot convert a mode of type None."

    raw_model_container = CommonSklearnModelContainer(model)

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.
    topology = Topology(raw_model_container)

    # Declare an object to provide variables' and operators' naming mechanism.
    # One global scope is enough for parsing scikit-learn models.
    scope = topology.declare_scope("__root__")

    # Declare input variables. Sklearn always gets as input a single dataframe,
    # therefore by default we start with a single `input` variable
    inputs = []
    inputs.append(scope.declare_local_variable("input"))

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the inputs of the scikit-learn's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model into its scope with the topology.
    # Get the outputs of the model.
    outputs = _parse_sklearn_api(scope, model, inputs)

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the outputs of the scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology


def parse_onnx_api_model(model):
    """
    Puts *ONNX* object into an abstract representation so that our framework can work seamlessly on models created
    with different machine learning tools.

    Args:
        model: A model object in onnx format

    Returns:
        A `onnxconverter_common.topology.Topology` object representing the input model
    """
    assert model is not None, "Cannot convert a mode of type None."

    raw_model_container = CommonONNXModelContainer(model)

    # We modify the ONNX model during translation
    model = deepcopy(model)

    # Declare a computational graph. It will become a representation of
    # the input ONNX model after parsing.
    topology = Topology(raw_model_container)

    # Declare an object to provide variables' and operators' naming mechanism.
    # One global scope is enough for parsing ONNX models.
    scope = topology.declare_scope("__root__")

    # Declare input variables.
    inputs = []
    for i in model.graph.input:
        inputs.append(scope.declare_local_variable(i.name))

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the inputs of the ONNX graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input ONNX model into its scope with the topology.
    _parse_onnx_api(scope, model, inputs)

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the output_names of the ONNX graph.
    for o in model.graph.output:
        raw_model_container.add_output(scope.declare_local_variable(o.name))

    return topology


def _parse_sklearn_api(scope, model, inputs):
    """
    This is a delegate function adding the model to the input scope.
    It does nothing but invokes the correct parsing function according to the input model's type.

    Args:
        scope: The `onnxconverter_common.topology.Scope` object where the model will be added
        model: A scikit-learn model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        The output `onnxconverter_common.topology.Variable`s produced by the input model
    """
    tmodel = type(model)
    if tmodel in sklearn_api_parsers_map:
        outputs = sklearn_api_parsers_map[tmodel](scope, model, inputs)
    else:
        outputs = _parse_sklearn_single_model(scope, model, inputs)

    return outputs


def _parse_sklearn_single_model(scope, model, inputs):
    """
    This function handles all sklearn objects composed by a single model.

    Args:
        scope: The ``onnxconverter_common.topology.Scope`` where the model will be added
        model: A scikit-learn model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        A list of output `onnxconverter_common.topology.Variable` which will be passed to next stage
    """
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a " "string '{0}'.".format(model))

    alias = get_sklearn_api_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    # We assume that all scikit-learn operators produce a single output.
    variable = scope.declare_local_variable("variable")
    this_operator.outputs.append(variable)

    return this_operator.outputs


def _parse_sklearn_pipeline(scope, model, inputs):
    """
    The basic ideas of scikit-learn pipeline parsing:
        1. Sequentially go though all stages defined in the considered
           scikit-learn pipeline
        2. The output `onnxconverter_common.topology.Variable`s of one stage will be fed into its next
           stage as the inputs.

    Args:
        scope: The ``onnxconverter_common.topology.Scope`` for the model
        model: A `sklearn.pipeline.Pipeline` object
        inputs: A list of `onnxconverter_common.topology.Variable` objects

    Returns:
        A list of output `onnxconverter_common.topology.Variable`s produced by the input pipeline
    """
    for step in model.steps:
        inputs = _parse_sklearn_api(scope, step[1], inputs)
    return inputs


def _build_sklearn_api_parsers_map():
    from sklearn import pipeline

    # Parsers for edge cases are going here.
    map_parser = {
        pipeline.Pipeline: _parse_sklearn_pipeline
        # More will go here as added.
    }
    return map_parser


def _parse_onnx_api(scope, model, inputs):
    """
    This function handles all input ONNX models.

    Args:
        scope: The ``onnxconverter_common.topology.Scope`` where the model will be added
        model: A ONNX model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        A list of output `onnxconverter_common.topology.Variable` which will be passed to next stage
    """
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a " "string '{0}'.".format(model))

    # Parse an ONNX-ML model into our internal data structure (i.e., LinkedNode)
    graph = model.graph
    inputs_names = [in_.raw_name for in_ in inputs]
    output_names = [] if graph.output is None else [o_.name for o_ in graph.output]
    initializers = [] if graph.initializer is None else [in_ for in_ in graph.initializer]
    node_list = LinkedNode.build_from_onnx(graph.node, [], inputs_names + [in_.name for in_ in initializers], output_names)

    # This a new node list but with some node been removed plus eventual variable renaming.
    new_node_list = _remove_zipmap(node_list)

    # Add each operator in the LinkedNode data structure to the topology.
    for node in new_node_list:
        _parse_onnx_single_operator(scope, node)


def _parse_onnx_single_operator(scope, operator):
    """
    This function handles the parsing of all ONNX operators.

    Args:
        scope: The ``onnxconverter_common.topology.Scope`` where the model will be added
        model: An ONNX operator
    """

    # Add the operator in the scope.
    alias = get_onnxml_api_operator_name(operator.op_type)
    this_operator = scope.declare_local_operator(alias, operator)

    # Register the operator's inputs.
    # LinkedNode uses dictionaries and with Python 3.5 the order is not deterministic.
    input_names = list(operator.input.keys())
    input_names.sort()
    this_operator.inputs = [scope.variables[in_] for in_ in input_names if in_ in scope.variables]

    # Register the operator's outpurs.
    output_names = list(operator.output.keys())
    output_names.sort()
    for output in output_names:
        variable = scope.declare_local_variable(output)
        this_operator.outputs.append(variable)


def _remove_zipmap(node_list):
    """
    Method used to remove ZipMap operators in the graph.

    """
    output_node_list = []

    for node_ in _topological_sort(node_list):
        if node_.op_type == "ZipMap":
            # We remove this map operator and just use an array.
            assert len(node_.input) == len(node_.output)
            # Check if in single path to output
            assert (
                len(node_.successor) == 1
                and node_.successor[0].in_or_out
                and len(node_.precedence) == 1
                and not node_.precedence[0].in_or_out
            )

            # We override the output names of the operator preceeding ZipMap with the output names of the ZipMap.
            # This will evenutally create problems if the output_names of the predecessor
            # are used somewhere else, but for the moment it works.
            # Perhaps a better strategy is to add an identity node.
            input_keys = list(node_.input.keys())
            for i in range(len(input_keys)):
                node_.precedence[0].output.pop(input_keys[i])
                node_.precedence[0].output[node_.origin.output[i]] = node_.origin.output[i]
            node_.precedence[0].origin.output[:] = node_.output.values()
        else:
            output_node_list.append(node_)

    return output_node_list


# Registered API parsers.
if sklearn_installed():
    sklearn_api_parsers_map = _build_sklearn_api_parsers_map()
