# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All functions used for parsing input models are listed here.
"""
from onnxconverter_common.container import CommonSklearnModelContainer
from onnxconverter_common.topology import Topology

from .supported import get_sklearn_api_operator_name


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


# Registered API parsers.
sklearn_api_parsers_map = _build_sklearn_api_parsers_map()
