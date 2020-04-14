# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from sklearn import pipeline

from ._supported_operators import get_sklearn_api_operator_name
from ._container import SklearnModelContainerNode
from ._topology import Topology


def parse_sklearn_api_model(model):
    """
    Puts *scikit-learn* object into an abstract representation so that
    our framework can work seamlessly on models created
    with different machine learning tools.

    :param model: A model object in scikit-learn format

    :return: :class:`Topology <hummingbird.common._topology.Topology>`

    """

    raw_model_container = SklearnModelContainerNode(model)

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.
    topology = Topology(raw_model_container)

    # Declare input variables. Sklearn always gets as input a single
    # dataframe, therefore by default we start with a single `input` variable
    inputs = []
    inputs.append(topology.declare_variable("input"))

    # The object raw_model_container is a part of the topology
    # we're going to return. We use it to store the inputs of
    # the scikit-learn's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model as a Topology object.
    outputs = parse_sklearn_api(topology, model, inputs)

    # The object raw_model_container is a part of the topology we're
    # going to return. We use it to store the outputs of the
    # scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology


def parse_sklearn_api(topology, model, inputs):
    """
    This is a delegate function. It does nothing but invokes the
    correct parsing function according to the input model's type.

    :param topology: The representation of the model graph
    :param model: A scikit-learn object
    :param inputs: A list of variables

    :return: The output variables produced by the input model
    """
    tmodel = type(model)
    if tmodel in sklearn_api_parsers_map:
        outputs = sklearn_api_parsers_map[tmodel](topology, model, inputs)
    else:
        outputs = _parse_sklearn_single_model(topology, model, inputs)

    return outputs


def _parse_sklearn_single_model(topology, model, inputs):
    """
    This function handles all sklearn objects composed by a single model.
    :param topology: The representation of the model graph
    :param model: A scikit-learn object
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    """

    # Alias can be None.
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a " "string '{0}'.".format(model))
    alias = get_sklearn_api_operator_name(type(model))
    this_operator = topology.declare_operator(alias, model)
    this_operator.inputs = inputs

    # We assume that all scikit-learn operators produce a single output.
    variable = topology.declare_variable("variable", None)
    this_operator.outputs.append(variable)

    return this_operator.outputs


def _parse_sklearn_pipeline(topology, model, inputs):
    """
    The basic ideas of scikit-learn API parsing:
        1. Sequentially go though all stages defined in the considered
           scikit-learn pipeline
        2. The output variables of one stage will be fed into its next
           stage as the inputs.

    :param topology: The internal abstract representation for the model
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    """
    for step in model.steps:
        inputs = parse_sklearn_api(topology, step[1], inputs)
    return inputs


def _build_sklearn_api_parsers_map():
    # Parsers for edge cases are going here.
    map_parser = {
        pipeline.Pipeline: _parse_sklearn_pipeline
        # More will go here as added.
    }
    return map_parser


# Registered API parsers.
sklearn_api_parsers_map = _build_sklearn_api_parsers_map()
