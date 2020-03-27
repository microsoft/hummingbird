# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from sklearn import pipeline

from ._supported_operators import _get_sklearn_operator_name
from .common._container import SklearnModelContainerNode
from .common._topology import Topology


def parse_sklearn_model(model, initial_types=None):
    """
    Puts *scikit-learn* object into an abstract container so that
    our framework can work seamlessly on models created
    with different machine learning tools.

    :param model: A scikit-learn model
    :param initial_types: a python list. Each element is a tuple of a
        variable name and a type defined in data_types.py

    :return: :class:`Topology <hummingbird.common._topology.Topology>`

    """

    raw_model_container = SklearnModelContainerNode(model)

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.

    topology = Topology(raw_model_container, initial_types)

    # Declare an object to provide variables' and operators' naming mechanism.
    # In contrast to CoreML, one global scope
    # is enough for parsing scikit-learn models.
    # scope = topology.declare_scope('__root__')

    # Declare input variables. They should be the inputs of the scikit-learn
    # model you want to convert into PyTorch.

    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(topology.declare_variable(var_name, initial_type))

    # The object raw_model_container is a part of the topology
    # we're going to return. We use it to store the inputs of
    # the scikit-learn's computational graph.

    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model as a Topology object.
    outputs = parse_sklearn(topology, model, inputs)

    # The object raw_model_container is a part of the topology we're
    # going to return. We use it to store the outputs of the
    # scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology


def _parse_sklearn_simple_model(topology, model, inputs):
    """
    This function handles all non-pipeline models.
    :param topology: Scope object
    :param model: A scikit-learn object (e.g., *OneHotEncoder*
        or *LogisticRegression*)
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next
        stage
    """

    # alias can be None
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a " "string '{0}'.".format(model))
    alias = _get_sklearn_operator_name(type(model))
    this_operator = topology.declare_operator(alias, model)
    this_operator.inputs = inputs

    # We assume that all scikit-learn operator produce a single output.
    variable = topology.declare_variable("variable", None)
    this_operator.outputs.append(variable)

    return this_operator.outputs


def parse_sklearn(topology, model, inputs):
    """
    This is a delegate function. It does nothing but invokes the
    correct parsing function according to the input model's type.

    :param topology: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder
        and LogisticRegression)
    :param inputs: A list of variables

    :return: The output variables produced by the input model
    """
    tmodel = type(model)
    if tmodel in sklearn_parsers_map:
        outputs = sklearn_parsers_map[tmodel](topology, model, inputs)
    else:
        outputs = _parse_sklearn_simple_model(topology, model, inputs)

    return outputs


def _parse_sklearn_pipeline(scope, model, inputs):
    """
    The basic ideas of scikit-learn parsing:
        1. Sequentially go though all stages defined in the considered
           scikit-learn pipeline
        2. The output variables of one stage will be fed into its next
           stage as the inputs.

    :param scope: Scope object defined in _topology.py
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    """
    for step in model.steps:
        inputs = parse_sklearn(scope, step[1], inputs)
    return inputs


def build_sklearn_parsers_map():
    # Adding parsers for edge cases
    map_parser = {
        pipeline.Pipeline: _parse_sklearn_pipeline
        # more will go here as added
    }
    return map_parser


# registered parsers
sklearn_parsers_map = build_sklearn_parsers_map()
