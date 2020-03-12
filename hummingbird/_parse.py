# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from ._supported_operators import _get_sklearn_operator_name
from .common._container import SklearnModelContainerNode
from .common._topology import Topology
from .common.data_types import TensorType
from .common.utils import get_column_indices


def _fetch_input_slice(scope, inputs, column_indices):
    if not isinstance(inputs, list):
        raise TypeError("Parameter inputs must be a list.")
    if len(inputs) == 0:
        raise RuntimeError("Operator ArrayFeatureExtractor requires at "
                           "least one inputs.")
    if len(inputs) != 1:
        raise RuntimeError("Operator ArrayFeatureExtractor does not support "
                           "multiple input tensors.")
    if (isinstance(inputs[0].type, TensorType) and
            inputs[0].type.shape[1] == len(column_indices)):
        # No need to extract.
        return inputs
    array_feature_extractor_operator = scope.declare_operator(
        'SklearnArrayFeatureExtractor')
    array_feature_extractor_operator.inputs = inputs
    array_feature_extractor_operator.column_indices = column_indices

    output_variable_name = scope.declare_variable(
        'extracted_feature_columns', None)
    array_feature_extractor_operator.outputs.append(output_variable_name)
    return array_feature_extractor_operator.outputs


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

    topology = Topology(raw_model_container, initial_types=initial_types)

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
        raise RuntimeError("Parameter model must be an object not a "
                           "string '{0}'.".format(model))
    alias = _get_sklearn_operator_name(type(model))
    this_operator = topology.declare_operator(alias, model)
    this_operator.inputs = inputs

    # We assume that all scikit-learn operator produce a single output.
    variable = topology.declare_variable('variable', None)
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


def _parse_sklearn_feature_union(scope, model, inputs):
    """
    :param scope: Scope object
    :param model: A scikit-learn FeatureUnion object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by feature union
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, transform in model.transformer_list:
        transformed_result_names.append(
            _parse_sklearn_simple_model(
                scope, transform, inputs)[0])
        if model.transformer_weights is not None and name in model.transformer_weights:
            transform_result = [transformed_result_names.pop()]
            # Create a Multiply PyTorch node
            multiply_operator = scope.declare_operator('SklearnMultiply')
            multiply_operator.inputs = transform_result
            multiply_operator.operand = model.transformer_weights[name]
            multiply_output = scope.declare_variable(
                'multiply_output', None)
            multiply_operator.outputs.append(multiply_output)
            transformed_result_names.append(multiply_operator.outputs[0])

    # Create a Concat PyTorch node
    concat_operator = scope.declare_operator('SklearnConcat')
    concat_operator.inputs = transformed_result_names

    # Declare output name of scikit-learn FeatureUnion
    union_name = scope.declare_variable('union', None)
    concat_operator.outputs.append(union_name)

    return concat_operator.outputs


def _prepare_columns_idx(column_indices):
    if isinstance(column_indices, slice):
        return list(range(
            column_indices.start
            if column_indices.start is not None else 0,
            column_indices.stop, column_indices.step
            if column_indices.step is not None else 1))
    elif isinstance(column_indices, (int, str)):
        return [column_indices]
    elif isinstance(column_indices, list):
        if len(column_indices) == 0:
            return column_indices
        elif isinstance(column_indices[0], bool):
            return [i for i in range(
                len(column_indices)) if column_indices[i]]


def _parse_sklearn_column_transformer(scope, model, inputs):
    """
    :param scope: Scope object
    :param model: A *scikit-learn* *ColumnTransformer* object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by column transformer
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    input_indices = []
    # Encode each transform as our IR object
    for name, _, column_indices in model.transformers:
        column_indices = _prepare_columns_idx(column_indices)
        names = get_column_indices(column_indices, inputs, multiple=True)
        for _, inp_ind in names.items():
            input_indices.extend(inp_ind)
        transform_inputs = []
        for pytorch_var, pytorch_is in names.items():
            tr_inputs = _fetch_input_slice(
                scope, [inputs[pytorch_var]], pytorch_is)
            transform_inputs.extend(tr_inputs)
        if len(transform_inputs) > 1:
            # Many PyTorch operators expect one input vector,
            # the default behaviour is to merge columns.
            ty = transform_inputs[0].type.__class__([None, None])

            conc_op = scope.declare_operator('SklearnConcat')
            conc_op.inputs = transform_inputs
            conc_names = scope.declare_variable('merged_columns', ty)
            conc_op.outputs.append(conc_names)
            transform_inputs = [conc_names]
        model_obj = model.named_transformers_[name]
        if isinstance(model_obj, str):
            if model_obj == "passthrough":
                model_obj = FunctionTransformer()
            else:
                raise RuntimeError("Unknown operator alias "
                                   "'{0}'. These are specified in "
                                   "_supported_operators.py."
                                   "".format(model_obj))

        var_out = parse_sklearn(scope, model_obj, transform_inputs)[0]
        transformed_result_names.append(var_out)

    if model.remainder == "passthrough":
        input_indices = set(input_indices)
        left_over = [i for i in range(len(inputs)) if i not in input_indices]
        if len(left_over) > 0:
            for i in sorted(left_over):
                pytorch_var, pytorch_is = get_column_indices(
                    [i], inputs, multiple=False)
                tr_inputs = _fetch_input_slice(
                    scope, [inputs[pytorch_var]], pytorch_is)
                transformed_result_names.extend(tr_inputs)

    # Create a Concat PyTorch node
    if len(transformed_result_names) > 1:
        concat_operator = scope.declare_operator('SklearnConcat')
        concat_operator.inputs = transformed_result_names

        # Declare output name of scikit-learn ColumnTransformer
        transformed_column_name = scope.declare_variable(
            'transformed_column', None)
        concat_operator.outputs.append(transformed_column_name)
        return concat_operator.outputs
    else:
        return transformed_result_names


def build_sklearn_parsers_map():
    # Adding parsers for edge cases
    map_parser = {
        pipeline.Pipeline: _parse_sklearn_pipeline,
        pipeline.FeatureUnion: _parse_sklearn_feature_union,
        ColumnTransformer: _parse_sklearn_column_transformer
    }
    return map_parser


def update_registered_parser(model, parser_fct):
    """
    Registers or updates a parser for a new model.
    A parser returns the expected output of a model.

    :param model: model class
    :param parser_fct: parser, signature is the same as
        :func:`parse_sklearn <hummingbird._parse.parse_sklearn>`
    """

    sklearn_parsers_map[model] = parser_fct


# registered parsers
sklearn_parsers_map = build_sklearn_parsers_map()
