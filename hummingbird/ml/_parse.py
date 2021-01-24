# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All functions used for parsing input models are listed here.
Some code here have been copied from https://github.com/onnx/sklearn-onnx/.
"""
from collections import OrderedDict
from copy import deepcopy
import pprint
from uuid import uuid4

import numpy as np
from onnxconverter_common.optimizer import LinkedNode, _topological_sort

from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.preprocessing import OneHotEncoder

from .containers import CommonSklearnModelContainer, CommonONNXModelContainer, CommonSparkMLModelContainer
from ._topology import Topology
from ._utils import sklearn_installed, sparkml_installed
from .operator_converters import constants
from .supported import get_sklearn_api_operator_name, get_onnxml_api_operator_name, get_sparkml_api_operator_name

do_not_merge_columns = tuple(filter(lambda op: op is not None, [OneHotEncoder, ColumnTransformer]))


def parse_sklearn_api_model(model, extra_config={}):
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

    # Declare input variables.
    inputs = _declare_input_variables(topology, raw_model_container, extra_config)

    # Parse the input scikit-learn model into a topology object.
    # Get the outputs of the model.
    outputs = _parse_sklearn_api(topology, model, inputs)

    # Declare output variables.
    _declare_output_variables(raw_model_container, extra_config, outputs)

    return topology


def parse_sparkml_api_model(model, extra_config={}):
    """
    Puts *Spark-ML* object into an abstract representation so that our framework can work seamlessly on models created
    with different machine learning tools.

    Args:
        model: A model object in Spark-ML format

    Returns:
        A `onnxconverter_common.topology.Topology` object representing the input model
    """
    assert model is not None, "Cannot convert a mode of type None."

    raw_model_container = CommonSparkMLModelContainer(model)

    # Declare a computational graph. It will become a representation of
    # the input Spark-ML model after parsing.
    topology = Topology(raw_model_container)

    # Declare input variables.
    inputs = _declare_input_variables(topology, raw_model_container, extra_config)

    # Parse the input Spark-ML model into its topology with the topology.
    # Get the outputs of the model.
    current_op_outputs, _ = _parse_sparkml_api(topology, model, inputs)

    # Declare output variables.
    _declare_output_variables(raw_model_container, extra_config, current_op_outputs)

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

    # Declare input variables.
    inputs = []
    for i in model.graph.input:
        inputs.append(topology.declare_logical_variable(i.name))

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the inputs of the ONNX graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input ONNX model into its topology with the topology.
    _parse_onnx_api(topology, model, inputs)

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the output_names of the ONNX graph.
    for o in model.graph.output:
        raw_model_container.add_output(topology.declare_logical_variable(o.name))

    return topology


def _declare_input_variables(topology, raw_model_container, extra_config):
    # Declare input variables.
    inputs = []
    n_inputs = extra_config[constants.N_INPUTS] if constants.N_INPUTS in extra_config else 1
    if constants.INPUT_NAMES in extra_config:
        assert n_inputs == len(extra_config[constants.INPUT_NAMES])
    if constants.TEST_INPUT in extra_config:
        from onnxconverter_common.data_types import (
            FloatTensorType,
            DoubleTensorType,
            Int32TensorType,
            Int64TensorType,
            StringTensorType,
        )

        test_input = extra_config[constants.TEST_INPUT] if n_inputs > 1 else [extra_config[constants.TEST_INPUT]]
        for i in range(n_inputs):
            input = test_input[i]
            input_name = (
                extra_config[constants.INPUT_NAMES][i] if constants.INPUT_NAMES in extra_config else "input_{}".format(i)
            )
            if input.dtype == np.float32:
                input_type = FloatTensorType(input.shape)
            elif input.dtype == np.float64:
                input_type = DoubleTensorType(input.shape)
            elif input.dtype == np.int32:
                input_type = Int32TensorType(input.shape)
            elif input.dtype == np.int64:
                input_type = Int64TensorType(input.shape)
            elif input.dtype.kind in constants.SUPPORTED_STRING_TYPES:
                input_type = StringTensorType(input.shape)
            else:
                raise NotImplementedError(
                    "Type {} not supported. Please fill an issue on https://github.com/microsoft/hummingbird/.".format(
                        input.dtype
                    )
                )
            inputs.append(topology.declare_logical_variable(input_name, type=input_type))
    else:
        # We have no information on the input. Sklearn/Spark-ML always gets as input a single dataframe,
        # therefore by default we start with a single `input` variable
        input_name = extra_config[constants.INPUT_NAMES][0] if constants.TEST_INPUT in extra_config else "input"
        var = topology.declare_logical_variable(input_name)
        inputs.append(var)

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the inputs of the Sklearn/Spark-ML's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    return inputs


def _declare_output_variables(raw_model_container, extra_config, outputs):
    # Declare output variables.
    # Use the output names specified by the user, if any
    if constants.OUTPUT_NAMES in extra_config:
        assert len(extra_config[constants.OUTPUT_NAMES]) == len(outputs)
        for i in range(len(outputs)):
            outputs[i].raw_name = extra_config[constants.OUTPUT_NAMES][i]

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the outputs of the Sklearn/Spark-ML's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)


def _parse_sklearn_api(topology, model, inputs):
    """
    This is a delegate function adding the model to the input topology.
    It does nothing but invokes the correct parsing function according to the input model's type.

    Args:
        topology: The ``hummingbitd.ml._topology.Topology`` object where the model will be added
        model: A scikit-learn model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        The output `onnxconverter_common.topology.Variable`s produced by the input model
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

    Args:
        topology: The ``hummingbitd.ml._topology.Topology`` object where the model will be added
        model: A scikit-learn model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        A list of output `onnxconverter_common.topology.Variable` which will be passed to next stage
    """
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a " "string '{0}'.".format(model))

    alias = get_sklearn_api_operator_name(type(model))
    this_operator = topology.declare_logical_operator(alias, model)
    this_operator.inputs = inputs

    # We assume that all scikit-learn operators produce a single output.
    variable = topology.declare_logical_variable("variable")
    this_operator.outputs.append(variable)

    return this_operator.outputs


def _parse_sklearn_pipeline(topology, model, inputs):
    """
    The basic ideas of scikit-learn parsing:
        1. Sequentially go though all stages defined in the considered
           scikit-learn pipeline
        2. The output variables of one stage will be fed into its next
           stage as the inputs.
    :param topology: Topology object defined in _topology.py
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    """
    for step in model.steps:
        inputs = _parse_sklearn_api(topology, step[1], inputs)
    return inputs


def _parse_sparkml_pipeline(topology, model, all_outputs):
    """
    The basic ideas of Spark-ML parsing:
        1. Sequentially go though all stages defined in the considered
           Spark-ML pipeline passing all outputs that has been generated so far
           as the input. Operator will pick which inputs to operates on.
        2. The output variables of one stage will be fed into its next
           stage as the inputs.
    :param topology: Topology object defined in _topology.py
    :param model: Spark-ML pipeline object
    :param all_outputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    """
    for step in model.stages:
        current_op_outputs, all_outputs = _parse_sparkml_api(topology, step, all_outputs)
    return current_op_outputs, all_outputs


def _parse_sklearn_feature_union(topology, model, inputs):
    """
    Taken from https://github.com/onnx/sklearn-onnx/blob/9939c089a467676f4ffe9f3cb91098c4841f89d8/skl2onnx/_parse.py#L199.
    :param topology: Topology object
    :param model: A scikit-learn FeatureUnion object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by feature union
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, transform in model.transformer_list:
        transformed_result_names.append(_parse_sklearn_single_model(topology, transform, inputs)[0])
        if model.transformer_weights is not None and name in model.transformer_weights:
            transform_result = [transformed_result_names.pop()]
            # Create a Multiply node
            multiply_operator = topology.declare_logical_operator("SklearnMultiply")
            multiply_operator.inputs = transform_result
            multiply_operator.operand = model.transformer_weights[name]
            multiply_output = topology.declare_logical_variable("multiply_output")
            multiply_operator.outputs.append(multiply_output)
            transformed_result_names.append(multiply_operator.outputs[0])

    # Create a Concat operator
    concat_operator = topology.declare_logical_operator("SklearnConcat")
    concat_operator.inputs = transformed_result_names

    # Declare output name of scikit-learn FeatureUnion
    union_name = topology.declare_logical_variable("union")
    concat_operator.outputs.append(union_name)

    return concat_operator.outputs


def _parse_sklearn_multi_output_regressor(topology, model, inputs):
    """
    :param topology: Topology object
    :param model: A *scikit-learn* *MultiOutputRegressor* object
    :param inputs: A list of Variable objects
    :return: Output produced by MultiOutputRegressor
    """
    outputs = []
    for estimator in model.estimators_:
        outputs.append(_parse_sklearn_api(topology, estimator, inputs)[0])

    conc_op = topology.declare_logical_operator("SklearnConcat")
    conc_op.inputs = outputs
    conc_names = topology.declare_logical_variable("concat_outputs")
    conc_op.outputs.append(conc_names)
    return conc_op.outputs


def _parse_sklearn_regressor_chain(topology, model, inputs):
    """
    :param topology: Topology object
    :param model: A *scikit-learn* *RegressorChain* object
    :param inputs: A list of Variable objects
    :return: Output produced by RegressorChain
    """
    outputs = []
    for estimator in model.estimators_:
        outputs.append(_parse_sklearn_api(topology, estimator, inputs)[0])
        conc_op = topology.declare_logical_operator("SklearnConcat")
        conc_op.inputs.extend(inputs)
        conc_op.inputs.append(outputs[-1])
        conc_names = topology.declare_logical_variable("concat_inputs")
        conc_op.outputs.append(conc_names)
        inputs = conc_op.outputs

    conc_op = topology.declare_logical_operator("SklearnConcat")
    if model.order is not None:
        reorderd_outputs = [None for _ in outputs]
        for i, pos in enumerate(model.order):
            reorderd_outputs[pos] = outputs[i]
        outputs = reorderd_outputs

    conc_op.inputs = outputs
    conc_names = topology.declare_logical_variable("concat_outputs")
    conc_op.outputs.append(conc_names)
    return conc_op.outputs


def _parse_sklearn_column_transformer(topology, model, inputs):
    """
    Taken from https://github.com/onnx/sklearn-onnx/blob/9939c089a467676f4ffe9f3cb91098c4841f89d8/skl2onnx/_parse.py#L238.
    :param topology: Topology object
    :param model: A *scikit-learn* *ColumnTransformer* object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by column transformer
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, op, column_indices in model.transformers_:
        if op == "drop":
            continue
        if isinstance(column_indices, slice):
            column_indices = list(
                range(
                    column_indices.start if column_indices.start is not None else 0,
                    column_indices.stop,
                    column_indices.step if column_indices.step is not None else 1,
                )
            )
        elif isinstance(column_indices, (int, str)):
            column_indices = [column_indices]
        if len(column_indices) == 0:
            continue
        names = _get_column_indices(column_indices, inputs, len(inputs) > 1)
        transform_inputs = []
        for onnx_var, onnx_is in names.items():
            tr_inputs = _fetch_input_slice(topology, [inputs[onnx_var]], onnx_is)
            transform_inputs.extend(tr_inputs)

        merged_cols = False
        if len(transform_inputs) > 1:
            if isinstance(op, pipeline.Pipeline):
                if not isinstance(op.steps[0][1], do_not_merge_columns):
                    merged_cols = True
            elif not isinstance(op, do_not_merge_columns):
                merged_cols = True

        if merged_cols:
            # Many ONNX operators expect one input vector, the default behaviour is to merge columns.
            ty = transform_inputs[0].type.__class__([None, None])

            conc_op = topology.declare_logical_operator("SklearnConcat")
            conc_op.inputs = transform_inputs
            conc_names = topology.declare_logical_variable("merged_columns", ty)
            conc_op.outputs.append(conc_names)
            transform_inputs = [conc_names]

        model_obj = model.named_transformers_[name]
        if isinstance(model_obj, str):
            if model_obj == "passthrough":
                var_out = transform_inputs[0]
            elif model_obj == "drop":
                var_out = None
            else:
                raise RuntimeError(
                    "Unknown operator alias " "'{0}'. These are specified in " "supported.py." "".format(model_obj)
                )
        else:
            var_out = _parse_sklearn_api(topology, model_obj, transform_inputs)[0]
            if model.transformer_weights is not None and name in model.transformer_weights:
                # Create a Multiply node
                multiply_operator = topology.declare_logical_operator("SklearnMultiply")
                multiply_operator.inputs.append(var_out)
                multiply_operator.operand = model.transformer_weights[name]
                var_out = topology.declare_logical_variable("multiply_output")
                multiply_operator.outputs.append(var_out)
        if var_out:
            transformed_result_names.append(var_out)

    # Create a Concat node
    if len(transformed_result_names) > 1:
        concat_operator = topology.declare_logical_operator("SklearnConcat")
        concat_operator.inputs = transformed_result_names

        # Declare output name of scikit-learn ColumnTransformer
        transformed_column_name = topology.declare_logical_variable("transformed_column")
        concat_operator.outputs.append(transformed_column_name)
        return concat_operator.outputs
    return transformed_result_names


def _build_sklearn_api_parsers_map():
    # Parsers for edge cases are going here.
    map_parser = {
        ColumnTransformer: _parse_sklearn_column_transformer,
        MultiOutputRegressor: _parse_sklearn_multi_output_regressor,
        pipeline.Pipeline: _parse_sklearn_pipeline,
        pipeline.FeatureUnion: _parse_sklearn_feature_union,
        RegressorChain: _parse_sklearn_regressor_chain,
        # More parsers will go here
    }

    return map_parser


def _build_sparkml_api_parsers_map():
    # Parsers for edge cases are going here.
    from pyspark.ml.pipeline import PipelineModel

    map_parser = {
        PipelineModel: _parse_sparkml_pipeline,
        # More parsers will go here
    }

    return map_parser


def _parse_onnx_api(topology, model, inputs):
    """
    This function handles all input ONNX models.

    Args:
        topology: The ``onnxconverter_common.topology.Topology`` where the model will be added
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
        _parse_onnx_single_operator(topology, node)


def _parse_onnx_single_operator(topology, operator):
    """
    This function handles the parsing of all ONNX operators.

    Args:
        topology: The ``onnxconverter_common.topology.Topology`` where the model will be added
        model: An ONNX operator
    """

    # Identify nodes can just be skipped.
    if operator.op_type == "Identity":
        return

    # Add the operator in the topology.
    alias = get_onnxml_api_operator_name(operator.op_type)
    this_operator = topology.declare_logical_operator(alias, operator)

    # Register the operator's inputs.
    input_names = list(operator.origin.input)
    this_operator.inputs = [topology.variables[in_] for in_ in input_names if in_ in topology.variables]

    # Register the operator's outpurs.
    output_names = list(operator.output.keys())
    output_names.sort()
    for output in output_names:
        variable = topology.declare_logical_variable(output)
        this_operator.outputs.append(variable)


def _parse_sparkml_api(topology, model, inputs):
    """
    This function handles all input Spark-ML models.

    Args:
        topology: The ``onnxconverter_common.topology.Topology`` where the model will be added
        model: A Spark-ML model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        A list of output `onnxconverter_common.topology.Variable` which will be passed to next stage
    """
    tmodel = type(model)
    if tmodel in sparkml_api_parsers_map:
        outputs = sparkml_api_parsers_map[tmodel](topology, model, inputs)
    else:
        outputs = _parse_sparkml_single_operator(topology, model, inputs)
    return outputs


def _parse_sparkml_single_operator(topology, operator, all_inputs):
    """
    This function handles the parsing of all Spark-ML operators.

    Args:
        topology: The ``onnxconverter_common.topology.Topology`` where the model will be added
        model: A Spark-ML operator
        all_inputs: A list of `onnxconverter_common.topology.Variable`s
    """
    import inspect

    if isinstance(operator, str):
        raise RuntimeError("Parameter operator must be an object not a " "string '{0}'.".format(operator))

    alias = get_sparkml_api_operator_name(type(operator))
    this_operator = topology.declare_logical_operator(alias, operator)

    if hasattr(operator, "getInputCol") and callable(operator.getInputCol):
        this_operator.inputs = [i for i in all_inputs if i.raw_name == operator.getInputCol()]
    elif hasattr(operator, "getInputCols") and callable(operator.getInputCols):
        temp = {i.raw_name: i for i in all_inputs if i.raw_name in operator.getInputCols()}
        this_operator.inputs = [temp[i] for i in operator.getInputCols()]
    elif operator.hasParam("featuresCol"):
        col_name = [param[1] for param in operator.extractParamMap().items() if param[0].name == "featuresCol"][0]
        this_operator.inputs = [i for i in all_inputs if i.raw_name == col_name]
    else:
        print(operator.getParam("featuresCol"))
        raise RuntimeError("Unable to determine inputs for the Spark-ML operator: {}".format(type(operator)))

    if hasattr(operator, "getOutputCol") and callable(operator.getOutputCol):
        variable = topology.declare_logical_variable(operator.getOutputCol())
        this_operator.outputs.append(variable)
    elif hasattr(operator, "getOutputCols") and callable(operator.getOutputCols):
        for output_col in operator.getOutputCols():
            variable = topology.declare_logical_variable(output_col)
            this_operator.outputs.append(variable)
    else:
        variable = topology.declare_logical_variable("variable")
        this_operator.outputs.append(variable)

    return this_operator.outputs, all_inputs + this_operator.outputs


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


def _fetch_input_slice(topology, inputs, column_indices):
    """
    Taken from https://github.com/onnx/sklearn-onnx/blob/9939c089a467676f4ffe9f3cb91098c4841f89d8/skl2onnx/_parse.py#L53.
    """
    if not isinstance(inputs, list):
        raise TypeError("Parameter inputs must be a list.")
    if len(inputs) == 0:
        raise RuntimeError("Operator ArrayFeatureExtractor requires at least one inputs.")
    if len(inputs) != 1:
        raise RuntimeError("Operator ArrayFeatureExtractor does not support multiple input tensors.")
    if (
        len(inputs) == 1
        and len(column_indices) == 1
        and inputs[0].type is not None
        and (len(inputs[0].type.shape) == 1 or inputs[0].type.shape[1] == 1)
    ):
        return inputs
    array_feature_extractor_operator = topology.declare_logical_operator("SklearnArrayFeatureExtractor")
    array_feature_extractor_operator.inputs = inputs
    array_feature_extractor_operator.column_indices = column_indices
    output_variable_name = topology.declare_logical_variable("extracted_feature_columns", inputs[0].type)
    array_feature_extractor_operator.outputs.append(output_variable_name)
    return array_feature_extractor_operator.outputs


def _get_column_index(i, inputs):
    """
    Taken from https://github.com/onnx/sklearn-onnx/blob/9939c089a467676f4ffe9f3cb91098c4841f89d8/skl2onnx/common/utils.py#L50.
    Returns a tuples (variable index, column index in that variable).
    The function has two different behaviours, one when *i* (column index)
    is an integer, another one when *i* is a string (column name).
    If *i* is a string, the function looks for input name with this name and returns (index, 0).
    If *i* is an integer, let's assume first we have two inputs
    *I0 = FloatTensorType([None, 2])* and *I1 = FloatTensorType([None, 3])*,
    in this case, here are the results:
    ::
        get_column_index(0, inputs) -> (0, 0)
        get_column_index(1, inputs) -> (0, 1)
        get_column_index(2, inputs) -> (1, 0)
        get_column_index(3, inputs) -> (1, 1)
        get_column_index(4, inputs) -> (1, 2)
    """
    if isinstance(i, int):
        if i == 0:
            # Useful shortcut, skips the case when end is None
            # (unknown dimension)
            return 0, 0
        if inputs[0].type is None:
            return 0, i
        vi = 0
        pos = 0
        end = inputs[0].type.shape[1]
        if end is None:
            raise RuntimeError(
                "Cannot extract a specific column {0} when " "one input ('{1}') has unknown " "dimension.".format(i, inputs[0])
            )
        while True:
            if pos <= i < end:
                return (vi, i - pos)
            vi += 1
            pos = end
            if vi >= len(inputs):
                raise RuntimeError(
                    "Input {} (i={}, end={}) is not available in\n{}".format(vi, i, end, pprint.pformat(inputs))
                )
            rel_end = inputs[vi].type.shape[1]
            if rel_end is None:
                raise RuntimeError(
                    "Cannot extract a specific column {0} when "
                    "one input ('{1}') has unknown "
                    "dimension.".format(i, inputs[vi])
                )
            end += rel_end
    else:
        assert isinstance(
            i, str
        ), "Type {} not supported. Please fill an issue on https://github.com/microsoft/hummingbird/.".format(type(i))
        for ind, inp in enumerate(inputs):
            if inp.onnx_name == i:
                return ind, 0
        raise RuntimeError("Unable to find column name '{0}'".format(i))


def _get_column_indices(indices, inputs, multiple=False):
    """
    Taken from https://github.com/onnx/sklearn-onnx/blob/9939c089a467676f4ffe9f3cb91098c4841f89d8/skl2onnx/common/utils.py#L105.
    Returns the requested graph inpudes based on their indices or names. See `_parse._get_column_index`.
    Args:
        indices: variables indices or names
        inputs: model inputs

    Returns:
        a tuple *(variable name, list of requested indices)* if *multiple* is False, a dictionary *{ var_index: [ list of
        requested indices ] }* if *multiple* is True
    """
    if multiple:
        res = OrderedDict()
        for p in indices:
            pt_var, pt_i = _get_column_index(p, inputs)
            if pt_var not in res:
                res[pt_var] = []
            res[pt_var].append(pt_i)
        return res
    else:
        pt_var = None
        pt_is = []
        for p in indices:
            pt_v, pt_i = _get_column_index(p, inputs)
            pt_is.append(pt_i)
            if pt_var is None:
                pt_var = pt_v
            elif pt_var != pt_v:
                raise NotImplementedError(
                    "Hummingbird is not able to merge multiple columns from "
                    "multiple variables ({0}). You should think about merging "
                    "initial types.".format([pt_var, pt_v])
                )
        return {pt_var: pt_is}


# Registered API parsers.
if sklearn_installed():
    sklearn_api_parsers_map = _build_sklearn_api_parsers_map()
if sparkml_installed():
    sparkml_api_parsers_map = _build_sparkml_api_parsers_map()
