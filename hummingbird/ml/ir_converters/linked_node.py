# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converters for LinkedNode IR are stored in this file.
"""

import os
from uuid import uuid4
from distutils.version import LooseVersion

from onnxconverter_common.optimizer import LinkedNode, _topological_sort
from onnxconverter_common.registration import get_converter
import onnxruntime as ort
import onnx
from onnx import helper, shape_inference
import torch

from ..operator_converters import constants
from ..supported import get_onnxml_api_operator_name
from ..exceptions import MissingConverter


def convert(
    node_list, input_tensors, initializers, output_names, test_data, output_model_name=None, target_opset=9, extra_config={},
):
    """
    This function converts an input list of `onnxconverter_common.optimizer.LinkedNode`s model into a ONNX model.
    The supported operators can be found at `hummingbird.ml.supported`.

    Args:
        node_list: A list of `onnxconverter_common.optimizer.LinkedNode`s representing a ONNX-ML model
        input_tensors: The input tensors of the model
        initializers: The initializers of the model
        output_names: A python list containing the output names expected from the translated model.
                      Should be a subset of the output variables in the input ONNX-ML model.
        test_data: Some input data used to trace the model execution
        output_model_name: The name of the ONNX model returned as output
        target_opset: The opset to use for the generated ONNX model
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Returns:
        A model containing only *ONNX* operators.
    """
    if output_model_name is None:
        output_model_name = str(uuid4().hex) + ".onnx"
    onnx_model_name = output_model_name
    # Container for the output IR operators. We will be using this to generate the final ONNX model.
    output_onnx_ir = []
    # During translation we need to make sure that each variable is unique across operators.
    variable_check = set()
    initializer_tensors = {}
    output_tensors = {}

    assert len(input_tensors) == 1, "Hummingbird ONNX converter currently support only graphs with a single input."

    # Used to pass tensor shapes over the converters.
    all_tensors = {in_.name: in_ for in_ in input_tensors}

    for init in initializers:
        initializer_tensors[init.name] = init

    # This a new node list but with some node been removed plus variable renaming.
    new_node_list = _check_and_prepare_ir_graph_for_conversion(node_list, input_tensors)

    for node_ in new_node_list:
        try:
            alias = get_onnxml_api_operator_name(node_.op_type)

            if alias is not None:
                # This is a supported operator
                # The workflow is the following:
                # (1) We generate a PyTorch model out of the ONNX-ML operator
                # (2) We generate a ONNX model using pytorch-to-onnx export. This requires tracing
                # (3) We check that the variable names for the ONNX model do not overlap
                #     with some previously generate model
                # (4) We append the ONNX model to the output
                converter = get_converter(alias)

                # Convert the model from PyTorch to ONNX.
                if constants.N_FEATURES not in extra_config:
                    extra_config[constants.ONNX_INPUTS] = all_tensors
                # Early version of pytorch have a bug with exporting gemm into ONNX.
                trimmed_version = torch.__version__.split("+")[0]  # For Win pytorch has a +cpu or +gpu postfix
                torch_version = LooseVersion(trimmed_version)
                gemm_min = LooseVersion("1.5.0")
                if torch_version <= gemm_min:
                    extra_config[constants.TREE_IMPLEMENTATION] = "tree_trav"
                pytorch_model = converter(node_, extra_config=extra_config)

                # Generate the inputs for the tracing.
                test_output_names = {node_.origin.input[i] for i in range(len(node_.input))}
                if len(test_output_names) == 1 and next(iter(test_output_names)) == input_tensors[0].name:
                    # This is the first operator: use the input parameter.
                    inputs = torch.from_numpy(test_data)
                else:
                    # We are in the middle of a graph: generate inputs using the part of the graph converted so far.
                    graph = helper.make_graph(
                        output_onnx_ir, "hb-tmp", input_tensors, output_tensors.values(), initializer_tensors.values()
                    )
                    model = helper.make_model(graph, producer_name="Hummingbird")
                    session = ort.InferenceSession(model.SerializeToString())
                    assert len(session.get_inputs()) == 1

                    test_input_name = session.get_inputs()[0].name
                    onnx_pred = session.run(list(test_output_names), {test_input_name: test_data})[0]
                    inputs = torch.from_numpy(onnx_pred)

                # LinkedNode uses dictionaries and with Python 3.5 the order is not deterministic.
                # Here we sort as an hack: label will always go before probabilities.
                # We will find a better approach when we unify the IRs.
                conversion_output = list(node_.output.values())
                conversion_output.sort

                # Export the ONNX model for the current operator.
                torch.onnx.export(
                    pytorch_model,
                    inputs,
                    onnx_model_name,
                    input_names=node_.input.values(),
                    output_names=conversion_output,
                    keep_initializers_as_inputs=False,
                    opset_version=target_opset,
                    do_constant_folding=True,
                )
                onnx_model = onnx.load(onnx_model_name)

                # Generate the IR for the exported ONNX model.
                initializers = [] if onnx_model.graph.initializer is None else [in_ for in_ in onnx_model.graph.initializer]
                inputs = [in_.name for in_ in onnx_model.graph.input] + [init.name for init in initializers]
                converted_model_nodes = LinkedNode.build_from_onnx(
                    onnx_model.graph.node,
                    [],
                    inputs,
                    [] if onnx_model.graph.output is None else [o_.name for o_ in onnx_model.graph.output],
                )

                # Since each operator is exported into ONNX separately, we need to do some check about the naming
                # since variable names can overlap across models.
                # We start with the initializers
                _check_and_rename_variables(onnx_model, converted_model_nodes, alias, variable_check)

                # Add the newly generated nodes to the output.
                for converted_node in _topological_sort(converted_model_nodes):
                    output_onnx_ir.append(converted_node.origin)

                # Take track of inputs, outputs and initializers.
                for output in onnx_model.graph.output:
                    output_tensors[output.name] = output
                for init in onnx_model.graph.initializer:
                    initializer_tensors[init.name] = init

                os.remove(onnx_model_name)
            else:
                # We don't support this operator, just attach it to the current model.
                # Should eventually go here only for ONNX operators.
                assert len(node_.input) == 1 or len(node_.output) == 1

                output_onnx_ir.append(node_.origin)

                # Compute the shape of the output variable.
                tmp_graph = helper.make_graph(
                    [node_.origin],
                    "hb-shape-inference",
                    all_tensors.values(),  # inputs
                    [],  # outputs
                    initializer_tensors.values(),  # initializers
                )
                tmp_model = helper.make_model(tmp_graph, producer_name="Hummingbird")

                # Apply shape inference on the model
                inferred_model = shape_inference.infer_shapes(tmp_model)

                for output in inferred_model.graph.value_info:
                    if output.name in node_.output:
                        output_tensors[output.name] = output

            all_tensors.update(output_tensors)
        except ValueError:
            raise MissingConverter(
                "Unable to find converter for alias '{}'."
                "You may raise an issue at "
                "https://github.com/microsoft/hummingbird."
                "".format(node_.op_type)
            )
        except Exception as e:
            raise e

    # Generate the model.
    output_graph = onnx.helper.make_graph(
        output_onnx_ir,
        onnx_model_name,
        input_tensors,
        [output_tensors[output] for output in output_names],
        list(initializer_tensors.values()),
    )
    output_model = helper.make_model(output_graph, producer_name="Hummingbird")

    return output_model


def _check_and_prepare_ir_graph_for_conversion(node_list, input_tensors, extra_config={}):
    """
    Method used to:
    (1) Check that we support the operators in the input IR
    (2) Optimize the input IR, e.g., by removing unused subgraphs.
    When we go from ONNXML to ONNX:
    (1) we can remove some of the operators
    (2) we have to check that all operators int the IR graph are supported

    """
    output_node_list = []
    skip_list = []
    input_names = {in_.name for in_ in input_tensors}

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
            # This will evenutally create problems if the outputs of the predecessor
            # are used somewhere else, but for the moment it works.
            # Perhaps a better strategy is to add an identity node.
            input_keys = list(node_.input.keys())
            for i in range(len(input_keys)):
                node_.precedence[0].output.pop(input_keys[i])
                node_.precedence[0].output[node_.origin.output[i]] = node_.origin.output[i]
            node_.precedence[0].origin.output[:] = node_.output.values()
        elif node_.name in skip_list:
            continue
        elif len(node_.input) == 1 and not node_.origin.input[0] in input_names:
            # Remove sub-graphs for which we don't have an input.
            current_node = node_
            while len(current_node.successor) > 0:
                assert len(current_node.successor) == 1

                if current_node.successor[0].op_type is not None:
                    skip_list.append(current_node.successor[0].name)
                current_node = current_node.successor[0]
        else:
            output_node_list.append(node_)
            for out_ in node_.output.keys():
                input_names.add(out_)

    return output_node_list


def _check_and_rename_variables(onnx_model, converted_model_nodes, alias, variable_check):
    """
    Method used to check that variable naming is consistent, i.e., two variables do not have the same name.
    In case there is some clash, rename the latested added variable.
    """
    initializers = {}
    for i in range(len(onnx_model.graph.initializer)):
        initializer_name = onnx_model.graph.initializer[i].name
        replace = False
        if initializer_name.isnumeric():
            if int(initializer_name) in variable_check:
                new_var_name = max(variable_check) + 1
                variable_check.add(new_var_name)
                new_var_name = str(new_var_name)
                replace = True
            else:
                variable_check.add(int(initializer_name))
        else:
            new_var_name = onnx_model.graph.initializer[i].name + alias
            replace = True

        if replace:
            # Initializers are part of the input in our IR, rename those as well
            for converted_node in converted_model_nodes:
                if initializer_name in converted_node.input:
                    converted_node.input[new_var_name] = new_var_name
                    del converted_node.input[initializer_name]

                    # Also origin can have initializers as input
                    for j in range(len(converted_node.origin.input)):
                        if converted_node.origin.input[j] == initializer_name:
                            converted_node.origin.input[j] = new_var_name
                            initializers[new_var_name] = (alias + converted_node.unique_name, j)
            onnx_model.graph.initializer[i].name = new_var_name

    # Then numeric variables.
    # Non-numeric are ok since we explicitly set those.
    for converted_node in _topological_sort(converted_model_nodes):
        for i in range(len(converted_node.output)):
            if converted_node.origin.output[i].isnumeric():
                # Only track numeric variables
                if int(converted_node.origin.output[i]) in variable_check:
                    new_var_name = max(variable_check) + 1
                    variable_check.add(new_var_name)

                    # Make sure that successor operators get the updated variable name
                    for succ_ in converted_node.successor:
                        for j in range(len(succ_.origin.input)):
                            if succ_.origin.input[j] == converted_node.origin.output[i] and not (
                                succ_.origin.input[j] in initializers
                                and initializers[succ_.origin.input[j]] == (alias + succ_.unique_name, j)
                            ):
                                del succ_.input[succ_.origin.input[j]]
                                succ_.input[str(new_var_name)] = str(new_var_name)
                                succ_.origin.input[j] = str(new_var_name)
                    del converted_node.output[converted_node.origin.output[i]]
                    converted_node.output[str(new_var_name)] = str(new_var_name)
                    converted_node.origin.output[i] = str(new_var_name)
                else:
                    variable_check.add(int(converted_node.origin.output[i]))
