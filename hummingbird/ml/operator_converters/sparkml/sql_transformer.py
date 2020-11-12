# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for SparkML SQLTransformer
"""
import torch
from onnxconverter_common.registration import register_converter
from .._base_operator import BaseOperator
from .. import constants


class SQLTransformerModel(BaseOperator, torch.nn.Module):

    # List of supported unary SQL ops.
    supported_unary_ops = {
        'org.apache.spark.sql.catalyst.expressions.Sqrt': torch.sqrt
    }

    # List of supported binary SQL ops.
    supported_binary_ops = {
        'org.apache.spark.sql.catalyst.expressions.Add': torch.add,
        'org.apache.spark.sql.catalyst.expressions.Subtract': torch.sub,
        'org.apache.spark.sql.catalyst.expressions.Multiply': torch.mul,
        'org.apache.spark.sql.catalyst.expressions.Divide': torch.div,
    }

    def __init__(self, input_names, project_node, device):
        super(SQLTransformerModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.project_node = project_node

    def forward(self, *x):
        inputs_dict = {name : tensor for name, tensor in zip(self.input_names, x)}
        # The first node of the project_node_def is the name definition node. Hence ignored.
        return self.calculate_output_feature(self.project_node[1:], inputs_dict)[0]

    def calculate_output_feature(self, project_node_def_stack, inputs_dict):
        # Handles reading of a feature column.
        if project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.AttributeReference':
            return inputs_dict[project_node_def_stack[0]['name']], project_node_def_stack[1:]
        # Handles reading of a constant value.
        elif project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Literal':
            return float(project_node_def_stack[0]['value']), project_node_def_stack[1:]
        # Handles binary ops e.g., Add/Subtract/Mul/Div.
        elif project_node_def_stack[0]['class'] in self.supported_binary_ops:
            op = self.supported_binary_ops[project_node_def_stack[0]['class']]
            project_node_def_stack = project_node_def_stack[1:]
            left, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            right, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            return op(left, right), project_node_def_stack
        # Handles unary ops e.g., Sqrt.
        elif project_node_def_stack[0]['class'] in self.supported_unary_ops:
            op = self.supported_unary_ops[project_node_def_stack[0]['class']]
            project_node_def_stack = project_node_def_stack[1:]
            param, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            return op(param), project_node_def_stack
        else:
            raise RuntimeError('SQLTransformer encountered unsupported operator type: {}'.format(project_node_def_stack[0]['class']))


def convert_sparkml_sql_transformer(operator, device, extra_config):
    """
    Converter for `pyspark.ml.feature.SQLTransformer`

    Args:
        operator: An operator wrapping a `pyspark.ml.feature.SQLTransformer`
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    input_names = [input.raw_name for input in operator.inputs]
    return SQLTransformerModel(input_names, operator.operand, device)


register_converter("SparkMLSQLTransformer", convert_sparkml_sql_transformer)
