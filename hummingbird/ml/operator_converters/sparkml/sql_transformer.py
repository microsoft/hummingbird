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

# List of supported unary SQL ops.
SUPPORTED_UNARY_OPS = {
    'org.apache.spark.sql.catalyst.expressions.Sqrt': torch.sqrt
}

# List of supported binary SQL ops.
SUPPORTED_BINARY_OPS = {
    # Arithmatic ops.
    'org.apache.spark.sql.catalyst.expressions.Add': torch.add,
    'org.apache.spark.sql.catalyst.expressions.Subtract': torch.sub,
    'org.apache.spark.sql.catalyst.expressions.Multiply': torch.mul,
    'org.apache.spark.sql.catalyst.expressions.Divide': torch.div,

    # Logical ops.
    'org.apache.spark.sql.catalyst.expressions.Or': torch.logical_or,
    'org.apache.spark.sql.catalyst.expressions.And': torch.logical_and,
    'org.apache.spark.sql.catalyst.expressions.LessThan': torch.lt,
    'org.apache.spark.sql.catalyst.expressions.LessThanOrEqual': torch.le,
    'org.apache.spark.sql.catalyst.expressions.GreaterThan': torch.gt,
    'org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual': torch.ge,
    'org.apache.spark.sql.catalyst.expressions.EqualTo': torch.eq
}


class SQLWhereModel(BaseOperator, torch.nn.Module):

    def __init__(self, input_names, condition_node, device):
        super(SQLWhereModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.select_op = SQLSelectModel(input_names, condition_node, device)

    def forward(self, *x):
        filter_confition = self.select_op(*x)
        return [torch.masked_select(c, filter_confition).reshape(-1, 1) for c in x]


class SQLSelectModel(BaseOperator, torch.nn.Module):

    def __init__(self, input_names, project_node, device):
        super(SQLSelectModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.project_node = project_node

    def forward(self, *x):
        inputs_dict = {name : tensor for name, tensor in zip(self.input_names, x)}
        return self.calculate_output_feature(self.project_node, inputs_dict)[0]

    def calculate_output_feature(self, project_node_def_stack, inputs_dict):
        # Handles reading of a feature column.
        if project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.AttributeReference':
            return inputs_dict[project_node_def_stack[0]['name']], project_node_def_stack[1:]
        # Handles reading of a constant value.
        elif project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Literal':
            return float(project_node_def_stack[0]['value']), project_node_def_stack[1:]
        # Handles binary ops e.g., Add/Subtract/Mul/Div.
        elif project_node_def_stack[0]['class'] in SUPPORTED_BINARY_OPS:
            op = SUPPORTED_BINARY_OPS[project_node_def_stack[0]['class']]
            project_node_def_stack = project_node_def_stack[1:]
            left, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            right, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            return op(left, right), project_node_def_stack
        # Handles unary ops e.g., Sqrt.
        elif project_node_def_stack[0]['class'] in SUPPORTED_UNARY_OPS:
            op = SUPPORTED_UNARY_OPS[project_node_def_stack[0]['class']]
            project_node_def_stack = project_node_def_stack[1:]
            param, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            return op(param), project_node_def_stack
        # Handles IN clause statement.
        elif project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.In':
            n_right_elements = len(project_node_def_stack[0]['list'])
            project_node_def_stack = project_node_def_stack[1:]
            left, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            right_elements = []
            for _ in range(n_right_elements):
                right, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
                right_elements.append(right)

            right_elements = torch.tensor(right_elements, dtype=left.dtype)
            x = left == right_elements
            x = x.float()
            x = torch.sum(x, axis=1, keepdim=True) >= 1.0
            return x, project_node_def_stack
        # Handles CASE statement.
        elif project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.CaseWhen':
            project_node_def_stack = project_node_def_stack[1:]
            nodes = []
            while len(project_node_def_stack) > 0:
                n, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
                nodes.append(n)

            conditions = []
            values = []
            for i in range(0, len(nodes) - 1, 2):
                conditions.append(nodes[i])
                if type(nodes[i + 1]) == torch.tensor:
                    values.append(nodes[i + 1])
                else:
                    values.append(nodes[i + 1] * torch.ones_like(conditions[0]))

            conditions.append(torch.ones_like(conditions[0]).bool())
            if len(nodes) % 2 == 0:
                # No defined else condition. We return NaN to confirm to the SQL semantics.
                # Division by zero results in a NaN tensor.
                values.append(torch.zeros_like(conditions[0]) / 0.0)
            else:
                if type(nodes[-1]) == torch.tensor:
                    values.append(nodes[-1])
                else:
                    values.append(nodes[-1] * torch.ones_like(conditions[0]))
            # In a tie, argmax returns the index of the first element.
            index = torch.argmax(torch.cat(conditions, dim=1).int(), dim=1, keepdim=True)
            values = torch.cat(values, dim=1)
            return torch.gather(values, 1, index), project_node_def_stack
        else:
            raise RuntimeError('SQLTransformer encountered unsupported operator type: {}'.format(project_node_def_stack[0]['class']))


def convert_sparkml_sql_select(operator, device, extra_config):
    """
    Converter for SELECT statement in `pyspark.ml.feature.SQLTransformer`

    Args:
        operator: JSON encoded filter condition extracted from SparkSQL parsed tree
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    input_names = [input.raw_name for input in operator.inputs]
    # First entry in the raw_operator is the Alias, which we ignore.
    return SQLSelectModel(input_names, operator.raw_operator[1:], device)


def convert_sparkml_sql_where(operator, device, extra_config):
    """
    Converter for WHERE statement in `pyspark.ml.feature.SQLTransformer`

    Args:
        operator: JSON encoded project_tree extracted from SparkSQL parsed tree
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    input_names = [input.raw_name for input in operator.inputs]
    return SQLWhereModel(input_names, operator.raw_operator, device)


register_converter("SparkMLSQLSelectModel", convert_sparkml_sql_select)
register_converter("SparkMLSQLWhereModel", convert_sparkml_sql_where)
