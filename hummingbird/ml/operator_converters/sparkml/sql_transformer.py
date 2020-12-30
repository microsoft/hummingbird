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
import numpy as np

# List of supported scalar SQL ops.
SUPPORTED_SCALAR_OPS = {
    'org.apache.spark.sql.catalyst.expressions.Sqrt': torch.sqrt,
    'org.apache.spark.sql.catalyst.expressions.IsNotNull': lambda x: torch.logical_not(torch.isnan(x)),
    # We rely on PyTorch's automatic type broadcast option.
    'org.apache.spark.sql.catalyst.expressions.Cast': lambda x: x
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
        filter_condition = self.select_op(*x)
        return [torch.masked_select(c, filter_condition).reshape(-1, 1) for c in x]


class SQLSelectModel(BaseOperator, torch.nn.Module):

    def __init__(self, input_names, project_node, device):
        super(SQLSelectModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.project_node = project_node

    def forward(self, *x):
        inputs_dict = {name.lower() : tensor for name, tensor in zip(self.input_names, x)}
        return self.calculate_output_feature(self.project_node, inputs_dict)[0]

    def calculate_output_feature(self, project_node_def_stack, inputs_dict):
        print("\ncurrent stack\n", project_node_def_stack[0])
        # Handles reading of a feature column.
        if project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.AttributeReference':
            if project_node_def_stack[0]['name'] not in inputs_dict:
                print('')
                pass
            return inputs_dict[project_node_def_stack[0]['name']], project_node_def_stack[1:]
        # Handles reading of a constant value.
        elif project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Literal':
            if project_node_def_stack[0]['dataType'] == 'timestamp':
                value = project_node_def_stack[0]['value'] + "0000000" # extra zeroes to match numpys precision
                
                time_as_int = np.datetime64(value).astype(np.int64) 
                print("filter date", value, time_as_int)
                return time_as_int, project_node_def_stack[1:]
                
            return float(project_node_def_stack[0]['value']), project_node_def_stack[1:]
        # Handles binary ops e.g., Add/Subtract/Mul/Div.
        elif project_node_def_stack[0]['class'] in SUPPORTED_BINARY_OPS:
            op = SUPPORTED_BINARY_OPS[project_node_def_stack[0]['class']]
            project_node_def_stack = project_node_def_stack[1:]
            left, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            right, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            print("\nbinary op:\n", op, "\n")
            return op(left, right), project_node_def_stack
        # Handles scalar ops e.g., Sqrt.
        elif project_node_def_stack[0]['class'] in SUPPORTED_SCALAR_OPS:
            op = SUPPORTED_SCALAR_OPS[project_node_def_stack[0]['class']]
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


class SQLOrderByModel(BaseOperator, torch.nn.Module):

    def __init__(self, input_names, sort_nodes, device):
        super(SQLOrderByModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.sort_nodes = sort_nodes
        sort_nodes.reverse()
        self.sort_nodes = []
        for n in sort_nodes:
            self.sort_nodes.append(
                (
                    SQLSelectModel(input_names, n[1:], device),
                    # Ascending ?
                    n[0]['direction']['object'] == 'org.apache.spark.sql.catalyst.expressions.Ascending$',
                    # NULLs last ?
                    n[0]['nullOrdering']['object'] == 'org.apache.spark.sql.catalyst.expressions.NullsLast$'
                )
            )

    def forward(self, *x):
        # TODO: Implement null ordering logic.
        order = None
        for select_node, asc, nulls_last in self.sort_nodes:
            vals = select_node(*x)
            if order is not None:
                vals = torch.index_select(vals, 0, order)
                order = torch.index_select(order, 0, torch.argsort(vals.view(-1), descending=not asc))
            else:
                order = torch.argsort(vals.view(-1), descending=not asc)

        return [torch.index_select(c, 0, order) for c in x]


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
    print("sql where statement\n", input_names, operator.inputs )
    print("sql where condition \n", operator.raw_operator)
    return SQLWhereModel(input_names, operator.raw_operator, device)


def convert_sparkml_sql_order_by(operator, device, extra_config):
    """
    Converter for ORDER BY statement in `pyspark.ml.feature.SQLTransformer`

    Args:
        operator: JSON encoded sort_order extracted from SparkSQL parsed tree
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    input_names = [input.raw_name for input in operator.inputs]
    return SQLOrderByModel(input_names, operator.raw_operator, device)


register_converter("SparkMLSQLSelectModel", convert_sparkml_sql_select)
register_converter("SparkMLSQLWhereModel", convert_sparkml_sql_where)
register_converter("SparkMLSQLOrderByModel", convert_sparkml_sql_order_by)
