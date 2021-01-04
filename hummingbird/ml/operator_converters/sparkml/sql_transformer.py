# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for SparkML SQLTransformer
"""
import torch
import math
import numpy as np
from onnxconverter_common.registration import register_converter
from .._base_operator import BaseOperator
from .. import constants

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

# List of supported SQL aggregation ops.
SUPPORTED_AGGEGATION_OPS = {
    'org.apache.spark.sql.catalyst.expressions.aggregate.Sum': lambda x: torch.sum(x, dim=0, keepdim=True),
    'org.apache.spark.sql.catalyst.expressions.aggregate.Count': lambda x: torch.sum(x, dim=0, keepdim=True),
    'org.apache.spark.sql.catalyst.expressions.aggregate.Average': lambda x: torch.mean(x, dim=0, keepdim=True),
    'org.apache.spark.sql.catalyst.expressions.aggregate.Min': lambda x: torch.min(x, dim=0, keepdim=True)[0],
    'org.apache.spark.sql.catalyst.expressions.aggregate.Max': lambda x: torch.max(x, dim=0, keepdim=True)[0]
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


class SQLGroupByMappingModel(BaseOperator, torch.nn.Module):

    def __init__(self, input_names, group_by_col_names, project_trees, device):
        super(SQLGroupByMappingModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.group_by_col_names = group_by_col_names
        self.project_trees = project_trees

    def forward(self, *x):
        inputs_dict = {k: t for k, t in zip(self.input_names, x)}

        # We need nan (NULL) value (if exists) to be the first element. Hence reverse sorting.
        reverse_sorted_unique_inputs = [torch.sort(torch.unique(inputs_dict[i]), descending=True)[0] for i in self.group_by_col_names]
        mapping_indices = [self.find_mapping_indices(inputs_dict[i], y_sorted) for i, y_sorted in zip(self.group_by_col_names, reverse_sorted_unique_inputs)]

        group_counts = [i.shape[0] for i in reverse_sorted_unique_inputs]
        total_group_count = np.prod(group_counts)
        group_counts = [np.prod(group_counts[i + 1:]) for i in range(len(group_counts))]

        group_indices = mapping_indices[-1]
        for i, g in zip(mapping_indices[:-1][::-1], group_counts[:-1][::-1]):
            group_indices = group_indices + i * g
        group_indices = group_indices.view(-1, 1)

        # TODO: For the following operation and the group by aggregation ops in `get_aggregate_value(...)`, instead of iterating through the
        # list of groups we can use scatter reductions using the `group_indices` e.g., bincount, scatter_add. But the support for various
        # scatter reductions in PyTorch is limited and they also face some issues related to deterministic execution.
        group_counts = torch.cat([torch.sum(group_indices == g, dim=0, keepdim=True) for g in range(total_group_count)])
        group_selection_mask = (group_counts > 0).view(-1, 1)
        group_keys = torch.meshgrid(*reverse_sorted_unique_inputs)

        # Updating the group by columns in the input dict.
        group_by_keys = {k : group_keys[i].flatten().view(-1, 1) for i, k in enumerate(self.group_by_col_names)}

        return [torch.masked_select(i, group_selection_mask).view(-1, 1) for i, _ in [self.get_aggregate_value(p, inputs_dict, group_by_keys, group_indices, total_group_count) for p in self.project_trees]]

    def get_aggregate_value(self, aggregate_node_def_stack, inputs_dict, group_by_keys, group_indices, total_group_count):
        # Returns group by columns.
        if aggregate_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.AttributeReference':
            if aggregate_node_def_stack[0]['name'] in self.group_by_col_names:
                return group_by_keys[aggregate_node_def_stack[0]['name']], aggregate_node_def_stack[1:]
            else:
                return inputs_dict[aggregate_node_def_stack[0]['name']], aggregate_node_def_stack[1:]
        # Returns a constant value of column size.
        elif aggregate_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Literal':
            dummy_input_col = inputs_dict[self.input_names[0]]
            return torch.ones_like(dummy_input_col) * float(aggregate_node_def_stack[0]['value']), aggregate_node_def_stack[1:]
        # Returns different aggregation values for each group.
        elif aggregate_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression':
            if aggregate_node_def_stack[1]['class'] in SUPPORTED_AGGEGATION_OPS:
                op = SUPPORTED_AGGEGATION_OPS[aggregate_node_def_stack[1]['class']]
                input_col, aggregate_node_def_stack = self.get_aggregate_value(aggregate_node_def_stack[2:], inputs_dict, group_by_keys, group_indices, total_group_count)
                # TODO: Iterating over the groups instead of using a scatter reduction can be inefficient.
                aggregated_col = torch.cat([op(torch.masked_select(input_col, group_indices == g)) for g in range(total_group_count)], dim=0)
                return aggregated_col, aggregate_node_def_stack
            else:
                raise RuntimeError('SQLTransformer encountered unsupported operator type: {}'.format(aggregate_node_def_stack[1]['class']))
        else:
            raise RuntimeError('SQLTransformer encountered unsupported operator type: {}'.format(aggregate_node_def_stack[0]['class']))

    def find_mapping_indices(self, x, reverse_sorted_y):
        N = reverse_sorted_y.shape[0]
        x = x.view(-1)
        N_prime = 2**math.ceil(np.log2(N))
        reverse_sorted_y = torch.nn.functional.pad(reverse_sorted_y, (N_prime - N, 0), value=reverse_sorted_y[-1] - 1)

        offset = N_prime // 2
        bins = (x <= reverse_sorted_y[offset])
        pos = bins * offset
        offset = int(math.ceil(offset / 2))

        for _ in range(1, int(np.log2(N_prime))):
            bins = x <= torch.index_select(reverse_sorted_y, 0, pos + offset)
            pos = pos + bins * offset
            offset = int(offset / 2)

        return pos


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


def convert_sparkml_sql_group_by_mapping(operator, device, extra_config):
    """
    Converter for ORDER BY statement in `pyspark.ml.feature.SQLTransformer`

    Args:
        operator: Tuple containing the group by column list as the first element and a list of JSON encoded aggregate columns to be projected.
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    group_by_col_names, project_trees = operator.raw_operator
    input_names = [input.raw_name for input in operator.inputs]
    # If the first entry in the project_tree is the Alias, we ignore.
    project_trees = [p[1:] if p[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Alias' else p for p in project_trees]
    return SQLGroupByMappingModel(input_names, group_by_col_names, project_trees, device)


register_converter("SparkMLSQLSelectModel", convert_sparkml_sql_select)
register_converter("SparkMLSQLWhereModel", convert_sparkml_sql_where)
register_converter("SparkMLSQLOrderByModel", convert_sparkml_sql_order_by)
register_converter("SparkMLSQLGroupByMappingModel", convert_sparkml_sql_group_by_mapping)
