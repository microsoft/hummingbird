# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for SparkML SQLTransformer
"""
import torch
import json
import numpy as np
from onnxconverter_common.registration import register_converter
from .._base_operator import BaseOperator


def get_input_output_col_names(operator):
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    parser = spark._jsparkSession.sessionState().sqlParser()
    plan = parser.parsePlan(operator.getStatement())
    plan_json = json.loads(plan.toJSON())

    for node in plan_json:
        if node['class'] == 'org.apache.spark.sql.catalyst.plans.logical.Project':
            projectList = node['projectList']
            input_feature_names = []
            output_feature_names = []
            for projectNode in projectList:
                if len(projectNode) == 1 and projectNode[0]['class'] == 'org.apache.spark.sql.catalyst.analysis.UnresolvedStar':
                    # Forwarding all input fields. Hence ignored.
                    continue
                else:
                    for node in projectNode:
                        if node['class'] == 'org.apache.spark.sql.catalyst.expressions.Alias':
                            if 'name' not in node:
                                raise RuntimeError('New feature definition in SQLTransformer should have an explicit name declaration.')
                            output_feature_names.append(node['name'])
                        elif node['class'] == 'org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute':
                            in_cols = [x.strip() for x in node['nameParts'][1:-1].split(',')]
                            for col in in_cols:
                                if col not in input_feature_names:
                                    input_feature_names.append(col)
            return input_feature_names, output_feature_names

    raise RuntimeError('Couldn\'t find the projectList from the SQLTransformer statement: {}.'.format(operator.getStatement()))


class SQLTransformerModel(BaseOperator, torch.nn.Module):

    # List of supported binary SQL ops.
    supported_binary_ops = {
        'org.apache.spark.sql.catalyst.expressions.Add': torch.add,
        'org.apache.spark.sql.catalyst.expressions.Subtract': torch.sub,
        'org.apache.spark.sql.catalyst.expressions.Multiply': torch.mul,
        'org.apache.spark.sql.catalyst.expressions.Divide': torch.div,
    }

    def __init__(self, input_names, project_list, device):
        super(SQLTransformerModel, self).__init__()
        self.transformer = True
        self.input_names = input_names
        self.project_list = project_list

    def forward(self, *x):
        inputs_dict = {name : tensor for name, tensor in zip(self.input_names, x)}
        # The first node of the project_node_def is the name definition node. Hence ignored.
        return [self.calculate_output_feature(project_node_def_stack[1:], inputs_dict)[0] for project_node_def_stack in self.project_list]

    def calculate_output_feature(self, project_node_def_stack, inputs_dict):
        if project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute':
            return inputs_dict[project_node_def_stack[0]['nameParts'][1:-1].strip()], project_node_def_stack[1:]
        elif project_node_def_stack[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Literal':
            return float(project_node_def_stack[0]['value']), project_node_def_stack[1:]
        elif project_node_def_stack[0]['class'] in self.supported_binary_ops:
            op = self.supported_binary_ops[project_node_def_stack[0]['class']]
            project_node_def_stack = project_node_def_stack[1:]
            left, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            right, project_node_def_stack = self.calculate_output_feature(project_node_def_stack, inputs_dict)
            return op(left, right), project_node_def_stack
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
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    parser = spark._jsparkSession.sessionState().sqlParser()
    statement = operator.raw_operator.getStatement()
    plan = parser.parsePlan(statement)
    plan_json = json.loads(plan.toJSON())

    project_node = [node for node in plan_json if node['class'] == 'org.apache.spark.sql.catalyst.plans.logical.Project']
    if len(project_node) == 0:
        raise RuntimeError('Couldn\'t find the projectList from the SQLTransformer statement: {}.'.format(statement))

    # Ignores select *
    project_list = [proj for proj in project_node[0]['projectList'] if proj[0]['class'] != 'org.apache.spark.sql.catalyst.analysis.UnresolvedStar']
    input_names = [input.raw_name for input in operator.inputs]
    return SQLTransformerModel(input_names, project_list, device)


register_converter("SparkMLSQLTransformer", convert_sparkml_sql_transformer)
