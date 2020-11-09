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
    def __init__(self, device):
        super(SQLTransformerModel, self).__init__()
        self.transformer = True

    def forward(self, x):
        return x


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

    return SQLTransformerModel(device)


register_converter("SparkMLSQLTransformer", convert_sparkml_sql_transformer)
