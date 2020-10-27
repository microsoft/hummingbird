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


def get_output_features(operator):
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    parser = spark._jsparkSession.sessionState().sqlParser()
    statement = operator.raw_operator.getStatement()
    plan = parser.parsePlan()
    plan_json = json.loads(plan.toJSON())

    for node in plan_json:
        if node['class'] == 'org.apache.spark.sql.catalyst.plans.logical.Project':
            projectList = node['projectList']
            output_feature_names = []
            for projectNode in projectList:
                if len(projectNode) == 1 and projectNode[0]['class'] == 'org.apache.spark.sql.catalyst.analysis.UnresolvedStar':
                    # Forwarding all input fields. Hence ignored.
                    continue
                elif projectNode[0]['class'] == 'org.apache.spark.sql.catalyst.expressions.Alias':
                    if 'name' not in projectNode[0]:
                        raise RuntimeError('New feature definition in SQLTransformer should have an explicit name declaration.')
                    output_feature_names.append(projectNode[0]['name'])

            return output_feature_names
    
    raise RuntimeError('Couldn\'t find the projectList from the SQLTransformer statement: {}.'.format(statement))


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

    return None


register_converter("SparkMLSQLTransformer", convert_sparkml_sql_transformer)
