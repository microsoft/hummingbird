# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for Spark-ML VectorAssembler
"""
import torch
import numpy as np
from onnxconverter_common.topology import Variable
from onnxconverter_common.registration import register_converter
from .._base_operator import BaseOperator
from .._array_feature_extractor_implementations import VectorAssemblerModel


def convert_sparkml_vector_assembler(operator, device, extra_config):
    """
    Converter for `pyspark.ml.feature.VectorAssembler`

    Args:
        operator: An operator wrapping a `pyspark.ml.feature.VectorAssembler`
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    def find_column_idx(col_name):
        for i in range(len(operator.inputs)):
            if operator.inputs[i].raw_name == col_name:
                return i
        raise RuntimeError('Column {} not found in the input data'.format(col_name))

    input_indices = [find_column_idx(col_name) for col_name in operator.raw_operator.getInputCols()]

    return VectorAssemblerModel(input_indices=input_indices, append_output=True)


register_converter("SparkMLVectorAssembler", convert_sparkml_vector_assembler)
