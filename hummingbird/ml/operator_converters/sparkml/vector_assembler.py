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
from .._physical_operator import PhysicalOperator
from .._pipeline_implementations import Concat


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

    return Concat(operator)


register_converter("SparkMLVectorAssembler", convert_sparkml_vector_assembler)
