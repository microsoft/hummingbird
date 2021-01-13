# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All rights reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for Spark-ML discretizers: Bucketizer.
"""
import torch
import numpy as np
from onnxconverter_common.topology import Variable
from onnxconverter_common.registration import register_converter
from .._physical_operator import PhysicalOperator
from .._discretizer_implementations import Binarizer, KBinsDiscretizer


def convert_sparkml_bucketizer(operator, device, extra_config):
    """
    Converter for `pyspark.ml.feature.Bucketizer`

    Args:
        operator: An operator wrapping a `pyspark.ml.feature.QuantileDiscretizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    bin_edges = [operator.raw_operator.getSplits()]
    max_bin_edges = len(bin_edges[0])
    labels = []

    for i in range(len(bin_edges)):
        bin_edges[i][0] = bin_edges[i][0] - 1e-3
        bin_edges[i][-1] = bin_edges[i][-1] + 1e-3
        labels.append(np.array([i for i in range(len(bin_edges[i]) - 1)]))
        if len(bin_edges[i]) < max_bin_edges:
            bin_edges[i] = bin_edges[i] + [np.inf for _ in range((max_bin_edges - len(bin_edges[i])))]

    return KBinsDiscretizer(operator, None, np.array(bin_edges), labels, device)


register_converter("SparkMLBucketizer", convert_sparkml_bucketizer)
