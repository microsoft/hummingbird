# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Input containers used to wrap input models are listed here.
These containers are used to represent different input models (e.g., onnx, spark-ml) within Hummingbird in a uniform way.
"""
from onnxconverter_common.container import CommonSklearnModelContainer


class CommonONNXModelContainer(CommonSklearnModelContainer):
    """
    Common container for input ONNX operators.
    """

    def __init__(self, onnx_model):
        super(CommonONNXModelContainer, self).__init__(onnx_model)


class CommonSparkMLModelContainer(CommonSklearnModelContainer):
    """
    Common container for input Spark-ML operators.
    """

    def __init__(self, sparkml_model):
        super(CommonSparkMLModelContainer, self).__init__(sparkml_model)
