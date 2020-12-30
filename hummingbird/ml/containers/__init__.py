# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
All custom model containers are listed here.
In Hummingbird we use two types of containers:
- containers for input models (e.g., `CommonONNXModelContainer`) used to represent input models in a unified way as DAG of containers
- containers for output models (e.g., `SklearnContainer`) used to surface output models as unified API format.
"""

# Register the containers used within Hummingbird.
# Input containers.
from ._input_containers import CommonSklearnModelContainer, CommonONNXModelContainer, CommonSparkMLModelContainer

# Output containers.
from .batch_container import BatchContainer
from .sklearn.pytorch_containers import (
    PyTorchSklearnContainer,
    PyTorchSklearnContainerTransformer,
    PyTorchSklearnContainerRegression,
    PyTorchSklearnContainerClassification,
    PyTorchSklearnContainerAnomalyDetection,
    TorchScriptSklearnContainerTransformer,
    TorchScriptSklearnContainerRegression,
    TorchScriptSklearnContainerClassification,
    TorchScriptSklearnContainerAnomalyDetection,
)
from .sklearn.onnx_containers import (
    ONNXSklearnContainer,
    ONNXSklearnContainerTransformer,
    ONNXSklearnContainerRegression,
    ONNXSklearnContainerClassification,
    ONNXSklearnContainerAnomalyDetection,
)
from .sklearn.tvm_containers import (
    TVMSklearnContainer,
    TVMSklearnContainerTransformer,
    TVMSklearnContainerRegression,
    TVMSklearnContainerClassification,
    TVMSklearnContainerAnomalyDetection,
)
