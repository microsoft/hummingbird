# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Hummingbird main (converters) API.
"""
from copy import deepcopy
import psutil
import numpy as np

from .operator_converters import constants
from ._parse import parse_sklearn_api_model, parse_onnx_api_model, parse_sparkml_api_model
from ._topology import convert as topology_converter
from ._utils import (
    torch_installed,
    lightgbm_installed,
    xgboost_installed,
    pandas_installed,
    sparkml_installed,
    is_pandas_dataframe,
    is_spark_dataframe,
    tvm_installed,
)
from .exceptions import MissingConverter, MissingBackend
from .supported import backends

# Invoke the registration of all our converters.
from . import operator_converters  # noqa

# Set up the converter dispatcher.
from .supported import xgb_operator_list  # noqa
from .supported import lgbm_operator_list  # noqa


def _is_onnx_model(model):
    """
    Function returning whether the input model is an ONNX model or not.
    """
    return type(model).__name__ == "ModelProto"


def _is_sparkml_model(model):
    """
    Function returning whether the input model is a Spark-ML model or not.
    """
    if sparkml_installed():
        from pyspark.ml import Model, Transformer
        from pyspark.ml.pipeline import PipelineModel

        return isinstance(model, Model) or isinstance(model, PipelineModel) or isinstance(model, Transformer)
    else:
        return False


def _supported_backend_check(backend):
    """
    Function used to check whether the specified backend is supported or not.
    """
    if backend is None:
        raise MissingBackend("Backend: {}".format(backend))


def _supported_backend_check_config(model, backend, extra_config):
    """
    Function used to check whether the specified backend and configuration pair is supported or not.
    """
    assert torch_installed(), "To use Hummingbird you need to install torch."
    import onnx
    import torch

    tvm_backend = None
    if tvm_installed():
        import tvm

        tvm_backend = tvm.__name__

    if (
        (backend == torch.jit.__name__ and not _is_onnx_model(model)) or backend == tvm_backend
    ) and constants.TEST_INPUT not in extra_config:
        raise RuntimeError("Backend {} requires test inputs. Please pass some test input to the convert.".format(backend))


def _convert_sklearn(model, backend, test_input, device, extra_config={}):
    """
    This function converts the specified *scikit-learn* (API) model into its *backend* counterpart.
    The supported operators and backends can be found at `hummingbird.ml.supported`.
    """
    assert model is not None
    assert torch_installed(), "To use Hummingbird you need to install torch."

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    # We modify the scikit learn model during translation.
    model = deepcopy(model)
    topology = parse_sklearn_api_model(model, extra_config)

    # Convert the Topology object into a PyTorch model.
    hb_model = topology_converter(topology, backend, test_input, device, extra_config=extra_config)
    return hb_model


def _convert_lightgbm(model, backend, test_input, device, extra_config={}):
    """
    This function is used to generate a *backend* model from a given input [LightGBM] model.
    [LightGBM]: https://lightgbm.readthedocs.io/
    """
    assert (
        lightgbm_installed()
    ), "To convert LightGBM models you need to install LightGBM (or `pip install hummingbird-ml[extra]`)."

    return _convert_sklearn(model, backend, test_input, device, extra_config)


def _convert_xgboost(model, backend, test_input, device, extra_config={}):
    """
    This function is used to generate a *backend* model from a given input [XGBoost] model.
    [XGBoost]: https://xgboost.readthedocs.io/
    """
    assert (
        xgboost_installed()
    ), "To convert XGboost models you need to instal XGBoost (or `pip install hummingbird-ml[extra]`)."

    # XGBoostRegressor and Classifier have different APIs for extracting the number of features.
    # In the former case we need to infer them from the test_input.
    if "_features_count" in dir(model):
        extra_config[constants.N_FEATURES] = model._features_count
    elif test_input is not None:
        if type(test_input) is np.ndarray and len(test_input.shape) == 2:
            extra_config[constants.N_FEATURES] = test_input.shape[1]
        else:
            raise RuntimeError(
                "XGBoost converter is not able to infer the number of input features.\
                    Please pass a test_input to convert or \
                    fill an issue at https://github.com/microsoft/hummingbird/."
            )
    else:
        raise RuntimeError(
            "XGBoost converter is not able to infer the number of input features.\
                Please pass some test_input to the converter."
        )
    return _convert_sklearn(model, backend, test_input, device, extra_config)


def _convert_onnxml(model, backend, test_input, device, extra_config={}):
    """
    This function converts the specified [ONNX-ML] model into its *backend* counterpart.
    The supported operators can be found at `hummingbird.ml.supported`.
    """
    assert model is not None
    assert torch_installed(), "To use Hummingbird you need to install torch."

    import onnx

    # The conversion requires some test input for tracing.
    # Test inputs can be either provided or generate from the input schema of the model.
    # Generate some test input if necessary.
    if test_input is None:
        import torch
        from onnxconverter_common.data_types import FloatTensorType, DoubleTensorType, Int32TensorType, Int64TensorType

        tvm_backend = None
        if tvm_installed():
            import tvm

            tvm_backend = tvm.__name__

        # Get the input information from the ONNX schema.
        initial_types = []
        for input in model.graph.input:
            name = input.name if hasattr(input, "name") else None
            data_type = (
                input.type.tensor_type.elem_type
                if hasattr(input, "type")
                and hasattr(input.type, "tensor_type")
                and hasattr(input.type.tensor_type, "elem_type")
                else None
            )
            if name is None:
                raise RuntimeError(
                    "Cannot fetch input name or data_type from the ONNX schema. Please provide some test input."
                )
            if data_type is None:
                raise RuntimeError(
                    "Cannot fetch input data_type from the ONNX schema, or data type is not tensor_type. Please provide some test input."
                )
            if not hasattr(input.type.tensor_type, "shape"):
                raise RuntimeError("Cannot fetch input shape from ONNX schema. Please provide some test input.")
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]

            if len(shape) == 1:
                shape = [1, shape[0]]
            assert len(shape) == 2
            # In ONNX dynamic dimensions will have a shape of 0. Fix the 0-shape in the batch dimension if they exist.
            if shape[0] == 0:
                shape[0] = 1

            if data_type == 1:
                initial_types.append((name, FloatTensorType(shape)))
            elif data_type == 11:
                initial_types.append((name, DoubleTensorType(shape)))
            elif data_type == 6:
                initial_types.append((name, Int32TensorType(shape)))
            elif data_type == 7:
                initial_types.append((name, Int64TensorType(shape)))
            else:
                raise RuntimeError(
                    "Input data type {} not supported. Please fill an issue at https://github.com/microsoft/hummingbird/, or pass some test_input".format(
                        data_type
                    )
                )

        first_shape = initial_types[0][1].shape
        assert all(
            map(lambda x: x[1].shape == first_shape, initial_types)
        ), "Hummingbird currently supports only inputs with same shape."
        extra_config[constants.N_INPUTS] = len(initial_types)
        extra_config[constants.N_FEATURES] = extra_config[constants.N_INPUTS] * first_shape[1]

        # Generate some random input data if necessary for the model conversion.
        if backend == onnx.__name__ or backend == tvm_backend or backend == torch.jit.__name__:
            test_input = []
            for i, it in enumerate(initial_types):
                if type(it[1]) is FloatTensorType:
                    test_input.append(np.array(np.random.rand(first_shape[0], first_shape[1]), dtype=np.float32))
                elif type(it[1]) is DoubleTensorType:
                    test_input.append(np.random.rand(first_shape[0], first_shape[1]))
                elif type(it[1]) is Int32TensorType:
                    test_input.append(np.array(np.random.randint(100, size=first_shape), dtype=np.int32))
                elif type(it[1]) is Int64TensorType:
                    test_input.append(np.random.randint(100, size=first_shape))
                else:
                    raise RuntimeError(
                        "Type {} not supported. Please fill an issue on https://github.com/microsoft/hummingbird/.".format(
                            type(it[1])
                        )
                    )
            if extra_config[constants.N_INPUTS] == 1:
                test_input = test_input[0]
            else:
                test_input = tuple(test_input)
            extra_config[constants.TEST_INPUT] = test_input

    # Set the number of features. Some converter requires to know in advance the number of features.
    if constants.N_FEATURES not in extra_config and test_input is not None:
        if len(test_input.shape) < 2:
            extra_config[constants.N_FEATURES] = 1
        else:
            extra_config[constants.N_FEATURES] = test_input.shape[1]

    # Set the initializers. Some converter requires the access to initializers.
    initializers = {} if model.graph.initializer is None else {in_.name: in_ for in_ in model.graph.initializer}
    extra_config[constants.ONNX_INITIALIZERS] = initializers

    # Parse ONNX model as our internal data structure (i.e., Topology).
    topology = parse_onnx_api_model(model)

    # Convert the Topology object into a PyTorch model.
    hb_model = topology_converter(topology, backend, test_input, device, extra_config=extra_config)
    return hb_model


def _convert_sparkml(model, backend, test_input, device, extra_config={}):
    """
    This function converts the specified *Spark-ML* (API) model into its *backend* counterpart.
    The supported operators and backends can be found at `hummingbird.ml.supported`.
    """
    assert model is not None
    assert torch_installed(), "To use Hummingbird you need to install torch."

    # Parse Spark-ML model as our internal data structure (i.e., Topology)
    # We modify the Spark-ML model during translation.
    model = model.copy()
    topology = parse_sparkml_api_model(model, extra_config)

    # Convert the Topology object into a PyTorch model.
    hb_model = topology_converter(topology, backend, test_input, device, extra_config=extra_config)
    return hb_model


def _convert_common(model, backend, test_input=None, device="cpu", extra_config={}):
    """
    A common function called by convert(...) and convert_batch(...) below.
    """
    assert model is not None

    # We destroy extra_config during conversion, we create a copy here.
    extra_config = deepcopy(extra_config)

    # Set some default configurations.
    # Add test input as extra configuration for conversion.
    if (
        test_input is not None
        and constants.TEST_INPUT not in extra_config
        and (is_spark_dataframe(test_input) or len(test_input) > 0)
    ):
        extra_config[constants.TEST_INPUT] = test_input
    # By default we return the converted model wrapped into a `hummingbird.ml._container.SklearnContainer` object.
    if constants.CONTAINER not in extra_config:
        extra_config[constants.CONTAINER] = True
    # By default we set num of intra-op parallelism to be the number of physical cores available
    if constants.N_THREADS not in extra_config:
        extra_config[constants.N_THREADS] = psutil.cpu_count(logical=False)

    # Fix the test_input type
    if constants.TEST_INPUT in extra_config:
        if type(extra_config[constants.TEST_INPUT]) == list:
            extra_config[constants.TEST_INPUT] = np.array(extra_config[constants.TEST_INPUT])
        elif type(extra_config[constants.TEST_INPUT]) == tuple:
            # We are passing multiple datasets.
            assert all(
                [type(input) == np.ndarray for input in extra_config[constants.TEST_INPUT]]
            ), "When passing multiple inputs only ndarrays are supported."
            assert all([len(input.shape) == 2 for input in extra_config[constants.TEST_INPUT]])
            extra_config[constants.N_FEATURES] = sum([input.shape[1] for input in extra_config[constants.TEST_INPUT]])
            extra_config[constants.N_INPUTS] = len(extra_config[constants.TEST_INPUT])
        elif pandas_installed() and is_pandas_dataframe(extra_config[constants.TEST_INPUT]):
            # We split the input dataframe into columnar ndarrays
            extra_config[constants.N_INPUTS] = len(extra_config[constants.TEST_INPUT].columns)
            extra_config[constants.N_FEATURES] = extra_config[constants.N_INPUTS]
            input_names = list(extra_config[constants.TEST_INPUT].columns)
            splits = [extra_config[constants.TEST_INPUT][input_names[idx]] for idx in range(extra_config[constants.N_INPUTS])]
            splits = [df.to_numpy().reshape(-1, 1) for df in splits]
            extra_config[constants.TEST_INPUT] = tuple(splits) if len(splits) > 1 else splits[0]
            extra_config[constants.INPUT_NAMES] = input_names
        elif sparkml_installed() and is_spark_dataframe(extra_config[constants.TEST_INPUT]):
            from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
            from pyspark.sql.types import ArrayType, FloatType, DoubleType, IntegerType, LongType

            df = extra_config[constants.TEST_INPUT]
            input_names = [field.name for field in df.schema.fields]
            extra_config[constants.N_INPUTS] = len(input_names)
            extra_config[constants.N_FEATURES] = extra_config[constants.N_INPUTS]

            size = df.count()
            row_dict = df.take(1)[0].asDict()
            splits = []
            for field in df.schema.fields:
                data_col = row_dict[field.name]
                spark_dtype = type(field.dataType)
                shape = 1
                if spark_dtype in [DenseVector, VectorUDT]:
                    np_dtype = np.float64
                    shape = data_col.array.shape[0]
                elif spark_dtype == SparseVector:
                    np_dtype = np.float64
                    shape = data_col.size
                elif spark_dtype == ArrayType:
                    np_dtype = np.float64
                    shape = len(data_col)
                elif spark_dtype == IntegerType:
                    np_dtype = np.int32
                elif spark_dtype == FloatType:
                    np_dtype = np.float32
                elif spark_dtype == DoubleType:
                    np_dtype = np.float64
                elif spark_dtype == LongType:
                    np_dtype = np.int64
                else:
                    raise ValueError("Unrecognized data type: {}".format(spark_dtype))

                splits.append(np.zeros((size, shape), np_dtype))

            extra_config[constants.TEST_INPUT] = tuple(splits) if len(splits) > 1 else splits[0]
            extra_config[constants.INPUT_NAMES] = input_names

        test_input = extra_config[constants.TEST_INPUT]

    # We do some normalization on backends.
    if type(backend) != str:
        raise ValueError("Backend must be a string: {}".format(backend))
    backend = backend.lower()
    backend = backends[backend]

    # Check whether we actually support the backend.
    _supported_backend_check(backend)
    _supported_backend_check_config(model, backend, extra_config)

    if type(model) in xgb_operator_list:
        return _convert_xgboost(model, backend, test_input, device, extra_config)

    if type(model) in lgbm_operator_list:
        return _convert_lightgbm(model, backend, test_input, device, extra_config)

    if _is_onnx_model(model):
        return _convert_onnxml(model, backend, test_input, device, extra_config)

    if _is_sparkml_model(model):
        return _convert_sparkml(model, backend, test_input, device, extra_config)

    return _convert_sklearn(model, backend, test_input, device, extra_config)


def convert(model, backend, test_input=None, device="cpu", extra_config={}):
    """
    This function converts the specified input *model* into an implementation targeting *backend*.
    *Convert* supports [Sklearn], [LightGBM], [XGBoost], [ONNX], and [SparkML] models.
    For *LightGBM* and *XGBoost* currently only the Sklearn API is supported.
    The detailed list of models and backends can be found at `hummingbird.ml.supported`.
    The *onnx* backend requires either a test_input of a the initial types set through the exta_config parameter.
    The *torch.jit* and *tvm* backends require a test_input.
    For *tvm* backend, the output container can do prediction only on the test data with the same size as test_input.
    [Sklearn]: https://scikit-learn.org/
    [LightGBM]: https://lightgbm.readthedocs.io/
    [XGBoost]: https://xgboost.readthedocs.io/
    [ONNX]: https://onnx.ai/
    [ONNX-ML]: https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md
    [ONNX operators]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    [Spark-ML]: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html

    Args:
        model: An input model
        backend: The target for the conversion
        test_input: Some input data used to trace the model execution.
                    Multiple inputs can be passed as `tuple` objects or pandas Dataframes.
                    When possible, (`numpy`)`arrays` are suggested.
                    The number of rows becomes the batch size when tracing PyTorch models and compiling with TVM.
        device: The target device the model should be run. This parameter is only used by the *torch** backends and *tvm*, and
                the devices supported are the one supported by PyTorch, i.e., 'cpu' or 'cuda'.
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> pytorch_model = convert(sklearn_model,`torch`)

    Returns:
        A model implemented in *backend*, which is equivalent to the input model
    """
    assert constants.REMAINDER_SIZE not in extra_config
    return _convert_common(model, backend, test_input, device, extra_config)


def convert_batch(model, backend, test_input, remainder_size=0, device="cpu", extra_config={}):
    """
    A convert function for batch by batch prediction use cases.
    For some backends such as TVM, a container returned by `convert(...)` function above has a strict requirement on the
    allowable input shape.
    The container returned by this function is more flexible in that it can predict on the input of size
    `test_input.shape[0] * k + remainder_size`, where `k` is any integer.
    `test_input.shape[0]`, the number of rows in the `test_input`, is interpreted as a batch size, and at test time
    prediction proceeds in a batch by batch fashion.
    See the documentation for *convert(...)* above for more information.

    Args:
        model: An input model
        backend: The target for the conversion
        test_input: Some input data used to trace the model execution.
                    Multiple inputs can be passed as `tuple` objects or pandas Dataframes.
                    When possible, (`numpy`)`arrays` are suggested.
                    The number of rows becomes the batch size when tracing PyTorch models and compiling with TVM.
        remainder_size: An integer that together with test_input determines the size of test data that can be predicted.
                    The input to the returned container can be of size `test_input.shape[0] * k + remainder_size`, where `k`
                    is any integer.
        device: The target device the model should be run. This parameter is only used by the *torch** backends and *tvm*, and
                the devices supported are the one supported by PyTorch, i.e., 'cpu' or 'cuda'.
        extra_config: Extra configurations to be used by the individual operator converters.
                      The set of supported extra configurations can be found at `hummingbird.ml.supported`

    Examples:
        >>> tvm_model = convert_batch(sklearn_model,`tvm`, X)
        >>> tvm_model = convert_batch(sklearn_model,`tvm`, X, remainder_size=50)

    Returns:
        A `BatchContainer` object that wraps one or two containers created by `convert(...)` function above.
    """
    extra_config[constants.REMAINDER_SIZE] = remainder_size
    return _convert_common(model, backend, test_input, device, extra_config)
