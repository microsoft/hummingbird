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

from abc import ABC
import numpy as np
from onnxconverter_common.container import CommonSklearnModelContainer
import torch

from hummingbird.ml.operator_converters import constants
from hummingbird.ml._utils import onnx_runtime_installed, tvm_installed, pandas_installed, _get_device

if pandas_installed():
    from pandas import DataFrame
else:
    DataFrame = None


# Input containers
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


# Output containers
class SklearnContainer(ABC):
    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        """
        Base container abstract class allowing to mirror the Sklearn API.
        *SklearnContainer* enables the use of `predict`, `predict_proba` etc. API of Sklearn
        also over the models generated by Hummingbird (irrespective of the selected backend).

        Args:
            model: Any Hummingbird supported model
            n_threads: How many threads should be used by the containter to run the model. None means use all threads.
            batch_size: If different than None, split the input into batch_size partitions and score one partition at a time.
            extra_config: Some additional configuration parameter.
        """
        self._model = model
        self._n_threads = n_threads
        self._batch_size = batch_size
        self._extra_config = extra_config

    @property
    def model(self):
        return self._model

    def _run(self, function, *inputs, reshape=False):
        """
        This function either score the full dataset at once or triggers batch inference.
        """
        if DataFrame is not None and type(inputs[0]) == DataFrame:
            # Split the dataframe into column ndarrays.
            inputs = inputs[0]
            input_names = list(inputs.columns)
            splits = [inputs[input_names[idx]] for idx in range(len(input_names))]
            inputs = [df.to_numpy().reshape(-1, 1) for df in splits]

        if self._batch_size is None:
            return function(*inputs)
        else:
            return self._run_batch_inference(function, *inputs, reshape=reshape)

    def _run_batch_inference(self, function, *inputs, reshape=False):
        """
        This function contains the code to run batched inference.
        """

        is_tuple = type(inputs) is tuple
        if is_tuple:
            total_size = inputs[0].shape[0]
        else:
            total_size = inputs.shape[0]

        iterations = total_size // self._batch_size
        iterations += 1 if total_size % self._batch_size > 0 else 0
        iterations = max(1, iterations)
        predictions = []

        for i in range(0, iterations):
            start = i * self._batch_size
            end = min(start + self._batch_size, total_size)
            if is_tuple:
                batch = tuple([input[start:end, :] for input in inputs])
            else:
                batch = inputs[start:end, :]
            predictions.extend(function(*batch).ravel())

        if reshape:
            return np.array(predictions).ravel().reshape(total_size, -1)
        return np.array(predictions).ravel()


class PyTorchTorchscriptSklearnContainer(SklearnContainer):
    """
    Base container for PyTorch and TorchScript models.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(PyTorchTorchscriptSklearnContainer, self).__init__(model, n_threads, batch_size, extra_config)

        assert self._n_threads is not None

        # We set intra op concurrency while we force operators to run sequentially.
        # We can revise this later, but in general we don't have graphs requireing inter-op parallelism.
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.set_num_threads(self._n_threads)


# PyTorch containers.
class PyTorchSklearnContainerTransformer(PyTorchTorchscriptSklearnContainer):
    """
    Container mirroring Sklearn transformers API.
    """

    def _transform(self, *inputs):
        return self.model.forward(*inputs).cpu().numpy()

    def transform(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On data transformers it returns transformed output data
        """
        return self._run(self._transform, *inputs, reshape=True)


class PyTorchSklearnContainerRegression(PyTorchTorchscriptSklearnContainer):
    """
    Container mirroring Sklearn regressors API.
    """

    def __init__(
        self, model, n_threads, batch_size, is_regression=True, is_anomaly_detection=False, extra_config={}, **kwargs
    ):
        super(PyTorchSklearnContainerRegression, self).__init__(model, n_threads, batch_size, extra_config)

        assert not (is_regression and is_anomaly_detection)

        self._is_regression = is_regression
        self._is_anomaly_detection = is_anomaly_detection

    def _predict(self, *inputs):
        if self._is_regression:
            return self.model.forward(*inputs).cpu().numpy().ravel()
        elif self._is_anomaly_detection:
            return self.model.forward(*inputs)[0].cpu().numpy().ravel()
        else:
            return self.model.forward(*inputs)[0].cpu().numpy().ravel()

    def predict(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On regression returns the predicted values.
        On classification tasks returns the predicted class labels for the input data.
        On anomaly detection (e.g. isolation forest) returns the predicted classes (-1 or 1).
        """
        return self._run(self._predict, *inputs)


class PyTorchSklearnContainerClassification(PyTorchSklearnContainerRegression):
    """
    Container mirroring Sklearn classifiers API.
    """

    def __init__(self, model, n_threads, batch_size, extra_config={}):
        super(PyTorchSklearnContainerClassification, self).__init__(
            model, n_threads, batch_size, is_regression=False, extra_config=extra_config
        )

    def _predict_proba(self, *input):
        return self.model.forward(*input)[1].cpu().numpy()

    def predict_proba(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On classification tasks returns the probability estimates.
        """
        return self._run(self._predict_proba, *inputs, reshape=True)


class PyTorchSklearnContainerAnomalyDetection(PyTorchSklearnContainerRegression):
    """
    Container mirroring Sklearn anomaly detection API.
    """

    def __init__(self, model, n_threads, batch_size, extra_config={}):
        super(PyTorchSklearnContainerAnomalyDetection, self).__init__(
            model, n_threads, batch_size, is_regression=False, is_anomaly_detection=True, extra_config=extra_config
        )

    def _decision_function(self, *inputs):
        return self.model.forward(*inputs)[1].cpu().numpy().ravel()

    def decision_function(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On anomaly detection (e.g. isolation forest) returns the decision function scores.
        """
        scores = self._run(self._decision_function, *inputs)

        # Backward compatibility for sklearn <= 0.21
        if constants.IFOREST_THRESHOLD in self._extra_config:
            scores += self._extra_config[constants.IFOREST_THRESHOLD]
        return scores

    def score_samples(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On anomaly detection (e.g. isolation forest) returns the decision_function score plus offset_
        """
        return self.decision_function(*inputs) + self._extra_config[constants.OFFSET]


# TorchScript containers.
def _torchscript_wrapper(device, function, *inputs):
    """
    This function contains the code to enable predictions over torchscript models.
    It is used to translates inputs in the proper torch format.
    """
    inputs = [*inputs]

    with torch.no_grad():
        if type(inputs) == DataFrame and DataFrame is not None:
            # Split the dataframe into column ndarrays
            inputs = inputs[0]
            input_names = list(inputs.columns)
            splits = [inputs[input_names[idx]] for idx in range(len(input_names))]
            splits = [df.to_numpy().reshape(-1, 1) for df in splits]
            inputs = tuple(splits)

        # Maps data inputs to the expected type and device.
        for i in range(len(inputs)):
            if type(inputs[i]) is np.ndarray:
                inputs[i] = torch.from_numpy(inputs[i]).float()
            elif type(inputs[i]) is not torch.Tensor:
                raise RuntimeError("Inputer tensor {} of not supported type {}".format(i, type(inputs[i])))
            if device.type != "cpu" and device is not None:
                inputs[i] = inputs[i].to(device)
        return function(*inputs)


class TorchScriptSklearnContainerTransformer(PyTorchSklearnContainerTransformer):
    """
    Container mirroring Sklearn transformers API.
    """

    def transform(self, *inputs):
        device = _get_device(self.model)
        f = super(TorchScriptSklearnContainerTransformer, self)._transform
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x)  # noqa: E731

        return self._run(f_wrapped, *inputs, reshape=True)


class TorchScriptSklearnContainerRegression(PyTorchSklearnContainerRegression):
    """
    Container mirroring Sklearn regressors API.
    """

    def predict(self, *inputs):
        device = _get_device(self.model)
        f = super(TorchScriptSklearnContainerRegression, self)._predict
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x)  # noqa: E731

        return self._run(f_wrapped, *inputs)


class TorchScriptSklearnContainerClassification(PyTorchSklearnContainerClassification):
    """
    Container mirroring Sklearn classifiers API.
    """

    def predict(self, *inputs):
        device = _get_device(self.model)
        f = super(TorchScriptSklearnContainerClassification, self)._predict
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x)  # noqa: E731

        return self._run(f_wrapped, *inputs)

    def predict_proba(self, *inputs):
        device = _get_device(self.model)
        f = super(TorchScriptSklearnContainerClassification, self)._predict_proba
        f_wrapped = lambda *x: _torchscript_wrapper(device, f, *x)  # noqa: E731

        return self._run(f_wrapped, *inputs, reshape=True)


class TorchScriptSklearnContainerAnomalyDetection(PyTorchSklearnContainerAnomalyDetection):
    """
    Container mirroring Sklearn anomaly detection API.
    """

    def predict(self, *inputs):
        device = _get_device(self.model)
        f = super(TorchScriptSklearnContainerAnomalyDetection, self)._predict
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x)  # noqa: E731

        return self._run(f_wrapped, *inputs)

    def decision_function(self, *inputs):
        device = _get_device(self.model)
        f = super(TorchScriptSklearnContainerAnomalyDetection, self)._decision_function
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x)  # noqa: E731

        return self._run(f_wrapped, *inputs)

    def score_samples(self, *inputs):
        device = _get_device(self.model)
        f = self.decision_function
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x)  # noqa: E731

        return self._run(f_wrapped, *inputs) + self._extra_config[constants.OFFSET]


# ONNX containers.
class ONNXSklearnContainer(SklearnContainer):
    """
    Base container for ONNX models.
    The container allows to mirror the Sklearn API.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(ONNXSklearnContainer, self).__init__(model, n_threads, batch_size, extra_config)

        if onnx_runtime_installed():
            import onnxruntime as ort

            self._model = model
            self._extra_config = extra_config

            sess_options = ort.SessionOptions()
            if self._n_threads is not None:
                sess_options.intra_op_num_threads = self._n_threads
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(self._model.SerializeToString(), sess_options=sess_options)
            self._output_names = [self._session.get_outputs()[i].name for i in range(len(self._session.get_outputs()))]
            self.input_names = [input.name for input in self._session.get_inputs()]
        else:
            raise RuntimeError("ONNX Container requires ONNX runtime installed.")

    def _get_named_inputs(self, inputs):
        """
        Retrieve the inputs names from the session object.
        """
        if len(inputs) < len(self.input_names):
            inputs = inputs[0]

        assert len(inputs) == len(self.input_names)

        named_inputs = {}

        for i in range(len(inputs)):
            named_inputs[self.input_names[i]] = np.array(inputs[i])

        return named_inputs


class ONNXSklearnContainerTransformer(ONNXSklearnContainer):
    """
    Container mirroring Sklearn transformers API.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(ONNXSklearnContainerTransformer, self).__init__(model, n_threads, batch_size, extra_config)

        assert len(self._output_names) == 1

    def _transform(self, *inputs):
        named_inputs = self._get_named_inputs(inputs)

        return np.array(self._session.run(self._output_names, named_inputs))

    def transform(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On data transformers it returns transformed output data
        """
        return self._run(self._transform, *inputs, reshape=True)


class ONNXSklearnContainerRegression(ONNXSklearnContainer):
    """
    Container mirroring Sklearn regressors API.
    """

    def __init__(
        self, model, n_threads=None, batch_size=None, is_regression=True, is_anomaly_detection=False, extra_config={}, **kwargs
    ):
        super(ONNXSklearnContainerRegression, self).__init__(model, n_threads, batch_size, extra_config)

        assert not (is_regression and is_anomaly_detection)
        if is_regression:
            assert len(self._output_names) == 1

        self._is_regression = is_regression
        self._is_anomaly_detection = is_anomaly_detection

    def _predict(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On regression returns the predicted values.
        On classification tasks returns the predicted class labels for the input data.
        On anomaly detection (e.g. isolation forest) returns the predicted classes (-1 or 1).
        """
        named_inputs = self._get_named_inputs(inputs)

        if self._is_regression:
            return np.array(self._session.run(self._output_names, named_inputs))
        elif self._is_anomaly_detection:
            return np.array(self._session.run([self._output_names[0]], named_inputs))[0].ravel()
        else:
            return np.array(self._session.run([self._output_names[0]], named_inputs))[0]

    def predict(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On data transformers it returns transformed output data
        """
        return self._run(self._predict, *inputs)


class ONNXSklearnContainerClassification(ONNXSklearnContainerRegression):
    """
    Container mirroring Sklearn classifiers API.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(ONNXSklearnContainerClassification, self).__init__(
            model, n_threads, batch_size, is_regression=False, extra_config=extra_config
        )

        assert len(self._output_names) == 2

    def _predict_proba(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On classification tasks returns the probability estimates.
        """
        named_inputs = self._get_named_inputs(inputs)

        return self._session.run([self._output_names[1]], named_inputs)[0]

    def predict_proba(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On data transformers it returns transformed output data
        """
        return self._run(self._predict_proba, *inputs, reshape=True)


class ONNXSklearnContainerAnomalyDetection(ONNXSklearnContainerRegression):
    """
    Container mirroring Sklearn anomaly detection API.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(ONNXSklearnContainerAnomalyDetection, self).__init__(
            model, n_threads, batch_size, is_regression=False, is_anomaly_detection=True, extra_config=extra_config
        )

        assert len(self._output_names) == 2

    def _decision_function(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On anomaly detection (e.g. isolation forest) returns the decision function scores.
        """
        named_inputs = self._get_named_inputs(inputs)

        scores = np.array(self._session.run([self._output_names[1]], named_inputs)[0]).flatten()
        # Backward compatibility for sklearn <= 0.21
        if constants.IFOREST_THRESHOLD in self._extra_config:
            scores += self._extra_config[constants.IFOREST_THRESHOLD]
        return scores

    def decision_function(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On data transformers it returns transformed output data
        """
        return self._run(self._decision_function, *inputs)

    def score_samples(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On anomaly detection (e.g. isolation forest) returns the decision_function score plus offset_
        """
        return self.decision_function(*inputs) + self._extra_config[constants.OFFSET]


# TVM containers.
class TVMSklearnContainer(ABC):
    """
    Base container for TVM models.
    The container allows to mirror the Sklearn API.
    """

    def __init__(self, model, n_threads, batch_size, extra_config={}):
        """
        Args:
            model: A TVM model
            extra_config: Some additional configuration parameter
        """
        assert tvm_installed()
        import tvm

        self._model = model
        self._extra_config = extra_config
        self._ctx = self._extra_config[constants.TVM_CONTEXT]
        self._input_names = self._extra_config[constants.TVM_INPUT_NAMES]
        self._to_tvm_array = lambda x: tvm.nd.array(x, self._ctx)

    @property
    def model(self):
        return self._model

    def _to_tvm_tensor(self, *inputs):
        return {self._input_names[0]: self._to_tvm_array(inputs[i]) for i in range(len(inputs))}


class TVMSklearnContainerTransformer(TVMSklearnContainer):
    """
    Container mirroring Sklearn transformers API.
    """

    def transform(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On data transformers it returns transformed output data
        """
        self.model.run(**self._to_tvm_tensor(*inputs))
        return self.model.get_output(0).asnumpy().squeeze()


class TVMSklearnContainerRegression(TVMSklearnContainer):
    """
    Container mirroring Sklearn regressors API.
    """

    def __init__(self, model, extra_config={}, is_regression=True, is_anomaly_detection=False, **kwargs):
        super(TVMSklearnContainerRegression, self).__init__(model, extra_config)

        assert not (is_regression and is_anomaly_detection)

        self._is_regression = is_regression
        self._is_anomaly_detection = is_anomaly_detection

    def predict(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On regression returns the predicted values.
        On classification tasks returns the predicted class labels for the input data.
        On anomaly detection (e.g. isolation forest) returns the predicted classes (-1 or 1).
        """
        self.model.run(**self._to_tvm_tensor(*inputs))
        if self._is_regression or self._is_anomaly_detection:
            return self.model.get_output(0).asnumpy().ravel()
        else:
            return self.model.get_output(0).asnumpy().ravel()


class TVMSklearnContainerClassification(TVMSklearnContainerRegression):
    """
    Container mirroring Sklearn classifiers API.
    """

    def __init__(self, model, extra_config={}):
        super(TVMSklearnContainerClassification, self).__init__(model, extra_config, is_regression=False)

    def predict_proba(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On classification tasks returns the probability estimates.
        """
        self.model.run(**self._to_tvm_tensor(*inputs))
        return self.model.get_output(1).asnumpy()


class TVMSklearnContainerAnomalyDetection(TVMSklearnContainerRegression):
    """
    Container mirroring Sklearn anomaly detection API.
    """

    def __init__(self, model, extra_config={}):
        super(TVMSklearnContainerAnomalyDetection, self).__init__(
            model, extra_config, is_regression=False, is_anomaly_detection=True
        )

    def decision_function(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On anomaly detection (e.g. isolation forest) returns the decision function scores.
        """
        self.model.run(**self._to_tvm_tensor(*inputs))
        scores = self.model.get_output(1).asnumpy().ravel()

        # Backward compatibility for sklearn <= 0.21
        if constants.IFOREST_THRESHOLD in self._extra_config:
            scores += self._extra_config[constants.IFOREST_THRESHOLD]
        return scores

    def score_samples(self, *inputs):
        """
        Utility functions used to emulate the behavior of the Sklearn API.
        On anomaly detection (e.g. isolation forest) returns the decision_function score plus offset_
        """
        self.model.run(**self._to_tvm_tensor(*inputs))
        return self.decision_function(inputs) + self._extra_config[constants.OFFSET]
