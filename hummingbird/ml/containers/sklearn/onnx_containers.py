# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
ONNX output containers for the sklearn API are listed here.
"""

import dill
import os
import numpy as np
import shutil

from hummingbird.ml._utils import onnx_runtime_installed, from_strings_to_ints
from hummingbird.ml.operator_converters import constants
from hummingbird.ml.containers._sklearn_api_containers import (
    SklearnContainer,
    SklearnContainerTransformer,
    SklearnContainerRegression,
    SklearnContainerClassification,
    SklearnContainerAnomalyDetection,
)
import onnx

if onnx_runtime_installed():
    import onnxruntime as ort


class ONNXSklearnContainer(SklearnContainer):
    """
    Base container for ONNX models.
    The container allows to mirror the Sklearn API.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(ONNXSklearnContainer, self).__init__(model, n_threads, batch_size, extra_config)

        assert onnx_runtime_installed(), "ONNX Container requires ONNX runtime installed."

        sess_options = ort.SessionOptions()
        if self._n_threads is not None:
            sess_options.intra_op_num_threads = self._n_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(self._model.SerializeToString(), sess_options=sess_options)
        self._output_names = [self._session.get_outputs()[i].name for i in range(len(self._session.get_outputs()))]
        self._input_names = [input.name for input in self._session.get_inputs()]
        self._extra_config = extra_config

    def save(self, location):
        """
        Method used to save the container for future use.

        Args:
            location: The location on the file system where to save the model.
        """
        assert self.model is not None, "Saving a None model is undefined."

        if constants.TEST_INPUT in self._extra_config:
            self._extra_config[constants.TEST_INPUT] = None

        if location.endswith("zip"):
            location = location[:-4]
        assert not os.path.exists(location), "Directory {} already exists.".format(location)
        os.makedirs(location)

        # Save the model type.
        with open(os.path.join(location, constants.SAVE_LOAD_MODEL_TYPE_PATH), "w") as file:
            file.write("onnx")

        # Save the actual model.
        onnx.save(self.model, os.path.join(location, constants.SAVE_LOAD_ONNX_PATH))

        model = self.model
        session = self._session
        self._model = None
        self._session = None

        # Save the container.
        with open(os.path.join(location, constants.SAVE_LOAD_CONTAINER_PATH), "wb") as file:
            dill.dump(self, file)

        # Zip the dir.
        shutil.make_archive(location, "zip", location)

        # Remove the directory.
        shutil.rmtree(location)

        self._model = model
        self._session = session

    @staticmethod
    def load(location, do_unzip_and_model_type_check=True):
        """
        Method used to load a container from the file system.

        Args:
            location: The location on the file system where to load the model.
            do_unzip_and_model_type_check: Whether to unzip the model and check the type.

        Returns:
            The loaded model.
        """

        assert onnx_runtime_installed
        import onnx
        import onnxruntime as ort

        container = None

        if do_unzip_and_model_type_check:
            # Unzip the dir.
            zip_location = location
            if not location.endswith("zip"):
                zip_location = location + ".zip"
            else:
                location = zip_location[:-4]
            shutil.unpack_archive(zip_location, location, format="zip")

            assert os.path.exists(location), "Model location {} does not exist.".format(location)

            # Load the model type.
            with open(os.path.join(location, constants.SAVE_LOAD_MODEL_TYPE_PATH), "r") as file:
                model_type = file.readline()
                assert model_type == "onnx", "Expected ONNX model type, got {}".format(model_type)

        # Load the actual model.
        model = onnx.load(os.path.join(location, constants.SAVE_LOAD_ONNX_PATH))

        # Load the container.
        with open(os.path.join(location, constants.SAVE_LOAD_CONTAINER_PATH), "rb") as file:
            container = dill.load(file)
        assert container is not None, "Failed to load the model container."

        # Setup the container.
        container._model = model
        sess_options = ort.SessionOptions()
        if container._n_threads is not None:
            # Need to set the number of threads to use as set in the original container.
            sess_options.intra_op_num_threads = container._n_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        container._session = ort.InferenceSession(container._model.SerializeToString(), sess_options=sess_options)

        return container

    def _get_named_inputs(self, inputs):
        """
        Retrieve the inputs names from the session object.
        """
        if len(inputs) < len(self._input_names):
            inputs = inputs[0]

        assert len(inputs) == len(self._input_names)

        named_inputs = {}

        for i in range(len(inputs)):
            input_ = np.array(inputs[i])
            if input_.dtype.kind in constants.SUPPORTED_STRING_TYPES:
                assert constants.MAX_STRING_LENGTH in self._extra_config

                input_ = from_strings_to_ints(input_, self._extra_config[constants.MAX_STRING_LENGTH])
            named_inputs[self._input_names[i]] = input_

        return named_inputs


class ONNXSklearnContainerTransformer(ONNXSklearnContainer, SklearnContainerTransformer):
    """
    Container for ONNX models mirroring Sklearn transformers API.
    """

    def _transform(self, *inputs):
        assert len(self._output_names) == 1
        named_inputs = self._get_named_inputs(inputs)
        return np.array(self._session.run(self._output_names, named_inputs))[0]


class ONNXSklearnContainerRegression(ONNXSklearnContainer, SklearnContainerRegression):
    """
    Container for ONNX models mirroring Sklearn regressors API.
    """

    def _predict(self, *inputs):
        named_inputs = self._get_named_inputs(inputs)

        if self._is_regression:
            assert len(self._output_names) == 1
            return np.array(self._session.run(self._output_names, named_inputs))[0].ravel()
        elif self._is_anomaly_detection:
            assert len(self._output_names) == 2
            return np.array(self._session.run([self._output_names[0]], named_inputs))[0].ravel()
        else:
            assert len(self._output_names) == 2
            return np.array(self._session.run([self._output_names[0]], named_inputs))[0]


class ONNXSklearnContainerClassification(ONNXSklearnContainerRegression, SklearnContainerClassification):
    """
    Container for ONNX models mirroring Sklearn classifiers API.
    """

    def _predict_proba(self, *inputs):
        assert len(self._output_names) == 2

        named_inputs = self._get_named_inputs(inputs)

        return self._session.run([self._output_names[1]], named_inputs)[0]


class ONNXSklearnContainerAnomalyDetection(ONNXSklearnContainerRegression, SklearnContainerAnomalyDetection):
    """
    Container for ONNX models mirroring Sklearn anomaly detection API.
    """

    def _decision_function(self, *inputs):
        assert len(self._output_names) == 2

        named_inputs = self._get_named_inputs(inputs)

        return np.array(self._session.run([self._output_names[1]], named_inputs)[0]).flatten()
