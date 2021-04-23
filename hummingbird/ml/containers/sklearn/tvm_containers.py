# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
TVM output containers for the sklearn API are listed here.
"""

import dill
import os
import numpy as np
import shutil

from hummingbird.ml._utils import tvm_installed
from hummingbird.ml.operator_converters import constants
from hummingbird.ml.containers._sklearn_api_containers import (
    SklearnContainer,
    SklearnContainerTransformer,
    SklearnContainerRegression,
    SklearnContainerClassification,
    SklearnContainerAnomalyDetection,
)

if tvm_installed():
    import tvm
    import tvm._ffi
    from tvm.contrib import graph_runtime


class TVMSklearnContainer(SklearnContainer):
    """
    Base container for TVM models.
    The container allows to mirror the Sklearn API.
    The test input size must be the same as the batch size this container is created.
    """

    def __init__(self, model, n_threads=None, batch_size=None, extra_config={}):
        super(TVMSklearnContainer, self).__init__(model, n_threads, batch_size, extra_config=extra_config)

        assert tvm_installed(), "TVM Container requires TVM installed."

        self._ctx = self._extra_config[constants.TVM_CONTEXT]
        self._to_tvm_array = lambda x: tvm.nd.array(x, self._ctx)
        self._input_names = self._extra_config[constants.TVM_INPUT_NAMES]
        self._tvm_tensors = {name: self._to_tvm_array(np.array([])) for name in self._input_names}
        self._pad_input = (
            self._extra_config[constants.TVM_PAD_INPUT] if constants.TVM_PAD_INPUT in self._extra_config else False
        )

        os.environ["TVM_NUM_THREADS"] = str(self._n_threads)

    def save(self, location):
        """
        Method used to save the container for future use.

        Args:
            location: The location on the file system where to save the model.
        """
        assert self.model is not None, "Saving a None model is undefined."

        if location.endswith("zip"):
            location = location[:-4]
        assert not os.path.exists(location), "Directory {} already exists.".format(location)
        os.makedirs(location)

        # Save the model type.
        with open(os.path.join(location, constants.SAVE_LOAD_MODEL_TYPE_PATH), "w") as file:
            file.write("tvm")

        # Save the actual model.
        path_lib = os.path.join(location, constants.SAVE_LOAD_TVM_LIB_PATH)
        self._extra_config[constants.TVM_LIB].export_library(path_lib)
        with open(os.path.join(location, constants.SAVE_LOAD_TVM_GRAPH_PATH), "w") as file:
            file.write(self._extra_config[constants.TVM_GRAPH])
        with open(os.path.join(location, constants.SAVE_LOAD_TVM_PARAMS_PATH), "wb") as file:
            file.write(tvm.relay.save_param_dict(self._extra_config[constants.TVM_PARAMS]))

        # Remove all information that cannot be pickled.
        if constants.TEST_INPUT in self._extra_config:
            self._extra_config[constants.TEST_INPUT] = None
        input_tensors = self._tvm_tensors
        lib = self._extra_config[constants.TVM_LIB]
        graph = self._extra_config[constants.TVM_GRAPH]
        params = self._extra_config[constants.TVM_PARAMS]
        ctx = self._extra_config[constants.TVM_CONTEXT]
        model = self._model
        self._tvm_tensors = None
        self._extra_config[constants.TVM_LIB] = None
        self._extra_config[constants.TVM_GRAPH] = None
        self._extra_config[constants.TVM_PARAMS] = None
        self._extra_config[constants.TVM_CONTEXT] = None
        self._ctx = "cpu" if self._ctx.device_type == 1 else "cuda"
        self._model = None

        # Save the container.
        with open(os.path.join(location, constants.SAVE_LOAD_CONTAINER_PATH), "wb") as file:
            dill.dump(self, file)

        # Zip the dir.
        shutil.make_archive(location, "zip", location)

        # Remove the directory.
        shutil.rmtree(location)

        # Restore the information
        self._tvm_tensors = input_tensors
        self._extra_config[constants.TVM_LIB] = lib
        self._extra_config[constants.TVM_GRAPH] = graph
        self._extra_config[constants.TVM_PARAMS] = params
        self._extra_config[constants.TVM_CONTEXT] = ctx
        self._ctx = ctx
        self._model = model

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
        assert tvm_installed(), "TVM Container requires TVM installed."

        _load_param_dict = tvm._ffi.get_global_func("tvm.relay._load_param_dict")

        # We borrow this function directly from Relay.
        # Relay when imported tryies to download schedules data,
        # but at inference time access to disk or network could be blocked.
        def load_param_dict(param_bytes):
            if isinstance(param_bytes, (bytes, str)):
                param_bytes = bytearray(param_bytes)
            load_arr = _load_param_dict(param_bytes)
            return {v.name: v.array for v in load_arr}

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
                assert model_type == "tvm", "Expected TVM model type, got {}".format(model_type)

        # Load the actual model.
        path_lib = os.path.join(location, constants.SAVE_LOAD_TVM_LIB_PATH)
        graph = open(os.path.join(location, constants.SAVE_LOAD_TVM_GRAPH_PATH)).read()
        lib = tvm.runtime.module.load_module(path_lib)
        params = load_param_dict(open(os.path.join(location, constants.SAVE_LOAD_TVM_PARAMS_PATH), "rb").read())

        # Load the container.
        with open(os.path.join(location, constants.SAVE_LOAD_CONTAINER_PATH), "rb") as file:
            container = dill.load(file)
        assert container is not None, "Failed to load the model container."

        # Setup the container.
        ctx = tvm.cpu() if container._ctx == "cpu" else tvm.gpu
        container._model = graph_runtime.create(graph, lib, ctx)
        container._model.set_input(**params)

        container._extra_config[constants.TVM_GRAPH] = graph
        container._extra_config[constants.TVM_LIB] = lib
        container._extra_config[constants.TVM_PARAMS] = params
        container._extra_config[constants.TVM_CONTEXT] = ctx
        container._ctx = ctx
        container._tvm_tensors = {name: container._to_tvm_array(np.array([])) for name in container._input_names}

        # Need to set the number of threads to use as set in the original container.
        os.environ["TVM_NUM_THREADS"] = str(container._n_threads)

        return container

    def _predict_common(self, output_index, *inputs):
        # Compute padding.
        padding_size = 0
        if self._pad_input:
            padding_size = self._batch_size - inputs[0].shape[0]

        # Prepare inputs.
        for i, input_ in enumerate(inputs):
            assert (
                self._pad_input or input_.shape[0] == self._batch_size
            ), "The number of input rows {} is different from the batch size {} the TVM model is compiled for.".format(
                input_.shape[0], self._batch_size
            )
            if padding_size > 0:
                input_ = np.pad(input_, [(0, padding_size), (0, 0)])
            self._tvm_tensors[self._input_names[i]] = self._to_tvm_array(input_)

        # Compute the predictions.
        self.model.run(**self._tvm_tensors)
        result = self.model.get_output(output_index).asnumpy()

        # Remove padding if necessary.
        if padding_size > 0:
            result = result[:-padding_size]
        return result


class TVMSklearnContainerTransformer(TVMSklearnContainer, SklearnContainerTransformer):
    """
    Container for TVM models mirroring Sklearn transformers API.
    """

    def _transform(self, *inputs):
        return self._predict_common(0, *inputs)


class TVMSklearnContainerRegression(TVMSklearnContainer, SklearnContainerRegression):
    """
    Container for TVM models mirroring Sklearn regressors API.
    """

    def _predict(self, *inputs):
        out = self._predict_common(0, *inputs)
        return out.ravel()


class TVMSklearnContainerClassification(TVMSklearnContainerRegression, SklearnContainerClassification):
    """
    Container for TVM models mirroring Sklearn classifiers API.
    """

    def _predict_proba(self, *inputs):
        return self._predict_common(1, *inputs)


class TVMSklearnContainerAnomalyDetection(TVMSklearnContainerRegression, SklearnContainerAnomalyDetection):
    """
    Container for TVM models mirroring Sklearn anomaly detection API.
    """

    def _decision_function(self, *inputs):
        out = self._predict_common(1, *inputs)
        return out.ravel()
