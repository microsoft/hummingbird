# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Pytorch and TorchScript output containers for the sklearn API are listed here.
"""

import dill
import os
import numpy as np
import shutil
import torch

from hummingbird.ml._utils import pandas_installed, get_device, from_strings_to_ints
from hummingbird.ml.operator_converters import constants
from hummingbird.ml.containers._sklearn_api_containers import (
    SklearnContainer,
    SklearnContainerTransformer,
    SklearnContainerRegression,
    SklearnContainerClassification,
    SklearnContainerAnomalyDetection,
)

if pandas_installed():
    from pandas import DataFrame
else:
    DataFrame = None


# PyTorch containers.
class PyTorchSklearnContainer(SklearnContainer):
    """
    Base container for PyTorch models.
    """

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

        if "torch.jit" in str(type(self.model)):
            # This is a torchscript model.
            # Save the model type.
            with open(os.path.join(location, constants.SAVE_LOAD_MODEL_TYPE_PATH), "w") as file:
                file.write("torch.jit")

            # Save the actual model.
            self.model.save(os.path.join(location, constants.SAVE_LOAD_TORCH_JIT_PATH))

            model = self.model
            self._model = None

            # Save the container.
            with open(os.path.join(location, "container.pkl"), "wb") as file:
                dill.dump(self, file)

            self._model = model
        elif "Executor" in str(type(self.model)):
            # This is a pytorch model.
            # Save the model type.
            with open(os.path.join(location, constants.SAVE_LOAD_MODEL_TYPE_PATH), "w") as file:
                file.write("torch")

            # Save the actual model plus the container
            with open(os.path.join(location, constants.SAVE_LOAD_TORCH_JIT_PATH), "wb") as file:
                dill.dump(self, file)
        else:
            raise RuntimeError("Model type {} not recognized.".format(type(self.model)))

        # Zip the dir.
        shutil.make_archive(location, "zip", location)

        # Remove the directory.
        shutil.rmtree(location)

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
        container = None

        # Unzip the dir.
        if do_unzip_and_model_type_check:
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

        if model_type == "torch.jit":
            # This is a torch.jit model
            model = torch.jit.load(os.path.join(location, constants.SAVE_LOAD_TORCH_JIT_PATH))
            with open(os.path.join(location, "container.pkl"), "rb") as file:
                container = dill.load(file)
            container._model = model
        elif model_type == "torch":
            # This is a pytorch  model
            with open(os.path.join(location, constants.SAVE_LOAD_TORCH_JIT_PATH), "rb") as file:
                container = dill.load(file)
        else:
            raise RuntimeError("Model type {} not recognized".format(model_type))

        # Need to set the number of threads to use as set in the original container.
        if container._n_threads is not None:
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.set_num_threads(container._n_threads)

        return container

    def to(self, device):
        self.model.to(device)
        return self


class PyTorchSklearnContainerTransformer(SklearnContainerTransformer, PyTorchSklearnContainer):
    """
    Container for PyTorch models mirroring Sklearn transformers API.
    """

    def _transform(self, *inputs):
        return self.model.forward(*inputs).cpu().numpy()


class PyTorchSklearnContainerRegression(SklearnContainerRegression, PyTorchSklearnContainer):
    """
    Container for PyTorch models mirroring Sklearn regressor API.
    """

    def _predict(self, *inputs):
        if self._is_regression:
            output = self.model.forward(*inputs).cpu().numpy()
            if len(output.shape) == 2 and output.shape[1] > 1:
                # Multioutput regression
                return output
            else:
                return output.ravel()
        elif self._is_anomaly_detection:
            return self.model.forward(*inputs)[0].cpu().numpy().ravel()
        else:
            return self.model.forward(*inputs)[0].cpu().numpy().ravel()


class PyTorchSklearnContainerClassification(SklearnContainerClassification, PyTorchSklearnContainerRegression):
    """
    Container for PyTorch models mirroring Sklearn classifiers API.
    """

    def _predict_proba(self, *input):
        return self.model.forward(*input)[1].cpu().numpy()


class PyTorchSklearnContainerAnomalyDetection(PyTorchSklearnContainerRegression, SklearnContainerAnomalyDetection):
    """
    Container for PyTorch models mirroning the Sklearn anomaly detection API.
    """

    def _decision_function(self, *inputs):
        return self.model.forward(*inputs)[1].cpu().numpy().ravel()


# TorchScript containers.
def _torchscript_wrapper(device, function, *inputs, extra_config={}):
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
            if type(inputs[i]) is list:
                inputs[i] = np.array(inputs[i])
            if type(inputs[i]) is np.ndarray:
                # Convert string arrays into int32.
                if inputs[i].dtype.kind in constants.SUPPORTED_STRING_TYPES:
                    assert constants.MAX_STRING_LENGTH in extra_config

                    inputs[i] = from_strings_to_ints(inputs[i], extra_config[constants.MAX_STRING_LENGTH])
                if inputs[i].dtype == np.float64:
                    # We convert double precision arrays into single precision. Sklearn does the same.
                    inputs[i] = inputs[i].astype("float32")
                inputs[i] = torch.from_numpy(inputs[i])
            elif type(inputs[i]) is not torch.Tensor:
                raise RuntimeError("Inputer tensor {} of not supported type {}".format(i, type(inputs[i])))
            if device.type != "cpu" and device is not None:
                inputs[i] = inputs[i].to(device)
        return function(*inputs)


class TorchScriptSklearnContainerTransformer(PyTorchSklearnContainerTransformer):
    """
    Container for TorchScript models mirroring Sklearn transformers API.
    """

    def transform(self, *inputs):
        device = get_device(self.model)
        f = super(TorchScriptSklearnContainerTransformer, self)._transform
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x, extra_config=self._extra_config)  # noqa: E731

        return self._run(f_wrapped, *inputs)


class TorchScriptSklearnContainerRegression(PyTorchSklearnContainerRegression):
    """
    Container for TorchScript models mirroring Sklearn regressors API.
    """

    def predict(self, *inputs):
        device = get_device(self.model)
        f = super(TorchScriptSklearnContainerRegression, self)._predict
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x, extra_config=self._extra_config)  # noqa: E731

        return self._run(f_wrapped, *inputs)


class TorchScriptSklearnContainerClassification(PyTorchSklearnContainerClassification):
    """
    Container for TorchScript models mirroring Sklearn classifiers API.
    """

    def predict(self, *inputs):
        device = get_device(self.model)
        f = super(TorchScriptSklearnContainerClassification, self)._predict
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x, extra_config=self._extra_config)  # noqa: E731

        return self._run(f_wrapped, *inputs)

    def predict_proba(self, *inputs):
        device = get_device(self.model)
        f = super(TorchScriptSklearnContainerClassification, self)._predict_proba
        f_wrapped = lambda *x: _torchscript_wrapper(device, f, *x, extra_config=self._extra_config)  # noqa: E731

        return self._run(f_wrapped, *inputs)


class TorchScriptSklearnContainerAnomalyDetection(PyTorchSklearnContainerAnomalyDetection):
    """
    Container for TorchScript models mirroring Sklearn anomaly detection API.
    """

    def predict(self, *inputs):
        device = get_device(self.model)
        f = super(TorchScriptSklearnContainerAnomalyDetection, self)._predict
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x, extra_config=self._extra_config)  # noqa: E731

        return self._run(f_wrapped, *inputs)

    def decision_function(self, *inputs):
        device = get_device(self.model)
        f = super(TorchScriptSklearnContainerAnomalyDetection, self)._decision_function
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x, extra_config=self._extra_config)  # noqa: E731

        scores = self._run(f_wrapped, *inputs)

        if constants.IFOREST_THRESHOLD in self._extra_config:
            scores += self._extra_config[constants.IFOREST_THRESHOLD]
        return scores

    def score_samples(self, *inputs):
        device = get_device(self.model)
        f = self.decision_function
        f_wrapped = lambda x: _torchscript_wrapper(device, f, x, extra_config=self._extra_config)  # noqa: E731

        return self._run(f_wrapped, *inputs) + self._extra_config[constants.OFFSET]
