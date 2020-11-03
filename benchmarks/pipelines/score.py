# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np

from benchmarks.timer import Timer

import hummingbird.ml


class ScoreBackend(ABC):
    @staticmethod
    def create(name):
        if name in ["torch", "torch.jit", "tvm", "onnx"]:
            return HBBackend(name)
        raise ValueError("Unknown backend: " + name)

    def __init__(self):
        self.model = None
        self.params = {}
        self.predictions = None

    def configure(self, data, model, args):
        self.params.update({"device": "cpu" if args.gpu is False else "cuda"})

    @staticmethod
    def get_data(data):
        np_data = data.to_numpy() if not isinstance(data, np.ndarray) else data

        return np_data

    @abstractmethod
    def convert(self, model, args):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass


class HBBackend(ScoreBackend):
    def __init__(self, backend_name):
        super(HBBackend, self).__init__()
        self._backend_name = backend_name

    def convert(self, model, data, args):
        self.configure(data, model, args)

        data = self.get_data(data)

        with Timer() as t:
            self.model = hummingbird.ml.convert(model, self._backend_name, data, device=self.params["device"])

        return t.interval

    def predict(self, data):
        assert self.model is not None

        data = self.get_data(data)

        with Timer() as t:
            self.predictions = self.model.predict(data)

        return t.interval

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
