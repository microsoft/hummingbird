# Copyright (c) 2019, Microsoft CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of Microsoft CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.svm.classes import LinearSVC, NuSVC, SVC
import sys

from benchmarks.timer import Timer
from benchmarks.datasets import LearningTask

from hummingbird.ml import constants
from hummingbird.ml import convert_batch


class ScoreBackend(ABC):
    @staticmethod
    def create(name):
        if name == "hb-pytorch":
            return HBBackend("torch")
        if name == "hb-torchscript":
            return HBBackend("torch.jit")
        if name == "hb-tvm":
            return HBBackend("tvm")
        if name == "hb-onnx":
            return HBBackend("onnx")
        if name == "onnx-ml":
            return ONNXMLBackend()
        raise ValueError("Unknown backend: " + name)

    def __init__(self):
        self.backend = None
        self.model = None
        self.params = {}
        self.predictions = None
        self.n_classes = None

    def configure(self, data, model, args):
        self.params.update(
            {
                "batch_size": len(data.X_test)
                if args.batch_size == -1 or len(data.X_test) < args.batch_size
                else args.batch_size,
                "input_size": data.X_test.shape[1] if isinstance(data.X_test, np.ndarray) else len(data.X_test.columns),
                "device": "cpu" if args.gpu is False else "cuda",
                "nthread": args.cpus,
                "extra_config": args.extra,
                "transform": True
                if args.operator
                not in [
                    "LogisticRegression",
                    "SGDClassifier",
                    "LogisticRegressionCV",
                    "SGDClassifier",
                    "LinearSVC",
                    "NuSVC",
                    "SVC",
                    "DecisionTreeClassifier",
                    "MLPClassifier",
                    "BernoulliNB",
                ]
                else False,
                "n_classes": 231
                if type(model) == PolynomialFeatures
                else 20
                if data.learning_task == LearningTask.REGRESSION
                else len(set(data.y_test)),
                "operator": args.operator,
            }
        )

    @staticmethod
    def get_data(data, size=-1):
        np_data = data.to_numpy() if not isinstance(data, np.ndarray) else data

        if size != -1:
            np_data = np_data[0:size, :]

        return np_data

    @abstractmethod
    def convert(self, model, args, model_name):
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
    def __init__(self, backend):
        super(HBBackend, self).__init__()
        self.backend = backend

    def convert(self, model, data, args, model_name):
        self.configure(data, model, args)

        test_data = self.get_data(data.X_test)
        remainder_size = test_data.shape[0] % self.params["batch_size"]

        with Timer() as t:
            self.model = convert_batch(
                model,
                self.backend,
                test_data,
                remainder_size,
                device=self.params["device"],
                extra_config={constants.N_THREADS: self.params["nthread"]},
            )

        return t.interval

    def predict(self, data):
        assert self.model is not None

        is_regression = data.learning_task == LearningTask.REGRESSION or "SVC" in self.params["operator"]

        with Timer() as t:
            predict_data = self.get_data(data.X_test)
            if self.params["transform"]:
                self.predictions = self.model.transform(predict_data)
            elif is_regression:
                self.predictions = self.model.predict(predict_data)
            else:
                self.predictions = self.model.predict_proba(predict_data)

        return t.interval

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class ONNXMLBackend(ScoreBackend):
    def __init__(self):
        super().__init__()
        self.remainder_model = None  # for batch inference in case we have remainder records

    def configure(self, data, model, args):
        super(ONNXMLBackend, self).configure(data, model, args)
        self.params.update({"operator": args.operator})

    def convert(self, model, data, args, model_name):
        from skl2onnx import convert_sklearn
        from onnxmltools.convert.common.data_types import FloatTensorType

        self.configure(data, model, args)

        with Timer() as t:
            batch = min(len(data.X_test), self.params["batch_size"])
            remainder = len(data.X_test) % batch
            initial_type = [("input", FloatTensorType([batch, self.params["input_size"]]))]

            self.model = convert_sklearn(model, initial_types=initial_type)
            if remainder > 0:
                initial_type = [("input", FloatTensorType([remainder, self.params["input_size"]]))]
                self.remainder_model = convert_sklearn(model, initial_types=initial_type, target_opset=11)
        return t.interval

    def predict(self, data):
        import onnxruntime as ort

        assert self.model is not None

        remainder_sess = None
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.params["nthread"]
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(self.model.SerializeToString(), sess_options=sess_options)
        if self.remainder_model is not None:
            remainder_sess = ort.InferenceSession(self.remainder_model.SerializeToString(), sess_options=sess_options)

        batch_size = 1 if self.params["operator"] == "xgb" else self.params["batch_size"]
        input_name = sess.get_inputs()[0].name
        is_regression = data.learning_task == LearningTask.REGRESSION or "SVC" in self.params["operator"]
        if is_regression:
            output_name_index = 0
        else:
            output_name_index = 1
        output_name = sess.get_outputs()[output_name_index].name

        with Timer() as t:
            predict_data = ScoreBackend.get_data(data.X_test)
            total_size = len(predict_data)
            iterations = total_size // batch_size
            iterations += 1 if total_size % batch_size > 0 else 0
            iterations = max(1, iterations)

            for i in range(0, iterations):
                start = i * batch_size
                end = min(start + batch_size, total_size)

                if self.params["operator"] == "xgb":
                    self.predictions[start:end, :] = sess.run([output_name], {input_name: predict_data[start:end, :]})
                else:
                    if i == iterations - 1 and self.remainder_model is not None:
                        pred = remainder_sess.run([output_name], {input_name: predict_data[start:end, :]})
                    else:
                        pred = sess.run([output_name], {input_name: predict_data[start:end, :]})

                    if is_regression:
                        self.predictions = pred[0]
                    else:
                        self.predictions = list(map(lambda x: list(x.values()), pred[0]))

        del sess
        if remainder_sess is not None:
            del remainder_sess

        return t.interval

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        if self.remainder_model is not None:
            del self.remainder_model
