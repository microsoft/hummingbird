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

from benchmarks.timer import Timer
from benchmarks.datasets import LearningTask

from hummingbird.ml import constants
from hummingbird.ml import convert, convert_batch


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
                "n_classes": 1 if data.learning_task == LearningTask.REGRESSION else model.n_classes_,
            }
        )

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
        self.predict_fn = None
        self.batch_benchmark = False

    def convert(self, model, data, test_data, args, model_name):
        self.configure(data, model, args)
        self.batch_benchmark = args.batch_benchmark
        extra_config = {constants.N_THREADS: self.params["nthread"]}

        with Timer() as t:
            if self.batch_benchmark:
                self.model = convert(model, self.backend, test_data, device=self.params["device"], extra_config=extra_config)
            else:
                remainder_size = test_data.shape[0] % self.params["batch_size"]
                self.model = convert_batch(
                    model, self.backend, test_data, remainder_size, device=self.params["device"], extra_config=extra_config
                )

        if data.learning_task == LearningTask.REGRESSION:
            self.predict_fn = self.model.predict
        else:
            self.predict_fn = self.model.predict_proba

        return t.interval

    def predict(self, predict_data):
        assert self.predict_fn is not None

        # For the batch by batch prediction case, we do not want to include the cost of
        # doing final outputs concatenation into time measurement
        with Timer() as t:
            if self.batch_benchmark:
                self.predictions = self.predict_fn(predict_data)
            else:
                self.predictions = self.predict_fn(predict_data, concatenate_outputs=False)

        if not self.batch_benchmark:
            self.predictions = np.concatenate(self.predictions)

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

    def convert(self, model, data, test_data, args, model_name):
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert import convert_lightgbm
        from skl2onnx import convert_sklearn
        from onnxmltools.convert.common.data_types import FloatTensorType

        self.configure(data, model, args)
        self.is_regression = data.learning_task == LearningTask.REGRESSION

        with Timer() as t:
            if self.params["operator"] == "xgb":
                initial_type = [("input", FloatTensorType([1, self.params["input_size"]]))]
                if model._Booster.feature_names is not None:
                    fixed_names = list(map(lambda x: str(x), range(len(model._Booster.feature_names))))
                    model._Booster.feature_names = fixed_names
                self.model = convert_xgboost(model, initial_types=initial_type, target_opset=11)
            else:
                batch = min(len(data.X_test), self.params["batch_size"])
                remainder = len(data.X_test) % batch
                initial_type = [("input", FloatTensorType([batch, self.params["input_size"]]))]

                if self.params["operator"] == "lgbm":
                    converter = convert_lightgbm
                elif self.params["operator"] == "rf":
                    converter = convert_sklearn

                self.model = converter(model, initial_types=initial_type)
                if remainder > 0:
                    initial_type = [("input", FloatTensorType([remainder, self.params["input_size"]]))]
                    self.remainder_model = converter(model, initial_types=initial_type, target_opset=11)

            import onnxruntime as ort

            self.remainder_sess = None
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.params["nthread"]
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.sess = ort.InferenceSession(self.model.SerializeToString(), sess_options=sess_options)
            if self.remainder_model is not None:
                self.remainder_sess = ort.InferenceSession(self.remainder_model.SerializeToString(), sess_options=sess_options)

        return t.interval

    def predict(self, predict_data):
        assert self.model is not None

        batch_size = 1 if self.params["operator"] == "xgb" else self.params["batch_size"]
        if self.is_regression:
            output_name_index = 0
        else:
            output_name_index = 1

        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[output_name_index].name

        with Timer() as t:
            total_size = len(predict_data)
            iterations = total_size // batch_size
            iterations += 1 if total_size % batch_size > 0 else 0
            iterations = max(1, iterations)
            self.predictions = np.empty([total_size, self.params["n_classes"]], dtype="f4")
            for i in range(0, iterations):
                start = i * batch_size
                end = min(start + batch_size, total_size)

                if self.params["operator"] == "xgb":
                    pred = self.sess.run([output_name], {input_name: predict_data[start:end, :]})
                    if type(pred) is list:
                        pred = pred[0]
                    self.predictions[start:end, :] = pred
                elif self.params["operator"] == "lgbm" or "rf":
                    if total_size > batch_size and i == iterations - 1 and self.remainder_model is not None:
                        pred = self.remainder_sess.run([output_name], {input_name: predict_data[start:end, :]})
                    else:
                        pred = self.sess.run([output_name], {input_name: predict_data[start:end, :]})

                    if self.is_regression:
                        self.predictions[start:end, :] = pred[0]
                    else:
                        self.predictions[start:end, :] = list(map(lambda x: list(x.values()), pred[0]))

        if self.is_regression:
            self.predictions = self.predictions.flatten()

        return t.interval

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        if self.remainder_model is not None:
            del self.remainder_model
