# BSD License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Miguel Gonzalez-Fierro nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) Microsoft Corporation. All rights reserved.


from abc import ABC, abstractmethod
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

from benchmarks.timer import Timer
from benchmarks.datasets import LearningTask


class TrainEnsembleAlgorithm(ABC):
    @staticmethod
    def create(name, learning_task):  # pylint: disable=too-many-return-statements
        if name == "xgb":
            return XgbAlgorithm(learning_task)
        if name == "lgbm":
            return LgbmAlgorithm(learning_task)
        if name == "rf":
            return RandomForestAlgorithm(learning_task)
        raise ValueError("Unknown algorithm: " + name)

    def __init__(self, learning_task):
        self.params = {}
        self.model = None
        self.predictions = []
        self.learning_task = learning_task

    @abstractmethod
    def fit(self, data, args):
        pass

    def test(self, data):
        assert self.model is not None
        return self.model.predict(data)

    def predict(self, model, predict_data, args):
        batch_size = args.batch_size

        if self.learning_task == LearningTask.REGRESSION:
            predict_fn = model.predict
        else:
            predict_fn = model.predict_proba

        with Timer() as t:
            total_size = len(predict_data)
            iterations = total_size // batch_size

            if total_size == batch_size:
                self.predictions = predict_fn(predict_data)
            else:
                iterations += 1 if total_size % batch_size > 0 else 0
                iterations = max(1, iterations)

                if self.learning_task == LearningTask.CLASSIFICATION:
                    self.predictions = np.empty([total_size, 2], dtype="f4")
                if self.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
                    self.predictions = np.empty([total_size, model.n_classes_], dtype="f4")
                if self.learning_task == LearningTask.REGRESSION:
                    self.predictions = np.empty([total_size], dtype="f4")

                for i in range(0, iterations):
                    start = i * batch_size
                    end = min(start + batch_size, total_size)
                    batch = predict_data[start:end]
                    self.predictions[start:end] = predict_fn(batch)

        return t.interval

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None:
            del self.model


# learning parameters shared by all algorithms, using the xgboost convention
shared_params = {"learning_rate": 0.1, "reg_lambda": 1}


class XgbAlgorithm(TrainEnsembleAlgorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        params.update({"tree_method": "hist"})
        params.update({"max_leaves": 256, "nthread": args.cpus, "max_depth": args.max_depth, "ntrees": args.ntrees})
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "reg:squarederror"
            params["args"] = {}
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary:logistic"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
            params["args"] = {"scale_pos_weight": params["scale_pos_weight"]}
        elif data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "multi:softmax"
            params["num_class"] = np.max(data.y_test) + 1
            params["args"] = {}
        else:
            raise ValueError("Unknown task: " + data.learning_task)

        params.update(args.extra)

        return params

    def fit(self, data, args):
        params = self.configure(data, args)

        tree_method = params["tree_method"]
        predictor = "cpu_predictor"

        if args.gpu:
            tree_method = "gpu_hist"
            predictor = "gpu_predictor"

        if data.learning_task == LearningTask.REGRESSION:
            self.model = xgb.XGBRegressor(
                max_depth=params["max_depth"],
                max_leaves=params["max_leaves"],
                n_estimators=params["ntrees"],
                learning_rate=params["learning_rate"],
                objective=params["objective"],
                nthread=params["nthread"],
                predictor=predictor,
                tree_method=tree_method,
                reg_lambda=params["reg_lambda"],
                **(params["args"])
            )
        else:
            self.model = xgb.XGBClassifier(
                max_depth=params["max_depth"],
                max_leaves=params["max_leaves"],
                n_estimators=params["ntrees"],
                learning_rate=params["learning_rate"],
                objective=params["objective"],
                nthread=params["nthread"],
                predictor=predictor,
                tree_method=tree_method,
                reg_lambda=params["reg_lambda"],
                **(params["args"])
            )

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class LgbmAlgorithm(TrainEnsembleAlgorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        params.update({"max_leaves": 256, "njobs": args.cpus, "max_depth": args.max_depth, "ntrees": args.ntrees})
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "regression"
            params["args"] = {}
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
            params["args"] = {"scale_pos_weight": params["scale_pos_weight"]}
        elif data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "multiclass"
            params["num_class"] = np.max(data.y_test) + 1
            params["args"] = {}
        params.update(args.extra)

        return params

    def fit(self, data, args):
        params = self.configure(data, args)

        if data.learning_task == LearningTask.REGRESSION:
            self.model = lgb.LGBMRegressor(
                max_depth=params["max_depth"],
                n_estimators=params["ntrees"],
                num_leaves=params["max_leaves"],
                learning_rate=params["learning_rate"],
                objective=params["objective"],
                n_jobs=params["njobs"],
                reg_lambda=params["reg_lambda"],
            )
        else:
            self.model = lgb.LGBMClassifier(
                max_depth=params["max_depth"],
                n_estimators=params["ntrees"],
                num_leaves=params["max_leaves"],
                learning_rate=params["learning_rate"],
                objective=params["objective"],
                n_jobs=params["njobs"],
                reg_lambda=params["reg_lambda"],
                **(params["args"])
            )

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class RandomForestAlgorithm(TrainEnsembleAlgorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        params.update({"njobs": args.cpus, "ntrees": args.ntrees, "max_depth": args.max_depth})
        params.update(args.extra)

        return params

    def fit(self, data, args):
        params = self.configure(data, args)

        if data.learning_task == LearningTask.REGRESSION:
            self.model = RandomForestRegressor(
                max_depth=params["max_depth"], n_estimators=params["ntrees"], n_jobs=params["njobs"]
            )
        else:
            self.model = RandomForestClassifier(
                max_depth=params["max_depth"], n_estimators=params["ntrees"], n_jobs=params["njobs"]
            )

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train.astype("|i4"))

        return t.interval
