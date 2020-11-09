#!/usr/bin/env python
# BSD License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
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

# Copyright (c) Microsoft Corporation. All rights reserved.

import os
import sys
import argparse
import json
import ast
from pathlib import Path
import psutil
import signal
import numpy as np
import warnings
import gc
from scipy import stats
from memory_profiler import memory_usage

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing.data import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm.classes import LinearSVC, NuSVC, SVC

import benchmarks.operators.train as train
import benchmarks.operators.score as score
from benchmarks.datasets import prepare_dataset, LearningTask

from hummingbird.ml._utils import sklearn_installed, onnx_ml_tools_installed, onnx_runtime_installed, tvm_installed

ROOT_PATH = Path(__file__).absolute().parent.parent.parent

ALL_OPS = [
    # Linear models
    LogisticRegression,
    SGDClassifier,
    LinearSVC,
    NuSVC,
    SVC,
    # Classifiers: Other
    BernoulliNB,
    MLPClassifier,
    # Trees
    DecisionTreeClassifier,
    # Feature Pre-processing
    Binarizer,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    StandardScaler,
]


class TimeoutException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)

        self.errors = errors


def get_number_processors(args):
    if args.cpus == 0:
        return psutil.cpu_count(logical=False)
    return args.cpus


def print_sys_info(args):
    import sklearn
    import torch

    print("System  : %s" % sys.version)
    print("OS  : %s" % sys.platform)
    print("Sklearn : %s" % sklearn.__version__)
    print("PyTorch : %s" % torch.__version__)

    # Optional imports
    try:
        import onnxruntime

        print("ORT   : %s" % onnxruntime.__version__)
    except ImportError:
        pass
    try:
        import tvm

        print("TVM : %s" % tvm.__version__)
    except ImportError:
        pass

    if args.gpu:
        print("Running on GPU")
    else:
        print("#CPU   : %d" % args.cpus)


def signal_handler(signum, frame):
    print("1 hour timeout triggered.")
    raise Exception("Timeout")


def set_alarm(timeout=0):
    if sys.platform == "linux":
        signal.alarm(timeout)


def set_signal():
    if sys.platform == "linux":
        signal.signal(signal.SIGALRM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sklearn/HB on iris")
    parser.add_argument("-dataset", default="iris", type=str, help="The dataset to be used for benchmarking.")
    parser.add_argument(
        "-datadir",
        default=os.path.join(ROOT_PATH, "benchmarks/operators/datasets/"),
        type=str,
        help="The root datasets folder",
    )
    parser.add_argument(
        "-modeldir", default=os.path.join(ROOT_PATH, "benchmarks/operators/models/"), type=str, help="The root models folder"
    )
    parser.add_argument(
        "-operator", default="all", type=str, help=("Comma-separated list of operators to run; " "'all' run all")
    )
    parser.add_argument(
        "-backend",
        default="all",
        type=str,
        help=("Comma-separated list of train algorithms to run; " "'all' run onnx-ml, hb-torchscript, hb-tvm"),
    )
    parser.add_argument(
        "-cpus", default=1, type=int, help=("#CPUs to use for the benchmarks; " "0 means psutil.cpu_count(logical=False)")
    )
    parser.add_argument(
        "-batch_size", default=1000000, type=int, help=("Supported batch size. By default we score one record at a time.")
    )
    parser.add_argument(
        "-gpu", default=False, action="store_true", help=("Whether to run scoring on SPU or not.  Adding this flag uses gpu")
    )
    parser.add_argument("-output", default=None, type=str, help="Output json file with runtime/accuracy stats")
    parser.add_argument(
        "-nrows",
        default=1000000,
        type=int,
        help=(
            "Subset of rows in the datasets to use. Useful for test running "
            "benchmarks on small amounts of data. WARNING: Some datasets will "
            "give incorrect accuracy results if nrows is specified as they have "
            "predefined train/test splits."
        ),
    )
    parser.add_argument("-niters", default=5, type=int, help=("Number of iterations for each experiment"))
    parser.add_argument("-validate", default=False, help="Validate prediction output and fails accordigly.")
    parser.add_argument("-extra", default="{}", help="Extra arguments as a python dictionary")
    args = parser.parse_args()
    # default value for output json file
    if not args.output:
        args.output = "result-{}-{}.json".format("gpu" if args.gpu else "cpu", args.nrows)
    return args


# benchmarks a single dataset
def benchmark(args, dataset_folder, model_folder, dataset):
    warnings.filterwarnings("ignore")
    data = prepare_dataset(dataset_folder, dataset, args.nrows)
    results = {}
    # "all" runs all operators
    args.dataset = dataset
    operators = args.operator
    if operators == "all":
        operators = {k.__name__: k for k in ALL_OPS}
        operators = ",".join(operators.keys())
    for op in operators.split(","):
        print("Running '%s' ..." % op)
        results[op] = {}
        model_name = op + "-" + str(args.cpus)
        trainer = train.CreateModel.create(op)
        with trainer:
            train_time = trainer.fit(data, args)
            results[op] = {"train_time": str(train_time)}
            model = trainer.model
            times = []
            mean = 0
            mem = 0
            try:
                for i in range(args.niters):
                    set_alarm(3600)
                    times.append(trainer.predict(data))
                    set_alarm(0)
                mean = stats.trim_mean(times, 1 / len(times)) if args.niters > 1 else times[0]
            except Exception as e:
                print(e)
                pass
            results[op].update({"prediction_time": mean})
            gc.collect()
            mem = max(memory_usage((trainer.predict, (data,))))
            results[op].update({"peak_mem": mem})
            outer_ops = args.operator
            args.operator = op

            if args.backend == "all":
                args.backend = "onnx-ml,hb-pytorch,hb-torchscript,hb-onnx"
            if "hb-tvm" in args.backend:
                assert (
                    tvm_installed
                ), "To run benchmark with TVM you need to have TVM installed. Either install TVM or remove it from the backends."
            if "hb-onnx" in args.backend:
                assert (
                    onnx_runtime_installed
                ), "To run benchmark with ONNX you need to have ONNX runtime installed. Either install ONNX runtime or remove ONNX from the backends."
            if "onnx-ml" in args.backend:
                assert (
                    onnx_runtime_installed and onnx_ml_tools_installed
                ), "To run benchmark with ONNX-ML you need to have ONNX runtime and ONNXMLTOOLS installed. Either install ONNX runtime and ONNXMLTOOLS or remove ONNX-ML from the backends."
            for backend in args.backend.split(","):
                print("Running '%s' ..." % backend)
                scorer = score.ScoreBackend.create(backend)
                with scorer:
                    try:
                        conversion_time = scorer.convert(model, data, args, os.path.join(model_folder, model_name))
                    except Exception as e:
                        print(e)
                        continue
                    times = []
                    prediction_time = 0
                    mem = 0
                    try:
                        for i in range(args.niters):
                            set_alarm(3600)
                            times.append(scorer.predict(data))
                            set_alarm(0)
                        prediction_time = times[0] if args.niters == 1 else stats.trim_mean(times, 1 / len(times))
                        gc.collect()
                        mem = max(memory_usage((scorer.predict, (data,))))
                    except Exception as e:
                        print(e)
                        pass

                    results[op][backend] = {
                        "conversion_time": str(conversion_time),
                        "prediction_time": str(prediction_time),
                        "peak_mem": str(mem),
                        "is_same_output": "None"
                        if len(trainer.predictions) == 0 or scorer is None or scorer.predictions is None
                        else np.allclose(trainer.predictions, scorer.predictions, atol=1e-6),
                    }

                    print(results[op][backend])

                    if args.validate:
                        np.testing.assert_allclose(
                            scorer.predictions, trainer.predictions, equal_nan=False, rtol=1e-5, atol=1e-6
                        )

            args.operator = outer_ops
    return results


def main():
    args = parse_args()
    args.cpus = get_number_processors(args)
    args.extra = ast.literal_eval(args.extra)
    print_sys_info(args)
    results = {}
    set_signal()

    for dataset in args.dataset.split(","):
        print("Dataset '%s' ..." % dataset)
        dataset_folder = os.path.join(args.datadir, dataset)
        model_folder = os.path.join(args.modeldir, dataset)
        results.update({dataset: benchmark(args, dataset_folder, model_folder, dataset)})
        print(json.dumps({dataset: results[dataset]}, indent=2, sort_keys=True))
        output = json.dumps(results, indent=2)
        output_file = open(args.output, "w")
        output_file.write(output + "\n")
        output_file.close()

    print("All results written to file '%s'" % args.output)


if __name__ == "__main__":
    assert sklearn_installed, "benchmark requires sklearn"
    assert onnx_ml_tools_installed and onnx_runtime_installed, "benchmark requires ORT and ONNXMLTOOLS"

    main()
