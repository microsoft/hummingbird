#!/usr/bin/env python
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

import os
import sys
import argparse
import json
import ast
import gc
import psutil
import signal
import pickle
import numpy as np
import warnings
from pathlib import Path
from scipy import stats
from memory_profiler import memory_usage

import benchmarks.trees.train as train
import benchmarks.trees.score as score
from benchmarks.trees.metrics import get_metrics
from benchmarks.datasets import prepare_dataset, LearningTask

from hummingbird.ml._utils import (
    xgboost_installed,
    lightgbm_installed,
    sklearn_installed,
    onnx_ml_tools_installed,
    onnx_runtime_installed,
    tvm_installed,
)


ROOT_PATH = Path(__file__).absolute().parent.parent.parent


class TimeoutException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)

        self.errors = errors


def get_number_processors(args):
    if args.cpus == 0:
        return psutil.cpu_count(logical=False)
    return args.cpus


def print_sys_info(args):
    import xgboost
    import lightgbm
    import sklearn
    import torch

    print("System  : %s" % sys.version)
    print("OS  : %s" % sys.platform)
    print("Xgboost : %s" % xgboost.__version__)
    print("LightGBM: %s" % lightgbm.__version__)
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
    parser = argparse.ArgumentParser(description="Benchmark xgboost/lightgbm/random forest on real datasets")
    parser.add_argument(
        "-dataset",
        default="all",
        type=str,
        help="The dataset to be used for benchmarking. 'all' for all datasets: "
        "fraud, epsilon, year, covtype, higgs, airline",
    )
    parser.add_argument(
        "-datadir", default=os.path.join(ROOT_PATH, "benchmarks/trees/datasets/"), type=str, help="The root datasets folder"
    )
    parser.add_argument(
        "-modeldir", default=os.path.join(ROOT_PATH, "benchmarks/trees/models/"), type=str, help="The root models folder"
    )
    parser.add_argument(
        "-operator", default="all", type=str, help=("Comma-separated list of operators to run; 'all' run rf, xgb and lgbm")
    )
    parser.add_argument(
        "-backend",
        default="all",
        type=str,
        help=(
            "Comma-separated list of frameworks to run against the baselines;" "'all' run onnx-ml, hb-pytorch, hb-torchscript"
        ),
    )
    parser.add_argument(
        "-cpus", default=6, type=int, help=("#CPUs to use for the benchmarks; 0 means psutil.cpu_count(logical=False)")
    )
    parser.add_argument(
        "-batch_size", default=10000, type=int, help=("Supported batch size. By default we score one record at a time.")
    )
    parser.add_argument(
        "-gpu",
        default=False,
        action="store_true",
        help=("Whether to run scoring on GPU (for the supported frameworks) or not"),
    )
    parser.add_argument("-output", default=None, type=str, help="Output json file with runtime/accuracy stats")
    parser.add_argument(
        "-ntrees",
        default=500,
        type=int,
        help=("Number of trees. Default is as specified in " "the respective dataset configuration"),
    )
    parser.add_argument(
        "-nrows",
        default=None,
        type=int,
        help=(
            "Subset of rows in the datasets to use. Useful for test running "
            "benchmarks on small amounts of data. WARNING: Some datasets will "
            "give incorrect accuracy results if nrows is specified as they have "
            "predefined train/test splits."
        ),
    )
    parser.add_argument("-niters", default=5, type=int, help=("Number of iterations for each experiment"))
    parser.add_argument(
        "-batch_benchmark",
        default=False,
        action="store_true",
        help=("Whether to do a single batch benchmark with specified batch_size and niters (not on the whole data)"),
    )
    parser.add_argument("-max_depth", default=8, type=int, help=("Maxmimum number of levels in the trees"))
    parser.add_argument(
        "-validate", default=False, action="store_true", help="Validate prediction output and fails accordingly."
    )
    parser.add_argument("-extra", default="{}", help="Extra arguments as a python dictionary")
    args = parser.parse_args()
    # Default value for output json file.
    if not args.output:
        args.output = "result-{}-{}-{}-{}.json".format("gpu" if args.gpu else args.cpus, args.ntrees, args.max_depth, args.batch_size)
    return args


def get_data(data, size=-1):
    np_data = data.to_numpy() if not isinstance(data, np.ndarray) else data

    if size != -1:
        msg = "Requested size bigger than the data size (%d vs %d)" % (size, np_data.shape[0])
        assert size <= np_data.shape[0], msg
        np_data = np_data[0:size]

    return np_data


# Benchmarks a single dataset.
def benchmark(args, dataset_folder, model_folder, dataset):
    warnings.filterwarnings("ignore")
    data = prepare_dataset(dataset_folder, dataset, args.nrows)
    results = {}
    args.dataset = dataset
    operators = args.operator
    if operators == "all":
        operators = "rf,lgbm,xgb"
    for op in operators.split(","):
        print("Running '%s' ..." % op)
        results[op] = {}
        model_name = op + "-" + str(args.ntrees) + "-" + str(args.max_depth) + "-" + str(args.cpus)
        model_full_name = os.path.join(model_folder, model_name + ".pkl")
        trainer = train.TrainEnsembleAlgorithm.create(op, data.learning_task)
        if args.batch_benchmark:
            test_size = args.batch_size
        else:
            test_size = data.X_test.shape[0]

        X_test = get_data(data.X_test, size=test_size)
        y_test = get_data(data.y_test, size=test_size)

        with trainer:
            if not os.path.exists(model_full_name):
                train_time = trainer.fit(data, args)
                pred = trainer.test(X_test)
                results[op] = {
                    "train_time": str(train_time),
                    "train_accuracy": str(get_metrics(y_test, pred, data.learning_task)),
                }
                model = trainer.model
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                pickle.dump(model, open(model_full_name, "wb"), protocol=4)
            else:
                model = pickle.load(open(model_full_name, "rb"))

            times = []
            mean = 0
            mem = 0

            try:
                for i in range(args.niters):
                    set_alarm(3600)
                    times.append(trainer.predict(model, X_test, args))
                    set_alarm(0)
                mean = stats.trim_mean(times, 1 / len(times)) if args.niters > 1 else times[0]
                gc.collect()
                mem = max(memory_usage((trainer.predict, (model, X_test, args))))
            except Exception as e:
                print(e)
                pass

            results[op].update({"prediction_time": mean})
            results[op].update({"peak_mem": mem})
            outer_ops = args.operator
            args.operator = op

            if args.backend == "all":
                args.backend = "onnx-ml,hb-pytorch,hb-torchscript,hb-onnx,hb-tvm"
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
                    conversion_time = scorer.convert(model, data, X_test, args, os.path.join(model_folder, model_name))

                    times = []
                    prediction_time = 0

                    try:
                        for i in range(args.niters):
                            set_alarm(3600)
                            times.append(scorer.predict(X_test))
                            set_alarm(0)
                        prediction_time = times[0] if args.niters == 1 else stats.trim_mean(times, 1 / len(times))
                        gc.collect()
                        mem = max(memory_usage((scorer.predict, (X_test,))))
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
                        np.testing.assert_allclose(scorer.predictions, trainer.predictions, rtol=1e-5, atol=1e-6)

            args.operator = outer_ops
    return results


def main():
    args = parse_args()
    args.cpus = get_number_processors(args)
    args.extra = ast.literal_eval(args.extra)
    print_sys_info(args)
    if args.dataset == "all":
        args.dataset = "fraud,epsilon,year,covtype,higgs,airline"
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
    assert xgboost_installed, "benchmark requires XGBoost"
    assert lightgbm_installed, "benchmark requires LightGBM"
    assert sklearn_installed, "benchmark requires sklearn"

    main()
