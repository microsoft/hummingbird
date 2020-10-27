# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import argparse
import json
import ast
import psutil
from pathlib import Path
import signal
import time
import numpy as np
import sklearn
import joblib
import torch
from scipy import stats
import gc

import benchmarks.pipelines.score as score
from benchmarks.timer import Timer


ROOT_PATH = Path(__file__).absolute().parent.parent.parent


def print_sys_info(args):
    print("System  : %s" % sys.version)
    print("OS  : %s" % sys.platform)
    print("Sklearn: %s" % sklearn.__version__)
    print("Torch: %s" % torch.__version__)
    if args.gpu:
        print("Running on GPU")
    else:
        print("#CPU {}".format(psutil.cpu_count(logical=False)))


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
    parser = argparse.ArgumentParser(description="Benchmark for OpenML pipelines")
    parser.add_argument(
        "-pipedir",
        default=os.path.join(ROOT_PATH, "benchmarks/pipelines/openml-cc18/"),
        type=str,
        help=("The root folder containing all pipelines"),
    )
    parser.add_argument("-backend", default="torch", type=str, help=("Comma-separated list of Hummingbird's backends to run"))
    parser.add_argument("-gpu", default=False, action="store_true", help=("Whether to run scoring on GPU or not"))
    parser.add_argument("-output", default=None, type=str, help="Output json file with runtime stats")
    parser.add_argument("-niters", default=5, type=int, help=("Number of iterations for each experiment"))
    parser.add_argument("-validate", default=False, help="Validate prediction output and fails accordigly.")

    args = parser.parse_args()
    # default value for output json file
    if not args.output:
        args.output = "result-{}.json".format("gpu" if args.gpu else "cpu")
    return args


def main():
    args = parse_args()
    print(args.gpu)
    print_sys_info(args)
    results = {}
    set_signal()
    skipped = 0
    total = 0

    if not os.path.exists(args.pipedir):
        raise Exception(args.pipedir + " directory not found")

    tasks = os.listdir(args.pipedir)
    tasks = list(map(lambda x: int(x), tasks))
    tasks.sort()
    for task in list(map(lambda x: str(x), tasks)):
        print("Task-{}".format(task))
        task_dir = os.path.join(args.pipedir, task)
        task_pip_dir = os.path.join(task_dir, "pipelines")

        X = np.load(os.path.join(task_dir, "data", "X.dat.npy"))
        results[task] = {"dataset_size": X.shape[0], "num_features": X.shape[1]}
        pipelines = os.listdir(os.path.join(task_pip_dir))
        pipelines = list(map(lambda x: int(x[:-4]), pipelines))
        pipelines.sort()
        res = []
        for pipeline in list(map(lambda x: str(x), pipelines)):
            total += 1
            with open(os.path.join(task_pip_dir, pipeline + ".pkl"), "rb") as f:
                model = joblib.load(f)

                assert model is not None

                results[task][pipeline] = {}
                times = []
                mean = 0
                print("Pipeline-{}".format(pipeline))

                try:
                    for i in range(args.niters):
                        set_alarm(3600)
                        with Timer() as t:
                            res = model.predict(X)
                        times.append(t.interval)
                        set_alarm(0)
                    mean = stats.trim_mean(times, 1 / len(times)) if args.niters > 1 else times[0]
                    gc.collect()
                except Exception as e:
                    print(e)
                    pass
                results[task][pipeline] = {"prediction_time": mean}

                for backend in args.backend.split(","):
                    print("Running '%s' ..." % backend)
                    scorer = score.ScoreBackend.create(backend)
                    with scorer:
                        try:
                            conversion_time = scorer.convert(model, X, args)
                        except Exception as e:
                            skipped += 1
                            print(e)
                            continue

                        times = []
                        prediction_time = 0
                        try:
                            for i in range(args.niters):
                                set_alarm(3600)
                                times.append(scorer.predict(X))
                                set_alarm(0)
                            prediction_time = times[0] if args.niters == 1 else stats.trim_mean(times, 1 / len(times))
                            gc.collect()
                        except Exception as e:
                            skipped += 1
                            print(e)
                            pass

                        results[task][pipeline][backend] = {
                            "conversion_time": str(conversion_time),
                            "prediction_time": str(prediction_time),
                            "speedup": "0"
                            if mean == 0 or prediction_time == 0
                            else str(mean / prediction_time)
                            if prediction_time < mean
                            else str(-prediction_time / mean),
                        }

                        print(results[task][pipeline][backend])

                        if args.validate:
                            np.testing.assert_allclose(scorer.predictions, res, rtol=1e-5, atol=1e-6)

    output = json.dumps(results, indent=2)
    output_file = open(args.output, "w")
    output_file.write(output + "\n")
    output_file.close()

    print("All results written to file '%s'" % args.output)
    print("Total num of pipelines: {}; skipped: {}".format(total, skipped))


if __name__ == "__main__":
    main()
