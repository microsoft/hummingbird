# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from packaging.version import Version, parse
import openml
import sklearn
import operator
import keyword
import re
from pathlib import Path
import numpy as np
import random
import os
import joblib
import warnings

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
    Normalizer,
    Binarizer,
    KBinsDiscretizer,
    PolynomialFeatures,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer as Imputer
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from hummingbird.ml.supported import sklearn_operator_list

warnings.filterwarnings("ignore")


def get_sk_op(comp_class, unsupported_ops, not_sk_ops):
    if not comp_class.startswith("sklearn.") or (
        comp_class.startswith("sklearn.pipeline.") and comp_class.split(".")[-1] != "Pipeline"
    ):
        if comp_class not in not_sk_ops:
            not_sk_ops.append(comp_class)
        return None
    elif comp_class.startswith("sklearn."):
        if comp_class.split(".")[-1] not in supported_ops and comp_class not in unsupported_ops:
            unsupported_ops.append(comp_class)
            return None

        comp = eval(comp_class.split(".")[-1] + "()")

        return comp


def get_column_transformer(
    flow, component_map, unsupported_ops, not_sk_ops, numeric_columns, nominal_columns,
):
    col_transformer_id = flow.flow_id
    transformers = []
    for key in flow.components:
        comp_id = flow.components[key].flow_id
        comp_class = flow.components[key].class_name
        if comp_class == "sklearn.pipeline.Pipeline":
            comp = get_pipeline(
                flow.components[key], component_map, unsupported_ops, not_sk_ops, numeric_columns, nominal_columns,
            )
        else:
            comp = get_sk_op(comp_class, unsupported_ops, not_sk_ops)

        if key == "numeric" and len(numeric_columns) > 0:
            transformers.append((key, comp, numeric_columns))
        elif key == "nominal" and len(nominal_columns) > 0:
            transformers.append((key, comp, nominal_columns))
        else:
            transformers.append((key, comp, []))

        component_map[comp_id] = comp

    col_transformer = ColumnTransformer(transformers)
    component_map[col_transformer_id] = col_transformer
    return col_transformer


def get_pipeline(
    flow, component_map, unsupported_ops, not_sk_ops, numeric_columns, nominal_columns,
):
    steps = []
    pipeline_id = flow.flow_id
    for prop in eval(flow.parameters["steps"].replace('["nonetype", null],', "")):
        key = prop["value"]["key"]
        comp_id = flow.components[prop["value"]["key"]].flow_id
        comp_class = flow.components[prop["value"]["key"]].class_name
        if comp_class == "sklearn.pipeline.Pipeline":
            comp = get_pipeline(
                flow.components[prop["value"]["key"]],
                component_map,
                unsupported_ops,
                not_sk_ops,
                numeric_columns,
                nominal_columns,
            )
        elif comp_class.endswith("ColumnTransformer"):
            comp = get_column_transformer(
                flow.components[prop["value"]["key"]],
                component_map,
                unsupported_ops,
                not_sk_ops,
                numeric_columns,
                nominal_columns,
            )
        else:
            comp = get_sk_op(comp_class, unsupported_ops, not_sk_ops)

        steps.append((key, comp))
        component_map[comp_id] = comp

    pipeline = sklearn.pipeline.Pipeline(steps)
    component_map[pipeline_id] = pipeline
    return pipeline


def init_parameters(p, run_id, component_map):
    run = openml.runs.get_run(run_id)
    for p in run.parameter_settings:
        name = p["oml:name"]
        value = p["oml:value"].strip()
        comp_id = int(p["oml:component"])
        comp = component_map[comp_id]

        if hasattr(comp, "n_jobs") and comp.n_jobs != -1:
            comp.n_jobs = -1

        if value == "true":
            value = "True"
        elif value == "false":
            value = "False"
        elif value in ["null", "Null", "NULL", "none", "NONE"]:
            value = "None"

        if (
            name == "validation_fraction"
            or name == "dtype"
            or value in ["deprecated", '"deprecated"']
            or isinstance(comp, sklearn.pipeline.Pipeline) and name == "steps"
            or isinstance(comp, sklearn.linear_model.SGDClassifier) and name == "max_iter" and value == "None"
        ):
            continue

        if str(type(comp).__name__) == "ColumnTransformer" and name == "transformers":
            value = value.replace("true", "True").replace("false", "False")
            for p in eval(value):
                key = p["value"]["key"]
                ids = eval(str(p["value"]["argument_1"]))

                for i in range(len(comp.transformers)):
                    if comp.transformers[i][0] == key:
                        comp.transformers[i] = (key, comp.transformers[i][1], ids)
                        break
            continue

        if isinstance(comp, sklearn.preprocessing.OneHotEncoder) and name == "categorical_features":
            idx = eval(value)
            if idx is not None and len(idx) > 0:
                comp.categorical_features = idx
            continue

        if value in ["NaN", '"NaN"']:
            exec("comp.{} = np.nan".format(name))
        else:
            try:
                exec("comp.{} = {}".format(name, value))
            except Exception:
                exec("comp.{} = '{}'".format(name, value.replace("'", "").replace('"', "")))


if __name__ == "__main__":

    DATA_ROOT = "{}/benchmarks/pipelines/openml-cc18/".format(Path(__file__).absolute().parent.parent.parent)
    DATA_SAMPLE_FRACTION = 0.2

    supported_ops = [op.__name__ for op in sklearn_operator_list]
    supported_ops.append("ColumnTransformer")
    supported_ops.append("FeatureUnion")

    benchmark_suite = openml.study.get_suite("OpenML-CC18")
    global_supported_ops = {}
    global_unsupported_ops = {}
    sum_unsupported = 0
    count_unsupported = 0
    sum_steps = 0
    count_steps = 0
    for task_id in benchmark_suite.tasks[:]:
        X, y = openml.tasks.get_task(task_id).get_X_and_y()
        X, X_test, y, y_test = train_test_split(X, y, test_size=DATA_SAMPLE_FRACTION)

        if not os.path.exists(DATA_ROOT + "/{}".format(task_id)):
            os.makedirs(DATA_ROOT + "/{}".format(task_id))

        if not os.path.exists(DATA_ROOT + "/{}/data".format(task_id)):
            os.makedirs(DATA_ROOT + "/{}/data".format(task_id))

        if not os.path.exists(DATA_ROOT + "/{}/pipelines".format(task_id)):
            os.makedirs(DATA_ROOT + "/{}/pipelines".format(task_id))

        np.save(DATA_ROOT + "/{}/data/X.dat".format(task_id), X_test)
        np.save(DATA_ROOT + "/{}/data/y.dat".format(task_id), y_test)

        log_file = open(DATA_ROOT + "/{}/{}.log".format(task_id, task_id), "w")

        numeric_columns = [
            i
            for i in range(len(openml.tasks.get_task(task_id).get_dataset().features))
            if openml.tasks.get_task(task_id).get_dataset().features[i].data_type == "numeric" and i < X.shape[1]
        ]

        nominal_columns = [
            i
            for i in range(len(openml.tasks.get_task(task_id).get_dataset().features))
            if openml.tasks.get_task(task_id).get_dataset().features[i].data_type == "nominal" and i < X.shape[1]
        ]

        vers = parse(openml.__version__)
        renamed_version = Version("0.11")
        if vers < renamed_version:
            runs_df = openml.evaluations.list_evaluations(
                function="predictive_accuracy", task=[task_id], output_format="dataframe"
            )
        else:
            runs_df = openml.evaluations.list_evaluations(
                function="predictive_accuracy", tasks=[task_id], output_format="dataframe"
            )
        sk_runs_df = runs_df[runs_df["flow_name"].str.startswith("sklearn.pipeline.Pipeline(")]
        sk_runs_df = sk_runs_df.loc[sk_runs_df.groupby("flow_id")["value"].idxmax()]

        log_file.write("===================================Task: {}=======================================\n".format(task_id))

        not_sk = 0
        unsupported = 0
        failures_on_train = 0
        for i in range(sk_runs_df.shape[0]):
            run_id = sk_runs_df["run_id"].values[i]
            flow_id = sk_runs_df["flow_id"].values[i]

            flow = openml.flows.get_flow(flow_id)

            component_map = {}
            unsupported_ops = []
            not_sk_ops = []
            p = get_pipeline(flow, component_map, unsupported_ops, not_sk_ops, numeric_columns, nominal_columns)
            sum_steps += len(p.named_steps)
            count_steps += 1

            found_unsupported_column_transformer = False
            for transf in p.named_steps.values():
                if isinstance(transf, sklearn.compose.ColumnTransformer):
                    for name, _, column_indices in transf.transformers:
                        if isinstance(column_indices, list):
                            if len(column_indices) > 0 and isinstance(column_indices[0], bool):
                                found_unsupported_column_transformer = True
                    sum_steps += len(transf.transformers) - 1
            if found_unsupported_column_transformer:
                unsupported += 1
                log_file.write("Run not translatable due to not supporter ColumnTransformer\n")
                continue
            if len(not_sk_ops) > 0:
                not_sk += 1
                log_file.write("Run {} not translatable due to non sk: ".format(run_id) + ",".join(not_sk_ops) + "\n")
                log_file.write(
                    "Run {} not translatable due to unsupported ops: ".format(run_id) + ",".join(unsupported_ops) + "\n"
                )
            elif len(unsupported_ops) > 0:
                unsupported += 1
                for op in unsupported_ops:
                    op = op.split(".")[-1]
                    if op in global_unsupported_ops:
                        global_unsupported_ops[op] += 1
                    else:
                        global_unsupported_ops[op] = 1
                log_file.write("Run {} not translatable due to non sk: ".format(run_id) + ",".join(not_sk_ops) + "\n")
                log_file.write(
                    "Run {} not translatable due to unsupported ops: ".format(run_id) + ",".join(unsupported_ops) + "\n"
                )
            else:
                init_parameters(p, run_id, component_map)
                try:
                    p.fit(X, y)
                    log_file.write(
                        "Run {} translated. Training Accuracy: {}\n".format(run_id, accuracy_score(p.predict(X), y))
                    )
                    for op in component_map.values():
                        if str(type(op).__name__) in global_supported_ops:
                            global_supported_ops[str(type(op).__name__)] += 1
                        else:
                            global_supported_ops[str(type(op).__name__)] = 1
                    joblib.dump(p, open(DATA_ROOT + "/{}/pipelines/{}.pkl".format(task_id, run_id), "wb"))
                except Exception as e:
                    failures_on_train += 1
                    log_file.write("{}, {}, {}{}".format(run_id, flow_id, p, "\n"))
                    log_file.write("{}{}".format(e, "\n"))

            log_file.write("--------------------------------------------------------------------------------------------\n")
            log_file.flush()

        message = "Task: {}, Total: {},  Not Pure SK: {}, Unsupported: {}, Train Failures: {}, Generated: {}, Coverage: {}\n".format(
            task_id,
            sk_runs_df.shape[0],
            not_sk,
            unsupported,
            failures_on_train,
            sk_runs_df.shape[0] - not_sk - unsupported - failures_on_train,
            1 - (unsupported) / (sk_runs_df.shape[0] - not_sk - failures_on_train),
        )
        log_file.write(message)
        print(message)
        sum_unsupported += 1 - (unsupported) / (sk_runs_df.shape[0] - not_sk - failures_on_train)
        count_unsupported += 1

        log_file.close()

    print("============================Supported operator usage================================")

    sorted_global_supported_ops = sorted(global_supported_ops.items(), key=operator.itemgetter(1), reverse=True)
    for k, v in sorted_global_supported_ops:
        print(k, v)

    print("============================Unsupported operator usage================================")

    sorted_global_unsupported_ops = sorted(global_unsupported_ops.items(), key=operator.itemgetter(1), reverse=True)
    for k, v in sorted_global_unsupported_ops:
        print(k, v)

    print("Average coverage: " + str(sum_unsupported / count_unsupported))
    print("Average pipeline length: " + str(sum_steps / count_steps))
