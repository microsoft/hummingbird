# MIT License
#
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from enum import Enum
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class LearningTask(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3


class Data:
    def __init__(self, X_train, X_test, y_train, y_test, learning_task, qid_train=None, qid_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.learning_task = learning_task
        # For ranking task
        self.qid_train = qid_train
        self.qid_test = qid_test


def prepare_dataset(dataset_folder, dataset, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    prepare_function = globals()["prepare_" + dataset]
    return prepare_function(dataset_folder, nrows)


def prepare_airline(dataset_folder, nrows):  # pylint: disable=too-many-locals
    url = "http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2"
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder, "airline" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))
    print("Preparing dataset ...")
    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)

    cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype,
        "Month": dtype,
        "DayofMonth": dtype,
        "DayofWeek": dtype,
        "CRSDepTime": dtype,
        "CRSArrTime": dtype,
        "FlightNum": dtype,
        "ActualElapsedTime": dtype,
        "Distance": dtype,
        "Diverted": dtype,
        "ArrDelay": dtype,
    }

    df = pd.read_csv(local_url, names=cols, dtype=dtype_columns, nrows=nrows)

    # Encode categoricals as numeric
    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)
    data = Data(
        X_train.astype("|f4").to_numpy(), X_test.astype("|f4").to_numpy(), y_train, y_test, LearningTask.CLASSIFICATION
    )
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_fraud(dataset_folder, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    filename = "creditcard.csv"
    local_url = os.path.join(dataset_folder, filename)
    pickle_url = os.path.join(dataset_folder, "fraud" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    print("Preparing dataset ...")

    os.system("kaggle datasets download mlg-ulb/creditcardfraud -f" + filename + " -p " + dataset_folder)
    df = pd.read_csv(local_url + ".zip", dtype=np.float32, nrows=nrows)
    X = df[[col for col in df.columns if col.startswith("V")]]
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)
    data = Data(
        X_train.astype("|f4").to_numpy(), X_test.astype("|f4").to_numpy(), y_train, y_test, LearningTask.CLASSIFICATION
    )
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_higgs(dataset_folder, nrows):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder, "higgs" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))
    print("Preparing dataset ...")

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    higgs = pd.read_csv(local_url, nrows=nrows, error_bad_lines=False)
    X = higgs.iloc[:, 1:]
    y = higgs.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)
    data = Data(
        X_train.astype("|f4").to_numpy(), X_test.astype("|f4").to_numpy(), y_train, y_test, LearningTask.CLASSIFICATION
    )
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_year(dataset_folder, nrows):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt" ".zip"
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder, "year" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))
    print("Preparing dataset ...")

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    year = pd.read_csv(local_url, nrows=nrows, header=None)
    X = year.iloc[:, 1:]
    y = year.iloc[:, 0]

    if nrows is None:
        # this dataset requires a specific train/test split,
        # with the specified number of rows at the start belonging to the train set,
        # and the rest being the test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=463715, test_size=51630)
    else:
        print("Warning: nrows is specified, not using predefined test/train split for " "YearPredictionMSD.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)

    data = Data(X_train.astype("|f4").to_numpy(), X_test.astype("|f4").to_numpy(), y_train, y_test, LearningTask.REGRESSION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_epsilon(dataset_folder, nrows):
    url_train = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary" "/epsilon_normalized.bz2"
    url_test = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary" "/epsilon_normalized.t.bz2"
    pickle_url = os.path.join(dataset_folder, "epsilon" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")
    local_url_train = os.path.join(dataset_folder, os.path.basename(url_train))
    local_url_test = os.path.join(dataset_folder, os.path.basename(url_test))

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))
    print("Preparing dataset ...")

    if not os.path.isfile(local_url_train):
        urlretrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        urlretrieve(url_test, local_url_test)

    X_train, y_train = load_svmlight_file(local_url_train, dtype=np.float32)
    X_test, y_test = load_svmlight_file(local_url_test, dtype=np.float32)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    y_train[y_train <= 0] = 0
    y_test[y_test <= 0] = 0

    if nrows is not None:
        print("Warning: nrows is specified, not using predefined test/train split for epsilon.")

        X_train = np.vstack((X_train, X_test))
        y_train = np.append(y_train, y_test)
        X_train = X_train[:nrows]
        y_train = y_train[:nrows]
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=77, test_size=0.2,)

    data = Data(X_train.astype("|f4"), X_test.astype("|f4"), y_train, y_test, LearningTask.CLASSIFICATION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_covtype(dataset_folder, nrows=None):
    from sklearn.datasets import fetch_covtype

    print("Preparing dataset ...")
    X, y = fetch_covtype(return_X_y=True)  # pylint: disable=unexpected-keyword-arg
    if nrows is not None:
        X = X[0:nrows]
        y = y[0:nrows]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)
    return Data(X_train.astype("|f4"), X_test.astype("|f4"), y_train, y_test, LearningTask.MULTICLASS_CLASSIFICATION)


def prepare_iris(dataset_folder, nrows):
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    X = _modify_dimension(X, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_test, y_test = _make_n_rows(X, nrows, y)

    return Data(
        X_train.astype("|f4"),
        X_test.astype("|f4"),
        y_train.astype("|i4"),
        y_test.astype("|i4"),
        LearningTask.MULTICLASS_CLASSIFICATION,
    )


def _modify_dimension(X, n_features):
    """
    Modifies the number of features to increase
    or reduce the number of features.
    """
    if n_features is None or n_features == X.shape[1]:
        return X
    if n_features < X.shape[1]:
        return X[:, :n_features]
    res = np.empty((X.shape[0], n_features), dtype=X.dtype)
    res[:, : X.shape[1]] = X[:, :]
    div = max((n_features // X.shape[1]) + 1, 2)
    for i in range(X.shape[1], res.shape[1]):
        j = i % X.shape[1]
        col = X[:, j]
        if X.dtype in (np.float32, np.float64):
            sigma = np.var(col) ** 0.5
            rnd = np.random.randn(len(col)) * sigma / div
            col2 = col + rnd
            res[:, j] -= col2 / div
            res[:, i] = col2
        elif X.dtype in (np.int32, np.int64):
            perm = np.random.permutation(col)
            h = np.random.randint(0, div) % X.shape[0]
            col2 = col.copy()
            col2[h::div] = perm[h::div]  # pylint: disable=E1136
            res[:, i] = col2
            h = (h + 1) % X.shape[0]
            res[h, j] = perm[h]  # pylint: disable=E1136
        else:
            raise NotImplementedError("Unable to add noise to a feature for this type {}".format(X.dtype))
    return res


def _make_n_rows(x, n, y=None):
    """
    Multiplies or reduces the rows of x to get
    exactly *n* rows.
    """
    if n < x.shape[0]:
        if y is None:
            return x[:n].copy()
        return x[:n].copy(), y[:n].copy()
    if len(x.shape) < 2:
        r = np.empty((n,), dtype=x.dtype)
        if y is not None:
            ry = np.empty((n,), dtype=y.dtype)
        for i in range(0, n, x.shape[0]):
            end = min(i + x.shape[0], n)
            r[i:end] = x[0 : end - i]
            if y is not None:
                ry[i:end] = y[0 : end - i]
    else:
        r = np.empty((n, x.shape[1]), dtype=x.dtype)
        if y is not None:
            if len(y.shape) < 2:
                ry = np.empty((n,), dtype=y.dtype)
            else:
                ry = np.empty((n, y.shape[1]), dtype=y.dtype)
        for i in range(0, n, x.shape[0]):
            end = min(i + x.shape[0], n)
            r[i:end, :] = x[0 : end - i, :]
            if y is not None:
                if len(y.shape) < 2:
                    ry[i:end] = y[0 : end - i]
                else:
                    ry[i:end, :] = y[0 : end - i, :]
    if y is None:
        return r
    return r, ry
