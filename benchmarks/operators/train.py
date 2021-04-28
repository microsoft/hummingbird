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
import time
import numpy as np
import pandas as pd
import pickle
import os.path

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
from sklearn.preprocessing._discretization import KBinsDiscretizer
from sklearn.svm.classes import LinearSVC, NuSVC, SVC
from sklearn.metrics import accuracy_score
from benchmarks.timer import Timer
from benchmarks.datasets import LearningTask


class CreateModel(ABC):
    @staticmethod
    def create(name):  # pylint: disable=too-many-return-statements
        if name == DecisionTreeClassifier.__name__:
            return CreateDecisionTreeClassifier()
        if name == LogisticRegression.__name__:
            return CreateLogisticRegression()
        if name == LogisticRegressionCV.__name__:
            return CreateLogisticRegressionCV()
        if name == SGDClassifier.__name__:
            return CreateSGDClassifier()
        if name == BernoulliNB.__name__:
            return CreateBernoulliNB()
        if name == MLPClassifier.__name__:
            return CreateMLPClassifier()
        if name == Binarizer.__name__:
            return CreateBinarizer()
        if name == KBinsDiscretizer.__name__:
            return CreateKBinsDiscretizer()
        if name == MaxAbsScaler.__name__:
            return CreateMaxAbsScaler()
        if name == MinMaxScaler.__name__:
            return CreateMinMaxScaler()
        if name == Normalizer.__name__:
            return CreateNormalizer()
        if name == PolynomialFeatures.__name__:
            return CreatePolynomialFeatures()
        if name == RobustScaler.__name__:
            return CreateRobustScaler()
        if name == StandardScaler.__name__:
            return CreateStandardScaler()
        if name == LinearSVC.__name__:
            return CreateLinearSVC()
        if name == NuSVC.__name__:
            return CreateNuSVC()
        if name == SVC.__name__:
            return CreateSVC()

    def __init__(self):
        self.params = {}
        self.model = None
        self.predictions = []

    @abstractmethod
    def fit(self, data, args):
        pass

    def test(self, data):
        assert self.model is not None

        return self.model.predict(data.X_test)

    def predict(self, data):
        assert self.model is not None

        with Timer() as t:
            self.predictions = self.model.predict_proba(data.X_test)

        return t.interval

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None:
            del self.model


class CreateDecisionTreeClassifier(CreateModel):
    def fit(self, data, args):
        self.model = DecisionTreeClassifier()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateLogisticRegression(CreateModel):
    def fit(self, data, args):
        self.model = LogisticRegression(solver="liblinear")

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateLogisticRegressionCV(CreateModel):
    def fit(self, data, args):
        self.model = LogisticRegressionCV()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateSGDClassifier(CreateModel):
    def fit(self, data, args):
        self.model = SGDClassifier(loss="log")

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateLinearSVC(CreateModel):
    def fit(self, data, args):
        self.model = LinearSVC()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def predict(self, data):
        assert self.model is not None

        with Timer() as t:
            self.predictions = self.test(data)

        return t.interval


class CreateNuSVC(CreateLinearSVC):
    def fit(self, data, args):
        self.model = NuSVC(probability=True)

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateSVC(CreateLinearSVC):
    def fit(self, data, args):
        self.model = SVC(probability=True)

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateBernoulliNB(CreateModel):
    def fit(self, data, args):
        self.model = BernoulliNB()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateMLPClassifier(CreateModel):
    def fit(self, data, args):
        self.model = MLPClassifier()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval


class CreateBinarizer(CreateModel):
    def fit(self, data, args):
        self.model = Binarizer()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreateKBinsDiscretizer(CreateModel):
    def fit(self, data, args):
        self.model = KBinsDiscretizer()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreateMaxAbsScaler(CreateModel):
    def fit(self, data, args):
        self.model = MaxAbsScaler()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreateMinMaxScaler(CreateModel):
    def fit(self, data, args):
        self.model = MinMaxScaler()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreateNormalizer(CreateModel):
    def fit(self, data, args):
        self.model = Normalizer(norm="l2")

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreatePolynomialFeatures(CreateModel):
    def fit(self, data, args):
        self.model = PolynomialFeatures()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreateRobustScaler(CreateModel):
    def fit(self, data, args):
        self.model = RobustScaler()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval


class CreateStandardScaler(CreateModel):
    def fit(self, data, args):
        self.model = StandardScaler()

        with Timer() as t:
            self.model.fit(data.X_train, data.y_train)

        return t.interval

    def test(self, data):
        assert self.model is not None

        return self.model.transform(data.X_test)

    def predict(self, data):
        with Timer() as t:
            self.predictions = self.test(data)

        data.learning_task = LearningTask.REGRESSION
        return t.interval
