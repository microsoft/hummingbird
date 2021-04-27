# BSD License
#
# Copyright (c) 2016-present, Miguel Gonzalez-Fierro. All rights reserved.
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


import numpy as np
import sklearn.metrics as sklm

from benchmarks.datasets import LearningTask


def get_metrics(y_test, pred, learning_task):
    if learning_task == LearningTask.REGRESSION:
        return regression_metrics(y_test, pred)
    if learning_task == LearningTask.CLASSIFICATION:
        return classification_metrics(y_test, pred)
    if learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
        return classification_metrics_multilabel(y_test, pred)
    raise ValueError("No metrics defined for learning task: " + str(learning_task))


def evaluate_metrics(y_true, y_pred, metrics):
    res = {}
    for metric_name, metric in metrics.items():
        res[metric_name] = metric(y_true, y_pred)
    return res


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = np.where(y_prob > threshold, 1, 0)
    metrics = {
        "Accuracy": sklm.accuracy_score,
        "Log_Loss": lambda real, pred: sklm.log_loss(real, y_prob, eps=1e-5),
        # yes, I'm using y_prob here!
        "AUC": lambda real, pred: sklm.roc_auc_score(real, y_prob),
        "Precision": sklm.precision_score,
        "Recall": sklm.recall_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_multilabel(y_true, y_pred):
    metrics = {
        "Accuracy": sklm.accuracy_score,
        "Precision": lambda real, pred: sklm.precision_score(real, pred, average="weighted"),
        "Recall": lambda real, pred: sklm.recall_score(real, pred, average="weighted"),
        "F1": lambda real, pred: sklm.f1_score(real, pred, average="weighted"),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def regression_metrics(y_true, y_pred):
    metrics = {
        "MeanAbsError": sklm.mean_absolute_error,
        "MeanSquaredError": sklm.mean_squared_error,
        "MedianAbsError": sklm.median_absolute_error,
    }
    return evaluate_metrics(y_true, y_pred, metrics)
