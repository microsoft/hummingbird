"""
Tests sklearn model selection ops (GridSearchCV, RandomizedSearchCV) converters.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import datasets

import hummingbird.ml


class TestSklearnModelSelection(unittest.TestCase):
    def _test_model_selection(
        self,
        model_selection_class,
        base_classifier,
        parameters,
        score_w_train_data=False,
        **kwargs
    ):
        for data in [datasets.load_breast_cancer(), datasets.load_iris()]:
            X, y = data.data, data.target
            X = X.astype(np.float32)

            model = model_selection_class(base_classifier, parameters, cv=5, **kwargs)

            n_train_rows = int(X.shape[0] * 0.6)
            model.fit(X[:n_train_rows, :], y[:n_train_rows])

            if not score_w_train_data:
                X = X[n_train_rows:, :]

            torch_model = hummingbird.ml.convert(model, "torch")
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-5)

    # GridSearchCV/DecisionTreeClassifier
    def test_gridsearchcv_decision_tree_1(self):
        self._test_model_selection(GridSearchCV, DecisionTreeClassifier(random_state=42), {'min_samples_split': range(2, 403, 10)})

    # GridSearchCV/DecisionTreeClassifier w/ extra configs
    def test_gridsearchcv_decision_tree_2(self):
        self._test_model_selection(GridSearchCV, DecisionTreeClassifier(random_state=42), {'min_samples_split': range(2, 403, 10)}, scoring={'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}, refit='AUC', return_train_score=True)

    # RandomizedSearchCV/DecisionTreeClassifier
    def test_randpmizedsearchcv_decision_tree_1(self):
        self._test_model_selection(RandomizedSearchCV, DecisionTreeClassifier(random_state=42), {'min_samples_split': range(2, 403, 10)})

    # RandomizedSearchCV/DecisionTreeClassifier w/ extra configs
    def test_randpmizedsearchcv_decision_tree_2(self):
        self._test_model_selection(RandomizedSearchCV, DecisionTreeClassifier(random_state=42), {'min_samples_split': range(2, 403, 10)}, scoring={'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}, refit='AUC', return_train_score=True)


if __name__ == "__main__":
    unittest.main()
