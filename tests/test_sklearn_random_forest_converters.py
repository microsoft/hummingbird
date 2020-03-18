"""
Tests scikit-RandomForestClassifier converter.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier

from hummingbird import convert_sklearn
from hummingbird.common.data_types import Float32TensorType
from sklearn.tree import DecisionTreeClassifier


class TestSklearnRandomForestConverter(unittest.TestCase):

    def _run_random_forest_classifier_converter(self, num_classes, extra_config={}):
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 20]))],
                extra_config=extra_config
            )
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict_proba(X), pytorch_model(
                torch.from_numpy(X))[1].data.numpy(), rtol=1e-06, atol=1e-06)

    # binary classifier

    def test_random_forest_classifier_binary_converter(self):
        self._run_random_forest_classifier_converter(2)

    # multi classifier
    def test_random_forest_classifier_multi_converter(self):
        self._run_random_forest_classifier_converter(3)

    # batch classifier
    def test_random_forest_batch_classifier_converter(self):
        self._run_random_forest_classifier_converter(3, extra_config={"tree_implementation": "batch"})

    # beam classifier
    def test_random_forest_beam_classifier_converter(self):
        self._run_random_forest_classifier_converter(3, extra_config={"tree_implementation": "beam"})

    # beam++ classifier
    def test_random_forest_beampp_classifier_converter(self):
        self._run_random_forest_classifier_converter(3, extra_config={"tree_implementation": "beam++"})

    def _run_random_forest_regressor_converter(self, num_classes, extra_config={}):
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = RandomForestRegressor(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 20]))],
                extra_config=extra_config
            )
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict(X), pytorch_model(
                torch.from_numpy(X)).numpy().flatten(), rtol=1e-06, atol=1e-06)

    # binary regressor
    def test_random_forest_regressor_binary_converter(self):
        self._run_random_forest_regressor_converter(2)

    # multi regressor
    def test_random_forest_regressor_multi_converter(self):
        self._run_random_forest_regressor_converter(3)

    # batch regressor
    def test_random_forest_batch_regressor_converter(self):
        self._run_random_forest_regressor_converter(3, extra_config={"tree_implementation": "batch"})

    # beam regressor
    def test_random_forest_beam_regressor_converter(self):
        self._run_random_forest_regressor_converter(3, extra_config={"tree_implementation": "beam"})

    # beam++ regressor
    def test_random_forest_beampp_regressor_converter(self):
        self._run_random_forest_regressor_converter(3, extra_config={"tree_implementation": "beam++"})

    # TODO consolidate
    def test_decision_tree_classifier_converter(self):
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = DecisionTreeClassifier(max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(3, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 20]))]
            )
            self.assertTrue(pytorch_model is not None)
            self.assertTrue(np.allclose(model.predict_proba(
                X), pytorch_model(torch.from_numpy(X))[1].data.numpy()))

    # TODO consolidate
    def test_extra_trees_classifier_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = ExtraTreesClassifier(n_estimators=10, max_depth=max_depth)
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(3, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 20]))]
            )
            self.assertTrue(pytorch_model is not None)
            self.assertTrue(np.allclose(model.predict_proba(
                X), pytorch_model(torch.from_numpy(X))[1].data.numpy()))


if __name__ == "__main__":
    unittest.main()
