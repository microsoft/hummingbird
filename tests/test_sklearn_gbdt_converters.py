"""
Tests scikit-GradientBoostingClassifier converter.
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

from hummingbird import convert_sklearn
from hummingbird.common.data_types import Float32TensorType


class TestSklearnGradientBoostingClassifier(unittest.TestCase):

    def _run_GB_trees_classifier_converter(self, num_classes, extra_config={}, labels_shift=0):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth, )
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(num_classes, size=100) + labels_shift

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 20]))],
                extra_config=extra_config
            )
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict_proba(
                X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-06, atol=1e-06)

    def test_GBDT_classifier_binary_converter(self):
        self._run_GB_trees_classifier_converter(2)

    # multi classifier
    def test_GBDT_classifier_multi_converter(self):
        self._run_GB_trees_classifier_converter(3)

    # batch classifier
    def test_GBDT_batch_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "batch"})

    # beam classifier
    def test_GBDT_beam_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "beam"})

    # beam++ classifier
    def test_GBDT_beampp_classifier_converter(self):
        self._run_GB_trees_classifier_converter(3, extra_config={"tree_implementation": "beam++"})

    # shifted classes
    def test_GBDT_shifted_labels_converter(self):
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "batch"})
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "beam"})
        self._run_GB_trees_classifier_converter(3, labels_shift=2, extra_config={"tree_implementation": "beam++"})

    def test_zero_init_GB_trees_classifier_converter(self):
        warnings.filterwarnings("ignore")
        for max_depth in [1, 3, 8, 10, 12, None]:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth, init='zero')
            X = np.random.rand(100, 200)
            X = np.array(X, dtype=np.float32)
            y = np.random.randint(3, size=100)

            model.fit(X, y)
            pytorch_model = convert_sklearn(
                model,
                [("input", Float32TensorType([1, 20]))]
            )
            self.assertTrue(pytorch_model is not None)
            np.testing.assert_allclose(model.predict_proba(
                X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
