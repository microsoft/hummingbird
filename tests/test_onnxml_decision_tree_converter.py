"""
Tests lightgbm->onnxmltools->hb conversion for lightgbm models.
"""
import unittest
import warnings

import sys
import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from onnxconverter_common.data_types import FloatTensorType

from hummingbird.ml import convert
from hummingbird.ml import constants
from hummingbird.ml._utils import onnx_ml_tools_installed, onnx_runtime_installed

if onnx_runtime_installed():
    import onnxruntime as ort
if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_sklearn


class TestONNXDecisionTreeConverter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestONNXDecisionTreeConverter, self).__init__(*args, **kwargs)

    # Base test implementation comparing ONNXML and ONNX models.
    def _test_decision_tree(self, X, model, extra_config={}):
        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=11
        )

        # Create ONNX model
        onnx_model = convert(onnx_ml_model, "onnx", X, extra_config)

        # Get the predictions for the ONNX-ML model
        session = ort.InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        onnx_ml_pred = [[] for i in range(len(output_names))]
        inputs = {session.get_inputs()[0].name: X}
        pred = session.run(output_names, inputs)
        for i in range(len(output_names)):
            if "label" in output_names[i]:
                onnx_ml_pred[1] = pred[i]
            else:
                onnx_ml_pred[0] = pred[i]

        # Get the predictions for the ONNX model
        onnx_pred = [[] for i in range(len(output_names))]
        if len(output_names) == 1:  # regression
            onnx_pred = onnx_model.predict(X)
        else:  # classification
            for i in range(len(output_names)):
                if "label" in output_names[i]:
                    onnx_pred[1] = onnx_model.predict(X)
                else:
                    onnx_pred[0] = onnx_model.predict_proba(X)

        return onnx_ml_pred, onnx_pred, output_names

    # Utility function for testing regression models.
    def _test_regressor(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_decision_tree(X, model, extra_config)

        # Check that predicted values match
        np.testing.assert_allclose(onnx_ml_pred[0].flatten(), onnx_pred, rtol=rtol, atol=atol)

    # Utility function for testing classification models.
    def _test_classifier(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_decision_tree(X, model, extra_config)

        np.testing.assert_allclose(onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol)  # labels
        np.testing.assert_allclose(
            list(map(lambda x: list(x.values()), onnx_ml_pred[0])), onnx_pred[0], rtol=rtol, atol=atol
        )  # probs

    # Regression.
    # Regression test with Decision Tree.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_decision_tree_regressor(self):
        warnings.filterwarnings("ignore")
        model = DecisionTreeRegressor()
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Basic regression test with decision tree.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_decision_tree_regressor_random(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(n_total, size=n_total)

        # Create DecisionTree model
        model = DecisionTreeRegressor()
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with Random Forest, 1 estimator.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_random_forest_regressor_1(self):
        warnings.filterwarnings("ignore")
        model = RandomForestRegressor(n_estimators=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = np.array([100, -10, 50], dtype=np.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Basic regression test with Random Forest.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_random_forest_regressor_random(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(n_total, size=n_total)

        # Create RandomForest model
        model = RandomForestRegressor()
        model.fit(X, y)
        self._test_regressor(X, model, rtol=1e-03, atol=1e-03)

    # Binary.
    # Binary classication test random.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_decision_tree_binary_random(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=n_total)

        # Create DecisionTree model
        model = DecisionTreeClassifier()
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test Decision Tree.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_decision_tree_binary(self):
        warnings.filterwarnings("ignore")
        model = DecisionTreeClassifier()
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test Random Forest with 3 estimators (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_random_forest_classifier(self):
        warnings.filterwarnings("ignore")
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        model = RandomForestClassifier(n_estimators=3)
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test Random Forest with 3 estimators random.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_random_forest_classifier_random(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=n_total)

        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test.
    # Multiclass classification test with DecisionTree, random.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_decision_tree_multi_random(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=n_total)

        # Create the DecisionTree model
        model = DecisionTreeClassifier()
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with DecisionTree (taken from ONNXMLTOOLS).
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_decision_tree_multi(self):
        warnings.filterwarnings("ignore")
        model = DecisionTreeClassifier()
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with Random Forest.
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_random_forest_multi_random(self):
        warnings.filterwarnings("ignore")
        n_features = 28
        n_total = 100
        np.random.seed(0)
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=n_total)

        # Create the RandomForest model
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        self._test_classifier(X, model)

    # # Used for small tree tests
    # # Commenting this test for the moment because it hits a bug in ORT / ONNXMLTOOLS (https://github.com/onnx/onnxmltools/issues/415)
    # @unittest.skipIf(
    #     not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    # )
    # def test_random_forest_classifier_single_node(self):
    #     warnings.filterwarnings("ignore")
    #     np.random.seed(0)
    #     X = np.random.rand(1, 1)
    #     X = np.array(X, dtype=np.float32)
    #     y = np.random.randint(1, size=1)
    #     model = RandomForestClassifier(n_estimators=5).fit(X, y)
    #     model.fit(X, y)
    #     self._test_classifier(X, model)


if __name__ == "__main__":
    unittest.main()
