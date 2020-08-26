import unittest
from distutils.version import StrictVersion
from io import StringIO
import numpy as np
from numpy.testing import assert_almost_equal
import pandas
from sklearn import __version__ as sklearn_version
from sklearn import datasets

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # not available in 0.19
    ColumnTransformer = None
from sklearn.decomposition import PCA

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

import hummingbird.ml


def check_scikit_version():
    # StrictVersion does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return StrictVersion(vers) >= StrictVersion("0.21.0")


class PipeConcatenateInput:
    def __init__(self, pipe):
        self.pipe = pipe

    def transform(self, inp):
        if isinstance(inp, (np.ndarray, pandas.DataFrame)):
            return self.pipe.transform(inp)
        elif isinstance(inp, dict):
            keys = list(sorted(inp.keys()))
            dim = inp[keys[0]].shape[0], len(keys)
            x2 = np.zeros(dim)
            for i in range(x2.shape[1]):
                x2[:, i] = inp[keys[i]].ravel()
            res = self.pipe.transform(x2)
            return res
        else:
            raise TypeError("Unable to predict with type {0}".format(type(inp)))


class TestSklearnPipeline(unittest.TestCase):
    def test_pipeline(self):
        data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data), rtol=1e-06, atol=1e-06,
        )

    def test_pipeline2(self):
        data = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data), rtol=1e-06, atol=1e-06,
        )

    def test_combine_inputs_union_in_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        model = Pipeline(
            [
                ("scaler1", StandardScaler()),
                ("union", FeatureUnion([("scaler2", StandardScaler()), ("scaler3", MinMaxScaler())])),
            ]
        )
        model.fit(data)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data), rtol=1e-06, atol=1e-06,
        )

    def test_combine_inputs_floats_ints(self):
        data = [[0, 0.0], [0, 0.0], [1, 1.0], [1, 1.0]]
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.transform(data), torch_model.transform(data), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(ColumnTransformer is None, reason="ColumnTransformer not available in 0.19")
    def test_pipeline_column_transformer(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = [0, 1, 2]  # ["vA", "vB", "vC"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="lbfgs", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict(X_test), torch_model.predict(X_test.values), rtol=1e-06, atol=1e-06,
        )


if __name__ == "__main__":
    unittest.main()
