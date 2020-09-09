import unittest
import numpy as np
from sklearn import datasets

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

import hummingbird.ml
from hummingbird.ml._utils import pandas_installed

if pandas_installed():
    import pandas


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

    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_1(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = [0, 1, 2]  # ["vA", "vB", "vC"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
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
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
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
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_weights(self):
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
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            transformer_weights={"num": 2, "cat": 3},
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_drop(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = [0, 1]  # ["vA", "vB"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            transformer_weights={"num": 2, "cat": 3},
            remainder="drop",
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_drop_noweights(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = [0, 1]  # ["vA", "vB"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(ColumnTransformer is None, reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_passthrough(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = [0, 1]  # ["vA", "vB"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            transformer_weights={"num": 2, "cat": 3},
            remainder="passthrough",
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(ColumnTransformer is None, reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_passthrough_noweights(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = [0, 1]  # ["vA", "vB"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(ColumnTransformer is None, reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_column_transformer_passthrough_slice(self):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2
        numeric_features = slice(0, 1)  # ["vA", "vB"]
        categorical_features = slice(3, 4)  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])), n_jobs=1, max_iter=10, solver="liblinear", tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(sparse=True, handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            transformer_weights={"num": 2, "cat": 3},
            remainder="passthrough",
        )

        model = Pipeline(steps=[("precprocessor", preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)

        X_test = X_train[:11]

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        )

    @unittest.skipIf(ColumnTransformer is None, reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_pipeline_many_inputs(self):
        n_features = 18
        X = np.random.rand(100, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(1000, size=100)

        scaler_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[("scaling", scaler_transformer, list(range(n_features)))])
        model = RandomForestRegressor(n_estimators=10, max_depth=9)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X, y)

        X_test = tuple(np.split(X, n_features, axis=1))

        hb_model = hummingbird.ml.convert(pipeline, "onnx", X_test)

        assert len(hb_model.model.graph.input) == n_features

        np.testing.assert_allclose(
            pipeline.predict(X), np.array(hb_model.predict(X_test)).flatten(), rtol=1e-06, atol=1e-06,
        )


if __name__ == "__main__":
    unittest.main()
