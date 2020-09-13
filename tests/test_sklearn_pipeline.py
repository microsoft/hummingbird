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
from hummingbird.ml._utils import pandas_installed, onnx_runtime_installed
from hummingbird.ml import constants

from onnxconverter_common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

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
    def test_pipeline_column_transformer_string(self):
        """
        TODO: Hummingbird does not yet support strings in this context. Should raise error.
        When this feature is complete, change this test.
        """
        # fit
        titanic_url = "https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv"
        data = pandas.read_csv(titanic_url)
        X = data.drop("survived", axis=1)
        y = data["survived"]

        # SimpleImputer on string is not available for string
        # in ONNX-ML specifications.
        # So we do it beforehand.
        for cat in ["embarked", "sex", "pclass"]:
            X[cat].fillna("missing", inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        numeric_features = ["age", "fare"]
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),])

        categorical_features = ["embarked", "sex", "pclass"]
        categorical_transformer = Pipeline(
            steps=[
                # --- SimpleImputer on string is not available
                # for string in ONNX-ML specifications.
                # ('imputer',
                #  SimpleImputer(strategy='constant', fill_value='missing')),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                # ("classifier", LogisticRegression(solver="lbfgs")),
            ]
        )

        # inputs

        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == "int64":
                    t = Int64TensorType([None, 1])
                elif v == "float64":
                    t = FloatTensorType([None, 1])
                else:
                    t = StringTensorType([None, 1])
                inputs.append((k, t))
            return inputs

        to_drop = {
            "parch",
            "sibsp",
            "cabin",
            "ticket",
            "name",
            "body",
            "home.dest",
            "boat",
        }

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["pclass"] = X_train["pclass"].astype(np.int64)
        X_test["pclass"] = X_test["pclass"].astype(np.int64)
        X_train = X_train.drop(to_drop, axis=1)
        X_test = X_test.drop(to_drop, axis=1)

        clf.fit(X_train, y_train)
        # inputs = convert_dataframe_schema(X_train, to_drop)

        self.assertRaises(NotImplementedError, hummingbird.ml.convert, clf, "torch")

        # torch_model = hummingbird.ml.convert(clf, "torch", inputs)
        # self.assertTrue(torch_model is not None)

        # np.testing.assert_allclose(
        #    model.predict_proba(X_test), torch_model.predict_proba(X_test.values), rtol=1e-06, atol=1e-06,
        # )

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
    @unittest.skipIf(not onnx_runtime_installed(), reason="Test requires ORT installed")
    def test_pipeline_many_inputs(self):
        n_features = 18
        X = np.random.rand(100, n_features)
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

    @unittest.skipIf(ColumnTransformer is None, reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_runtime_installed(), reason="Test requires ORT installed")
    def test_pipeline_many_inputs_with_schema(self):
        n_features = 5
        X = np.random.rand(100, n_features)
        y = np.random.randint(1000, size=100)
        input_column_names = ["A", "B", "C", "D", "E"]
        output_column_names = ["score"]

        scaler_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[("scaling", scaler_transformer, list(range(n_features)))])
        model = RandomForestRegressor(n_estimators=10, max_depth=9)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X, y)

        X_test = tuple(np.split(X, n_features, axis=1))
        extra_config = {constants.INPUT_NAMES: input_column_names, constants.OUTPUT_NAMES: output_column_names}

        hb_model = hummingbird.ml.convert(pipeline, "onnx", X_test, extra_config=extra_config)

        graph_inputs = [input.name for input in hb_model.model.graph.input]
        graph_outputs = [output.name for output in hb_model.model.graph.output]

        assert len(hb_model.model.graph.input) == n_features
        assert graph_inputs == input_column_names
        assert graph_outputs == output_column_names


if __name__ == "__main__":
    unittest.main()
