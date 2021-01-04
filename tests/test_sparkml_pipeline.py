"""
Tests Spark-ML Pipeline converters
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.datasets import load_iris

from hummingbird.ml._utils import sparkml_installed, pandas_installed
from hummingbird.ml import convert
from distutils.version import LooseVersion

if sparkml_installed():
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import QuantileDiscretizer, VectorAssembler

    sc = SparkContext.getOrCreate()
    sql = SQLContext(sc)

if pandas_installed():
    import pandas as pd


class TestSparkMLPipeline(unittest.TestCase):
    @unittest.skipIf(not sparkml_installed(), reason="Spark-ML test requires pyspark")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    def test_pipeline_1(self):
        n_features = 10
        n_total = 100
        classes = 2
        np.random.seed(0)
        warnings.filterwarnings("ignore")
        X = np.random.rand(n_total, n_features)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(classes, size=(n_total, 1))

        arr = np.concatenate([y, X], axis=1).reshape(n_total, -1)
        df = map(lambda x: (int(x[0]), Vectors.dense(x[1:])), arr)
        df = sql.createDataFrame(df, schema=["label", "features"])

        pipeline = Pipeline(stages=[LogisticRegression()])
        model = pipeline.fit(df)

        test_df = df.select("features").limit(1)
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            np.array(model.transform(df).select("prediction").collect()).reshape(-1),
            torch_model.predict(X),
            rtol=1e-06,
            atol=1e-06,
        )

        np.testing.assert_allclose(
            np.array(model.transform(df).select("probability").collect()).reshape(-1, classes),
            torch_model.predict_proba(X),
            rtol=1e-06,
            atol=1e-06,
        )

    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    def test_pipeline2(self):
        iris = load_iris()
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        pd_df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=features + ["label"])
        df = sql.createDataFrame(pd_df)

        quantile = QuantileDiscretizer(inputCol="sepal_length", outputCol="sepal_length_bucket", numBuckets=2)
        features = ["sepal_length_bucket"] + features
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        pipeline = Pipeline(stages=[quantile, assembler, LogisticRegression()])
        model = pipeline.fit(df)

        df = df.select(["sepal_length", "sepal_width", "petal_length", "petal_width"])
        pd_df = pd_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        torch_model = convert(model, "torch", df)
        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            np.array(model.transform(df).select("prediction").collect()).reshape(-1),
            torch_model.predict(pd_df),
            rtol=1e-06,
            atol=1e-06,
        )

        np.testing.assert_allclose(
            np.array(model.transform(df).select("probability").collect()).reshape(-1, 3),
            torch_model.predict_proba(pd_df),
            rtol=1e-06,
            atol=1e-05,
        )

    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    def test_pipeline3(self):
        iris = load_iris()
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        pd_df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=features + ["label"])
        df = sql.createDataFrame(pd_df)

        quantile1 = QuantileDiscretizer(inputCol="sepal_length", outputCol="sepal_length_bucket", numBuckets=2)
        quantile2 = QuantileDiscretizer(inputCol="sepal_width", outputCol="sepal_width_bucket", numBuckets=2)
        features = ["sepal_length_bucket", "sepal_width_bucket"] + features
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        pipeline = Pipeline(stages=[quantile1, quantile2, assembler, LogisticRegression()])
        model = pipeline.fit(df)

        df = df.select(["sepal_length", "sepal_width", "petal_length", "petal_width"])
        pd_df = pd_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        torch_model = convert(model, "torch", df)
        self.assertTrue(torch_model is not None)

        np.testing.assert_allclose(
            np.array(model.transform(df).select("prediction").collect()).reshape(-1),
            torch_model.predict(pd_df),
            rtol=1e-06,
            atol=1e-06,
        )

        np.testing.assert_allclose(
            np.array(model.transform(df).select("probability").collect()).reshape(-1, 3),
            torch_model.predict_proba(pd_df),
            rtol=1e-06,
            atol=1e-05,
        )


if __name__ == "__main__":
    unittest.main()
