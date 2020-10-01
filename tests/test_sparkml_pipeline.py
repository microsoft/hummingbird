"""
Tests Spark-ML Pipeline converters
"""
import unittest
import warnings

import numpy as np
import torch

from hummingbird.ml._utils import sparkml_installed
from hummingbird.ml import convert

if sparkml_installed():
    import pandas as pd
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import LogisticRegression

    sc = SparkContext.getOrCreate()
    sql = SQLContext(sc)


class TestSparkMLPipeline(unittest.TestCase):
    @unittest.skipIf(not sparkml_installed(), reason="Spark-ML test requires pyspark")
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
            np.array(model.transform(df).select("probability").collect()).reshape(-1, classes),
            torch_model.predict_proba(X), rtol=1e-06, atol=1e-06
        )


if __name__ == "__main__":
    unittest.main()
