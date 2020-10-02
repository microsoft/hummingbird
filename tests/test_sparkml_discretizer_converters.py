"""
Tests Spark-ML discretizer converters: QuantileDiscretizer
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.datasets import load_iris

from hummingbird.ml._utils import sparkml_installed, pandas_installed
from hummingbird.ml import convert

if sparkml_installed():
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark.ml.feature import QuantileDiscretizer

    sc = SparkContext.getOrCreate()
    sql = SQLContext(sc)

if pandas_installed():
    import pandas as pd


class TestSparkMLDiscretizers(unittest.TestCase):
    # Test QuantileDiscretizer
    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark")
    def test_quantilediscretizer_converter(self):
        iris = load_iris()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])
        df = sql.createDataFrame(pd_df)

        quantile = QuantileDiscretizer(inputCol='sepal_length', outputCol='sepal_length_bucket', numBuckets=2)
        model = quantile.fit(df)

        test_df = df

        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()
        spark_output_np = tuple([spark_output[col_name].to_numpy().reshape(-1, 1) for col_name in spark_output.columns])
        torch_output_np = torch_model.transform(pd_df)
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
