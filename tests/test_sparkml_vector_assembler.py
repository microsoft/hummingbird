"""
Tests Spark-ML VectorAssembler
"""
import unittest
import warnings

import numpy as np
import torch
from sklearn.datasets import load_iris
from distutils.version import LooseVersion

from hummingbird.ml._utils import sparkml_installed, pandas_installed
from hummingbird.ml import convert

if sparkml_installed():
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark.ml.feature import VectorAssembler

    sc = SparkContext.getOrCreate()
    sql = SQLContext(sc)

if pandas_installed():
    import pandas as pd


class TestSparkMLVectorAssembler(unittest.TestCase):
    # Test VectorAssembler
    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    def test_vectorassembler_converter(self):
        iris = load_iris()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        df = sql.createDataFrame(pd_df)

        model = VectorAssembler(inputCols=features, outputCol='features')

        test_df = df
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()
        spark_output['features'] = spark_output['features'].map(lambda x: np.array(x.toArray()))
        spark_output_np = spark_output['features'].to_numpy()
        torch_output_np = torch_model.transform(pd_df)

        np.testing.assert_allclose(np.vstack(spark_output_np), torch_output_np, rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
