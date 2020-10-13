"""
Tests Spark-ML SQLTransformer
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
    from pyspark.sql import SQLContext, SparkSession
    from pyspark.ml.feature import SQLTransformer

    spark = SparkSession.builder.getOrCreate()
    sql = SQLContext(spark.sparkContext)

if pandas_installed():
    import pandas as pd


class TestSparkMLSQLTransformer(unittest.TestCase):
    # Test SQLTransformer
    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    def test_sql_transformer_converter(self):
        # iris = load_iris()
        # features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        # pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        # df = sql.createDataFrame(pd_df)

        # WIP.
        parser = spark._jsparkSession.sessionState().sqlParser()
        print(parser.parsePlan("select * from table"))


if __name__ == "__main__":
    unittest.main()
