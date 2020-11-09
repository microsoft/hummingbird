"""
Tests Spark-ML SQLTransformer
"""
import unittest
import warnings
import json
import pprint
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
        iris = load_iris()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        df = sql.createDataFrame(pd_df)

        model = SQLTransformer(statement="SELECT *, sepal_length*(sepal_length/sepal_width) as new_feature1,"
                                         " petal_length*(petal_length/petal_width) as new_feature2,"
                                         " petal_length + 10.0*(sepal_width - petal_width) as new_feature3 from __THIS__")

        output_col_names = ['new_feature1', 'new_feature2', 'new_feature3']
        test_df = df
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()[output_col_names]
        spark_output_np = [spark_output[x].to_numpy().reshape(-1, 1) for x in output_col_names]
        torch_output_np = torch_model.transform(pd_df)
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
