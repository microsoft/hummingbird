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
from hummingbird.ml import constants

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
    def test_sql_transformer_converter1(self):
        iris = load_iris()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        df = sql.createDataFrame(pd_df)

        model = SQLTransformer(statement="SELECT *, sepal_length*(sepal_length/sepal_width) as new_feature1,"
                                         " petal_length*(petal_length/petal_width) as new_feature2,"
                                         " petal_length + 10.0*(sepal_width - petal_width) as new_feature3,"
                                         " sqrt(petal_length) as new_feature4 from __THIS__"
                                         " WHERE (sepal_length < 4.5 AND petal_length < 4.5) OR sepal_width >= 4.5 OR sepal_width = 4.5")

        output_col_names = ['new_feature1', 'new_feature2', 'new_feature3', 'new_feature4']

        test_df = df
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()[output_col_names]
        spark_output_np = [spark_output[x].to_numpy().reshape(-1, 1) for x in output_col_names]
        torch_output_np = torch_model.transform(pd_df)
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)

    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.7.0"), reason="Spark-ML test requires torch >= 1.7.0")
    def test_sql_transformer_converter2(self):
        iris = load_iris()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        df = sql.createDataFrame(pd_df)

        model = SQLTransformer(statement="SELECT *,"
                                         " CASE"
                                         "   WHEN sepal_length < 1 THEN 1"
                                         "   WHEN sepal_length < 2 THEN 2"
                                         "   ELSE sepal_width + 2"
                                         " END AS new_feature1,"
                                         " CASE"
                                         "   WHEN petal_width < 4 THEN 1"
                                         "   WHEN petal_width < 4.5 THEN 2"
                                         " END AS new_feature2,"
                                         " CASE"
                                         "   WHEN SQRT(petal_width) < 4 THEN 1/SQRT(sepal_width)"
                                         "   WHEN petal_width - 0.2 < 4.5 THEN 2"
                                         " END AS new_feature3"
                                         " FROM __THIS__")
        output_col_names = ['new_feature1', 'new_feature2', 'new_feature3']

        test_df = df
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()[output_col_names]
        spark_output_np = [spark_output[x].to_numpy().reshape(-1, 1) for x in output_col_names]
        torch_output_np = torch_model.transform(pd_df)
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)

    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    def test_sql_transformer_converter3(self):
        iris = load_iris()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        pd_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=features + ['target'])[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        df = sql.createDataFrame(pd_df)

        model = SQLTransformer(statement="SELECT *,"
                                         " CASE"
                                         "   WHEN 1.0 + (sepal_length/2) IN (4.0, 4.1, 4.2, 4.3) THEN 1"
                                         "   WHEN SQRT(sepal_length) < 2 THEN 2"
                                         "   ELSE sepal_width + 2"
                                         " END AS new_feature1"
                                         " FROM __THIS__")
        output_col_names = ['new_feature1']

        test_df = df
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()[output_col_names]
        spark_output_np = [spark_output[x].to_numpy().reshape(-1, 1) for x in output_col_names][0]
        torch_output_np = torch_model.transform(pd_df)
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)

    @unittest.skipIf((not sparkml_installed()) or (not pandas_installed()), reason="Spark-ML test requires pyspark and pandas")
    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Spark-ML test requires torch >= 1.6.0")
    def test_sql_transformer_converter4(self):
        df = spark.createDataFrame(
            [
                (1, 101, 5),  # create your data here, be consistent in the types.
                (2, 100, 7),
                (3, None, 1),
                (4, 100, 9),
                (5, 102, 8),
            ],
            ['id', 'val1', 'val2']  # add your columns label here
        )

        model = SQLTransformer(statement="SELECT *, val2 AS val FROM __THIS__ WHERE id > 0 ORDER BY val1/2 NULLS LAST, id DESC")
        output_col_names = ['val']

        test_df = df
        torch_model = convert(model, "torch", test_df)
        self.assertTrue(torch_model is not None)

        spark_output = model.transform(test_df).toPandas()[output_col_names]
        spark_output_np = [spark_output[x].to_numpy().reshape(-1, 1) for x in output_col_names][0]
        torch_output_np = torch_model.transform(df.toPandas())
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)

    def test_sql_transformer_tpc_h_6(self):
        from datetime import datetime
        to_date = lambda x: datetime.strptime(x, '%m/%d/%Y')
        # lineitem_df = spark.createDataFrame(
        #     [
        #         (1,1, 155190, 7706, 1, 17, 21168.23, 0.04, 0.02, "N", "O", to_date("3/13/1996"), to_date("2/12/1996"), to_date("3/22/1996"),"DELIVER IN PERSON", "TRUCK","egular courts above the|"),
        #         (2,2, 106170, 1191, 1, 38, 44694.46,  0.0, 0.05, "N", "O", to_date("1/28/1997"), to_date("1/14/1997"), to_date("2/2/1997"),"TAKE BACK RETURN", "RAIL", "ven requests. deposits breach a|"),
        #         (3,3,   4297, 1798, 1, 45, 54058.05, 0.06,  0.0, "R", "F", to_date("2/2/1994"), to_date("1/4/1994"), to_date("2/23/1994"),"NONE", "AIR","ongside of the furiously brave acco|"),
        #         (4,4,  88035, 5560, 1, 30,  30690.9, 0.03, 0.08, "N", "O", to_date("1/10/1996"), to_date("12/14/1995"), to_date("1/18/1996"), "DELIVER IN PERSON", "REG AIR", "- quickly regular packages sleep. idly|"),
        #         (5,5, 108570, 8571, 1, 15, 23678.55, 0.02, 0.04, "R", "F", to_date("10/31/1994"), to_date("8/31/1994"), to_date("11/20/1994"), "NONE", "AIR", "ts wake furiously |")
        #     ],
        #     [ "rownumber", "L_ORDERKEY", "L_PARTKEY", "L_SUPPKEY", "L_LINENUMBER", "L_QUANTITY", 
        #     "L_EXTENDEDPRICE", "L_DISCOUNT", "L_TAX", "L_RETURNFLAG", "L_LINESTATUS", "L_SHIPDATE", "L_COMMITDATE", "L_RECEIPTDATE", 
        #     "L_SHIPINSTRUCT", "L_SHIPMODE", "L_COMMENT"]  # add your columns label here
        # )
        from pyspark.sql.types import StructType, StructField
        from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType

        schema = StructType() \
            .add("rownumber",IntegerType(),True) \
            .add("L_ORDERKEY",IntegerType(),True) \
            .add("L_PARTKEY",IntegerType(),True) \
            .add("L_SUPPKEY",IntegerType(),True) \
            .add("L_LINENUMBER",IntegerType(),True) \
            .add("L_QUANTITY",DoubleType(),True) \
            .add("L_EXTENDEDPRICE",DoubleType(),True) \
            .add("L_DISCOUNT",DoubleType(),True) \
            .add("L_TAX",DoubleType(),True) \
            .add("L_RETURNFLAG",StringType(),True) \
            .add("L_LINESTATUS",StringType(),True) \
            .add("L_SHIPDATE",TimestampType(),True) \
            .add("L_COMMITDATE",TimestampType(),True) \
            .add("L_RECEIPTDATE",TimestampType(),True) \
            .add("L_SHIPINSTRUCT",StringType(),True) \
            .add("L_SHIPMODE",StringType(),True) \
            .add("L_COMMENT",StringType(),True)
        
        print("my schema", schema)

        lineitem_df = spark.read.options(header=True, inferSchema=True)\
                            .option("dateFormat",  "MM/dd/yyyy").csv("tests/resources/LINEITEM_1ST1M.csv")
        # TPC-H Query 6
        query = """select
                sum(l_extendedprice * l_discount) as revenue
                from
                __THIS__
                where
                l_shipdate >= date '1994-01-01'
                and l_shipdate < date '1994-01-01' + interval '1' year
                and l_discount between .06 - 0.01 and .06 + 0.01
                and l_quantity < 24"""       

        model = SQLTransformer(statement=query)
        output_col_names = ['revenue']
        test_df = lineitem_df.limit(1000)

        # run spark
        spark_output = model.transform(test_df).toPandas()[output_col_names]
        spark_output_np = [spark_output[x].to_numpy().reshape(-1, 1) for x in output_col_names][0]
        x = test_df.toPandas()
        assert len(x) > 3 # check that not everything is filtered out

        # run hummingbird conversion
        torch_model = convert(model, "torch", test_df, extra_config={constants.MAX_STRING_LENGTH: 100})
        self.assertTrue(torch_model is not None)

        torch_output_np = torch_model.transform(x)
        np.testing.assert_allclose(spark_output_np, torch_output_np, rtol=1e-06, atol=1e-06)

if __name__ == "__main__":
    unittest.main()
