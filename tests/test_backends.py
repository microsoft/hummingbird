"""
Tests Hummingbird's backends.
"""
import unittest
import warnings
import os
import sys
import numpy as np
from typing import Iterator
from distutils.version import LooseVersion
import shutil

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from onnxconverter_common.data_types import (
    FloatTensorType,
    DoubleTensorType,
    Int64TensorType,
    Int32TensorType,
    StringTensorType,
)

import hummingbird.ml
from hummingbird.ml._utils import (
    onnx_ml_tools_installed,
    onnx_runtime_installed,
    tvm_installed,
    sparkml_installed,
    pandas_installed,
    prophet_installed,
)
from hummingbird.ml.exceptions import MissingBackend

if onnx_ml_tools_installed():
    from onnxmltools.convert import convert_sklearn

    try:
        from skl2onnx.sklapi import CastTransformer
    except ImportError:
        CastTransformer = None

if sparkml_installed():
    import pyspark
    from pyspark import SparkFiles
    from pyspark.sql import SparkSession, SQLContext
    from pyspark.sql.functions import pandas_udf, col, expr

    spark = SparkSession.builder.master("local[*]").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    sql_context = SQLContext(spark)

if pandas_installed():
    import pandas as pd


class TestBackends(unittest.TestCase):
    # Test backends are browsable
    def test_backends(self):
        warnings.filterwarnings("ignore")
        self.assertTrue(len(hummingbird.ml.backends) > 0)

    # Test backends are not case sensitive
    def test_backends_case_sensitive(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tOrCh")
        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test pytorch is still a valid backend name
    def test_backends_pytorch(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "pytOrCh")
        self.assertIsNotNone(hb_model)
        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

    # Test pytorch save and load
    def test_pytorch_save_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        hb_model.save("pt-tmp")

        hb_model_loaded = hummingbird.ml.TorchContainer.load("pt-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("pt-tmp.zip")

    # Test pytorch save and load
    def test_pytorch_save_load_delete_folder(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        model_location_name = "pt-tmp-delete-folder"
        hb_model.save(model_location_name)

        # Default behavior
        hummingbird.ml.TorchContainer.load(model_location_name, delete_unzip_location_folder=True)
        assert not os.path.exists(model_location_name)

        hummingbird.ml.TorchContainer.load(model_location_name, delete_unzip_location_folder=False)
        assert os.path.exists(model_location_name)

        os.remove(f"{model_location_name}.zip")
        shutil.rmtree(model_location_name)

    # Test pytorch save and generic load
    def test_pytorch_save_generic_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        hb_model.save("pt-tmp")

        hb_model_loaded = hummingbird.ml.load("pt-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("pt-tmp.zip")

    def test_pytorch_save_load_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        hb_model.save("pt-tmp")

        hummingbird.ml.load("pt-tmp")
        hummingbird.ml.load("pt-tmp")

        os.remove("pt-tmp.zip")

    def test_pytorch_save_load_more_versions(self):
        from hummingbird.ml.operator_converters import constants

        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        hb_model.save("pt-tmp")

        shutil.unpack_archive("pt-tmp.zip", "pt-tmp", format="zip")

        # Adding a new library does not create problems.
        with open(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH), "r") as file:
            configuration = file.readlines()
        configuration.append("\nlibx=1.3")
        os.remove(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH))
        with open(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH), "w") as file:
            file.writelines(configuration)
        shutil.make_archive("pt-tmp", "zip", "pt-tmp")

        hummingbird.ml.load("pt-tmp")
        os.remove("pt-tmp.zip")

    def test_pytorch_save_load_less_versions(self):
        from hummingbird.ml.operator_converters import constants

        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        hb_model.save("pt-tmp")

        shutil.unpack_archive("pt-tmp.zip", "pt-tmp", format="zip")

        # Removing a library does not create problems.
        with open(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH), "r") as file:
            configuration = file.readlines()
        configuration = configuration[-1]
        os.remove(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH))
        with open(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH), "w") as file:
            file.writelines(configuration)
        shutil.make_archive("pt-tmp", "zip", "pt-tmp")

        hummingbird.ml.load("pt-tmp")
        os.remove("pt-tmp.zip")

    def test_pytorch_save_load_different_versions(self):
        from hummingbird.ml.operator_converters import constants

        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertIsNotNone(hb_model)
        hb_model.save("pt-tmp")

        shutil.unpack_archive("pt-tmp.zip", "pt-tmp", format="zip")

        # Changing the version of a library does not create problems.
        with open(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH), "r") as file:
            configuration = file.readlines()
        configuration[0] = "hummingbird=0.0.0.1\n"
        os.remove(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH))
        with open(os.path.join("pt-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH), "w") as file:
            file.writelines(configuration)
        shutil.make_archive("pt-tmp", "zip", "pt-tmp")

        hummingbird.ml.load("pt-tmp")
        os.remove("pt-tmp.zip")

    # Test torchscript save and load
    def test_torchscript_save_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("ts-tmp")

        hb_model_loaded = hummingbird.ml.TorchContainer.load("ts-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("ts-tmp.zip")

    # Test torchscript save and generic load
    def test_torchscript_save_generic_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("ts-tmp")

        hb_model_loaded = hummingbird.ml.load("ts-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("ts-tmp.zip")

    def test_load_fails_bad_path(self):
        # Asserts for bad path with extension
        self.assertRaises(AssertionError, hummingbird.ml.load, "nonsense.zip")
        self.assertRaises(AssertionError, hummingbird.ml.TorchContainer.load, "nonsense.zip")

        # Asserts for bad path with no extension
        self.assertRaises(AssertionError, hummingbird.ml.load, "nonsense")
        self.assertRaises(AssertionError, hummingbird.ml.TorchContainer.load, "nonsense")

    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_load_fails_bad_path_onnx(self):
        self.assertRaises(AssertionError, hummingbird.ml.ONNXContainer.load, "nonsense.zip")
        self.assertRaises(AssertionError, hummingbird.ml.ONNXContainer.load, "nonsense")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_load_fails_bad_path_tvm(self):
        self.assertRaises(AssertionError, hummingbird.ml.TVMContainer.load, "nonsense.zip")
        self.assertRaises(AssertionError, hummingbird.ml.TVMContainer.load, "nonsense")

    # Test not supported backends
    def test_unsupported_backend(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test scala backend rises an exception
        self.assertRaises(MissingBackend, hummingbird.ml.convert, model, "scala")

    # Test torchscript requires test_data
    def test_torchscript_test_data(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test torcscript requires test_input
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch.jit")

    # Test TVM requires test_data
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_test_data(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Test tvm requires test_input
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "tvm")

    # Test tvm save and load
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_save_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tvm", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("tvm-tmp")

        hb_model_loaded = hummingbird.ml.TVMContainer.load("tvm-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("tvm-tmp.zip")

    # Test tvm save and generic load
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_save_generic_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tvm", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("tvm-tmp")

        hb_model_loaded = hummingbird.ml.load("tvm-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("tvm-tmp.zip")

    # Test tvm save and load zip file
    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_save_load_zip(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tvm", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("tvm-tmp.zip")

        hb_model_loaded = hummingbird.ml.TVMContainer.load("tvm-tmp.zip")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("tvm-tmp.zip")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_save_load_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tvm", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("tvm-tmp.zip")

        hummingbird.ml.TVMContainer.load("tvm-tmp.zip")
        hummingbird.ml.TVMContainer.load("tvm-tmp.zip")

        os.remove("tvm-tmp.zip")

    @unittest.skipIf(not tvm_installed(), reason="TVM test requires TVM installed")
    def test_tvm_save_load_no_versions(self):
        from hummingbird.ml.operator_converters import constants

        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "tvm", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("tvm-tmp")

        shutil.unpack_archive("tvm-tmp.zip", "tvm-tmp", format="zip")

        # Removing the configuration file with the versions does not create problems.
        os.remove(os.path.join("tvm-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH))

        hummingbird.ml.load("tvm-tmp")
        os.remove("tvm-tmp.zip")

    # Test onnx requires test_data or initial_types
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_no_test_data_float(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", FloatTensorType([X.shape[0], X.shape[1]]))], target_opset=11
        )

        # Test onnx requires no test_data
        hb_model = hummingbird.ml.convert(onnx_ml_model, "onnx")
        assert hb_model

    # Test onnx no test_data, double input
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_no_test_data_double(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        if CastTransformer is None:
            model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        else:
            # newer version of sklearn-onnx
            model = make_pipeline(
                CastTransformer(dtype=np.float32), GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
            )
        np.random.seed(0)
        X = np.random.rand(100, 200)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", DoubleTensorType([None, X.shape[1]]))], target_opset=11
        )

        # Test onnx requires no test_data
        hb_model = hummingbird.ml.convert(onnx_ml_model, "onnx")
        assert hb_model

    # Test onnx no test_data, long input
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_no_test_data_long(self):
        warnings.filterwarnings("ignore")
        model = model = StandardScaler(with_mean=True, with_std=True)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.int64)

        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", Int64TensorType([X.shape[0], X.shape[1]]))], target_opset=11
        )

        # Test onnx requires no test_data
        hb_model = hummingbird.ml.convert(onnx_ml_model, "onnx")
        assert hb_model

    # Test onnx no test_data, int input
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_no_test_data_int(self):
        warnings.filterwarnings("ignore")
        model = OneHotEncoder()
        X = np.array([[1, 2, 3]], dtype=np.int32)
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", Int32TensorType([X.shape[0], X.shape[1]]))], target_opset=11
        )

        # Test onnx requires no test_data
        hb_model = hummingbird.ml.convert(onnx_ml_model, "onnx")
        assert hb_model

    # Test onnx no test_data, string input
    @unittest.skipIf(
        not (onnx_ml_tools_installed() and onnx_runtime_installed()), reason="ONNXML test require ONNX, ORT and ONNXMLTOOLS"
    )
    def test_onnx_no_test_data_string(self):
        warnings.filterwarnings("ignore")
        model = OneHotEncoder()
        X = np.array([["a", "b", "c"]])
        model.fit(X)

        # Create ONNX-ML model
        onnx_ml_model = convert_sklearn(
            model, initial_types=[("input", StringTensorType([X.shape[0], X.shape[1]]))], target_opset=11
        )

        # Test backends are not case sensitive
        self.assertRaises(RuntimeError, hummingbird.ml.convert, onnx_ml_model, "onnx")

    # Test ONNX save and load
    @unittest.skipIf(not onnx_runtime_installed(), reason="ONNX test requires ORT")
    def test_onnx_save_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("onnx-tmp")

        hb_model_loaded = hummingbird.ml.ONNXContainer.load("onnx-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("onnx-tmp.zip")

    # Test ONNX save and generic load
    @unittest.skipIf(not onnx_runtime_installed(), reason="ONNX test requires ORT")
    def test_onnx_save_generic_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("onnx-tmp")

        hb_model_loaded = hummingbird.ml.load("onnx-tmp")
        np.testing.assert_allclose(hb_model_loaded.predict_proba(X), hb_model.predict_proba(X), rtol=1e-06, atol=1e-06)

        os.remove("onnx-tmp.zip")

    # Test ONNX save and generic load
    @unittest.skipIf(not onnx_runtime_installed(), reason="ONNX test requires ORT")
    def test_onnx_save_load_load(self):
        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("onnx-tmp")

        hummingbird.ml.load("onnx-tmp")
        hummingbird.ml.load("onnx-tmp")

        os.remove("onnx-tmp.zip")

    @unittest.skipIf(not onnx_runtime_installed(), reason="ONNX test requires ORT")
    def test_onnx_save_load_no_versions(self):
        from hummingbird.ml.operator_converters import constants

        warnings.filterwarnings("ignore")
        max_depth = 10
        num_classes = 2
        model = GradientBoostingClassifier(n_estimators=10, max_depth=max_depth)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        hb_model = hummingbird.ml.convert(model, "onnx", X)
        self.assertIsNotNone(hb_model)
        hb_model.save("onnx-tmp")

        shutil.unpack_archive("onnx-tmp.zip", "onnx-tmp", format="zip")

        # Removing the configuration file with the versions does not create problems.
        os.remove(os.path.join("onnx-tmp", constants.SAVE_LOAD_MODEL_CONFIGURATION_PATH))

        hummingbird.ml.load("onnx-tmp")
        os.remove("onnx-tmp.zip")

    # Test for when the user forgets to add a target (ex: convert(model, output) rather than convert(model, 'torch')) due to API change
    def test_forgotten_backend_string(self):
        from sklearn.preprocessing import LabelEncoder

        model = LabelEncoder()
        data = np.array([1, 4, 5, 2, 0, 2], dtype=np.int32)
        model.fit(data)

        self.assertRaises(ValueError, hummingbird.ml.convert, model, [("input", Int32TensorType([6, 1]))])

    # Test ONNX
    @unittest.skipIf(not onnx_runtime_installed(), reason="ONNX test requires ORT")
    def test_onnx(self):
        import numpy as np
        import lightgbm as lgb
        from hummingbird.ml import convert

        # Create some random data for binary classification.
        num_classes = 2
        X = np.array(np.random.rand(10000, 28), dtype=np.float32)
        y = np.random.randint(num_classes, size=10000)

        model = lgb.LGBMClassifier()
        model.fit(X, y)

        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "onnx")

    # Test Spark UDF
    @unittest.skipIf(
        os.name == "nt" or not sparkml_installed() or LooseVersion(pyspark.__version__) < LooseVersion("3"),
        reason="UDF Test requires spark >= 3",
    )
    @unittest.skipIf(
        prophet_installed() and sys.platform == "darwin",
        reason="Spark has problems with Prophet on Mac",
    )
    def test_udf_torch(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=77,
            test_size=0.2,
        )
        spark_df = sql_context.createDataFrame(pd.DataFrame(data=X_train))
        sql_context.registerDataFrameAsTable(spark_df, "IRIS")

        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        hb_model = hummingbird.ml.convert(model, "torch")

        # Broadcast the model.
        broadcasted_model = spark.sparkContext.broadcast(hb_model)

        # UDF definition.
        @pandas_udf("long")
        def udf_hb_predict(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            model = broadcasted_model.value
            for args in iterator:
                data_unmangled = pd.concat([feature for feature in args], axis=1)
                predictions = model.predict(data_unmangled)
                yield pd.Series(np.array(predictions))

        # Register the UDF.
        sql_context.udf.register("PREDICT", udf_hb_predict)

        # Run the query.
        sql_context.sql("SELECT SUM(prediction) FROM (SELECT PREDICT(*) as prediction FROM IRIS)").show()

    @unittest.skipIf(
        os.name == "nt" or not sparkml_installed() or LooseVersion(pyspark.__version__) < LooseVersion("3"),
        reason="UDF Test requires spark >= 3",
    )
    def test_udf_torch_jit_broadcast(self):
        import pickle

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=77,
            test_size=0.2,
        )
        spark_df = sql_context.createDataFrame(pd.DataFrame(data=X_train))
        sql_context.registerDataFrameAsTable(spark_df, "IRIS")

        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X_test)

        # Broadcast the model returns an error.
        self.assertRaises(pickle.PickleError, spark.sparkContext.broadcast, hb_model)

    @unittest.skipIf(
        os.name == "nt" or not sparkml_installed() or LooseVersion(pyspark.__version__) < LooseVersion("3"),
        reason="UDF Test requires spark >= 3",
    )
    @unittest.skipIf(
        prophet_installed() and sys.platform == "darwin",
        reason="Spark has problems with Prophet on Mac",
    )
    def test_udf_torch_jit_spark_file(self):
        import dill
        import torch.jit

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=77,
            test_size=0.2,
        )
        spark_df = sql_context.createDataFrame(pd.DataFrame(data=X_train))
        sql_context.registerDataFrameAsTable(spark_df, "IRIS")

        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        hb_model = hummingbird.ml.convert(model, "torch.jit", X_test)

        # Save the file locally.
        if os.path.exists("deployed_model.zip"):
            os.remove("deployed_model.zip")
        torch.jit.save(hb_model.model, "deployed_model.zip")
        hb_model._model = None

        # Share the model using spark file and broadcast the container.
        spark.sparkContext.addFile("deployed_model.zip")
        broadcasted_container = spark.sparkContext.broadcast(hb_model)

        # UDF definition.
        @pandas_udf("long")
        def udf_hb_predict(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            location = SparkFiles.get("deployed_model.zip")
            torch_model = torch.jit.load(location)
            container = broadcasted_container.value
            container._model = torch_model
            model = container
            for args in iterator:
                data_unmangled = pd.concat([feature for feature in args], axis=1)
                predictions = model.predict(data_unmangled.values)
                yield pd.Series(np.array(predictions))

        # Register the UDF.
        sql_context.udf.register("PREDICT", udf_hb_predict)

        # Run the query.
        sql_context.sql("SELECT SUM(prediction) FROM (SELECT PREDICT(*) as prediction FROM IRIS)").show()

        os.remove("deployed_model.zip")


if __name__ == "__main__":
    unittest.main()
