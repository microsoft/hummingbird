# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Collection of utility functions used throughout Hummingbird.
"""

from distutils.version import LooseVersion
import numpy as np
import torch
import os
import warnings
import shutil

from hummingbird.ml.exceptions import ConstantError


def torch_installed():
    """
    Checks that *PyTorch* is available.
    """
    try:
        import torch

        return True
    except ImportError:
        return False


def onnx_ml_tools_installed():
    """
    Checks that *ONNXMLTools* is available.
    """
    try:
        import onnxmltools

        return True
    except ImportError:
        print("ONNXMLTOOLS not installed. Please check https://github.com/onnx/onnxmltools for instructions.")
        return False


def onnx_runtime_installed():
    """
    Checks that *ONNX Runtime* is available.
    """
    try:
        import onnxruntime

        return True
    except ImportError:
        return False


def sparkml_installed():
    """
    Checks that *Spark ML/PySpark* is available.
    """
    try:
        import pyspark

        return True
    except ImportError:
        return False


def sklearn_installed():
    """
    Checks that *Sklearn* is available.
    """
    try:
        import sklearn

        return True
    except ImportError:
        return False


def lightgbm_installed():
    """
    Checks that *LightGBM* is available.
    """
    try:
        import lightgbm

        return True
    except ImportError:
        return False


def xgboost_installed():
    """
    Checks that *XGBoost* is available.
    """
    try:
        import xgboost
    except ImportError:
        return False
    from xgboost.core import _LIB

    try:
        _LIB.XGBoosterDumpModelEx
    except AttributeError:
        # The version is not recent enough even though it is version 0.6.
        # You need to install xgboost from github and not from pypi.
        return False
    from xgboost import __version__

    vers = LooseVersion(__version__)
    allowed_min = LooseVersion("0.90")
    if vers < allowed_min:
        warnings.warn("The converter works for xgboost >= 0.9. Different versions might not.")
    return True


def tvm_installed():
    """
    Checks that *TVM* is available.
    """
    try:
        import tvm
    except ImportError:
        return False
    return True


def pandas_installed():
    """
    Checks that *Pandas* is available.
    """
    try:
        import pandas
    except ImportError:
        return False
    return True


def is_pandas_dataframe(df):
    import pandas as pd

    if type(df) == pd.DataFrame:
        return True
    else:
        return False


def is_spark_dataframe(df):
    if not sparkml_installed():
        return False

    import pyspark

    if type(df) == pyspark.sql.DataFrame:
        return True
    else:
        return False


def get_device(model):
    """
    Convenient function used to get the runtime device for the model.
    """
    assert issubclass(model.__class__, torch.nn.Module)

    device = None
    if len(list(model.parameters())) > 0:
        device = next(model.parameters()).device  # Assuming we are using a single device for all parameters

    return device


def from_strings_to_ints(input, max_string_length):
    """
    Utility function used to transform string inputs into a numerical representation.
    """
    shape = list(input.shape)
    shape.append(max_string_length // 4)
    return np.array(input, dtype="|S" + str(max_string_length)).view(np.int32).reshape(shape)


def load(location):
    """
    Utility function used to load arbitrary Hummingbird models.
    """
    # Add load capabilities.
    from hummingbird.ml.containers import PyTorchSklearnContainer
    from hummingbird.ml.containers import TVMSklearnContainer
    from hummingbird.ml.containers import ONNXSklearnContainer
    from hummingbird.ml.operator_converters import constants

    model = None
    model_type = None

    # Unzip the dir.
    zip_location = location
    if not location.endswith("zip"):
        zip_location = location + ".zip"
    else:
        location = zip_location[:-4]
    shutil.unpack_archive(zip_location, location, format="zip")

    assert os.path.exists(location), "Model location {} does not exist.".format(location)

    # Load the model type.
    with open(os.path.join(location, constants.SAVE_LOAD_MODEL_TYPE_PATH), "r") as file:
        model_type = file.readline()

    if "torch" in model_type:
        model = PyTorchSklearnContainer.load(location, do_unzip_and_model_type_check=False)
    elif "onnx" in model_type:
        model = ONNXSklearnContainer.load(location, do_unzip_and_model_type_check=False)
    elif "tvm" in model_type:
        model = TVMSklearnContainer.load(location, do_unzip_and_model_type_check=False)
    else:
        raise RuntimeError("Model type {} not recognized.".format(model_type))

    assert model.model is not None
    return model


class _Constants(object):
    """
    Class enabling the proper definition of constants.
    """

    def __init__(self, constants, other_constants=None):
        for constant in dir(constants):
            if constant.isupper():
                setattr(self, constant, getattr(constants, constant))
        for constant in dir(other_constants):
            if constant.isupper():
                setattr(self, constant, getattr(other_constants, constant))

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ConstantError("Overwriting a constant is not allowed {}".format(name))
        self.__dict__[name] = value
