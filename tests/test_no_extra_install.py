"""
Test Hummingbird when no extra dependencies are installed.
"""
import unittest
import warnings

import numpy as np

from hummingbird.ml._utils import lightgbm_installed, xgboost_installed, onnx_runtime_installed, onnx_ml_tools_installed


class TestNoExtra(unittest.TestCase):
    """
    These tests are meant to be run on a clean container after doing
    `pip install hummingbird-ml` without any of the `extra` packages
    """

    # Test no LGBM returns false on lightgbm_installed()
    @unittest.skipIf(lightgbm_installed(), reason="Test when LightGBM is not installed")
    def test_lightgbm_installed_false(self):
        warnings.filterwarnings("ignore")
        assert not lightgbm_installed()

    # Test no XGB returns false on xgboost_installed()
    @unittest.skipIf(xgboost_installed(), reason="Test when XGBoost is not installed")
    def test_xgboost_installed_false(self):
        warnings.filterwarnings("ignore")
        assert not xgboost_installed()

    # Test no ONNX returns false on onnx_installed()
    @unittest.skipIf(onnx_runtime_installed(), reason="Test when ONNX is not installed")
    def test_onnx_installed_false(self):
        warnings.filterwarnings("ignore")
        assert not onnx_runtime_installed()

    # Test no ONNXMLTOOLS returns false on onnx_ml_tools_installed()
    @unittest.skipIf(onnx_ml_tools_installed(), reason="Test when ONNXMLTOOLS is not installed")
    def test_onnx_ml_installed_false(self):
        warnings.filterwarnings("ignore")
        assert not onnx_ml_tools_installed()

    # Test that we can import the converter successfully without installing [extra]
    def test_import_convert_no_extra(self):
        try:
            from hummingbird.ml import convert
        except Exception:  # TODO something more specific?
            self.fail("Unexpected Error on importing convert without extra packages")


if __name__ == "__main__":
    unittest.main()
