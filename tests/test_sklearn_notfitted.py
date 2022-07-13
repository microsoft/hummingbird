"""
Tests sklearn convertion incase of model not fitted.
"""
import unittest
import warnings

from hummingbird.ml._utils import sklearn_installed
from sklearn.exceptions import NotFittedError
from hummingbird.ml import convert

if sklearn_installed():
    from sklearn.preprocessing import Binarizer


class TestSklearnNotfitted(unittest.TestCase):
    @unittest.skipIf(not sklearn_installed(), reason="Sklearn test requires Sklearn installed")
    def test_binarizer_notfitted(self):
        warnings.filterwarnings("ignore")
        model = Binarizer()
        self.assertRaises(NotFittedError, convert, model, "torch")


if __name__ == "__main__":
    unittest.main()
