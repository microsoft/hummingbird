"""
Tests sklearn linear classifiers (LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV) converters.
"""
import unittest
import warnings
from distutils.version import LooseVersion

import numpy as np
import torch
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    LogisticRegressionCV,
    RidgeCV,
    Lasso,
    ElasticNet,
    Ridge,
)
from sklearn import datasets

import hummingbird.ml
from hummingbird.ml._utils import tvm_installed, pandas_installed
from hummingbird.ml import constants

if pandas_installed():
    import pandas


class TestSklearnLinearClassifiers(unittest.TestCase):

    # LogisticRegression test function to be parameterized
    def _test_logistic_regression(
        self, num_classes, solver="liblinear", multi_class="auto", labels_shift=0, fit_intercept=True
    ):
        if num_classes > 2:
            model = LogisticRegression(solver=solver, multi_class=multi_class, fit_intercept=fit_intercept)
        else:
            model = LogisticRegression(solver="liblinear", fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # LogisticRegression binary
    def test_logistic_regression_bi(self):
        self._test_logistic_regression(2)

    # LogisticRegression multiclass with auto
    def test_logistic_regression_multi_auto(self):
        self._test_logistic_regression(3)

    # LogisticRegression with class labels shifted
    def test_logistic_regression_shifted_classes(self):
        self._test_logistic_regression(3, labels_shift=2)

    # LogisticRegression with multi+ovr
    def test_logistic_regression_multi_ovr(self):
        self._test_logistic_regression(3, multi_class="ovr")

    # LogisticRegression with multi+multinomial+sag
    def test_logistic_regression_multi_multin_sag(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(3, multi_class="multinomial", solver="sag")

    # LogisticRegression binary lbfgs
    def test_logistic_regression_bi_lbfgs(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(2, solver="lbfgs")

    # LogisticRegression with multi+lbfgs
    def test_logistic_regression_multi_lbfgs(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(3, solver="lbfgs")

    # LogisticRegression with multi+multinomial+lbfgs
    def test_logistic_regression_multi_multin_lbfgs(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(3, multi_class="multinomial", solver="lbfgs")

    # LogisticRegression with multi+ovr+lbfgs
    def test_logistic_regression_multi_ovr_lbfgs(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(3, multi_class="ovr", solver="lbfgs")

    # LogisticRegression with fit_intercept set to False
    def test_logistic_regression_no_intercept(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression(3, fit_intercept=False)

    # LinearRegression test function to be parameterized
    def _test_linear_regression(self, y_input, fit_intercept=True):
        model = LinearRegression(fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = y_input

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # LinearRegression with ints
    def test_linear_regression_int(self):
        np.random.seed(0)
        self._test_linear_regression(np.random.randint(2, size=100))

    # LinearRegression with floats
    def test_linear_regression_float(self):
        np.random.seed(0)
        self._test_linear_regression(np.random.rand(100))

    # LinearRegression with fit_intercept set to False
    def test_linear_regression_no_intercept(self):
        np.random.seed(0)
        self._test_linear_regression(np.random.rand(100), fit_intercept=False)

    # Lasso test function to be parameterized
    def _test_lasso(self, y_input, fit_intercept=True):
        model = Lasso(fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = y_input

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # Lasso with ints
    def test_lasso_int(self):
        np.random.seed(0)
        self._test_lasso(np.random.randint(2, size=100))

    # Lasso with floats
    def test_lasso_float(self):
        np.random.seed(0)
        self._test_lasso(np.random.rand(100))

    # Lasso with fit_intercept set to False
    def test_lasso_no_intercept(self):
        np.random.seed(0)
        self._test_lasso(np.random.rand(100), fit_intercept=False)

    # Ridge test function to be parameterized
    def _test_ridge(self, y_input, fit_intercept=True):
        model = Ridge(fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = y_input

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # Ridge with ints
    def test_ridge_int(self):
        np.random.seed(0)
        self._test_ridge(np.random.randint(2, size=100))

    # Ridge with floats
    def test_ridge_float(self):
        np.random.seed(0)
        self._test_ridge(np.random.rand(100))

    # Ridge with fit_intercept set to False
    def test_ridge_no_intercept(self):
        np.random.seed(0)
        self._test_ridge(np.random.rand(100), fit_intercept=False)

    # ElasticNet test function to be parameterized
    def _test_elastic_net(self, y_input, fit_intercept=True):
        model = ElasticNet(fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = y_input

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # ElasticNet with ints
    def test_elastic_net_int(self):
        np.random.seed(0)
        self._test_elastic_net(np.random.randint(2, size=100))

    # ElasticNet with floats
    def test_elastic_net_float(self):
        np.random.seed(0)
        self._test_elastic_net(np.random.rand(100))

    # ElasticNet with fit_intercept set to False
    def test_elastic_net_no_intercept(self):
        np.random.seed(0)
        self._test_elastic_net(np.random.rand(100), fit_intercept=False)

    # RidgeCV test function to be parameterized
    def _test_ridge_cv(self, y_input, fit_intercept=True):
        model = RidgeCV(fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = y_input

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # RidgeCV with ints
    def test_ridge_cv_int(self):
        np.random.seed(0)
        self._test_ridge_cv(np.random.randint(2, size=100))

    # RidgeCV with floats
    def test_ridge_cv_float(self):
        np.random.seed(0)
        self._test_ridge_cv(np.random.rand(100))

    # RidgeCV with fit_intercept set to False
    def test_ridge_cv_no_intercept(self):
        np.random.seed(0)
        self._test_ridge_cv(np.random.rand(100), fit_intercept=False)

    # LogisticRegressionCV test function to be parameterized
    def _test_logistic_regression_cv(
        self, num_classes, solver="liblinear", multi_class="auto", labels_shift=0, fit_intercept=True
    ):
        if num_classes > 2:
            model = LogisticRegressionCV(solver=solver, multi_class=multi_class, fit_intercept=fit_intercept)
        else:
            model = LogisticRegressionCV(solver="liblinear", fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # LogisticRegressionCV with 2 classes
    def test_logistic_regression_cv_bi(self):
        self._test_logistic_regression_cv(2)

    # LogisticRegressionCV with 3 classes
    def test_logistic_regression_cv_multi(self):
        self._test_logistic_regression_cv(3)

    # LogisticRegressionCV with shifted classes
    def test_logistic_regression_cv_shifted_classes(self):
        self._test_logistic_regression_cv(3, labels_shift=2)

    # LogisticRegressionCV with multi+ovr
    def test_logistic_regression_cv_multi_ovr(self):
        self._test_logistic_regression_cv(3, multi_class="ovr")

    # LogisticRegressionCV with multi+multinomial
    def test_logistic_regression_cv_multi_multin(self):
        warnings.filterwarnings("ignore")
        # this will not converge due to small test size
        self._test_logistic_regression_cv(3, multi_class="multinomial", solver="sag")

    # LogisticRegressionCV with fit_intercept set to False
    def test_logistic_regression_cv_no_intercept(self):
        self._test_logistic_regression_cv(3, fit_intercept=False)

    # SGDClassifier test function to be parameterized
    def _test_sgd_classifier(self, num_classes, fit_intercept=True):

        model = SGDClassifier(loss="log_loss", fit_intercept=fit_intercept)

        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict_proba(X), torch_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # SGDClassifier with 2 classes
    def test_sgd_classifier_bi(self):
        self._test_sgd_classifier(2)

    # SGDClassifier with 3 classes
    def test_sgd_classifier_multi(self):
        self._test_sgd_classifier(3)

    # SGDClassifier with fit_intercept set to False
    def test_sgd_classifier_no_intercept(self):
        self._test_sgd_classifier(3, fit_intercept=False)

    # SGDClassifier with modified huber loss
    def test_modified_huber(self):
        X = np.array([[-0.5, -1], [-1, -1], [-0.1, -0.1], [0.1, -0.2], [0.5, 1], [1, 1], [0.1, 0.1], [-0.1, 0.2]])
        Y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        model = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3)
        model.fit(X, Y)

        # Use Hummingbird to convert the model to PyTorch
        hb_model = hummingbird.ml.convert(model, "torch")

        inputs = [[-1, -1], [1, 1], [-0.2, 0.1], [0.2, -0.1]]
        np.testing.assert_allclose(model.predict_proba(inputs), hb_model.predict_proba(inputs), rtol=1e-6, atol=1e-6)

    def test_modified_huber2(self):
        X = np.array([[-0.5, -1], [-1, -1], [-0.1, -0.1], [0.1, -0.2], [0.5, 1], [1, 1], [0.1, 0.1], [-0.1, 0.2]])
        Y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        model = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3)
        model.fit(X, Y)

        # Use Hummingbird to convert the model to PyTorch
        hb_model = hummingbird.ml.convert(model, "torch")

        np.testing.assert_allclose(model.predict_proba(X), hb_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    def test_modified_huber_multi(self):
        X = np.array([[-0.5, -1], [-1, -1], [-0.1, -0.1], [0.1, -0.2], [0.5, 1], [1, 1], [0.1, 0.1], [-0.1, 0.2]])
        Y = np.array([0, 1, 1, 1, 2, 2, 2, 2])

        model = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3)
        model.fit(X, Y)

        # Use Hummingbird to convert the model to PyTorch
        hb_model = hummingbird.ml.convert(model, "torch")

        inputs = [[-1, -1], [1, 1], [-0.2, 0.1], [0.2, -0.1]]
        np.testing.assert_allclose(model.predict_proba(inputs), hb_model.predict_proba(inputs), rtol=1e-6, atol=1e-6)

    # Failure cases
    def test_sklearn_linear_model_raises_wrong_type(self):
        warnings.filterwarnings("ignore")
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100).astype(np.float32)  # y must be int, not float, should error
        model = SGDClassifier().fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch")

    # Float 64 data tests
    def test_float64_linear_regression(self):
        model = LinearRegression()

        np.random.seed(0)
        X = np.random.rand(100, 200)
        y = np.random.randint(2, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    def test_float64_sgd_classifier(self):

        model = SGDClassifier(loss="log_loss")

        np.random.seed(0)
        num_classes = 3
        X = np.random.rand(100, 200)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # Multioutput regression tests
    def test_multioutput_linear_regression(self):
        for n_targets in [1, 2, 7]:
            model = LinearRegression()
            X, y = datasets.make_regression(
                n_samples=100, n_features=10, n_informative=5, n_targets=n_targets, random_state=2021
            )
            model.fit(X, y)

            torch_model = hummingbird.ml.convert(model, "torch")
            self.assertTrue(torch_model is not None)
            np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-5, atol=1e-5)

    # Test Pandas input
    @unittest.skipIf(not pandas_installed(), reason="Test requires pandas installed")
    def test_logistic_regression_pandas(self):
        model = LogisticRegression(solver="liblinear")

        data = datasets.load_iris()
        X, y = data.data[:, :3], data.target
        X = X.astype(np.float32)
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: 1 if x > 0.5 else 2)
        X_train["vcat2"] = X_train["vB"].apply(lambda x: 3 if x > 0.5 else 4)
        y_train = y % 2

        model.fit(X_train, y_train)

        hb_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(hb_model is not None)
        np.testing.assert_allclose(model.predict(X_train), hb_model.predict(X_train), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.predict_proba(X_train), hb_model.predict_proba(X_train), rtol=1e-6, atol=1e-6)

    # Test Torschscript backend.
    def test_logistic_regression_ts(self):

        model = LogisticRegression(solver="liblinear")

        data = datasets.load_iris()
        X, y = data.data, data.target
        X = X.astype(np.float32)

        model.fit(X, y)

        ts_model = hummingbird.ml.convert(model, "torch.jit", X)
        self.assertTrue(ts_model is not None)
        np.testing.assert_allclose(model.predict(X), ts_model.predict(X), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.predict_proba(X), ts_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    # Test TVM backends.
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_sgd_classifier_tvm(self):

        model = SGDClassifier(loss="log_loss")

        np.random.seed(0)
        num_classes = 3
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        tvm_model = hummingbird.ml.convert(model, "tvm", X)
        self.assertTrue(tvm_model is not None)
        np.testing.assert_allclose(model.predict(X), tvm_model.predict(X), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.predict_proba(X), tvm_model.predict_proba(X), rtol=1e-6, atol=1e-6)

    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_lr_tvm(self):

        model = LinearRegression()

        np.random.seed(0)
        num_classes = 1000
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)

        tvm_model = hummingbird.ml.convert(model, "tvm", X, extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})
        self.assertTrue(tvm_model is not None)

        np.testing.assert_allclose(model.predict(X), tvm_model.predict(X), rtol=1e-6, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
