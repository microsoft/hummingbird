"""
Tests sklearn SVC classifiers (LinearSVC, SVC, NuSVC) converters.
"""
import unittest

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC, NuSVC

import hummingbird.ml
from hummingbird.ml import constants
from hummingbird.ml._utils import tvm_installed


class TestSklearnSVC(unittest.TestCase):
    def _test_linear_svc(self, num_classes, labels_shift=0):
        model = LinearSVC()
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # LinearSVC binary
    def test_linear_svc_bi(self):
        self._test_linear_svc(2)

    # LinearSVC multiclass
    def test_linear_svc_multi(self):
        self._test_linear_svc(3)

    # LinearSVC with class labels shifted
    def test_linear_svc_shifted(self):
        self._test_linear_svc(3, labels_shift=2)

    # SVC test function to be parameterized
    def _test_svc(self, num_classes, kernel="rbf", gamma=None, backend="torch", labels_shift=0, extra_config={}):

        if gamma:
            model = SVC(kernel=kernel, gamma=gamma)
        else:
            model = SVC(kernel=kernel)
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100) + labels_shift

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, backend, X, extra_config=extra_config)

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # SVC binary
    def test_svc_bi(self):
        self._test_svc(2)

    # SVC multiclass
    def test_svc_multi(self):
        self._test_svc(3)

    # SVC linear kernel
    def test_svc_linear(self):
        self._test_svc(2, kernel="linear")

    # SVC sigmoid kernel
    def test_svc_sigmoid(self):
        self._test_svc(3, kernel="sigmoid")

    # SVC poly kernel
    def test_svc_poly(self):
        self._test_svc(3, kernel="poly")

    # SVC with class labels shifted
    def test_svc_shifted(self):
        self._test_svc(3, labels_shift=2)

    # SVC with different gamma (default=’scale’)
    def test_svc_gamma(self):
        self._test_svc(3, gamma="auto")

    # NuSVC test function to be parameterized
    def _test_nu_svc(self, num_classes, backend="torch", extra_config={}):
        model = NuSVC()
        np.random.seed(0)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(num_classes, size=100)

        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, backend, X, extra_config=extra_config)

        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # NuSVC binary
    def test_nu_svc_bi(self):
        self._test_nu_svc(2)

    # NuSVC multiclass
    def test_nu_svc_multi(self):
        self._test_nu_svc(3)

    # assert fail on unsupported kernel
    def test_sklearn_linear_model_raises_wrong_type(self):

        np.random.seed(0)
        X = np.random.rand(10, 10)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=10)
        model = SVC(kernel="precomputed").fit(X, y)
        self.assertRaises(RuntimeError, hummingbird.ml.convert, model, "torch")

    # Float 64 data tests
    def test_float64_linear_svc(self):
        np.random.seed(0)
        num_classes = 3
        X = np.random.rand(100, 200)
        y = np.random.randint(num_classes, size=100)

        model = LinearSVC()
        model.fit(X, y)
        torch_model = hummingbird.ml.convert(model, "torch")
        self.assertTrue(torch_model is not None)
        np.testing.assert_allclose(model.predict(X), torch_model.predict(X), rtol=1e-6, atol=1e-6)

    # Torchscript backend
    def test_svc_ts(self):
        self._test_svc(2, backend="torch.jit")

    # SVC linear kernel
    def test_svc_linear_ts(self):
        self._test_svc(2, kernel="linear", backend="torch.jit")

    # SVC sigmoid kernel
    def test_svc_sigmoid_ts(self):
        self._test_svc(2, kernel="sigmoid", backend="tvtorch.jit")

    # SVC poly kernel
    def test_svc_poly_ts(self):
        self._test_svc(2, kernel="poly", backend="torch.jit")

    # NuSVC binary
    def test_nu_svc_bi_ts(self):
        self._test_nu_svc(2, backend="torch.jit")

    def test_svc_multi_ts(self):
        self._test_svc(3, backend="torch.jit")

    # TVM backend.
    # SVC binary
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_svc_tvm(self):
        self._test_svc(2, backend="tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

    # SVC linear kernel
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_svc_linear_tvm(self):
        self._test_svc(2, kernel="linear", backend="tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

    # SVC sigmoid kernel
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_svc_sigmoid_tvm(self):
        self._test_svc(2, kernel="sigmoid", backend="tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

    # SVC poly kernel
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_svc_poly_tvm(self):
        self._test_svc(2, kernel="poly", backend="tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

    # NuSVC binary
    @unittest.skipIf(not (tvm_installed()), reason="TVM tests require TVM")
    def test_nu_svc_bi_tvm(self):
        self._test_nu_svc(2, backend="tvm", extra_config={constants.TVM_MAX_FUSE_DEPTH: 30})

    # # SVC multiclass
    # def test_svc_multi(self):
    #     self._test_svc(3)

    # # SVC sigmoid kernel
    # def test_svc_sigmoid(self):
    #     self._test_svc(3, kernel="sigmoid", backend="tvm")

    # # SVC poly kernel
    # def test_svc_poly(self):
    #     self._test_svc(3, kernel="poly", backend="tvm")

    # # SVC with class labels shifted
    # def test_svc_shifted(self):
    #     self._test_svc(3, labels_shift=2, backend="tvm")

    # # SVC with different gamma (default=’scale’)
    # def test_svc_gamma(self):
    #     self._test_svc(3, gamma="auto", backend="tvm")


if __name__ == "__main__":
    unittest.main()
