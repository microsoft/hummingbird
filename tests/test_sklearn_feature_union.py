# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import hummingbird.ml


class TestSklearnFeatureUnion(unittest.TestCase):
    def test_feature_union_default(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(np.float32)
        X_train, X_test, *_ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = FeatureUnion([("standard", StandardScaler()), ("minmax", MinMaxScaler())]).fit(X_train)

        torch_model = hummingbird.ml.convert(model, "torch")

        np.testing.assert_allclose(
            model.transform(X_test),
            torch_model.transform(X_test),
            rtol=1e-06,
            atol=1e-06,
        )

    def test_feature_union_transformer_weights(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(np.float32)
        X_train, X_test, *_ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = FeatureUnion(
            [("standard", StandardScaler()), ("minmax", MinMaxScaler())], transformer_weights={"standard": 2, "minmax": 4}
        ).fit(X_train)

        torch_model = hummingbird.ml.convert(model, "torch")

        np.testing.assert_allclose(
            model.transform(X_test),
            torch_model.transform(X_test),
            rtol=1e-06,
            atol=1e-06,
        )


if __name__ == "__main__":
    unittest.main()
