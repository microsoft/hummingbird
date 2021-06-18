import unittest
import numpy as np
import os
import sys
import torch
from distutils.version import LooseVersion

import hummingbird
from hummingbird.ml._utils import pandas_installed, prophet_installed, onnx_runtime_installed

if pandas_installed():
    import pandas as pd

if prophet_installed():
    from prophet import Prophet

if onnx_runtime_installed():
    import onnxruntime

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class TestProphet(unittest.TestCase):
    def _get_data(self):
        local_path = "tests/resources"
        local_data = os.path.join(local_path, "example_wp_log_peyton_manning.csv")
        url = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_peyton_manning.csv"
        if not os.path.isfile(local_data):
            os.makedirs(local_path)
            urlretrieve(url, local_data)
        data = pd.read_csv(local_data)
        return data

    @unittest.skipIf(not (pandas_installed() and prophet_installed()), reason="Test requires Prophet and Pandas")
    def test_prophet_trend(self):
        df = self._get_data()

        m = Prophet()
        m.fit(df)

        # Convert with Hummingbird.
        hb_model = hummingbird.ml.convert(m, "torch")

        # Predictions.
        future = m.make_future_dataframe(periods=365)
        prophet_trend = m.predict(future)["trend"].values
        hb_trend = hb_model.predict(future)

        np.testing.assert_allclose(prophet_trend, hb_trend, rtol=1e-06, atol=1e-06)

    @unittest.skipIf(
        not (pandas_installed() and prophet_installed()), reason="Test requires Prophet, Pandas and ONNX runtime.",
    )
    @unittest.skipIf(
        LooseVersion(torch.__version__) < LooseVersion("1.8.1"), reason="Test requires Torch 1.8.1.",
    )
    @unittest.skipIf(
        not onnx_runtime_installed() or LooseVersion(onnxruntime.__version__) < LooseVersion("1.7.0"),
        reason="Prophet test requires onnxruntime => 1.7.0",
    )
    def test_prophet_trend_onnx(self):
        df = self._get_data()

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=365)
        future_np = (future.values - np.datetime64("1970-01-01T00:00:00.000000000")).astype(np.int64) / 1000000000

        # Convert with Hummingbird.
        hb_model = hummingbird.ml.convert(m, "onnx", future_np)

        # Predictions.
        prophet_trend = m.predict(future)["trend"]
        hb_trend = hb_model.predict(future_np)
        import onnx

        onnx.save(hb_model.model, "prophet.onnx")

        np.testing.assert_allclose(prophet_trend, hb_trend, rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
    unittest.main()
