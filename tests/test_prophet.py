import unittest
import numpy as np

import hummingbird
from hummingbird.ml._utils import pandas_installed, prophet_installed, onnx_runtime_installed

if pandas_installed():
    import pandas as pd

if prophet_installed():
    from prophet import Prophet

DATA = "tests/resources/example_wp_log_peyton_manning.csv"


class TestProphet(unittest.TestCase):
    @unittest.skipIf(not (pandas_installed() and prophet_installed()), reason="Tests requires Prophet and Pandas")
    def test_prophet_trend(self):
        df = pd.read_csv(DATA)

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
        not (pandas_installed() and prophet_installed() and onnx_runtime_installed()),
        reason="Tests requires Prophet and Pandas",
    )
    def test_prophet_trend_onnx(self):
        df = pd.read_csv(DATA)

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
