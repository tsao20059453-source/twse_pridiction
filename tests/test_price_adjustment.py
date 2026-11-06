import unittest
from unittest.mock import patch

import pandas as pd

import pipeline_data


class PriceAdjustmentTests(unittest.TestCase):
    def test_missing_ex_dividend_snapshot_still_adjusts_prior_prices(self):
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "symbol": ["2330", "2330"],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "volume": [1000.0, 1000.0],
        })
        dividends = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["2330"],
            "f": [0.9],
        })
        with patch.dict(pipeline_data._DIVIDEND_CACHE, {"table": dividends}, clear=True):
            result = pipeline_data.adjust_for_corporate_actions(prices, verbose=False)

        self.assertAlmostEqual(float(result.loc[0, "close"]), 90.0)
        self.assertAlmostEqual(float(result.loc[1, "close"]), 100.0)
        self.assertAlmostEqual(float(result.loc[0, "volume"]), 1000.0 / 0.9)


if __name__ == "__main__":
    unittest.main()
