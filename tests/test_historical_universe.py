import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline_historical_universe import (
    append_unscored_execution_rows,
    build_daily_top_n_membership,
    load_shared_market_index,
    schedule_oos_signals_for_next_open,
    set_shared_snapshot_cache_dir,
)


class DailyUniverseTests(unittest.TestCase):
    @patch("pipeline_historical_universe.fetch_twse_daily_snapshot")
    @patch("pipeline_historical_universe.snapshot_cache_available", return_value=True)
    def test_daily_membership_uses_each_date_rank(self, _, snapshot_mock):
        snapshot_mock.side_effect = [
            pd.DataFrame({
                "symbol": ["2330", "2317", "2454"],
                "stock_name": ["A", "B", "C"],
                "turnover": [100.0, 300.0, 200.0],
                "volume": [10.0, 30.0, 20.0],
            }),
            pd.DataFrame({
                "symbol": ["2330", "2317", "2454"],
                "stock_name": ["A", "B", "C"],
                "turnover": [400.0, 100.0, 200.0],
                "volume": [40.0, 10.0, 20.0],
            }),
        ]

        result = build_daily_top_n_membership(
            pd.to_datetime(["2024-01-02", "2024-01-03"]),
            top_n=2,
            sleep_sec=0,
        )

        by_day = result.groupby("date")["symbol"].apply(list).to_dict()
        self.assertEqual(by_day[pd.Timestamp("2024-01-02")], ["2317", "2454"])
        self.assertEqual(by_day[pd.Timestamp("2024-01-03")], ["2330", "2454"])
        self.assertTrue(result["historical_universe_eligible"].all())

    def test_non_prediction_rows_are_retained_for_exit_prices(self):
        predicted = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["2330"],
            "final_decision_score": [0.8],
        })
        source = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["2330", "2330"],
            "close": [600.0, 610.0],
            "historical_universe_eligible": [True, False],
        })

        result = append_unscored_execution_rows(predicted, source).set_index("date")

        self.assertTrue(result.loc[pd.Timestamp("2024-01-02"), "is_oos_prediction"])
        self.assertFalse(result.loc[pd.Timestamp("2024-01-03"), "is_oos_prediction"])
        self.assertTrue(pd.isna(result.loc[pd.Timestamp("2024-01-03"), "final_decision_score"]))
        self.assertEqual(float(result.loc[pd.Timestamp("2024-01-03"), "close"]), 610.0)

    def test_shared_market_index_is_loaded_without_network(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            snapshot_dir = raw_dir / "twse_daily_market_snapshots"
            snapshot_dir.mkdir(parents=True)
            pd.DataFrame({
                "date": ["2020-01-02", "2020-03-31"],
                "market_close": [12000.0, 9700.0],
            }).to_csv(raw_dir / "TWSE_TAIEX_raw.csv", index=False)
            set_shared_snapshot_cache_dir(snapshot_dir)
            result = load_shared_market_index("2020-01-01", "2020-03-31")
            self.assertEqual(len(result), 2)
            self.assertEqual(float(result.iloc[-1]["market_close"]), 9700.0)
        set_shared_snapshot_cache_dir(None)

    def test_signal_is_executed_at_next_trading_open(self):
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["2330"],
            "close": [100.0],
            "final_decision_score": [0.8],
            "entry_stop_loss_price": [95.0],
        })
        source = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["2330", "2330"],
            "open": [99.0, 105.0],
            "high": [101.0, 110.0],
            "low": [98.0, 104.0],
            "close": [100.0, 108.0],
        })

        result = schedule_oos_signals_for_next_open(signals, source)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "signal_date"], pd.Timestamp("2024-01-02"))
        self.assertEqual(result.loc[0, "date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(float(result.loc[0, "signal_close"]), 100.0)
        self.assertEqual(float(result.loc[0, "entry_execution_price"]), 105.0)
        self.assertEqual(float(result.loc[0, "close"]), 108.0)
        self.assertEqual(result.loc[0, "entry_execution_timing"], "next_open")


if __name__ == "__main__":
    unittest.main()
