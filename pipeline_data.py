from __future__ import annotations

"""
Clean Stock ML Pipeline
=======================

目的：
1. 抓取台股 TWSE 近 N 年日線資料
2. 計算技術指標與市場結構特徵
3. 整理成乾淨、不重複欄位的 final_stock_table
4. 建立未來 10 個交易日風險報酬結構 target_score
5. 訓練 Random Forest 模型並輸出最新 target_score

目前版本重點：
- horizon 統一為 10
- 欄位命名統一為 *_10d
- 避免欄位重複 merge
- 將原本 stock.py / indicator.py / output.py / ML.py 整合成單一 pipeline
- 先保留 RandomForestRegressor，後續可替換 Ridge / XGBoost

注意：
- 目前 TPEx 尚未接資料源，本版先支援 TWSE。
- BDay 只排除週末，不排除台股國定假日。
- 本程式是研究與教學用途，不構成投資建議。
"""

import os
import re
import io
import json
import time
import hashlib
import argparse
import warnings
import random
import contextlib
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import requests
import urllib3
from pandas.tseries.offsets import BDay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# 0. 全域設定
# ============================================================

FEATURE_SET_CHOICES = [
    "basic",
    "structure",
    "full",
    "full_no_market",
    "full_market_light",
    "full_amount_relative",
    "full_amount_relative_selected",
    "technical_core",
    "structural",
    "selected",
    "selected_v2",
    "selected_risk_v1",
]
DEFAULT_COMPARE_FEATURE_SETS = ["selected_v2"]
DEPRECATED_FEATURE_SETS = {
    "full": "experimental/deprecated: full includes broad market features that may dominate stock-level signals.",
    "full_market_light": "experimental: light market context is retained for comparison only.",
    "full_error_v1": "deprecated: realized history feature experiment underperformed.",
    "full_error_v1_symbol": "deprecated: symbol categorical feature experiment underperformed.",
}
MODEL_PROFILE_CHOICES = ["quick", "conservative", "medium", "flexible"]
MODEL_TYPE_CHOICES = ["random_forest", "extra_trees", "hist_gradient_boosting"]
DIAGNOSTIC_SCOPE_CHOICES = ["single", "common"]
DISPLAY_MODE_CHOICES = ["research", "presentation"]

@dataclass
class PipelineConfig:
    stock_no: str = "2330"
    market: str = "TWSE"
    years: int = 3
    horizon: int = 10
    output_dir: str = "output"
    sleep_sec: float = 0.3
    atr_period: int = 14
    atr_multiplier: float = 2.0
    ob_lookback_days: int = 10
    fib_lookback_days: int = 10
    sr_lookback_days: int = 150
    sr_tolerance_pct: float = 0.003
    fvg_min_gap_pct: float = 0.002
    risk_penalty: float = 1.0
    opportunity_threshold: float = 0.05
    target_col: str = "target_score_v2"
    feature_set: str = "selected_v2"
    mode: Optional[str] = None
    analysis_mode: Optional[str] = None
    target_mode: Optional[str] = None
    use_sensitivity: bool = False
    require_sensitivity_data: bool = False
    model_type: str = "extra_trees"
    model_profile: str = "medium"
    random_state: int = 42
    run_feature_importance: bool = False
    target_compare_cols: str = "target_score_v2,target_score_raw_10d"
    compare_feature_sets: Optional[str] = None
    compare_model_types: Optional[str] = None
    diagnostic_scope: str = "common"
    skip_classification: bool = False
    display_mode: str = "presentation"
    backtest_initial_capital: float = 1_000_000.0
    backtest_max_positions: int = 8
    backtest_atr_trailing_multiplier: float = 2.0
    backtest_ma_exit_period: int = 20
    backtest_partial_exit_ratio: float = 0.5
    backtest_buy_fee_rate: float = 0.001425    # 買進手續費 0.1425%
    backtest_sell_fee_rate: float = 0.001425   # 賣出手續費 0.1425%
    backtest_tax_rate: float = 0.003           # 證交稅 0.3%（賣出時收取）
    backtest_slippage_rate: float = 0.001      # 滑價 0.1%（買賣各一半）
    backtest_fee_discount: float = 0.6         # 手續費折扣，實際費率 × 0.6
    backtest_profile_name: str = "balanced_v2_combined"
    backtest_portfolio_config_version: str = "balanced_v2_combined"
    backtest_initial_position_pct: float = 0.08      # 試倉 8%
    backtest_add1_position_pct: float = 0.06         # 第一次加碼 6%
    backtest_add2_position_pct: float = 0.06         # 第二次加碼 6%
    backtest_max_single_position_pct: float = 0.20   # 單一股票最大倉位 20%
    backtest_max_total_exposure: float = 0.85        # 總曝險上限 85%
    backtest_min_cash_reserve: float = 0.15          # 最低現金保留 15%
    backtest_entry_mode: str = "balanced_support"
    backtest_return_20d_top_entry_pct: float = 0.30
    backtest_return_20d_top_add_pct: float = 0.30
    backtest_return_20d_top_strict_pct: float = 0.20
    use_return_20d: bool = True   # 是否啟用 20 日報酬輔助分數（return_20d_score）
    use_symbol_eligibility_filter: bool = True
    eligibility_window_months: int = 6
    exclude_eligibility_window_from_official_backtest: bool = True


CONFIG = PipelineConfig()

ENTRY_DIAGNOSTIC_DAILY_COLUMNS = [
    "candidate_entry_signals",
    "executed_entries",
    "support_entry_candidates",
    "strong_continuation_candidates",
    "accepted_strong_continuation_entries",
    "rejected_by_market_filter",
    "rejected_by_support_zone",
    "rejected_by_stop_loss_pct",
    "rejected_by_rr",
    "rejected_by_cash",
    "rejected_by_position_limit",
    "rejected_by_total_exposure",
    "rejected_by_single_position_limit",
    "rejected_by_existing_position",
    "executed_by_strict_support",
    "executed_by_support_entry",
    "executed_by_near_support_relaxed",
    "executed_by_relaxed_support_entry",
    "executed_by_strong_momentum",
    "executed_by_strong_continuation",
    "volume_confirmed_support_entry",
    "volume_confirmed_strong_entry",
    "rejected_by_momentum_condition",
    "executed_by_quiet_pullback_support",
    "executed_by_volume_rebound_support",
    "executed_by_volume_breakout_momentum",
    "rejected_by_volume_confirmation",
    "rejected_strong_by_trend",
    "rejected_strong_by_return_rank",
    "rejected_strong_by_high_position",
    "rejected_strong_by_atr_risk",
    "rejected_strong_by_market_filter",
    "rejected_strong_by_cash",
    "rejected_strong_by_position_limit",
    "rejected_strong_by_total_exposure",
    "rejected_strong_by_existing_position",
    "rejected_strong_by_rr",
    "rejected_strong_by_volume_soft_condition",
    "candidate_with_volume_confirmation",
    "volume_confirmation_rate",
    "zone_entry_count",
    "support_proximity_count",
    "breakout_entry_count",
]
ENTRY_REJECTION_KEYS = [
    "rejected_by_market_filter",
    "rejected_by_support_zone",
    "rejected_by_stop_loss_pct",
    "rejected_by_rr",
    "rejected_by_cash",
    "rejected_by_position_limit",
    "rejected_by_total_exposure",
    "rejected_by_single_position_limit",
    "rejected_by_existing_position",
    "rejected_by_momentum_condition",
    "rejected_by_volume_confirmation",
]
ENTRY_DIAGNOSTIC_LABELS = {
    "candidate_entry_signals": "候選進場訊號數",
    "executed_entries": "實際進場次數",
    "entry_execution_rate": "進場執行率",
    "executed_by_strict_support": "原本支撐區間進場次數",
    "executed_by_near_support_relaxed": "靠近支撐放寬進場次數",
    "executed_by_strong_momentum": "強勢延續進場次數",
    "executed_by_strong_continuation": "強勢延續進場次數",
    "executed_by_quiet_pullback_support": "縮量回測支撐進場次數",
    "executed_by_volume_rebound_support": "放量止跌承接進場次數",
    "executed_by_volume_breakout_momentum": "放量強勢延續進場次數",
    "candidate_with_volume_confirmation": "符合成交量確認候選數",
    "volume_confirmation_rate": "成交量確認率",
    "rejected_by_market_filter": "被大盤過濾擋下",
    "rejected_by_support_zone": "被支撐區間擋下",
    "rejected_by_stop_loss_pct": "被停損距離條件擋下",
    "rejected_by_rr": "被 R/R 條件擋下",
    "rejected_by_cash": "被現金限制擋下",
    "rejected_by_position_limit": "被最大持股數限制擋下",
    "rejected_by_total_exposure": "被總曝險限制擋下",
    "rejected_by_single_position_limit": "被單檔倉位限制擋下",
    "rejected_by_existing_position": "被既有持倉限制擋下",
    "rejected_by_momentum_condition": "被強勢延續條件擋下",
    "rejected_by_volume_confirmation": "被成交量確認擋下",
}
ENTRY_REJECTION_LABELS = {
    "rejected_by_market_filter": "大盤過濾",
    "rejected_by_support_zone": "支撐區間",
    "rejected_by_stop_loss_pct": "停損距離",
    "rejected_by_rr": "R/R 條件",
    "rejected_by_cash": "現金限制",
    "rejected_by_position_limit": "最大持股數",
    "rejected_by_total_exposure": "總曝險",
    "rejected_by_single_position_limit": "單檔倉位",
    "rejected_by_existing_position": "既有持倉",
    "rejected_by_momentum_condition": "強勢延續條件",
    "rejected_by_volume_confirmation": "成交量確認",
}
ENTRY_DIAGNOSTIC_LABELS.update({
    "support_entry_candidates": "support_entry candidates",
    "strong_continuation_candidates": "strong_continuation candidates",
    "accepted_strong_continuation_entries": "accepted strong_continuation entries",
    "executed_by_support_entry": "support_entry entries",
    "executed_by_relaxed_support_entry": "relaxed_support_entry entries",
    "volume_confirmed_support_entry": "volume_confirmed_support_entry entries",
    "volume_confirmed_strong_entry": "volume_confirmed_strong_entry entries",
    "rejected_strong_by_trend": "rejected strong by trend",
    "rejected_strong_by_return_rank": "rejected strong by return rank",
    "rejected_strong_by_high_position": "rejected strong by high position",
    "rejected_strong_by_atr_risk": "rejected strong by atr risk",
    "rejected_strong_by_market_filter": "rejected strong by market filter",
    "rejected_strong_by_cash": "rejected strong by cash",
    "rejected_strong_by_position_limit": "rejected strong by position limit",
    "rejected_strong_by_total_exposure": "rejected strong by total exposure",
    "rejected_strong_by_existing_position": "rejected strong by existing position",
    "rejected_strong_by_rr": "rejected strong by rr",
    "rejected_strong_by_volume_soft_condition": "rejected strong by volume soft condition",
})

ENTRY_REJECTION_LABELS.update({
    "rejected_strong_by_trend": "rejected strong by trend",
    "rejected_strong_by_return_rank": "rejected strong by return rank",
    "rejected_strong_by_high_position": "rejected strong by high position",
    "rejected_strong_by_atr_risk": "rejected strong by atr risk",
    "rejected_strong_by_rr": "rejected strong by rr",
})

PORTFOLIO_EXPERIMENT_PROFILES = [
    {
        "profile_name": "balanced_v1_baseline",
        "entry_mode": "strict_support",
        "backtest_max_positions": 5,
    },
    {
        "profile_name": "balanced_v2_max8",
        "entry_mode": "strict_support",
        "backtest_max_positions": 8,
    },
    {
        "profile_name": "balanced_v2_support_relaxed",
        "entry_mode": "balanced_support",
        "backtest_max_positions": 5,
    },
    {
        "profile_name": "balanced_v2_combined",
        "entry_mode": "balanced_support",
        "backtest_max_positions": 8,
    },
    {
        "profile_name": "balanced_v2_support_v2",
        "entry_mode": "balanced_support_v2",
        "backtest_max_positions": 5,
    },
    {
        "profile_name": "balanced_v2_combined_support_v2",
        "entry_mode": "balanced_support_v2",
        "backtest_max_positions": 8,
    },
    {
        "profile_name": "balanced_v2_volume_confirmed",
        "entry_mode": "volume_confirmed_entry_v1",
        "backtest_max_positions": 5,
    },
    {
        "profile_name": "balanced_v2_combined_volume_confirmed",
        "entry_mode": "volume_confirmed_entry_v1",
        "backtest_max_positions": 8,
    },
]

OUTPUT_DIR = Path(CONFIG.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True)
TOP_TURNOVER_CACHE_DIR = OUTPUT_DIR / "cache"
TOP_TURNOVER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ENABLE_IMPORTANCE_FEATURE_SELECTION = False
MODEL_FLOW_AUDITS: dict[str, dict] = {}

STRUCTURAL_MODEL_FEATURES = [
    "return_1d", "return_3d", "return_5d", "return_10d",
    "ma_week_distance_pct", "ma_month_distance_pct", "ma_season_distance_pct",
    "ma_week_slope", "ma_month_slope", "ma_season_slope",
    "close_to_5d_high", "close_to_5d_low", "close_to_10d_high", "close_to_10d_low",
    "support_distance_pct", "resistance_distance_pct",
    "atr_pct", "risk_stoploss_pct", "risk_atr_stop_distance_pct",
    "fib_position", "fib_zone",
    "rsi_zone", "macd_cross_signal", "macd_hist_trend", "kdj_cross_signal", "bias_zone",
    "volume_ratio", "volume_trend_signal",
    "VR_14", "VR_26", "VR_14_change", "VR_26_change",
]

STRUCTURAL_CATEGORICAL_FEATURES = ["fib_zone", "rsi_zone", "bias_zone", "VR_14_zone"]

EXCLUDED_RAW_FEATURES = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma20", "ma60", "MA6", "MA12", "MA24",
    "EMA12", "EMA26", "DIF", "DEA", "MACD",
    "K", "D", "J", "RSI14", "BIAS6", "BIAS12", "BIAS24",
    "atr_14", "true_range", "vol_ma5", "vol_ma20",
    "nearest_support_level", "nearest_support_zone_low", "nearest_support_zone_high",
    "nearest_resistance_level", "nearest_resistance_zone_low", "nearest_resistance_zone_high",
    "risk_entry_price", "risk_atr_14", "risk_ob_bottom", "risk_atr_stop",
    "risk_stoploss_price", "risk_per_share",
]

MARKET_RAW_FEATURES = [
    "market_close",
    "market_ma5",
    "market_ma20",
    "market_ma60",
    "market_overheat_threshold",
]

MARKET_NUMERIC_FEATURES = [
    "market_return_1d",
    "market_return_3d",
    "market_return_5d",
    "market_return_10d",
    "market_return_20d",
    "market_ma5_distance",
    "market_ma20_distance",
    "market_ma60_distance",
    "market_volatility_10d",
    "market_volatility_20d",
    "market_above_ma20",
    "market_above_ma60",
]

MARKET_CATEGORICAL_FEATURES = [
    "market_trend_regime",
]

AMOUNT_FEATURES = [
    "trading_amount",
    "amount_ma5",
    "amount_ma20",
    "amount_ratio_5",
    "amount_ratio_20",
    "amount_change_5d",
    "amount_change_10d",
    "amount_zscore_20",
]

CROSS_SECTIONAL_AMOUNT_FEATURES = [
    "amount_rank",
    "amount_rank_pct",
    "amount_rank_change_5d",
    "amount_rank_change_10d",
    "amount_share_in_pool",
]

RELATIVE_STRENGTH_FEATURES = [
    "return_20d",
    "relative_return_5d",
    "relative_return_10d",
    "relative_return_20d",
    "stock_vs_market_ma20_distance",
    "stock_vs_market_ma60_distance",
]

CROSS_SECTIONAL_RELATIVE_STRENGTH_FEATURES = [
    "relative_strength_rank_10d",
]

REALIZED_HISTORY_FEATURES = [
    "realized_score_v2_lag10",
    "realized_score_v2_lag20",
    "realized_score_v2_roll20_lag10",
    "realized_score_v2_roll60_lag10",
    "realized_score_v2_std20_lag10",
    "realized_return_10d_lag10",
    "realized_return_10d_roll20_lag10",
    "realized_drawdown_10d_roll20_lag10",
    "realized_risk_adjusted_return_roll20_lag10",
    "realized_win_rate_20_lag10",
]

LOW_SUBJECTIVE_TARGETS = [
    "target_return_10d",
    "target_return_10d_cs_rank",
    "target_downside_risk_10d",
    "target_downside_risk_10d_cs_rank",
    "target_safety_10d_cs_rank",
    "target_abs_drawdown_safety_10d",
    "target_rar_10d",
    "target_rar_10d_cs_rank",
]

NEW_DIAGNOSTIC_FEATURES = (
    MARKET_RAW_FEATURES
    + MARKET_NUMERIC_FEATURES
    + MARKET_CATEGORICAL_FEATURES
    + AMOUNT_FEATURES
    + CROSS_SECTIONAL_AMOUNT_FEATURES
    + RELATIVE_STRENGTH_FEATURES
    + CROSS_SECTIONAL_RELATIVE_STRENGTH_FEATURES
    + REALIZED_HISTORY_FEATURES
)

TWSE_STOCK_DAY_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
TWSE_STOCK_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_TAIEX_HIST_URLS = [
    "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
    "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
]
SESSION = requests.Session()

COLUMN_RENAME_MAP = {
    "日期": "date",
    "成交股數": "volume",
    "成交金額": "turnover",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
    "漲跌價差": "change",
    "成交筆數": "transactions",
}

NUMERIC_COLUMNS = [
    "volume", "turnover", "open", "high", "low", "close", "change", "transactions"
]


# ============================================================
# 1. 資料抓取與清理
# ============================================================

def roc_to_ad_date(roc_date: str) -> pd.Timestamp:
    roc_date = str(roc_date).strip()
    year, month, day = roc_date.split("/")
    return pd.Timestamp(year=int(year) + 1911, month=int(month), day=int(day))


def clean_numeric(value) -> Optional[float]:
    if value is None:
        return pd.NA
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "---", "X0.00", "除權息"}:
        return pd.NA
    try:
        return float(text)
    except ValueError:
        return pd.NA


def get_month_start_str(dt: datetime) -> str:
    return dt.strftime("%Y%m") + "01"


def shift_month(dt: datetime, months_back: int) -> datetime:
    year = dt.year
    month = dt.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    return datetime(year, month, 1)


def validate_inputs(stock_no: str, market: str) -> None:
    if market not in {"TWSE", "TPEX"}:
        raise ValueError("market 只能是 TWSE 或 TPEX")
    if not stock_no:
        raise ValueError("股票代碼不能為空")
    if not stock_no.isdigit():
        raise ValueError("股票代碼應為數字，例如 2330")


def normalize_stock_symbol_input(value: str) -> str:
    text = "" if value is None else str(value).strip().upper()
    text = text.replace("\\", "").replace("/", "")
    text = re.sub(r"[^0-9A-Z*]", "", text)
    return text


def is_all_stock_symbol_input(value: str) -> bool:
    text = normalize_stock_symbol_input(value)
    return text in {"ALL", "*"} or str(value).strip() in {"全部", "全"}


def prompt_single_stock_symbol(default_symbol: str = "2330") -> str:
    while True:
        user_symbol = input("請輸入要分析的股票代碼，例如 2330：").strip()
        if user_symbol and is_all_stock_symbol_input(user_symbol):
            print("個股深度分析需要指定單一股票代碼，例如 2330。")
            continue
        symbol = normalize_stock_symbol_input(user_symbol) if user_symbol else default_symbol
        if not symbol.isdigit():
            print("股票代碼格式不正確，請輸入數字股票代碼，例如 2330。")
            continue
        return symbol


def parse_interactive_top_n(value: str, default_top_n: int = 30, min_top_n: int = 5) -> int:
    text = "" if value is None else str(value).strip()
    if not text:
        return default_top_n
    try:
        top_n = int(text)
    except ValueError:
        print("Top N 輸入不是整數，已改用預設值 30。")
        return default_top_n
    if top_n < min_top_n:
        print("目前 target 使用橫斷面排名分數，至少需要 5 檔股票才能建立共同模型；已自動將 Top N 調整為 5。")
        return min_top_n
    return top_n


def request_twse_month_json(stock_no: str, date_yyyymm01: str) -> dict:
    params = {
        "response": "json",
        "date": date_yyyymm01,
        "stockNo": stock_no,
    }
    response = SESSION.get(
        TWSE_STOCK_DAY_URL,
        params=params,
        timeout=30,
        verify=False,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("stat") != "OK":
        raise ValueError(f"TWSE 回傳失敗：{data.get('stat')}")
    return data


def fetch_twse_month_raw(stock_no: str, dt: datetime) -> pd.DataFrame:
    date_yyyymm01 = get_month_start_str(dt)
    data = request_twse_month_json(stock_no, date_yyyymm01)
    return pd.DataFrame(data["data"], columns=data["fields"])


def fetch_month_data(stock_no: str, market: str, dt: datetime) -> pd.DataFrame:
    market = market.upper()
    if market == "TWSE":
        return fetch_twse_month_raw(stock_no, dt)
    if market == "TPEX":
        raise NotImplementedError("TPEx 尚未接好資料源，本版請先使用 TWSE。")
    raise ValueError("market 只能是 TWSE 或 TPEX")


def fetch_twse_stock_day_all() -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            print(f"第 {attempt} 次嘗試連線 TWSE OpenAPI...")

            response = SESSION.get(
                TWSE_STOCK_DAY_ALL_URL,
                timeout=10,
                verify=False
            )
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, list):
                raise ValueError("TWSE OpenAPI 回傳格式不符合預期")

            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("TWSE OpenAPI 沒有回傳資料")

            return df

        except requests.exceptions.RequestException as e:
            print(f"第 {attempt} 次連線失敗：{e}")
            time.sleep(3)

    raise ValueError("TWSE OpenAPI 連線失敗，無法取得當日成交金額排行")


def get_twse_top_turnover_cache_path(limit: int) -> Path:
    return TOP_TURNOVER_CACHE_DIR / f"twse_top_turnover_{int(limit)}_latest.csv"


def normalize_top_turnover_stock_list(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "stock_name", "today_turnover"])

    work = df.copy()
    if "symbol" not in work.columns:
        code_col = "Code" if "Code" in work.columns else "code" if "code" in work.columns else None
        if code_col is None:
            return pd.DataFrame(columns=["symbol", "stock_name", "today_turnover"])
        work["symbol"] = work[code_col]

    if "stock_name" not in work.columns:
        if "name" in work.columns:
            work["stock_name"] = work["name"]
        elif "Name" in work.columns:
            work["stock_name"] = work["Name"]
        else:
            work["stock_name"] = ""

    if "today_turnover" not in work.columns:
        if "turnover" in work.columns:
            work["today_turnover"] = work["turnover"]
        elif "today_turnover" not in work.columns:
            work["today_turnover"] = np.nan

    work["symbol"] = work["symbol"].astype(str).str.strip()
    work["stock_name"] = work["stock_name"].fillna("").astype(str).str.strip()
    work["today_turnover"] = pd.to_numeric(work["today_turnover"], errors="coerce")
    work = work[work["symbol"].str.fullmatch(r"[1-9]\d{3}", na=False)]
    work = work.drop_duplicates(subset=["symbol"], keep="first")
    if "today_turnover" in work.columns and work["today_turnover"].notna().any():
        work = work.sort_values("today_turnover", ascending=False, na_position="last")
    return work.head(limit)[["symbol", "stock_name", "today_turnover"]].reset_index(drop=True)


def write_twse_top_turnover_cache(stock_list: pd.DataFrame, limit: int, source_date=None) -> Path:
    cache_path = get_twse_top_turnover_cache_path(limit)
    out = normalize_top_turnover_stock_list(stock_list, limit=limit)
    if out.empty:
        return cache_path
    cached_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    out["name"] = out["stock_name"]
    out["turnover"] = out["today_turnover"]
    out["source_date"] = source_date if source_date is not None else pd.Timestamp.today().strftime("%Y-%m-%d")
    out["cached_at"] = cached_at
    cache_cols = ["symbol", "name", "turnover", "source_date", "cached_at", "stock_name", "today_turnover"]
    out[cache_cols].to_csv(cache_path, index=False, encoding="utf-8-sig")
    return cache_path


def read_twse_top_turnover_cache(limit: int) -> pd.DataFrame:
    exact_path = get_twse_top_turnover_cache_path(limit)
    candidates = [exact_path] if exact_path.exists() else []
    candidates.extend(
        path for path in sorted(TOP_TURNOVER_CACHE_DIR.glob("twse_top_turnover_*_latest.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if path != exact_path
    )
    for path in candidates:
        try:
            cached = pd.read_csv(path)
            stock_list = normalize_top_turnover_stock_list(cached, limit=limit)
            if not stock_list.empty:
                return stock_list
        except Exception as exc:
            print(f"讀取 Top turnover 快取失敗，略過：{path}；原因：{exc}")
    return pd.DataFrame(columns=["symbol", "stock_name", "today_turnover"])


def read_latest_ranking_symbols_as_stock_list(limit: int) -> pd.DataFrame:
    candidates = sorted(OUTPUT_DIR.glob("*_ranking.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            ranking_df = pd.read_csv(path)
            stock_list = normalize_top_turnover_stock_list(ranking_df, limit=limit)
            if not stock_list.empty:
                print(f"已從最新 ranking CSV 取得股票清單：{path}")
                return stock_list
        except Exception as exc:
            print(f"讀取 ranking CSV 備援失敗，略過：{path}；原因：{exc}")
    return pd.DataFrame(columns=["symbol", "stock_name", "today_turnover"])


def fetch_twse_top_turnover_stocks(limit: int = 30) -> pd.DataFrame:
    print(f"抓取 TWSE 當日成交金額排行前 {limit} 檔...")
    try:
        df = fetch_twse_stock_day_all()

        code_col = "Code" if "Code" in df.columns else "code" if "code" in df.columns else None
        name_col = "Name" if "Name" in df.columns else "name" if "name" in df.columns else None

        turnover_candidates = ["TradeValue", "trade_value", "成交金額", "Turnover", "turnover", "TradeAmount", "trade_amount"]
        turnover_col = next((col for col in turnover_candidates if col in df.columns), None)

        if code_col is None:
            raise ValueError(f"找不到股票代碼欄位，目前欄位有：{df.columns.tolist()}")
        if turnover_col is None:
            raise ValueError(f"找不到成交金額欄位，目前欄位有：{df.columns.tolist()}")

        work = df.copy()
        work["symbol"] = work[code_col].astype(str).str.strip()
        work["stock_name"] = work[name_col].astype(str).str.strip() if name_col else ""
        work["today_turnover"] = work[turnover_col].apply(clean_numeric)
        work["today_turnover"] = pd.to_numeric(work["today_turnover"], errors="coerce")

        work = work[work["symbol"].str.fullmatch(r"[1-9]\d{3}", na=False)]
        work = work.dropna(subset=["today_turnover"])
        work = work[work["today_turnover"] > 0]
        work = work.sort_values("today_turnover", ascending=False).head(limit)
        work = work[["symbol", "stock_name", "today_turnover"]].reset_index(drop=True)

        if work.empty:
            raise ValueError("沒有篩出任何 TWSE 上市普通股成交金額資料")

        write_twse_top_turnover_cache(work, limit=limit)

        print("今日成交金額排行：")
        for idx, row in work.iterrows():
            print(f"{idx + 1:>2}. {row['symbol']} {row['stock_name']} 成交金額 {int(row['today_turnover']):,}")

        return work
    except Exception as exc:
        print(f"TWSE OpenAPI 成交金額排行取得失敗：{exc}")

    cached_stock_list = read_twse_top_turnover_cache(limit=limit)
    if not cached_stock_list.empty:
        print("TWSE OpenAPI 連線失敗，改用最近一次成交金額排行快取")
        return cached_stock_list

    ranking_stock_list = read_latest_ranking_symbols_as_stock_list(limit=limit)
    if not ranking_stock_list.empty:
        print("TWSE OpenAPI 連線失敗，改用最新 ranking CSV 股票清單")
        return ranking_stock_list

    raise ValueError("無法取得成交金額排行，且沒有可用快取；請先執行一次 predict 或檢查網路連線。")


def fetch_twse_top_volume_stocks(limit: int = 30) -> pd.DataFrame:
    return fetch_twse_top_turnover_stocks(limit=limit)

def standardize_columns(df: pd.DataFrame, stock_no: str, market: str) -> pd.DataFrame:
    df = df.copy().rename(columns=COLUMN_RENAME_MAP)

    if "date" not in df.columns:
        raise ValueError("找不到日期欄位，請檢查資料來源格式是否改變。")

    df["date"] = df["date"].apply(roc_to_ad_date)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    df["symbol"] = stock_no
    df["market"] = market.upper()

    if "volume" in df.columns:
        df["volume_lots"] = df["volume"] / 1000

    return df.sort_values("date").reset_index(drop=True)


def clean_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """保留第一個同名欄位，避免 merge 或 concat 後欄位重複。"""
    return df.loc[:, ~df.columns.duplicated()].copy()


def get_raw_cache_path(market: str, stock_no: str) -> Path:
    return RAW_DATA_DIR / f"{market.upper()}_{stock_no}_raw.csv"


def read_raw_cache(cache_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_path)
    if "date" not in df.columns:
        raise ValueError(f"快取檔缺少 date 欄位：{cache_path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return clean_price_dataframe(df)


def write_raw_cache(df: pd.DataFrame, cache_path: Path) -> None:
    out = df.copy()
    out = out.dropna(subset=["date"])
    if "symbol" in out.columns:
        out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    else:
        out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")


def filter_to_recent_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    cutoff = pd.Timestamp(datetime.today().date()) - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)


def fetch_trading_data_months(
    stock_no: str,
    market: str,
    months_to_fetch: int,
    start_from_month: Optional[datetime] = None,
    only_after_date: Optional[pd.Timestamp] = None,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    today = datetime.today()
    monthly_frames: list[pd.DataFrame] = []

    for i in range(months_to_fetch):
        target_month = shift_month(start_from_month or today, i)
        try:
            raw_df = fetch_month_data(stock_no, market, target_month)
            std_df = standardize_columns(raw_df, stock_no, market)
            if only_after_date is not None:
                std_df = std_df[std_df["date"] > only_after_date]
            if not std_df.empty:
                monthly_frames.append(std_df)
            print(f"成功抓取 {target_month.strftime('%Y-%m')}，新增筆數：{len(std_df)}")
        except Exception as e:
            print(f"抓取 {target_month.strftime('%Y-%m')} 失敗：{e}")

        if sleep_sec > 0:
            time.sleep(sleep_sec + random.uniform(0, 0.4))

    if not monthly_frames:
        return pd.DataFrame()

    return clean_price_dataframe(pd.concat(monthly_frames, ignore_index=True))


def get_last_n_years_trading_data(config: PipelineConfig) -> pd.DataFrame:
    validate_inputs(config.stock_no, config.market)

    cache_path = get_raw_cache_path(config.market, config.stock_no)
    months_to_fetch = config.years * 12

    if cache_path.exists():
        print("\n使用本地快取資料：")
        print(cache_path)
        cached_df = read_raw_cache(cache_path)
        cached_df["symbol"] = config.stock_no
        cached_df["market"] = config.market.upper()

        if cached_df.empty:
            last_date = None
            print("本地快取為空，改抓完整歷史資料。")
        else:
            last_date = pd.to_datetime(cached_df["date"]).max()
            print(f"本地資料最後日期：{last_date.strftime('%Y-%m-%d')}")
            if last_date == pd.Timestamp.today().normalize():
                print(f'[CACHE HIT] {config.stock_no} 今日資料已是最新，跳過網路請求')
                return filter_to_recent_years(cached_df, config.years)

        if last_date is None:
            new_df = fetch_trading_data_months(
                stock_no=config.stock_no,
                market=config.market,
                months_to_fetch=months_to_fetch,
                sleep_sec=config.sleep_sec,
            )
        else:
            current_month = datetime.today().replace(day=1)
            last_month = pd.Timestamp(last_date).to_pydatetime().replace(day=1)
            months_after_last = (current_month.year - last_month.year) * 12 + current_month.month - last_month.month + 1
            new_df = fetch_trading_data_months(
                stock_no=config.stock_no,
                market=config.market,
                months_to_fetch=max(1, months_after_last),
                start_from_month=current_month,
                only_after_date=last_date,
                sleep_sec=config.sleep_sec,
            )

        print(f"本次新增資料筆數：{len(new_df)}")
        combined_df = pd.concat([cached_df, new_df], ignore_index=True) if not new_df.empty else cached_df
        combined_df = clean_price_dataframe(combined_df)
        combined_df["symbol"] = config.stock_no
        combined_df["market"] = config.market.upper()
        write_raw_cache(combined_df, cache_path)
        print(f"更新後總資料筆數：{len(combined_df)}")
        return filter_to_recent_years(combined_df, config.years)

    print("\n找不到本地快取，開始抓完整歷史資料：")
    print(cache_path)
    df_all = fetch_trading_data_months(
        stock_no=config.stock_no,
        market=config.market,
        months_to_fetch=months_to_fetch,
        sleep_sec=config.sleep_sec,
    )

    if df_all.empty:
        raise ValueError("沒有成功抓到任何資料，請檢查股票代碼或市場設定。")

    df_all["symbol"] = config.stock_no
    df_all["market"] = config.market.upper()
    write_raw_cache(df_all, cache_path)
    print("已建立本地快取資料。")
    print(f"本次新增資料筆數：{len(df_all)}")
    print(f"更新後總資料筆數：{len(df_all)}")
    return filter_to_recent_years(df_all, config.years)


def parse_market_index_date(value) -> pd.Timestamp:
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "nat", "none"}:
        return pd.NaT
    text = text.replace("-", "/")
    parts = text.split("/")
    if len(parts) == 3:
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            if year < 1911:
                year += 1911
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return pd.to_datetime(value, errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def get_market_index_cache_path(market: str) -> Path:
    index_symbol = "TAIEX" if market.upper() == "TWSE" else "MARKET_INDEX"
    return RAW_DATA_DIR / f"{market.upper()}_{index_symbol}_raw.csv"


def standardize_market_index_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "market_close"])

    work = df.copy()
    date_col = "date" if "date" in work.columns else None
    if date_col is None:
        date_col = next(
            (
                col for col in work.columns
                if "date" in str(col).lower() or "\u65e5\u671f" in str(col)
            ),
            work.columns[0],
        )

    close_candidates = [
        "market_close",
        "close",
        "Close",
        "\u6536\u76e4\u6307\u6578",
        "\u6536\u76e4\u50f9",
    ]
    close_col = next((col for col in close_candidates if col in work.columns), None)
    if close_col is None:
        close_col = next(
            (col for col in work.columns if "\u6536\u76e4" in str(col)),
            None,
        )
    if close_col is None:
        non_date_cols = [col for col in work.columns if col != date_col]
        close_col = non_date_cols[-1] if non_date_cols else date_col

    out = pd.DataFrame({
        "date": work[date_col].apply(parse_market_index_date),
        "market_close": work[close_col].apply(clean_numeric),
    })
    out["market_close"] = pd.to_numeric(out["market_close"], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def read_market_index_cache(cache_path: Path) -> pd.DataFrame:
    return standardize_market_index_data(pd.read_csv(cache_path))


def write_market_index_cache(df: pd.DataFrame, cache_path: Path) -> None:
    out = standardize_market_index_data(df)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")


def fetch_twse_taiex_month_raw(dt: datetime) -> pd.DataFrame:
    date_yyyymm01 = get_month_start_str(dt)
    last_error = None
    for url in TWSE_TAIEX_HIST_URLS:
        try:
            response = SESSION.get(
                url,
                params={"response": "json", "date": date_yyyymm01},
                timeout=30,
                verify=False,
            )
            response.raise_for_status()
            data = response.json()
            rows = data.get("data") if isinstance(data, dict) else None
            fields = data.get("fields") if isinstance(data, dict) else None
            if rows:
                return pd.DataFrame(rows, columns=fields)
            last_error = ValueError(
                f"empty TAIEX payload: {data.get('stat') if isinstance(data, dict) else type(data)}"
            )
        except Exception as e:
            last_error = e
            continue
    raise ValueError(f"TAIEX fetch failed for {date_yyyymm01}: {last_error}")


def fetch_market_index_months(
    market: str,
    months_to_fetch: int,
    start_from_month: Optional[datetime] = None,
    only_after_date: Optional[pd.Timestamp] = None,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    if market.upper() != "TWSE":
        print(f"WARNING: market index features currently support TWSE only, got market={market}")
        return pd.DataFrame(columns=["date", "market_close"])

    today = datetime.today()
    monthly_frames: list[pd.DataFrame] = []
    for i in range(months_to_fetch):
        target_month = shift_month(start_from_month or today, i)
        try:
            raw_df = fetch_twse_taiex_month_raw(target_month)
            std_df = standardize_market_index_data(raw_df)
            if only_after_date is not None:
                std_df = std_df[std_df["date"] > only_after_date]
            if not std_df.empty:
                monthly_frames.append(std_df)
        except Exception as e:
            print(f"WARNING: TAIEX market index fetch failed for {target_month.strftime('%Y-%m')}: {e}")
        if sleep_sec > 0:
            time.sleep(sleep_sec + random.uniform(0, 0.2))

    if not monthly_frames:
        return pd.DataFrame(columns=["date", "market_close"])
    return standardize_market_index_data(pd.concat(monthly_frames, ignore_index=True))


def get_market_index_data(config: PipelineConfig) -> pd.DataFrame:
    if config.market.upper() != "TWSE":
        print(f"WARNING: no market index cache/fetch flow for market={config.market}; market features will be NaN")
        return pd.DataFrame(columns=["date", "market_close"])

    cache_path = get_market_index_cache_path(config.market)
    months_to_fetch = config.years * 12
    cached_df = pd.DataFrame(columns=["date", "market_close"])

    if cache_path.exists():
        try:
            cached_df = read_market_index_cache(cache_path)
        except Exception as e:
            print(f"WARNING: failed to read market index cache {cache_path}: {e}")

    try:
        if cached_df.empty:
            new_df = fetch_market_index_months(
                market=config.market,
                months_to_fetch=months_to_fetch,
                sleep_sec=config.sleep_sec,
            )
            if not new_df.empty:
                write_market_index_cache(new_df, cache_path)
                cached_df = new_df
        else:
            last_date = pd.to_datetime(cached_df["date"]).max()
            if last_date == pd.Timestamp.today().normalize():
                print(f'[CACHE HIT] {config.market}_INDEX 今日資料已是最新，跳過網路請求')
                return filter_to_recent_years(standardize_market_index_data(cached_df), config.years)
            current_month = datetime.today().replace(day=1)
            last_month = pd.Timestamp(last_date).to_pydatetime().replace(day=1)
            months_after_last = (
                (current_month.year - last_month.year) * 12
                + current_month.month
                - last_month.month
                + 1
            )
            new_df = fetch_market_index_months(
                market=config.market,
                months_to_fetch=max(1, months_after_last),
                start_from_month=current_month,
                only_after_date=last_date,
                sleep_sec=config.sleep_sec,
            )
            if not new_df.empty:
                cached_df = standardize_market_index_data(
                    pd.concat([cached_df, new_df], ignore_index=True)
                )
                write_market_index_cache(cached_df, cache_path)
    except Exception as e:
        print(f"WARNING: market index update failed; using available cache only: {e}")

    if cached_df.empty:
        print("WARNING: market index data unavailable; market environment features will be NaN")
        return cached_df

    return filter_to_recent_years(standardize_market_index_data(cached_df), config.years)


__all__ = [name for name in globals() if not name.startswith('__')]
