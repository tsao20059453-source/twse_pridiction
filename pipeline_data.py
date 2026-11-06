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
from datetime import datetime, timedelta
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
    sleep_sec: float = 2.0
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
    # Kept configurable so walk-forward score optimization can evaluate an
    # entry threshold without changing the execution, cost, or exit rules.
    backtest_entry_threshold: float = 0.55
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
SHARED_RAW_CACHE_DIR: Optional[Path] = None
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


# ============================================================
# 上櫃（TPEx）資料（2026-10-26 新增）
# ------------------------------------------------------------
# 個股歷史：FinMind TaiwanStockPrice（與 ETF 專案相同的資料集，上市上櫃皆可，一檔一次請求取得整段期間）。
# 上櫃成交金額排行：櫃買中心 dailyQuotes（ETF 專案每日使用的端點）。
# 市場判別：股票池檔案的 market 欄 → 本機快取檔名（TWSE_/TPEX_）→ data/cache/market_map.csv
#           → FinMind TaiwanStockInfo（一次取得全部代號的上市／上櫃別）。都沒有時沿用預設市場。
# 大盤過濾仍使用加權指數（TAIEX）。
# ============================================================
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
TPEX_DAILY_QUOTES_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
SYMBOL_MARKET: dict = {}
OTC_SETTINGS: dict = {"include_otc": False}


def set_include_otc(enabled: bool = True) -> None:
    OTC_SETTINGS["include_otc"] = bool(enabled)


def _load_env_once() -> None:
    if os.environ.get("_PIPELINE_ENV_LOADED"):
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ["_PIPELINE_ENV_LOADED"] = "1"


def finmind_request(dataset: str, **params) -> pd.DataFrame:
    """FinMind API（token 從 .env 的 FINMIND_TOKEN 讀取，不會印出）。"""
    _load_env_once()
    query = {"dataset": dataset, **{k: v for k, v in params.items() if v is not None}}
    token = os.environ.get("FINMIND_TOKEN", "")
    if token:
        query["token"] = token
    last_error = None
    for attempt in range(1, 4):
        try:
            response = SESSION.get(FINMIND_API_URL, params=query, timeout=60)
            payload = response.json()
            status = payload.get("status")
            if status == 402:
                raise ValueError("FinMind 額度用完（未登入每小時 300 次），請稍後再試或在 .env 設定 FINMIND_TOKEN")
            if status not in (200, None):
                raise ValueError(f"FinMind 回傳 status={status}：{payload.get('msg')}")
            return pd.DataFrame(payload.get("data", []))
        except ValueError:
            raise
        except Exception as exc:                          # 網路錯誤：重試
            last_error = exc
            time.sleep(3 * attempt)
    raise ValueError(f"FinMind 連線失敗：{last_error}")


def fetch_finmind_price_range(stock_no: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """個股日線（FinMind，上市上櫃皆可），輸出欄位與證交所月資料標準化後相同。"""
    raw = finmind_request("TaiwanStockPrice", data_id=stock_no, start_date=start_date, end_date=end_date)
    if raw.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], errors="coerce"),
        "volume": pd.to_numeric(raw["Trading_Volume"], errors="coerce"),
        "turnover": pd.to_numeric(raw["Trading_money"], errors="coerce"),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["max"], errors="coerce"),
        "low": pd.to_numeric(raw["min"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        "change": pd.to_numeric(raw.get("spread"), errors="coerce"),
        "transactions": pd.to_numeric(raw["Trading_turnover"], errors="coerce"),
    })
    for col in ("open", "high", "low", "close"):              # 無成交日 FinMind 給 0 價格，視為缺值並剔除
        df.loc[df[col] <= 0, col] = np.nan
    df = df.dropna(subset=["date", "close"])
    df["volume_lots"] = df["volume"] / 1000
    return df.sort_values("date").reset_index(drop=True)


# ---- 股票名稱（2026-10-27）：顯示時一律寫成「名稱 代號」，例如「台積電 2330」 ----
# 來源：data/cache/stock_names.csv（證交所＋櫃買中心收盤資料的代號與名稱）；查不到時（線上執行）
# 再從 FinMind TaiwanStockInfo 補齊並寫回同一個檔案。
NAME_SETTINGS: dict = {"allow_fetch": False}
_NAME_CACHE: dict = {}


def _stock_names_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "cache" / "stock_names.csv"


def _load_stock_names() -> dict:
    if "_loaded" not in _NAME_CACHE:
        _NAME_CACHE["_loaded"] = True
        _NAME_CACHE["names"], _NAME_CACHE["markets"] = {}, {}
        path = _stock_names_path()
        if path.exists():
            try:
                t = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
                _NAME_CACHE["names"] = dict(zip(t["symbol"].str.strip(), t["stock_name"].str.strip()))
                if "market" in t.columns:
                    _NAME_CACHE["markets"] = {k: v for k, v in zip(t["symbol"].str.strip(), t["market"].str.strip().str.upper()) if v}
            except Exception as exc:
                print(f"WARNING: 股票名稱對照表讀取失敗：{exc}")
    return _NAME_CACHE["names"]


def _save_stock_names(new_rows: dict, source: str) -> None:
    path = _stock_names_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("") if path.exists() else \
        pd.DataFrame(columns=["symbol", "stock_name", "market", "source"])
    add = pd.DataFrame([{"symbol": k, "stock_name": n, "market": m, "source": source} for k, (n, m) in new_rows.items()])
    merged = pd.concat([old, add], ignore_index=True).drop_duplicates("symbol", keep="last").sort_values("symbol")
    merged.to_csv(path, index=False, encoding="utf-8-sig")


def get_stock_name(symbol) -> str:
    sym = str(symbol).strip()
    names = _load_stock_names()
    name = names.get(sym, "")
    if not name and NAME_SETTINGS.get("allow_fetch") and not is_offline_as_of_mode() and not _NAME_CACHE.get("fetched"):
        _NAME_CACHE["fetched"] = True
        try:
            refresh_market_map()
            name = _load_stock_names().get(sym, "")
        except Exception as exc:
            print(f"WARNING: 無法取得股票名稱（{exc}）")
    return name.rstrip("*").strip()


def stock_label(symbol) -> str:
    """「名稱 代號」，查不到名稱時只顯示代號。"""
    sym = str(symbol).strip()
    name = get_stock_name(sym)
    return f"{name} {sym}" if name else sym


def _market_map_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "cache" / "market_map.csv"


def register_symbol_markets(mapping: dict) -> None:
    for sym, mkt in (mapping or {}).items():
        mkt = str(mkt).strip().upper()
        if mkt in {"TWSE", "TPEX"}:
            SYMBOL_MARKET[str(sym).strip()] = mkt


def _read_market_map() -> dict:
    path = _market_map_path()
    if not path.exists():
        return {}
    try:
        m = pd.read_csv(path, dtype=str)
        return dict(zip(m["symbol"].str.strip(), m["market"].str.strip().str.upper()))
    except Exception:
        return {}


def refresh_market_map() -> dict:
    """FinMind TaiwanStockInfo：全部代號的上市（twse）／上櫃（tpex）別，存到 data/cache/market_map.csv。"""
    info = finmind_request("TaiwanStockInfo")
    if info.empty or "type" not in info.columns:
        return {}
    info = info[info["type"].isin(["twse", "tpex"])]
    m = (info.assign(symbol=info["stock_id"].astype(str).str.strip(), market=info["type"].str.upper())
             [["symbol", "market"]].drop_duplicates("symbol", keep="last"))
    if "stock_name" in info.columns:                      # 同時補齊股票名稱對照表
        try:
            names = {str(r.stock_id).strip(): (str(r.stock_name).strip(), str(r.type).upper())
                     for r in info.itertuples() if str(r.stock_id).strip()}
            known = _load_stock_names()
            _save_stock_names({k: v for k, v in names.items() if not known.get(k)}, "finmind")
            _NAME_CACHE.pop("_loaded", None)
        except Exception as exc:
            print(f"WARNING: 股票名稱對照表更新失敗：{exc}")
    path = _market_map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已更新上市／上櫃對照表：{path}（{len(m)} 檔）")
    return dict(zip(m["symbol"], m["market"]))


def lookup_symbol_market(stock_no: str, default: str = "TWSE") -> str:
    sym = str(stock_no).strip()
    if sym in SYMBOL_MARKET:
        return SYMBOL_MARKET[sym]
    twse_cache = get_raw_cache_path("TWSE", sym).exists()
    tpex_cache = get_raw_cache_path("TPEX", sym).exists()
    if twse_cache != tpex_cache:
        SYMBOL_MARKET[sym] = "TWSE" if twse_cache else "TPEX"
        return SYMBOL_MARKET[sym]
    _load_stock_names()
    if sym in _NAME_CACHE.get("markets", {}):               # 名稱對照表也記錄了上市／上櫃別
        SYMBOL_MARKET[sym] = _NAME_CACHE["markets"][sym]
        return SYMBOL_MARKET[sym]
    mapping = _read_market_map()
    if sym not in mapping and not is_offline_as_of_mode():
        try:
            mapping = refresh_market_map()
        except Exception as exc:
            print(f"WARNING: 無法判別 {sym} 的上市／上櫃別（{exc}），沿用 {default}")
    if sym in mapping:
        SYMBOL_MARKET[sym] = mapping[sym]
        return mapping[sym]
    return default.upper()


def fetch_tpex_daily_quotes(target: datetime) -> pd.DataFrame:
    """櫃買中心單日全部上櫃個股收盤（代號、名稱、成交金額）。查無資料（假日）回傳空表。"""
    params = {"date": target.strftime("%Y/%m/%d"), "type": "EW", "response": "json"}
    response = SESSION.get(TPEX_DAILY_QUOTES_URL, params=params, timeout=30, verify=False)
    response.raise_for_status()
    payload = response.json()
    for table in payload.get("tables") or []:
        fields = [str(f).strip() for f in table.get("fields", [])]
        if {"代號", "名稱", "成交金額(元)"}.issubset(fields) and table.get("data"):
            df = pd.DataFrame(table["data"], columns=fields)
            return pd.DataFrame({
                "symbol": df["代號"].astype(str).str.strip(),
                "stock_name": df["名稱"].astype(str).str.strip(),
                "today_turnover": pd.to_numeric(df["成交金額(元)"].astype(str).str.replace(",", ""), errors="coerce"),
                "market": "TPEX",
            })
    return pd.DataFrame()


def fetch_tpex_top_turnover_stocks(limit: int = 30, max_lookback_days: int = 7) -> pd.DataFrame:
    today = datetime.today()
    for back in range(max_lookback_days + 1):
        day = today - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        try:
            df = fetch_tpex_daily_quotes(day)
        except Exception as exc:
            print(f"櫃買中心 {day.date()} 收盤資料取得失敗：{exc}")
            continue
        if not df.empty:
            df = df[df["symbol"].str.fullmatch(r"[1-9]\d{3}", na=False)].dropna(subset=["today_turnover"])
            df = df[df["today_turnover"] > 0].sort_values("today_turnover", ascending=False).head(limit)
            print(f"櫃買中心成交金額資料日期：{day.date()}")
            return df.reset_index(drop=True)
    return pd.DataFrame(columns=["symbol", "stock_name", "today_turnover", "market"])


def fetch_month_data(stock_no: str, market: str, dt: datetime) -> pd.DataFrame:
    market = market.upper()
    if market == "TWSE":
        return fetch_twse_month_raw(stock_no, dt)
    if market == "TPEX":
        start = dt.replace(day=1)
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        return fetch_finmind_price_range(stock_no, start.strftime("%Y-%m-%d"), end)
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


# ============================================================
# 可重現模式（2026-10-25 新增）
#   --pool-csv：固定股票池，不再抓「執行當天」的成交金額排行
#   --as-of   ：把指定日期視為「今天」——只用到這天為止的快取資料、
#               不上網補資料、不寫回快取，近 N 年區間也以這天為基準
#   兩個參數都不給時，行為與原本完全相同。
# ============================================================
REPRO_SETTINGS: dict = {"pool_csv": None, "as_of": None}


def set_reproducible_mode(pool_csv: Optional[str] = None, as_of: Optional[str] = None) -> None:
    REPRO_SETTINGS["pool_csv"] = None
    REPRO_SETTINGS["as_of"] = None
    if pool_csv:
        pool_path = Path(pool_csv)
        if not pool_path.exists():
            raise FileNotFoundError(f"找不到固定股票池檔案：{pool_path}")
        REPRO_SETTINGS["pool_csv"] = str(pool_path)
    if as_of:
        REPRO_SETTINGS["as_of"] = pd.Timestamp(str(as_of)[:10]).normalize()
    if REPRO_SETTINGS["pool_csv"] or REPRO_SETTINGS["as_of"] is not None:
        print("=" * 70)
        print("可重現模式：開啟")
        print(f"  固定股票池：{REPRO_SETTINGS['pool_csv'] or '（未指定，沿用當日成交金額排行）'}")
        as_of_text = REPRO_SETTINGS['as_of'].strftime('%Y-%m-%d') if REPRO_SETTINGS['as_of'] is not None else '（未指定，以今天為準）'
        print(f"  資料基準日：{as_of_text}")
        if REPRO_SETTINGS["as_of"] is not None:
            print("  僅使用本地快取：不上網補資料、不寫回快取")
        print("=" * 70)


def set_shared_raw_cache_dir(path: Optional[str] = None) -> None:
    """Configure a read-only raw-data cache for reproducible replay runs."""
    global SHARED_RAW_CACHE_DIR
    if path is None:
        SHARED_RAW_CACHE_DIR = None
        return
    cache_dir = Path(path).expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"共享原始快取目錄不存在：{cache_dir}")
    SHARED_RAW_CACHE_DIR = cache_dir
    print(f"[可重現模式] 使用唯讀共享原始快取：{SHARED_RAW_CACHE_DIR}")


def get_reference_today() -> pd.Timestamp:
    """回傳程式視為「今天」的日期：可重現模式下為 --as-of，否則為系統日期。"""
    if REPRO_SETTINGS.get("as_of") is not None:
        return REPRO_SETTINGS["as_of"]
    return pd.Timestamp(datetime.today().date())


def is_offline_as_of_mode() -> bool:
    return REPRO_SETTINGS.get("as_of") is not None


def _clip_to_as_of(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns or not is_offline_as_of_mode():
        return df
    dates = pd.to_datetime(df["date"])
    return df[dates <= REPRO_SETTINGS["as_of"]].copy()


def read_fixed_stock_pool(limit: int) -> pd.DataFrame:
    pool_path = Path(REPRO_SETTINGS["pool_csv"])
    raw = pd.read_csv(pool_path, dtype={"symbol": str, "Code": str, "code": str})
    stock_list = normalize_top_turnover_stock_list(raw, limit=limit)
    if stock_list.empty:
        raise ValueError(f"固定股票池檔案沒有可用的股票代碼：{pool_path}")
    print(f"使用固定股票池：{pool_path}（{len(stock_list)} 檔）")
    for idx, row in stock_list.iterrows():
        print(f"{idx + 1:>2}. {row['stock_name']} {row['symbol']}")
    return stock_list


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
    names = _load_stock_names()                                     # 空白名稱以對照表補齊
    empty = work["stock_name"].isin(["", "nan", "None"])
    work.loc[empty, "stock_name"] = work.loc[empty, "symbol"].map(lambda s: names.get(s, "").rstrip("*"))
    work["today_turnover"] = pd.to_numeric(work["today_turnover"], errors="coerce")
    work = work[work["symbol"].str.fullmatch(r"[1-9]\d{3}", na=False)]
    work = work.drop_duplicates(subset=["symbol"], keep="first")
    if "today_turnover" in work.columns and work["today_turnover"].notna().any():
        work = work.sort_values("today_turnover", ascending=False, na_position="last")
    cols = ["symbol", "stock_name", "today_turnover"]
    if "market" in work.columns:                       # 含上櫃的股票池：保留並登記上市／上櫃別
        work["market"] = work["market"].fillna("TWSE").astype(str).str.upper()
        register_symbol_markets(dict(zip(work["symbol"], work["market"])))
        cols.append("market")
    return work.head(limit)[cols].reset_index(drop=True)


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
    if REPRO_SETTINGS.get("pool_csv"):
        return read_fixed_stock_pool(limit=limit)
    if OTC_SETTINGS.get("include_otc"):
        return fetch_combined_top_turnover_stocks(limit=limit)
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
            print(f"{idx + 1:>2}. {row['stock_name']} {row['symbol']} 成交金額 {int(row['today_turnover']):,}")

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


def fetch_combined_top_turnover_stocks(limit: int = 30) -> pd.DataFrame:
    """上市＋上櫃合併，依當日成交金額取前 N 檔（--include-otc）。"""
    OTC_SETTINGS["include_otc"] = False                  # 避免遞迴：先取上市排行
    try:
        twse = fetch_twse_top_turnover_stocks(limit=limit).copy()
    finally:
        OTC_SETTINGS["include_otc"] = True
    twse["market"] = "TWSE"
    print(f"抓取櫃買中心當日成交金額排行前 {limit} 檔...")
    tpex = fetch_tpex_top_turnover_stocks(limit=limit)
    if tpex.empty:
        print("WARNING: 櫃買中心資料取得失敗，本次只使用上市股票")
    combined = pd.concat([twse, tpex], ignore_index=True)
    combined = combined.sort_values("today_turnover", ascending=False).drop_duplicates("symbol").head(limit)
    combined = combined.reset_index(drop=True)
    register_symbol_markets(dict(zip(combined["symbol"], combined["market"])))
    cache_dir = TOP_TURNOVER_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"twse_tpex_top_turnover_{limit}_{datetime.today().strftime('%Y%m%d')}.csv"
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    print("上市＋上櫃成交金額排行：")
    for idx, row in combined.iterrows():
        tag = "上櫃" if row["market"] == "TPEX" else "上市"
        print(f"{idx + 1:>2}. {row['stock_name']} {row['symbol']}（{tag}）成交金額 {int(row['today_turnover']):,}")
    print(f"已存成股票池檔案（可用 --pool-csv 重現）：{path}")
    return combined


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
    filename = f"{market.upper()}_{stock_no}_raw.csv"
    local_path = RAW_DATA_DIR / filename
    if is_offline_as_of_mode() and SHARED_RAW_CACHE_DIR is not None:
        shared_path = SHARED_RAW_CACHE_DIR / filename
        if shared_path.exists():
            return shared_path
    return local_path


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


# ============================================================
# 除權、面額變更、減資的價格還原（2026-10-26 新增）
# ------------------------------------------------------------
# 證交所提供的是未還原價格。台股漲跌幅 10%，單日變動超過 ±10.5% 必定是公司行為
# （面額變更／股票分割、大額配股、減資），例如 2327 國巨 2025-08-25 一拆四（546 → 143）、
# 6669 緯穎 2026-09-02 除權一張變三張（7800 → 2610）。不還原時，回測會把持有部位記成
# 單日虧損 70% 以上，技術指標也會出現假訊號。
# 做法（同 ETF 專案 esun/data/adjust.py 的公司行為推估）：
#   f_t = 當日開盤 / 前一日收盤（假設參考價 ≈ 開盤價，誤差約為一般的隔夜跳空）
#   向後還原：adj_price(s) = price(s) × Π_{t>s} f_t，最新價格不變；成交量同步除以該倍數。
# 現金股利與小額配股（缺口 < 10%）：data/chip/dividends.csv 有資料時，以 f = 除權息參考價 / 前一日收盤 還原
#   （FinMind 除權息結果，同 ETF 專案）；先套用除權息，剩下仍超過 ±10.5% 的才用開盤價推估。
# ============================================================
PRICE_ADJUST_SETTINGS: dict = {"enabled": True}
PRICE_LIMIT_TOLERANCE = 0.105
_DIVIDEND_CACHE: dict = {}


def _dividend_factors(symbol: str) -> pd.Series:
    """回傳 {除權息日: 調整因子}；沒有資料時為空。"""
    if "table" not in _DIVIDEND_CACHE:
        path = Path(__file__).resolve().parent / "data" / "chip" / "dividends.csv"
        table = pd.DataFrame()
        if path.exists():
            table = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            table["date"] = pd.to_datetime(table["date"], errors="coerce")
            table["f"] = pd.to_numeric(table["after_price"], errors="coerce") / pd.to_numeric(table["before_price"], errors="coerce")
            table = table[(table["f"] > 0.2) & (table["f"] <= 1.0)]
        _DIVIDEND_CACHE["table"] = table
    table = _DIVIDEND_CACHE["table"]
    if table.empty:
        return pd.Series(dtype=float)
    sub = table[table["code"] == str(symbol)]
    return sub.groupby("date")["f"].prod()


def set_price_adjust(enabled: bool = True) -> None:
    PRICE_ADJUST_SETTINGS["enabled"] = bool(enabled)


def _adjust_one_symbol(g: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    dates = pd.to_datetime(g["date"], errors="coerce")
    close = pd.to_numeric(g["close"], errors="coerce")
    open_ = pd.to_numeric(g["open"], errors="coerce")
    prev = close.shift(1)
    sym = str(g["symbol"].iloc[0]) if "symbol" in g.columns else ""
    dividend_factors = _dividend_factors(sym).copy()
    dividend_factors.index = pd.to_datetime(dividend_factors.index, errors="coerce")
    dividend_factors = dividend_factors[
        dividend_factors.index.notna()
        & dividend_factors.index.to_series().between(dates.min(), dates.max())
    ]
    f_div = dates.map(dividend_factors).astype(float).fillna(1.0)
    f_div.index = g.index
    jump = (close / (prev * f_div) - 1).abs() > PRICE_LIMIT_TOLERANCE      # 除權息後仍超過漲跌幅 → 面額變更／減資等
    event = jump & prev.notna() & (prev > 0) & open_.notna() & (open_ > 0)
    event_total_factor = (open_ / prev).clip(0.05, 20.0)
    # event_total_factor already includes any same-day cash dividend. Divide
    # by f_div so a known dividend and an inferred split are not counted twice.
    inferred_event_factors = pd.Series(
        event_total_factor[event].to_numpy() / f_div[event].to_numpy(),
        index=dates[event],
        dtype=float,
    )
    all_factors = pd.concat([dividend_factors, inferred_event_factors])
    all_factors = all_factors.groupby(level=0).prod().sort_index()
    if all_factors.empty:
        after = pd.Series(1.0, index=g.index, dtype=float)
    else:
        # Daily Top-N snapshots can omit a security on its ex-dividend date. Use
        # the event calendar rather than only rows currently present in g, so the
        # adjustment still reaches all earlier retained observations.
        after = pd.Series(
            [float(all_factors.loc[all_factors.index > date].prod()) for date in dates],
            index=g.index,
            dtype=float,
        ).fillna(1.0)                                  # Π_{event date > t} f_t
    g["adj_factor"] = after.astype(float)
    g["corporate_action_flag"] = event.astype(bool)
    g["is_ex_dividend_date"] = (f_div < 1.0) & ~event
    for col in ("open", "high", "low", "close"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce") * g["adj_factor"]
    for col in ("volume", "volume_lots"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce") / g["adj_factor"]
    if verbose:
        n_div = int(g["is_ex_dividend_date"].sum())
        if n_div:
            print(f"[價格還原] {sym} 套用除權息 {n_div} 次（FinMind 除權息參考價）")
        for i in np.flatnonzero(event.to_numpy()):
            print(f"[價格還原] {sym} {pd.Timestamp(g.loc[i, 'date']).date()} 單日 {prev.iloc[i]:.2f} → {close.iloc[i]:.2f}"
                  f"（超過漲跌幅，視為除權／面額變更），之前價格 × {event_total_factor.iloc[i]:.4f}")
    return g


def adjust_for_corporate_actions(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """個股價量資料的公司行為還原；大盤指數（沒有 open 欄位）或關閉時原樣回傳。"""
    if (df is None or df.empty or not PRICE_ADJUST_SETTINGS["enabled"]
            or not {"date", "open", "close"}.issubset(df.columns)):
        return df
    if "symbol" in df.columns and df["symbol"].nunique() > 1:
        parts = [_adjust_one_symbol(g, verbose) for _, g in df.groupby("symbol", sort=False)]
        return pd.concat(parts, ignore_index=True)
    return _adjust_one_symbol(df, verbose)


def filter_to_recent_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    df = _clip_to_as_of(df)
    df = adjust_for_corporate_actions(df)          # 先截到資料基準日再還原，不使用基準日之後的公司行為
    cutoff = get_reference_today() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)


def fetch_trading_data_months(
    stock_no: str,
    market: str,
    months_to_fetch: int,
    start_from_month: Optional[datetime] = None,
    only_after_date: Optional[pd.Timestamp] = None,
    sleep_sec: float = 2.0,
) -> pd.DataFrame:
    today = datetime.today()
    monthly_frames: list[pd.DataFrame] = []

    if market.upper() == "TPEX":
        # 上櫃：FinMind 一次請求取得整段期間（避免逐月請求耗用額度）
        base = start_from_month or today
        start = shift_month(base, max(months_to_fetch - 1, 0))
        if only_after_date is not None:
            start = max(start, (pd.Timestamp(only_after_date) + pd.Timedelta(days=1)).to_pydatetime())
        end = (pd.Timestamp(base.replace(day=1)) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        try:
            df = fetch_finmind_price_range(stock_no, start.strftime("%Y-%m-%d"), end)
        except Exception as e:
            print(f"上櫃股票 {stock_no} 抓取失敗：{e}")
            return pd.DataFrame()
        if only_after_date is not None and not df.empty:
            df = df[df["date"] > pd.Timestamp(only_after_date)]
        if df.empty:
            return pd.DataFrame()
        df["symbol"] = stock_no
        df["market"] = "TPEX"
        print(f"成功抓取上櫃 {stock_no} {start.strftime('%Y-%m-%d')} ~ {end}，新增筆數：{len(df)}")
        return clean_price_dataframe(df)

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
    market = lookup_symbol_market(config.stock_no, config.market)
    if market != config.market.upper():
        print(f"{config.stock_no} 為{'上櫃' if market == 'TPEX' else '上市'}股票，改用 {market} 資料")
        config = replace(config, market=market)

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
            if is_offline_as_of_mode():
                as_of = REPRO_SETTINGS["as_of"]
                if last_date < as_of - pd.Timedelta(days=7):
                    print(f"WARNING: {config.stock_no} 快取只到 {last_date.date()}，早於資料基準日 {as_of.date()}（可重現模式不會上網補資料）")
                print(f"[可重現模式] {config.stock_no} 只使用快取，截至 {as_of.date()}")
                return filter_to_recent_years(cached_df, config.years)
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

    if is_offline_as_of_mode():
        raise FileNotFoundError(
            f"可重現模式找不到 {config.stock_no} 的本地快取：{cache_path}。"
            "請先在一般模式（不加 --as-of）執行一次以建立快取。"
        )

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
    filename = f"{market.upper()}_{index_symbol}_raw.csv"
    local_path = RAW_DATA_DIR / filename
    if is_offline_as_of_mode() and SHARED_RAW_CACHE_DIR is not None:
        shared_path = SHARED_RAW_CACHE_DIR / filename
        if shared_path.exists():
            return shared_path
    return local_path


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
    sleep_sec: float = 2.0,
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

    if is_offline_as_of_mode():
        if cached_df.empty:
            print("WARNING: 可重現模式找不到大盤指數快取；market environment features will be NaN")
            return cached_df
        print(f"[可重現模式] {config.market}_INDEX 只使用快取，截至 {REPRO_SETTINGS['as_of'].date()}")
        return filter_to_recent_years(standardize_market_index_data(cached_df), config.years)

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


def get_historical_trading_data(
    config: PipelineConfig,
    start_date: str,
    end_date: str,
    burn_in_months: int = 12,
) -> pd.DataFrame:
    """
    抓取 [start_date - burn_in_months, end_date] 區間的個股資料。
    不呼叫 filter_to_recent_years()，不受 datetime.today() 影響。
    burn_in_months 讓 RSI14/MACD/ATR/BIAS 在 start_date 前有足夠 lookback。
    """
    validate_inputs(config.stock_no, config.market)
    config = replace(config, market=lookup_symbol_market(config.stock_no, config.market))
    start_dt = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
    end_dt   = datetime.strptime(str(end_date)[:10],   "%Y-%m-%d")
    fetch_start = shift_month(start_dt, burn_in_months)
    total_months = (
        (end_dt.year - fetch_start.year) * 12
        + (end_dt.month - fetch_start.month)
        + 1
    )
    print(
        f"[HIST] {config.stock_no} 抓取範圍：{fetch_start.strftime('%Y-%m')} ~ "
        f"{end_dt.strftime('%Y-%m')}（{total_months} 個月，含 burn-in {burn_in_months} 個月）"
    )
    # fetch_trading_data_months 以 start_from_month 為最新月，往回抓 months_to_fetch 個月
    df = fetch_trading_data_months(
        stock_no=config.stock_no,
        market=config.market,
        months_to_fetch=total_months,
        start_from_month=end_dt,
        sleep_sec=config.sleep_sec,
    )
    if df.empty:
        raise ValueError(
            f"歷史資料抓取失敗：{config.stock_no} {start_date}~{end_date}（含 burn-in）"
        )
    df["symbol"] = config.stock_no
    df["market"] = config.market.upper()
    end_ts = pd.Timestamp(end_dt)
    df = df[df["date"] <= end_ts].sort_values("date").reset_index(drop=True)
    return adjust_for_corporate_actions(df)


def get_historical_market_index_data(
    config: PipelineConfig,
    start_date: str,
    end_date: str,
    burn_in_months: int = 12,
) -> pd.DataFrame:
    """
    抓取 [start_date - burn_in_months, end_date] 區間的 TAIEX 大盤資料。
    不受 datetime.today() 影響。
    """
    if config.market.upper() != "TWSE":
        print(f"WARNING: 歷史市場指數目前只支援 TWSE，market={config.market}；回傳空 DataFrame")
        return pd.DataFrame(columns=["date", "market_close"])
    start_dt = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
    end_dt   = datetime.strptime(str(end_date)[:10],   "%Y-%m-%d")
    fetch_start = shift_month(start_dt, burn_in_months)
    total_months = (
        (end_dt.year - fetch_start.year) * 12
        + (end_dt.month - fetch_start.month)
        + 1
    )
    print(
        f"[HIST] TAIEX 抓取範圍：{fetch_start.strftime('%Y-%m')} ~ "
        f"{end_dt.strftime('%Y-%m')}（{total_months} 個月，含 burn-in {burn_in_months} 個月）"
    )
    # fetch_market_index_months 以 start_from_month 為最新月，往回抓 months_to_fetch 個月
    df = fetch_market_index_months(
        market=config.market,
        months_to_fetch=total_months,
        start_from_month=end_dt,
        sleep_sec=config.sleep_sec,
    )
    if df.empty:
        print(f"WARNING: TAIEX 歷史資料抓取結果為空：{start_date}~{end_date}")
        return df
    end_ts = pd.Timestamp(end_dt)
    df = df[df["date"] <= end_ts].sort_values("date").reset_index(drop=True)
    return standardize_market_index_data(df)


__all__ = [name for name in globals() if not name.startswith('__')]
