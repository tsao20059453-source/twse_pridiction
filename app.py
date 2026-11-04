from __future__ import annotations

import itertools
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


_PLOTLY_CHART_COUNTER = itertools.count(1)

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = BASE_DIR / "Pipeline_clean.py"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


CORE_MODEL_SETTINGS = {
    "feature_set": "selected_v2",
    "model_type": "extra_trees",
    "model_profile": "medium",
    "target_col": "target_safety_10d_cs_rank",
    "diagnostic_scope": "common",
    "display_mode": "presentation",
}

BACKTEST_SETTINGS = {
    "backtest_profile": "balanced_v2_combined",
    "backtest_entry_mode": "balanced_support",
    "max_positions": 8,
    "initial_position_pct": 0.08,
    "add1_position_pct": 0.06,
    "add2_position_pct": 0.06,
    "max_single_position_pct": 0.20,
    "max_total_exposure": 0.85,
    "min_cash_reserve": 0.15,
}

FALLBACK_VALIDATION_METRICS = pd.DataFrame(
    [
        {
            "model": "Safety model",
            "MAE": 0.2389,
            "Baseline MAE": 0.2495,
            "RMSE": 0.2797,
            "Baseline RMSE": 0.2881,
            "R2": 0.0578,
            "Spearman": 0.2405,
            "Top5 Lift": 0.1149,
            "Top10 Lift": 0.0774,
            "Long-Short Spread": 0.1994,
        },
        {
            "model": "Return model",
            "MAE": 0.2482,
            "Baseline MAE": 0.2499,
            "RMSE": 0.2884,
            "Baseline RMSE": 0.2883,
            "R2": -0.0009,
            "Spearman": 0.0798,
            "Top5 Lift": 0.0301,
            "Top10 Lift": 0.0281,
            "Long-Short Spread": 0.0620,
        },
    ]
)

RANKING_COLUMNS = [
    "rank",
    "symbol",
    "stock_name",
    "date",
    "close",
    "safety_score",
    "return_score",
    "combined_decision_score",
    "final_decision_score",
    "final_decision_label",
    "risk_reward_ratio",
    "practical_risk_reward_ratio",
    "trade_veto_flag",
    "trade_veto_reason",
    "position_sizing_label",
]

COMMON_MODEL_COLUMNS = [
    "rank",
    "symbol",
    "stock_name",
    "date",
    "close",
    "safety_score",
    "return_score",
    "combined_decision_score",
    "final_decision_label",
    "trade_veto_flag",
    "trade_veto_reason",
]

FINAL_DECISION_COLUMNS = [
    "symbol",
    "stock_name",
    "date",
    "close",
    "combined_decision_score",
    "final_decision_score",
    "final_decision_label",
    "risk_reward_ratio",
    "practical_risk_reward_ratio",
    "trade_veto_flag",
    "trade_veto_reason",
    "position_sizing_label",
]

SINGLE_METRIC_ROWS = [
    ["final_decision_score", "combined_decision_score", "safety_score"],
    ["return_score", "rule_filter_score", "risk_reward_ratio"],
    ["practical_risk_reward_ratio", "stop_loss_pct", "suggested_position_pct"],
]

FEATURE_GROUPS = {
    "趨勢類": {
        "features": ["ma60_distance", "ma20_distance", "return_5d", "return_10d"],
        "description": "價格相對均線與近期報酬，用來判斷趨勢方向與延續性。",
    },
    "動能類": {
        "features": ["RSI14", "MACD", "DIF", "DEA", "K", "J", "BIAS24", "VR_14", "VR_26"],
        "description": "衡量買盤動能、量價強度與短中期技術狀態。",
    },
    "波動類": {
        "features": ["atr_pct", "volatility_5d", "volatility_10d", "risk_atr_stop_distance_pct"],
        "description": "描述價格波動、ATR 停損距離與交易風險。",
    },
    "位置類": {
        "features": [
            "support_distance",
            "resistance_distance",
            "close_to_10d_high",
            "close_to_10d_low",
            "fib_position",
            "fib_score",
        ],
        "description": "衡量股價相對支撐、壓力、近期高低點與 Fibonacci 區間的位置。",
    },
    "結構類": {
        "features": [
            "recent_2w_ob_volume_ratio",
            "recent_2w_ob_strength_score",
            "recent_2w_ob_type",
            "recent_active_gap_size_pct",
            "recent_active_gap_type",
        ],
        "description": "觀察近期 Order Block、缺口與價格結構。",
    },
    "類別特徵": {
        "features": ["fib_zone", "recent_2w_ob_type", "recent_active_gap_type"],
        "description": "將區間、結構與型態轉為模型可使用的類別訊號。",
    },
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 96vw !important;
            padding-top: 3.2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 2rem !important;
        }

        html, body, [class*="css"] {
            font-size: 16px !important;
            color: #111827 !important;
        }

        h1 {
            font-size: 2.2rem !important;
            line-height: 1.25 !important;
            color: #0f172a !important;
            font-weight: 800 !important;
            margin-bottom: 0.75rem !important;
        }

        h2 {
            font-size: 1.55rem !important;
            color: #0f172a !important;
            font-weight: 700 !important;
            margin-top: 1.4rem !important;
            margin-bottom: 0.8rem !important;
        }

        h3 {
            font-size: 1.25rem !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 14px !important;
            margin-top: 0.2rem !important;
            margin-bottom: 1.2rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.2rem !important;
            overflow: visible !important;
            height: auto !important;
        }

        .stTabs [data-baseweb="tab"] {
            height: 46px !important;
            padding: 8px 14px !important;
            line-height: 1.2 !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab"] p {
            font-size: 17px !important;
            font-weight: 700 !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #111827 !important;
        }

        .stTabs [aria-selected="true"] p {
            color: #ef4444 !important;
        }

        [data-testid="stSidebar"] {
            font-size: 16px !important;
        }

        [data-testid="stSidebar"] label {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #111827 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #374151 !important;
            font-weight: 600 !important;
            font-size: 15px !important;
        }

        [data-testid="stMetricValue"] {
            color: #111827 !important;
            font-weight: 700 !important;
            font-size: 1.75rem !important;
        }

        .small-note {
            color: #4b5563 !important;
            font-size: 0.95rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def find_latest_file(
    patterns: Iterable[str],
    prefer_after: float | None = None,
    search_dirs: Iterable[Path] | None = None,
) -> Path | None:
    roots = list(search_dirs) if search_dirs is not None else [OUTPUT_DIR]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and path not in seen:
                    candidates.append(path)
                    seen.add(path)
    if not candidates:
        return None
    if prefer_after is not None:
        fresh_candidates = [path for path in candidates if path.stat().st_mtime >= prefer_after]
        if fresh_candidates:
            return max(fresh_candidates, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_latest_ranking_csv(analysis_mode: str | None = None) -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    if analysis_mode == "多股票排名分析":
        patterns = ["*COMMON_MODEL_TRAIN_*_selected_v2_ranking.csv", "*_ranking.csv", "*ranking*.csv"]
    else:
        patterns = ["*_ranking.csv", "*ranking*.csv"]
    path = find_latest_file(patterns, prefer_after=prefer_after)
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_single_diagnostic_csv(symbol: str | None = None) -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    patterns: list[str] = []
    clean_symbol = (symbol or "").strip()
    if clean_symbol:
        patterns.extend(
            [
                f"*{clean_symbol}*common_single_diagnostic*.csv",
                f"*{clean_symbol}*single_diagnostic*.csv",
            ]
        )
    patterns.extend(["*common_single_diagnostic*.csv", "*single_diagnostic*.csv"])
    path = find_latest_file(patterns, prefer_after=prefer_after)
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_validation_csv() -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    path = find_latest_file(["*validation*.csv", "*walk_forward*.csv", "*metrics*.csv"], prefer_after=prefer_after)
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_backtest_equity_curve() -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    path = find_latest_file(
        ["*equity_curve*.csv", "*portfolio*equity*.csv"],
        prefer_after=prefer_after,
        search_dirs=[OUTPUT_DIR, BASE_DIR],
    )
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_backtest_trade_log() -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    path = find_latest_file(
        ["*trade_log*.csv", "*trades*.csv"],
        prefer_after=prefer_after,
        search_dirs=[OUTPUT_DIR, BASE_DIR],
    )
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_portfolio_metrics() -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    path = find_latest_file(
        ["*portfolio_metrics*.csv", "*backtest_metrics*.csv", "*metrics*.csv", "*perf*.csv"],
        prefer_after=prefer_after,
        search_dirs=[OUTPUT_DIR, BASE_DIR],
    )
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_entry_diagnostic() -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    path = find_latest_file(
        [
            "*entry_diagnostics_daily*.csv",
            "*daily_diagnostic*.csv",
            "*entry_diagnostic*.csv",
            "*rejection*.csv",
        ],
        prefer_after=prefer_after,
        search_dirs=[OUTPUT_DIR, BASE_DIR],
    )
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def load_latest_sensitivity_summary() -> tuple[Path | None, pd.DataFrame | None, str | None]:
    prefer_after = st.session_state.get("run_started_at")
    path = find_latest_file(
        ["*sensitivity*.csv", "*profile_comparison*.csv"],
        prefer_after=prefer_after,
        search_dirs=[OUTPUT_DIR, BASE_DIR],
    )
    if path is None:
        return None, None, None
    try:
        return path, read_csv_safely(path), None
    except Exception as exc:
        return path, None, str(exc)


def get_latest_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if "date" not in df.columns:
        return df.iloc[-1]
    work = df.copy()
    work["_date_sort"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values("_date_sort", na_position="first")
    return work.drop(columns=["_date_sort"]).iloc[-1]


def output_csv_files(limit: int = 20) -> list[Path]:
    return sorted((p for p in OUTPUT_DIR.glob("*.csv") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def file_time(path: Path | None) -> str:
    if path is None or not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    pct_mask = text.str.endswith("%", na=False)
    cleaned = text.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    values.loc[pct_mask] = values.loc[pct_mask] / 100.0
    return values


def date_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[column], errors="coerce")


def to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            if text.endswith("%"):
                return float(text[:-1]) / 100.0
            return float(text)
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        text = "" if value is None else str(value)
        return text if text else "-"
    return f"{number:,.{digits}f}"


def format_pct(value, digits: int = 2) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    display = number * 100 if abs(number) <= 1 else number
    return f"{display:,.{digits}f}%"


def safe_plotly_chart(fig, key_prefix: str, height: int | None = None) -> None:
    """
    Streamlit 1.4x+ needs each plotly_chart to have a unique key.
    key_prefix + counter avoids DuplicateElementId when chart functions render
    in multiple tabs or are called more than once on the same page.
    """
    safe_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", str(key_prefix)).strip("_").lower()
    if not safe_prefix:
        safe_prefix = "plotly_chart"
    chart_key = f"{safe_prefix}_{next(_PLOTLY_CHART_COUNTER)}"
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    number = to_float(value)
    if number is not None:
        return number != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y", "veto", "blocked"}


def normalize_symbol(value: str) -> str:
    text = "" if value is None else str(value).strip().upper()
    return "".join(ch for ch in text if ch.isalnum())


def stock_display_label(df: pd.DataFrame) -> pd.Series:
    symbols = df["symbol"].astype(str) if "symbol" in df.columns else pd.Series("", index=df.index)
    if "stock_name" not in df.columns:
        return symbols
    names = df["stock_name"].fillna("").astype(str).str.strip()
    return pd.Series(np.where(names.ne(""), symbols + " " + names, symbols), index=df.index)


def row_stock_label(row: pd.Series | None) -> str:
    if row is None:
        return "-"
    symbol = str(row.get("symbol", "")).strip()
    stock_name = str(row.get("stock_name", "")).strip()
    if symbol and stock_name and stock_name.lower() != "nan":
        return f"{symbol} {stock_name}"
    return symbol or stock_name or "-"


def prepare_common_model_ranking(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["rank", "close", "safety_score", "return_score", "combined_decision_score"]:
        if col in work.columns:
            work[col] = numeric_series(work, col)
    if "combined_decision_score" not in work.columns and {"safety_score", "return_score"}.issubset(work.columns):
        work["combined_decision_score"] = 0.65 * work["safety_score"] + 0.35 * work["return_score"]

    if "rank" in work.columns and work["rank"].notna().any():
        sort_cols = ["rank"]
        ascending = [True]
        if "combined_decision_score" in work.columns:
            sort_cols.append("combined_decision_score")
            ascending.append(False)
        return work.sort_values(sort_cols, ascending=ascending, na_position="last")
    if "combined_decision_score" in work.columns:
        return work.sort_values("combined_decision_score", ascending=False, na_position="last")
    return work


def prepare_final_decision_ranking(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in [
        "close",
        "combined_decision_score",
        "final_decision_score",
        "risk_reward_ratio",
        "practical_risk_reward_ratio",
    ]:
        if col in work.columns:
            work[col] = numeric_series(work, col)
    if "final_decision_score" in work.columns:
        return work.sort_values("final_decision_score", ascending=False, na_position="last")
    return work


def render_common_model_bar(common_df: pd.DataFrame, key_prefix: str = "ranking_common_model_bar") -> None:
    if "combined_decision_score" not in common_df.columns:
        st.warning("ranking CSV 缺少 combined_decision_score，也無法由 safety_score / return_score 建立共同模型分數。")
        return

    chart_df = common_df.dropna(subset=["combined_decision_score"]).head(15).copy()
    if chart_df.empty:
        st.info("目前沒有可繪製的 combined_decision_score。")
        return

    chart_df["display_label"] = stock_display_label(chart_df)
    chart_df = chart_df.sort_values("combined_decision_score", ascending=True)
    fig = px.bar(
        chart_df,
        x="combined_decision_score",
        y="display_label",
        orientation="h",
        text=chart_df["combined_decision_score"].map(lambda x: f"{x:.4f}"),
        labels={"display_label": "股票", "combined_decision_score": "Combined Score"},
        title="Top 15 共同模型分數（Combined Score）",
    )
    fig.update_traces(marker_color="#1f4e79", textposition="outside", cliponaxis=False)
    fig.update_layout(height=430, margin=dict(l=20, r=40, t=50, b=20), yaxis=dict(title=None), font=dict(size=13))
    safe_plotly_chart(fig, key_prefix)


def cli_mode_for_analysis(analysis_mode: str) -> str:
    if analysis_mode == "個股深度分析":
        return "single_diagnostic"
    if analysis_mode == "回測分析":
        return "backtest"
    return "train"


def make_command(analysis_mode: str, market: str, symbol: str, top_n: int) -> list[str]:
    cli_mode = cli_mode_for_analysis(analysis_mode)
    command_symbol = normalize_symbol(symbol) or "2330"
    if cli_mode == "train":
        command_symbol = "ALL"
    command_top_n = 30 if cli_mode == "single_diagnostic" else max(int(top_n), 5)
    return [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--mode",
        cli_mode,
        "--market",
        market,
        "--symbol",
        command_symbol,
        "--top-n",
        str(command_top_n),
        "--feature-set",
        CORE_MODEL_SETTINGS["feature_set"],
        "--model-type",
        CORE_MODEL_SETTINGS["model_type"],
        "--model-profile",
        CORE_MODEL_SETTINGS["model_profile"],
        "--diagnostic-scope",
        CORE_MODEL_SETTINGS["diagnostic_scope"],
        "--display-mode",
        CORE_MODEL_SETTINGS["display_mode"],
        "--target-col",
        CORE_MODEL_SETTINGS["target_col"],
    ]


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if command else "-"


def init_run_state() -> None:
    defaults = {
        "model_process": None,
        "model_running": False,
        "last_stdout": "",
        "last_stderr": "",
        "last_command": "",
        "last_command_list": [],
        "last_returncode": None,
        "last_status": "尚未執行",
        "last_started_at": None,
        "last_finished_at": None,
        "last_pid": None,
        "run_started_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_process_state() -> None:
    process = st.session_state.model_process
    if process is None:
        st.session_state.model_running = False
        return

    returncode = process.poll()
    if returncode is None:
        st.session_state.model_running = True
        st.session_state.last_status = "執行中"
        return

    stdout, stderr = process.communicate()
    st.session_state.last_stdout = stdout or ""
    st.session_state.last_stderr = stderr or ""
    st.session_state.last_returncode = returncode
    st.session_state.model_running = False
    st.session_state.model_process = None
    st.session_state.last_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.last_status = "執行成功" if returncode == 0 else "執行失敗"


def start_model_process(command: list[str]) -> tuple[bool, str]:
    if st.session_state.model_running:
        return False, "目前已有模型正在背景執行。"
    run_started_at = time.time()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=BASE_DIR,
        )
    except Exception as exc:
        st.session_state.last_stdout = ""
        st.session_state.last_stderr = f"啟動 subprocess 失敗：{exc}"
        st.session_state.last_returncode = None
        st.session_state.last_status = "執行失敗"
        st.session_state.model_running = False
        st.session_state.model_process = None
        st.session_state.last_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return False, str(exc)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.run_started_at = run_started_at
    st.session_state.model_process = process
    st.session_state.model_running = True
    st.session_state.last_stdout = ""
    st.session_state.last_stderr = ""
    st.session_state.last_command = command_text(command)
    st.session_state.last_command_list = command
    st.session_state.last_returncode = None
    st.session_state.last_status = "執行中"
    st.session_state.last_started_at = now
    st.session_state.last_finished_at = None
    st.session_state.last_pid = process.pid
    return True, "模型已在背景開始執行。"


def stop_model_process() -> tuple[bool, str]:
    process = st.session_state.model_process
    if process is None or process.poll() is not None:
        st.session_state.model_running = False
        st.session_state.model_process = None
        return False, "目前沒有正在執行的模型。"

    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()

    st.session_state.last_stdout = stdout or ""
    st.session_state.last_stderr = stderr or ""
    st.session_state.last_returncode = process.returncode
    st.session_state.last_status = "已停止"
    st.session_state.model_running = False
    st.session_state.model_process = None
    st.session_state.last_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return True, "已停止背景模型執行。"


def run_status_label() -> str:
    if st.session_state.model_running:
        return "執行中"
    return st.session_state.last_status


def render_top15_bar(ranking_df: pd.DataFrame, key_prefix: str = "ranking_final_score_top15") -> None:
    if "final_decision_score" not in ranking_df.columns:
        st.warning("ranking CSV 缺少 final_decision_score，無法繪製 Top 15 排名圖。")
        return

    chart_df = ranking_df.copy()
    chart_df["final_decision_score"] = numeric_series(chart_df, "final_decision_score")
    chart_df = chart_df.dropna(subset=["final_decision_score"]).sort_values("final_decision_score", ascending=False).head(15)
    if chart_df.empty:
        st.info("目前沒有可繪製的 final_decision_score。")
        return

    chart_df["display_label"] = stock_display_label(chart_df)
    chart_df = chart_df.sort_values("final_decision_score", ascending=True)
    fig = px.bar(
        chart_df,
        x="final_decision_score",
        y="display_label",
        orientation="h",
        text=chart_df["final_decision_score"].map(lambda x: f"{x:.4f}"),
        labels={"display_label": "股票", "final_decision_score": "Final Decision Score"},
        title="Top 15 最終決策分數（Final Decision Score）",
    )
    fig.update_traces(marker_color="#1f4e79", textposition="outside", cliponaxis=False)
    fig.update_layout(height=430, margin=dict(l=20, r=40, t=50, b=20), yaxis=dict(title=None), font=dict(size=13))
    safe_plotly_chart(fig, key_prefix)


def render_score_rr_scatter(ranking_df: pd.DataFrame, key_prefix: str = "ranking_score_rr_scatter") -> None:
    rr_col = "risk_reward_ratio" if "risk_reward_ratio" in ranking_df.columns else "practical_risk_reward_ratio"
    if "final_decision_score" not in ranking_df.columns or rr_col not in ranking_df.columns:
        st.info("ranking CSV 缺少 final_decision_score 或 Risk/Reward 欄位，略過散點圖。")
        return

    chart_df = ranking_df.copy()
    chart_df["final_decision_score"] = numeric_series(chart_df, "final_decision_score")
    chart_df[rr_col] = numeric_series(chart_df, rr_col)
    chart_df = chart_df.dropna(subset=["final_decision_score", rr_col])
    if chart_df.empty:
        st.info("目前沒有可繪製的 Final Score / Risk Reward 資料。")
        return

    chart_df["display_label"] = stock_display_label(chart_df)
    color_col = "final_decision_label" if "final_decision_label" in chart_df.columns else None
    fig = px.scatter(
        chart_df,
        x="final_decision_score",
        y=rr_col,
        color=color_col,
        hover_name="display_label",
        hover_data=existing_columns(chart_df, ["rank", "trade_veto_flag", "position_sizing_label"]),
        title="Final Score vs Risk/Reward",
        labels={"final_decision_score": "Final Decision Score", rr_col: "Risk/Reward"},
    )
    fig.add_vline(x=0.65, line_dash="dash", line_color="#b91c1c")
    fig.add_hline(y=1.5, line_dash="dash", line_color="#d97706")
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=50, b=20), font=dict(size=13))
    safe_plotly_chart(fig, key_prefix)


def render_veto_reason_chart(ranking_df: pd.DataFrame, key_prefix: str = "ranking_veto_reason") -> None:
    if "trade_veto_reason" not in ranking_df.columns:
        st.info("ranking CSV 缺少 trade_veto_reason，略過 veto reason 圖。")
        return
    reasons = ranking_df["trade_veto_reason"].fillna("").astype(str).str.strip()
    reasons = reasons[reasons.ne("")]
    if reasons.empty:
        st.info("目前沒有 veto reason。")
        return
    counts = reasons.value_counts().head(10).reset_index()
    counts.columns = ["trade_veto_reason", "count"]
    fig = px.bar(
        counts.sort_values("count", ascending=True),
        x="count",
        y="trade_veto_reason",
        orientation="h",
        title="Veto reason 分布",
        labels={"count": "數量", "trade_veto_reason": "Veto reason"},
    )
    fig.update_traces(marker_color="#b91c1c")
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=50, b=20), yaxis=dict(title=None), font=dict(size=13))
    safe_plotly_chart(fig, key_prefix)


def apply_chart_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        font=dict(size=13),
        legend_title_text="",
    )
    return fig


def render_value_count_bar(
    df: pd.DataFrame,
    column: str,
    title: str,
    max_rows: int = 15,
    key_prefix: str | None = None,
) -> None:
    if column not in df.columns:
        st.info(f"CSV 缺少 {column}，略過「{title}」。")
        return
    values = df[column].fillna("未標示").astype(str).str.strip().replace({"": "未標示"})
    counts = values.value_counts().head(max_rows).reset_index()
    counts.columns = [column, "count"]
    if counts.empty:
        st.info(f"目前沒有可繪製的 {column} 分布。")
        return
    fig = px.bar(
        counts.sort_values("count", ascending=True),
        x="count",
        y=column,
        orientation="h",
        title=title,
        labels={"count": "數量", column: column},
    )
    fig.update_traces(marker_color="#2563eb")
    fig.update_layout(yaxis=dict(title=None))
    safe_plotly_chart(apply_chart_layout(fig, 380), key_prefix or f"value_count_{column}")


def render_numeric_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    nbins: int = 30,
    key_prefix: str | None = None,
) -> None:
    if column not in df.columns:
        st.info(f"CSV 缺少 {column}，略過「{title}」。")
        return
    chart_df = pd.DataFrame({column: numeric_series(df, column)}).dropna()
    if chart_df.empty:
        st.info(f"目前沒有可繪製的 {column} 分布。")
        return
    fig = px.histogram(chart_df, x=column, nbins=nbins, title=title, labels={column: column})
    fig.update_traces(marker_color="#0f766e")
    safe_plotly_chart(apply_chart_layout(fig, 360), key_prefix or f"numeric_histogram_{column}")


def render_top_numeric_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    top_n: int = 15,
    key_prefix: str | None = None,
) -> None:
    if value_col not in df.columns:
        st.info(f"CSV 缺少 {value_col}，略過「{title}」。")
        return
    chart_df = df.copy()
    chart_df[value_col] = numeric_series(chart_df, value_col)
    chart_df = chart_df.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)
    if chart_df.empty:
        st.info(f"目前沒有可繪製的 {value_col}。")
        return
    chart_df["display_label"] = stock_display_label(chart_df)
    chart_df = chart_df.sort_values(value_col, ascending=True)
    fig = px.bar(
        chart_df,
        x=value_col,
        y="display_label",
        orientation="h",
        text=chart_df[value_col].map(lambda x: f"{x:.4f}"),
        title=title,
        labels={value_col: value_col, "display_label": "股票"},
    )
    fig.update_traces(marker_color="#1f4e79", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(title=None))
    safe_plotly_chart(apply_chart_layout(fig, 430), key_prefix or f"top_numeric_{value_col}")


def render_safety_return_scatter(
    ranking_df: pd.DataFrame,
    key_prefix: str = "ranking_safety_return_scatter",
) -> None:
    required = ["safety_score", "return_score"]
    missing = [col for col in required if col not in ranking_df.columns]
    if missing:
        st.info(f"ranking CSV 缺少 {', '.join(missing)}，略過 safety / return 散點圖。")
        return
    chart_df = ranking_df.copy()
    chart_df["safety_score"] = numeric_series(chart_df, "safety_score")
    chart_df["return_score"] = numeric_series(chart_df, "return_score")
    if "final_decision_score" in chart_df.columns:
        chart_df["final_decision_score"] = numeric_series(chart_df, "final_decision_score")
    chart_df = chart_df.dropna(subset=["safety_score", "return_score"])
    if chart_df.empty:
        st.info("目前沒有可繪製的 safety_score / return_score。")
        return
    chart_df["display_label"] = stock_display_label(chart_df)
    size_col = "final_decision_score" if "final_decision_score" in chart_df.columns and chart_df["final_decision_score"].notna().any() else None
    color_col = "final_decision_label" if "final_decision_label" in chart_df.columns else None
    fig = px.scatter(
        chart_df,
        x="safety_score",
        y="return_score",
        size=size_col,
        color=color_col,
        hover_name="display_label",
        hover_data=existing_columns(chart_df, ["rank", "combined_decision_score", "final_decision_score", "trade_veto_flag"]),
        title="Safety Score vs Return Score",
        labels={"safety_score": "Safety Score", "return_score": "Return Score"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
    safe_plotly_chart(apply_chart_layout(fig, 420), key_prefix)


def render_trade_veto_distribution(
    ranking_df: pd.DataFrame,
    key_prefix: str = "ranking_trade_veto_distribution",
) -> None:
    if "trade_veto_flag" not in ranking_df.columns:
        st.info("ranking CSV 缺少 trade_veto_flag，略過 veto flag 分布圖。")
        return
    counts = (
        ranking_df["trade_veto_flag"]
        .map(lambda value: "Veto" if truthy_flag(value) else "Pass")
        .value_counts()
        .reset_index()
    )
    counts.columns = ["trade_veto_flag", "count"]
    fig = px.bar(
        counts,
        x="trade_veto_flag",
        y="count",
        text="count",
        title="trade_veto_flag 分布",
        labels={"trade_veto_flag": "狀態", "count": "數量"},
    )
    fig.update_traces(marker_color=["#b91c1c" if label == "Veto" else "#0f766e" for label in counts["trade_veto_flag"]])
    safe_plotly_chart(apply_chart_layout(fig, 340), key_prefix)


def render_support_resistance_scatter(
    ranking_df: pd.DataFrame,
    key_prefix: str = "ranking_support_resistance_scatter",
) -> None:
    required = ["support_distance", "resistance_distance"]
    missing = [col for col in required if col not in ranking_df.columns]
    if missing:
        st.info(f"ranking CSV 缺少 {', '.join(missing)}，略過支撐/壓力距離散點圖。")
        return
    chart_df = ranking_df.copy()
    for col in required + ["final_decision_score"]:
        if col in chart_df.columns:
            chart_df[col] = numeric_series(chart_df, col)
    chart_df = chart_df.dropna(subset=required)
    if chart_df.empty:
        st.info("目前沒有可繪製的 support_distance / resistance_distance。")
        return
    chart_df["display_label"] = stock_display_label(chart_df)
    fig = px.scatter(
        chart_df,
        x="support_distance",
        y="resistance_distance",
        size="final_decision_score" if "final_decision_score" in chart_df.columns else None,
        color="final_decision_label" if "final_decision_label" in chart_df.columns else None,
        hover_name="display_label",
        title="支撐距離 vs 壓力距離",
        labels={"support_distance": "Support Distance", "resistance_distance": "Resistance Distance"},
    )
    safe_plotly_chart(apply_chart_layout(fig, 420), key_prefix)


def render_single_metric_timeseries(
    single_df: pd.DataFrame,
    key_prefix: str = "single_metric_timeseries",
) -> None:
    score_cols = existing_columns(
        single_df,
        ["final_decision_score", "combined_decision_score", "safety_score", "return_score"],
    )
    if "date" not in single_df.columns or not score_cols:
        st.info("single diagnostic CSV 缺少 date 或分數欄位，略過個股分數時序圖。")
        return
    work = single_df.copy()
    work["date"] = date_series(work, "date")
    for col in score_cols:
        work[col] = numeric_series(work, col)
    work = work.dropna(subset=["date"])
    long_df = work.melt(id_vars=["date"], value_vars=score_cols, var_name="metric", value_name="value").dropna()
    if long_df.empty:
        st.info("目前沒有可繪製的個股分數時序資料。")
        return
    fig = px.line(long_df, x="date", y="value", color="metric", markers=True, title="個股分數時序")
    fig.update_yaxes(range=[0, 1])
    safe_plotly_chart(apply_chart_layout(fig, 390), key_prefix)


def render_single_close_ma_chart(
    single_df: pd.DataFrame,
    key_prefix: str = "single_close_ma_chart",
) -> None:
    cols = existing_columns(single_df, ["close", "ma20", "ma60"])
    if "date" not in single_df.columns or "close" not in cols:
        st.info("single diagnostic CSV 缺少 date 或 close，略過收盤價時序圖。")
        return
    work = single_df.copy()
    work["date"] = date_series(work, "date")
    for col in cols:
        work[col] = numeric_series(work, col)
    work = work.dropna(subset=["date"])
    long_df = work.melt(id_vars=["date"], value_vars=cols, var_name="metric", value_name="price").dropna()
    if long_df.empty:
        st.info("目前沒有可繪製的 close / MA 時序資料。")
        return
    fig = px.line(long_df, x="date", y="price", color="metric", markers=True, title="收盤價與均線")
    safe_plotly_chart(apply_chart_layout(fig, 390), key_prefix)


def render_single_indicator_chart(
    single_df: pd.DataFrame,
    columns: list[str],
    title: str,
    reference_lines: list[tuple[float, str]] | None = None,
    key_prefix: str | None = None,
) -> None:
    cols = existing_columns(single_df, columns)
    if "date" not in single_df.columns or not cols:
        st.info(f"single diagnostic CSV 缺少 {', '.join(columns)}，略過「{title}」。")
        return
    work = single_df.copy()
    work["date"] = date_series(work, "date")
    for col in cols:
        work[col] = numeric_series(work, col)
    long_df = work.dropna(subset=["date"]).melt(id_vars=["date"], value_vars=cols, var_name="metric", value_name="value").dropna()
    if long_df.empty:
        st.info(f"目前沒有可繪製的「{title}」資料。")
        return
    fig = px.line(long_df, x="date", y="value", color="metric", markers=True, title=title)
    for y_value, label in reference_lines or []:
        fig.add_hline(y=y_value, line_dash="dash", line_color="#94a3b8", annotation_text=label)
    safe_plotly_chart(apply_chart_layout(fig, 370), key_prefix or f"single_indicator_{'_'.join(columns)}")


def render_single_price_plan_lines(
    single_df: pd.DataFrame,
    key_prefix: str = "single_price_plan_lines",
) -> None:
    cols = existing_columns(single_df, ["entry_price", "stop_loss_price_adjusted", "take_profit_price"])
    if "date" not in single_df.columns or not cols:
        st.info("single diagnostic CSV 缺少進場/停損/停利價格欄位，略過價格線圖。")
        return
    work = single_df.copy()
    work["date"] = date_series(work, "date")
    for col in cols:
        work[col] = numeric_series(work, col)
    long_df = work.dropna(subset=["date"]).melt(id_vars=["date"], value_vars=cols, var_name="metric", value_name="price").dropna()
    if long_df.empty:
        st.info("目前沒有可繪製的進場/停損/停利價格線。")
        return
    fig = px.line(long_df, x="date", y="price", color="metric", markers=True, title="進場價 / 停損價 / 停利價")
    safe_plotly_chart(apply_chart_layout(fig, 370), key_prefix)


def prepare_equity_curve(equity_df: pd.DataFrame) -> pd.DataFrame:
    work = equity_df.copy()
    if "date" in work.columns:
        work["date"] = date_series(work, "date")
    for col in [
        "total_equity",
        "cash",
        "market_value",
        "portfolio_value",
        "daily_return",
        "exposure",
        "capital_usage",
        "num_positions",
        "taiex_equity",
        "top30_equity",
        "tsmc_equity",
    ]:
        if col in work.columns:
            work[col] = numeric_series(work, col)
    if "market_value" not in work.columns and "portfolio_value" in work.columns:
        work["market_value"] = work["portfolio_value"]
    if "total_equity" not in work.columns and {"cash", "market_value"}.issubset(work.columns):
        work["total_equity"] = work["cash"] + work["market_value"]
    if "daily_return" not in work.columns and "total_equity" in work.columns:
        work["daily_return"] = work["total_equity"].pct_change().fillna(0.0)
    if "capital_usage" not in work.columns and {"market_value", "total_equity"}.issubset(work.columns):
        work["capital_usage"] = work["market_value"] / work["total_equity"].replace(0, np.nan)
    if "total_equity" in work.columns:
        cumulative_max = work["total_equity"].cummax().replace(0, np.nan)
        work["drawdown"] = work["total_equity"] / cumulative_max - 1.0
    return work


def metrics_lookup(metrics_df: pd.DataFrame | None) -> dict[str, float]:
    if metrics_df is None or metrics_df.empty:
        return {}
    if "股票代碼" in metrics_df.columns and len(metrics_df) > 1:
        return {}
    metrics: dict[str, float] = {}
    if {"metric", "value"}.issubset(metrics_df.columns):
        for _, row in metrics_df.iterrows():
            key = str(row.get("metric", "")).strip()
            value = to_float(row.get("value"))
            if key and value is not None:
                normalized = key.lower().replace(" ", "_").replace("-", "_")
                metrics[normalized] = value
                metrics[key] = value
        return metrics
    row = metrics_df.iloc[0]
    for col in metrics_df.columns:
        value = to_float(row.get(col))
        if value is not None:
            normalized = str(col).strip().lower().replace(" ", "_").replace("-", "_")
            metrics[normalized] = value
            metrics[str(col)] = value
    return metrics


def metric_lookup_value(metrics: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
        normalized = key.lower().replace(" ", "_").replace("-", "_")
        if normalized in metrics and metrics[normalized] is not None:
            return metrics[normalized]
    return None


def equity_kpis(equity_df: pd.DataFrame, metrics_df: pd.DataFrame | None) -> dict[str, float | None]:
    work = prepare_equity_curve(equity_df)
    metrics = metrics_lookup(metrics_df)
    initial = to_float(work["total_equity"].dropna().iloc[0]) if "total_equity" in work.columns and work["total_equity"].notna().any() else None
    final = to_float(work["total_equity"].dropna().iloc[-1]) if "total_equity" in work.columns and work["total_equity"].notna().any() else None
    total_return = metric_lookup_value(metrics, "total_return", "總報酬率")
    if total_return is None and initial not in (None, 0) and final is not None:
        total_return = final / initial - 1.0
    mdd = metric_lookup_value(metrics, "max_drawdown", "mdd", "最大回撤")
    if mdd is None and "drawdown" in work.columns and work["drawdown"].notna().any():
        mdd = float(work["drawdown"].min())
    cagr = metric_lookup_value(metrics, "cagr", "annual_return", "annualized_return")
    if cagr is None and initial not in (None, 0) and final is not None and "date" in work.columns:
        dates = work["date"].dropna()
        if len(dates) >= 2:
            days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
            cagr = (final / initial) ** (365.0 / days) - 1.0
    sharpe = metric_lookup_value(metrics, "sharpe_ratio", "sharpe", "Sharpe")
    calmar = metric_lookup_value(metrics, "calmar_ratio", "calmar")
    if calmar is None and cagr is not None and mdd not in (None, 0):
        calmar = cagr / abs(mdd)
    return {
        "initial": initial,
        "final": final,
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def render_backtest_kpis(equity_df: pd.DataFrame, metrics_df: pd.DataFrame | None) -> None:
    kpis = equity_kpis(equity_df, metrics_df)
    cols = st.columns(7)
    cols[0].metric("初始資金", format_value(kpis["initial"], 0))
    cols[1].metric("最終資產", format_value(kpis["final"], 0))
    cols[2].metric("總報酬率", format_pct(kpis["total_return"]))
    cols[3].metric("CAGR", format_pct(kpis["cagr"]))
    cols[4].metric("MDD", format_pct(kpis["mdd"]))
    cols[5].metric("Sharpe", format_value(kpis["sharpe"], 2))
    cols[6].metric("Calmar", format_value(kpis["calmar"], 2))


def render_equity_curve_chart(
    equity_df: pd.DataFrame,
    key_prefix: str = "equity_curve",
) -> None:
    work = prepare_equity_curve(equity_df)
    if "date" not in work.columns or "total_equity" not in work.columns:
        st.info("equity curve CSV 缺少 date 或 total_equity。")
        return
    chart_df = work.dropna(subset=["date", "total_equity"])
    if chart_df.empty:
        st.info("目前沒有可繪製的資金曲線。")
        return
    fig = px.line(chart_df, x="date", y="total_equity", title="資金曲線", labels={"total_equity": "Total Equity"})
    safe_plotly_chart(apply_chart_layout(fig, 430), key_prefix)


def render_drawdown_chart(
    equity_df: pd.DataFrame,
    key_prefix: str = "backtest_drawdown",
) -> None:
    work = prepare_equity_curve(equity_df)
    if "date" not in work.columns or "drawdown" not in work.columns:
        st.info("equity curve CSV 缺少 drawdown 可用欄位。")
        return
    chart_df = work.dropna(subset=["date", "drawdown"])
    if chart_df.empty:
        st.info("目前沒有可繪製的 drawdown。")
        return
    fig = px.area(chart_df, x="date", y="drawdown", title="Drawdown", labels={"drawdown": "Drawdown"})
    fig.update_traces(line_color="#b91c1c", fillcolor="rgba(185,28,28,0.25)")
    safe_plotly_chart(apply_chart_layout(fig, 360), key_prefix)


def render_cash_market_equity_chart(
    equity_df: pd.DataFrame,
    key_prefix: str = "backtest_cash_market_equity",
) -> None:
    work = prepare_equity_curve(equity_df)
    cols = existing_columns(work, ["cash", "market_value", "total_equity"])
    if "date" not in work.columns or len(cols) < 2:
        st.info("equity curve CSV 缺少 cash / market_value / total_equity，略過資產組成圖。")
        return
    long_df = work.dropna(subset=["date"]).melt(id_vars=["date"], value_vars=cols, var_name="metric", value_name="value").dropna()
    if long_df.empty:
        st.info("目前沒有可繪製的資產組成資料。")
        return
    fig = px.line(long_df, x="date", y="value", color="metric", title="Cash / Market Value / Total Equity")
    safe_plotly_chart(apply_chart_layout(fig, 400), key_prefix)


def render_exposure_chart(
    equity_df: pd.DataFrame,
    key_prefix: str = "backtest_exposure",
) -> None:
    work = prepare_equity_curve(equity_df)
    col = "exposure" if "exposure" in work.columns else "capital_usage" if "capital_usage" in work.columns else None
    if "date" not in work.columns or col is None:
        st.info("equity curve CSV 缺少 exposure 或 capital_usage，略過曝險圖。")
        return
    chart_df = work.dropna(subset=["date", col])
    if chart_df.empty:
        st.info("目前沒有可繪製的曝險資料。")
        return
    fig = px.line(chart_df, x="date", y=col, title="Exposure / Capital Usage", labels={col: col})
    fig.update_yaxes(tickformat=".0%")
    safe_plotly_chart(apply_chart_layout(fig, 360), key_prefix)


def render_daily_return_histogram(
    equity_df: pd.DataFrame,
    key_prefix: str = "backtest_daily_return_histogram",
) -> None:
    work = prepare_equity_curve(equity_df)
    if "daily_return" not in work.columns:
        st.info("equity curve CSV 缺少 daily_return，略過每日報酬率分布。")
        return
    render_numeric_histogram(work, "daily_return", "每日報酬率分布", nbins=35, key_prefix=key_prefix)


def prepare_trade_log(trade_df: pd.DataFrame) -> pd.DataFrame:
    work = trade_df.copy()
    for col in ["return_pct", "pnl", "holding_days", "position_value", "shares"]:
        if col in work.columns:
            work[col] = numeric_series(work, col)
    for col in ["entry_date", "exit_date"]:
        if col in work.columns:
            work[col] = date_series(work, col)
    return work


def _aggregate_trades_by_entry_internal(trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    以 trade_id（單次進場）為單位，把同一次進場的所有出場記錄（含分批出場）
    合併為一筆「真實交易」記錄。

    回傳欄位：
      trade_id, layer_id, entry_layer_id, symbol, entry_id, entry_date, entry_price,
      final_exit_date, final_exit_price, total_holding_days,
      total_return_pct, is_win, exit_reasons, partial_exit_count
    """
    agg_cols = [
        "trade_id", "layer_id", "entry_layer_id", "symbol", "entry_id", "entry_date", "entry_price",
        "final_exit_date", "final_exit_price", "total_holding_days",
        "total_return_pct", "is_win", "exit_reasons", "partial_exit_count",
    ]
    if trade_log is None or trade_log.empty or "trade_id" not in trade_log.columns:
        return pd.DataFrame(columns=agg_cols)

    group_cols = ["trade_id"]
    if "entry_layer_id" in trade_log.columns:
        group_cols.append("entry_layer_id")
    elif "layer_id" in trade_log.columns:
        group_cols.append("layer_id")

    rows: list[dict] = []
    for group_key, g in trade_log.groupby(group_cols, sort=False, dropna=False):
        trade_id = group_key[0] if isinstance(group_key, tuple) else group_key
        if "exit_sequence" in g.columns:
            g_sorted = g.sort_values("exit_sequence")
        else:
            g_sorted = g

        n_exits = len(g_sorted)
        entry_date  = g_sorted["entry_date"].iloc[0]
        entry_price = float(g_sorted["entry_price"].iloc[0])

        if "shares" in g_sorted.columns:
            exit_shares = pd.to_numeric(g_sorted["shares"], errors="coerce").fillna(0.0)
            total_exit_shares = float(exit_shares.sum())
            weights = (exit_shares / total_exit_shares).tolist() if total_exit_shares > 0 else None
        else:
            weights = None
        if weights is None:
            weights = [1.0 / n_exits] * n_exits

        exit_prices = pd.to_numeric(g_sorted["exit_price"], errors="coerce").tolist()
        weighted_exit_price = sum(
            w * p for w, p in zip(weights, exit_prices) if pd.notna(p)
        )

        final_row = g_sorted.iloc[-1]
        final_exit_date = final_row.get("exit_date")
        total_holding_days = final_row.get("holding_days", np.nan)

        total_return_pct = (
            (weighted_exit_price - entry_price) / entry_price
            if entry_price else np.nan
        )
        is_win = bool(total_return_pct > 0) if pd.notna(total_return_pct) else False

        exit_reasons = " / ".join(
            str(r) for r in g_sorted["exit_reason"].tolist()
        ) if "exit_reason" in g_sorted.columns else ""

        rows.append({
            "trade_id":            trade_id,
            "layer_id":            final_row.get("layer_id", final_row.get("layer_at_exit")),
            "entry_layer_id":      final_row.get("entry_layer_id", final_row.get("layer_id")),
            "symbol":              g_sorted["symbol"].iloc[0] if "symbol" in g_sorted.columns else None,
            "entry_id":            g_sorted["entry_id"].iloc[0] if "entry_id" in g_sorted.columns else None,
            "entry_date":          entry_date,
            "entry_price":         entry_price,
            "final_exit_date":     final_exit_date,
            "final_exit_price":    weighted_exit_price,
            "total_holding_days":  total_holding_days,
            "total_return_pct":    total_return_pct,
            "is_win":              is_win,
            "exit_reasons":        exit_reasons,
            "partial_exit_count":  n_exits,
        })

    return pd.DataFrame(rows, columns=agg_cols)


def aggregate_real_trades(trade_df: pd.DataFrame) -> pd.DataFrame:
    work = prepare_trade_log(trade_df)
    # 只保留初始試倉層（entry_layer_id=1），與 PyCharm 回測勝率口徑一致
    if "entry_layer_id" in work.columns:
        work = work[pd.to_numeric(work["entry_layer_id"], errors="coerce") == 1].copy()
    result = _aggregate_trades_by_entry_internal(work)
    if result.empty:
        return work
    rename_map = {
        "total_return_pct": "return_pct",
        "total_holding_days": "holding_days",
        "final_exit_date": "exit_date",
    }
    for old, new in rename_map.items():
        if old in result.columns and new not in result.columns:
            result = result.rename(columns={old: new})
    return result


def render_trade_kpis(trade_df: pd.DataFrame) -> None:
    real_trades = aggregate_real_trades(trade_df)
    if real_trades.empty:
        st.info("trade log 目前沒有可統計的交易。")
        return
    returns = numeric_series(real_trades, "return_pct") if "return_pct" in real_trades.columns else pd.Series(dtype=float)
    wins = real_trades["is_win"] if "is_win" in real_trades.columns else (returns > 0)
    pnl = numeric_series(real_trades, "pnl") if "pnl" in real_trades.columns else pd.Series(dtype=float)
    holding = numeric_series(real_trades, "holding_days") if "holding_days" in real_trades.columns else pd.Series(dtype=float)
    cols = st.columns(4)
    cols[0].metric("真實交易數", f"{len(real_trades):,}")
    cols[1].metric("勝率", format_pct(wins.mean() if len(wins) else None))
    cols[2].metric("平均單筆報酬", format_pct(returns.mean() if not returns.dropna().empty else None))
    cols[3].metric("平均持有天數", format_value(holding.mean() if not holding.dropna().empty else None, 1))

    # 試倉層勝率
    entry1_win_rate = None
    if "entry_layer_id" in trade_df.columns:
        entry1_df = trade_df[pd.to_numeric(trade_df["entry_layer_id"], errors="coerce") == 1].copy()
        if not entry1_df.empty:
            entry1_agg = aggregate_real_trades(entry1_df)
            if "is_win" in entry1_agg.columns and not entry1_agg.empty:
                entry1_win_rate = float(entry1_agg["is_win"].mean())

    cols2 = st.columns(2)
    cols2[0].metric(
        "試倉層勝率（entry_layer_id=1）",
        format_pct(entry1_win_rate) if entry1_win_rate is not None else "N/A"
    )
    cols2[1].metric(
        "全層勝率（含加碼）",
        format_pct(wins.mean() if len(wins) else None)
    )

    if not pnl.dropna().empty:
        st.caption(f"累計 PnL：{format_value(pnl.sum(), 0)}")


def render_symbol_avg_return_bar(
    trade_df: pd.DataFrame,
    key_prefix: str = "trade_symbol_avg_return",
) -> None:
    real_trades = aggregate_real_trades(trade_df)
    if not {"symbol", "return_pct"}.issubset(real_trades.columns):
        st.info("trade log 缺少 symbol 或 return_pct，略過每檔平均報酬。")
        return
    work = real_trades.copy()
    work["return_pct"] = numeric_series(work, "return_pct")
    chart_df = work.dropna(subset=["return_pct"]).groupby("symbol", as_index=False)["return_pct"].mean()
    chart_df = chart_df.sort_values("return_pct", ascending=False).head(20)
    if chart_df.empty:
        st.info("目前沒有可繪製的每檔平均報酬。")
        return
    fig = px.bar(chart_df, x="symbol", y="return_pct", title="每檔股票平均 return_pct", labels={"return_pct": "平均報酬"})
    fig.update_yaxes(tickformat=".1%")
    safe_plotly_chart(apply_chart_layout(fig, 380), key_prefix)


def render_symbol_trade_count_bar(
    trade_df: pd.DataFrame,
    key_prefix: str = "trade_symbol_count",
) -> None:
    real_trades = aggregate_real_trades(trade_df)
    if "symbol" not in real_trades.columns:
        st.info("trade log 缺少 symbol，略過每檔交易次數。")
        return
    counts = real_trades["symbol"].astype(str).value_counts().head(20).reset_index()
    counts.columns = ["symbol", "count"]
    fig = px.bar(counts, x="symbol", y="count", text="count", title="每檔股票交易次數")
    safe_plotly_chart(apply_chart_layout(fig, 380), key_prefix)


def render_trade_date_distribution(
    trade_df: pd.DataFrame,
    key_prefix: str = "trade_date_distribution",
) -> None:
    work = prepare_trade_log(trade_df)
    date_cols = existing_columns(work, ["entry_date", "exit_date"])
    if not date_cols:
        st.info("trade log 缺少 entry_date / exit_date，略過交易時間分布。")
        return
    rows = []
    for col in date_cols:
        counts = work.dropna(subset=[col]).groupby(work[col].dt.date).size()
        rows.extend({"date": date, "count": count, "type": col} for date, count in counts.items())
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        st.info("目前沒有可繪製的交易時間分布。")
        return
    fig = px.line(chart_df, x="date", y="count", color="type", markers=True, title="Entry / Exit 日期分布")
    safe_plotly_chart(apply_chart_layout(fig, 370), key_prefix)


def render_cumulative_pnl_curve(
    trade_df: pd.DataFrame,
    key_prefix: str = "trade_cumulative_pnl",
) -> None:
    real_trades = aggregate_real_trades(trade_df)
    if "pnl" not in real_trades.columns:
        st.info("trade log 缺少 pnl，略過 cumulative pnl curve。")
        return
    work = real_trades.copy()
    work["pnl"] = numeric_series(work, "pnl")
    if "exit_date" in work.columns:
        work = work.sort_values("exit_date")
    work = work.dropna(subset=["pnl"])
    if work.empty:
        st.info("目前沒有可繪製的累積 PnL。")
        return
    work["cumulative_pnl"] = work["pnl"].cumsum()
    x_col = "exit_date" if "exit_date" in work.columns and work["exit_date"].notna().any() else work.index
    fig = px.line(work, x=x_col, y="cumulative_pnl", markers=True, title="Cumulative PnL")
    safe_plotly_chart(apply_chart_layout(fig, 370), key_prefix)


def entry_summary_values(entry_df: pd.DataFrame) -> dict[str, float]:
    if entry_df is None or entry_df.empty:
        return {}
    if {"metric", "value"}.issubset(entry_df.columns):
        return metrics_lookup(entry_df)
    summary: dict[str, float] = {}
    for col in entry_df.columns:
        values = numeric_series(entry_df, col)
        if values.notna().any():
            summary[col] = float(values.sum())
    if "candidate_entry_signals" in summary and summary["candidate_entry_signals"]:
        summary["entry_execution_rate"] = summary.get("executed_entries", 0.0) / summary["candidate_entry_signals"]
        summary["volume_confirmation_rate"] = summary.get("candidate_with_volume_confirmation", 0.0) / summary["candidate_entry_signals"]
    return summary


def render_entry_kpis(entry_df: pd.DataFrame) -> None:
    summary = entry_summary_values(entry_df)
    candidates = metric_lookup_value(summary, "candidate_entry_signals")
    executed = metric_lookup_value(summary, "executed_entries")
    rate = metric_lookup_value(summary, "entry_execution_rate")
    if rate is None and candidates not in (None, 0) and executed is not None:
        rate = executed / candidates
    cols = st.columns(3)
    cols[0].metric("候選訊號數", format_value(candidates, 0))
    cols[1].metric("實際進場數", format_value(executed, 0))
    cols[2].metric("進場執行率", format_pct(rate))


def render_entry_daily_line(
    entry_df: pd.DataFrame,
    key_prefix: str = "entry_daily_line",
) -> None:
    cols = existing_columns(entry_df, ["candidate_entry_signals", "executed_entries"])
    if "date" not in entry_df.columns or len(cols) < 2:
        st.info("entry diagnostic CSV 缺少 date / candidate_entry_signals / executed_entries，略過每日進場圖。")
        return
    work = entry_df.copy()
    work["date"] = date_series(work, "date")
    for col in cols:
        work[col] = numeric_series(work, col)
    long_df = work.dropna(subset=["date"]).melt(id_vars=["date"], value_vars=cols, var_name="metric", value_name="count").dropna()
    if long_df.empty:
        st.info("目前沒有可繪製的每日進場診斷資料。")
        return
    fig = px.line(long_df, x="date", y="count", color="metric", title="候選訊號 vs 實際進場", markers=True)
    safe_plotly_chart(apply_chart_layout(fig, 390), key_prefix)


def render_entry_reason_bar(
    entry_df: pd.DataFrame,
    columns: list[str],
    title: str,
    key_prefix: str | None = None,
) -> None:
    existing = existing_columns(entry_df, columns)
    if not existing:
        st.info(f"entry diagnostic CSV 缺少「{title}」需要的欄位。")
        return
    rows = []
    for col in existing:
        value = numeric_series(entry_df, col).sum()
        rows.append({"reason": col, "count": value})
    chart_df = pd.DataFrame(rows).sort_values("count", ascending=True)
    chart_df = chart_df[chart_df["count"] > 0]
    if chart_df.empty:
        st.info(f"目前沒有可繪製的「{title}」。")
        return
    fig = px.bar(chart_df, x="count", y="reason", orientation="h", title=title, labels={"count": "數量", "reason": ""})
    fig.update_traces(marker_color="#1f4e79")
    safe_plotly_chart(apply_chart_layout(fig, 430), key_prefix or f"entry_reason_{title}")


def render_volume_confirmation_rate(
    entry_df: pd.DataFrame,
    key_prefix: str = "entry_volume_confirmation_rate",
) -> None:
    if "date" not in entry_df.columns or "volume_confirmation_rate" not in entry_df.columns:
        st.info("entry diagnostic CSV 缺少 volume_confirmation_rate，略過成交量確認率時序圖。")
        return
    work = entry_df.copy()
    work["date"] = date_series(work, "date")
    work["volume_confirmation_rate"] = numeric_series(work, "volume_confirmation_rate")
    chart_df = work.dropna(subset=["date", "volume_confirmation_rate"])
    if chart_df.empty:
        st.info("目前沒有可繪製的成交量確認率。")
        return
    fig = px.line(chart_df, x="date", y="volume_confirmation_rate", title="Volume Confirmation Rate", markers=True)
    fig.update_yaxes(tickformat=".0%")
    safe_plotly_chart(apply_chart_layout(fig, 360), key_prefix)


def render_single_radar(row: pd.Series, key_prefix: str = "single_score_radar") -> None:
    metrics = ["final_decision_score", "combined_decision_score", "safety_score", "return_score", "rule_filter_score"]
    values: list[float] = []
    labels: list[str] = []
    for metric in metrics:
        if metric in row.index:
            value = to_float(row.get(metric))
            if value is not None:
                labels.append(metric)
                values.append(max(0.0, min(value, 1.0)))
    if len(values) < 3:
        st.info("診斷資料不足，無法繪製雷達圖。")
        return
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + values[:1],
                theta=labels + labels[:1],
                fill="toself",
                name="score",
                line_color="#1f4e79",
            )
        ]
    )
    fig.update_layout(
        title="分數輪廓",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=390,
        margin=dict(l=30, r=30, t=55, b=25),
        font=dict(size=13),
    )
    safe_plotly_chart(fig, key_prefix)


def render_price_level_chart(row: pd.Series, key_prefix: str = "single_price_level") -> None:
    price_columns = [
        ("收盤價", "close"),
        ("進場價", "entry_price"),
        ("停損價", "stop_loss_price_adjusted"),
        ("停利價", "take_profit_price"),
        ("支撐", "support_price"),
        ("壓力", "resistance_price"),
    ]
    rows = []
    for label, column in price_columns:
        if column in row.index:
            value = to_float(row.get(column))
            if value is not None:
                rows.append({"level": label, "price": value})
    if len(rows) < 2:
        st.info("價格區間資料不足，無法繪製價格圖。")
        return
    chart_df = pd.DataFrame(rows)
    fig = px.scatter(chart_df, x="price", y="level", text="price", title="價格區間", labels={"price": "價格", "level": ""})
    fig.update_traces(marker=dict(size=14, color="#d97706"), texttemplate="%{text:.2f}", textposition="middle right")
    fig.update_layout(height=390, margin=dict(l=20, r=40, t=55, b=25), yaxis=dict(title=None), font=dict(size=13))
    safe_plotly_chart(fig, key_prefix)


def render_trade_plan(row: pd.Series) -> None:
    columns = [
        "entry_price",
        "stop_loss_price_adjusted",
        "stop_loss_pct",
        "take_profit_price",
        "risk_reward_ratio",
        "suggested_position_pct",
    ]
    rows = []
    for column in columns:
        if column in row.index and pd.notna(row.get(column)):
            value = row.get(column)
            if column in {"stop_loss_pct", "suggested_position_pct"}:
                display = format_pct(value)
            elif "ratio" in column:
                display = format_value(value, 2)
            else:
                display = format_value(value, 2)
            rows.append({"項目": column, "數值": display})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("目前診斷 CSV 未提供完整交易計畫欄位。")


def prepare_validation_metrics(validation_df: pd.DataFrame | None) -> pd.DataFrame:
    if validation_df is None or validation_df.empty:
        return FALLBACK_VALIDATION_METRICS.copy()

    display_map = {
        "mae": "MAE",
        "baseline_mae": "Baseline MAE",
        "rmse": "RMSE",
        "baseline_rmse": "Baseline RMSE",
        "r2": "R2",
        "spearman_corr": "Spearman",
        "top5_lift": "Top5 Lift",
        "top10_lift": "Top10 Lift",
        "long_short_spread": "Long-Short Spread",
    }
    lower_to_col = {str(col).lower(): col for col in validation_df.columns}
    row = {"model": "Safety model"}
    for source, display in display_map.items():
        col = lower_to_col.get(source)
        if col is not None:
            values = numeric_series(validation_df, col).dropna()
            row[display] = values.mean() if not values.empty else np.nan
    if len(row) == 1:
        return FALLBACK_VALIDATION_METRICS.copy()
    return pd.DataFrame([row])


def render_validation_charts(metrics_df: pd.DataFrame, key_prefix: str = "validation") -> None:
    if "model" not in metrics_df.columns:
        st.info("validation metrics 缺少 model 欄位，略過驗證圖。")
        return

    for col in ["MAE", "Baseline MAE", "RMSE", "Baseline RMSE", "Spearman", "Top5 Lift", "Top10 Lift", "Long-Short Spread"]:
        if col in metrics_df.columns:
            metrics_df[col] = numeric_series(metrics_df, col)

    if {"MAE", "Baseline MAE"}.issubset(metrics_df.columns):
        mae_df = metrics_df.melt(id_vars=["model"], value_vars=["MAE", "Baseline MAE"], var_name="metric", value_name="value")
        fig_mae = px.bar(mae_df, x="model", y="value", color="metric", barmode="group", title="MAE vs Baseline MAE")
        safe_plotly_chart(apply_chart_layout(fig_mae, 360), f"{key_prefix}_mae_metrics")
    else:
        st.info("validation CSV 缺少 MAE / baseline_mae，略過 MAE 比較圖。")

    if {"RMSE", "Baseline RMSE"}.issubset(metrics_df.columns):
        rmse_df = metrics_df.melt(id_vars=["model"], value_vars=["RMSE", "Baseline RMSE"], var_name="metric", value_name="value")
        fig_rmse = px.bar(rmse_df, x="model", y="value", color="metric", barmode="group", title="RMSE vs Baseline RMSE")
        safe_plotly_chart(apply_chart_layout(fig_rmse, 360), f"{key_prefix}_rmse_metrics")
    else:
        st.info("validation CSV 缺少 RMSE / baseline_rmse，略過 RMSE 比較圖。")

    perf_cols = [col for col in ["Spearman", "Top5 Lift", "Top10 Lift", "Long-Short Spread"] if col in metrics_df.columns]
    if perf_cols:
        perf_df = metrics_df.melt(id_vars=["model"], value_vars=perf_cols, var_name="metric", value_name="value")
        fig_perf = px.bar(perf_df, x="metric", y="value", color="model", barmode="group", title="排序與投組效果")
        safe_plotly_chart(apply_chart_layout(fig_perf, 380), f"{key_prefix}_ranking_metrics")
    else:
        st.info("validation CSV 缺少 Spearman / Top Lift / Long-Short Spread 欄位。")


def render_validation_fold_chart(
    validation_df: pd.DataFrame | None,
    key_prefix: str = "validation_fold_metrics",
) -> None:
    if validation_df is None or validation_df.empty or "fold" not in validation_df.columns:
        st.info("validation CSV 沒有 fold 欄位，略過 fold-level 圖。")
        return
    cols = existing_columns(validation_df, ["mae", "spearman_corr"])
    if not cols:
        st.info("validation CSV 缺少 fold-level MAE / Spearman 欄位。")
        return
    work = validation_df.copy()
    work["fold"] = numeric_series(work, "fold")
    for col in cols:
        work[col] = numeric_series(work, col)
    long_df = work.melt(id_vars=["fold"], value_vars=cols, var_name="metric", value_name="value").dropna()
    if long_df.empty:
        st.info("目前沒有可繪製的 fold-level validation 資料。")
        return
    fig = px.line(long_df, x="fold", y="value", color="metric", markers=True, title="Fold-level MAE / Spearman")
    safe_plotly_chart(apply_chart_layout(fig, 380), key_prefix)


def feature_dataframe() -> pd.DataFrame:
    rows = []
    for category, config in FEATURE_GROUPS.items():
        for feature in config["features"]:
            rows.append({"類別": category, "特徵": feature, "說明": config["description"]})
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="台股 ML 投資決策支援系統",
    page_icon="📈",
    layout="wide",
)
try:
    st.set_option("client.toolbarMode", "minimal")
except Exception:
    pass
inject_css()
init_run_state()
sync_process_state()

with st.sidebar:
    st.title("模型執行設定")
    analysis_mode = st.radio(
        label="分析模式",
        options=["個股深度分析", "多股票排名分析", "回測分析"],
        index=0,
        horizontal=False,
    )
    market = st.selectbox(label="市場", options=["TWSE", "TPEX"], index=0)

    if analysis_mode == "個股深度分析":
        symbol = normalize_symbol(st.text_input(label="股票代碼", value="2330"))
        top_n = 30
    elif analysis_mode == "多股票排名分析":
        symbol = "ALL"
        top_n = st.number_input(label="Top N", min_value=5, value=30, step=1)
    else:
        backtest_target = st.radio(
            label="回測對象",
            options=["Top N 股票池", "指定個股"],
            index=0,
            horizontal=False,
        )
        if backtest_target == "指定個股":
            symbol = normalize_symbol(st.text_input(label="回測股票代碼", value="2330"))
            top_n = 30
        else:
            symbol = "ALL"
            top_n = st.number_input(label="回測 Top N", min_value=5, value=30, step=1)

    command = make_command(analysis_mode, market, symbol or "2330", int(top_n))

    st.caption(f"目前狀態：{run_status_label()}")
    start_disabled = not PIPELINE_SCRIPT.exists() or st.session_state.model_running
    if st.button("開始執行模型", type="primary", use_container_width=True, disabled=start_disabled):
        ok, message = start_model_process(command)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    if st.session_state.model_running:
        st.info("模型背景執行中。")
        if st.button("重新整理狀態", use_container_width=True):
            st.rerun()
        if st.button("停止模型執行", use_container_width=True):
            ok, message = stop_model_process()
            if ok:
                st.warning(message)
                st.rerun()
            else:
                st.info(message)

ranking_path, ranking_df, ranking_error = load_latest_ranking_csv(analysis_mode)
single_path, single_df, single_error = load_latest_single_diagnostic_csv(symbol if symbol != "ALL" else None)
validation_path, validation_df, validation_error = load_latest_validation_csv()
equity_path, equity_df, equity_error = load_latest_backtest_equity_curve()
trade_path, trade_df, trade_error = load_latest_backtest_trade_log()
metrics_path, metrics_df, metrics_error = load_latest_portfolio_metrics()
entry_path, entry_df, entry_error = load_latest_entry_diagnostic()
sensitivity_path, sensitivity_df, sensitivity_error = load_latest_sensitivity_summary()

if st.session_state.model_running:
    st.info("模型正在背景執行中。你可以切換分頁查看目前已存在的圖表與 CSV；需要更新狀態時請按側邊欄的「重新整理狀態」。")
elif st.session_state.last_status == "執行成功":
    st.success("上一輪模型已執行成功，Dashboard 已讀取 output 資料夾中的最新 CSV。")
elif st.session_state.last_status == "執行失敗":
    st.error("上一輪模型執行失敗，請到「執行紀錄」查看 stderr。")
elif st.session_state.last_status == "已停止":
    st.warning("上一輪模型已由使用者停止。")

(
    tab_overview,
    tab_ranking,
    tab_single,
    tab_backtest,
    tab_trades,
    tab_entry,
    tab_validation,
    tab_features,
    tab_logs,
) = st.tabs(
    ["系統總覽", "多股票排名", "個股診斷", "回測績效", "交易明細", "進場診斷", "模型驗證", "特徵解釋", "執行紀錄"]
)

with tab_overview:
    st.title("台股機器學習投資決策支援系統")
    st.caption("Machine Learning Based Stock Investment Decision Support Dashboard")
    st.info("本系統整合機器學習模型、技術指標、價格結構、風險控管與 Risk/Reward，輸出可解釋的投資決策建議。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型核心", "Extra Trees")
    c2.metric("特徵組合", CORE_MODEL_SETTINGS["feature_set"])
    c3.metric("主目標", CORE_MODEL_SETTINGS["target_col"])
    c4.metric("決策架構", "模型 + 規則 + 風控")

    st.markdown("#### 目前執行與後端設定")
    settings_rows = [
        {"參數": "mode", "值": cli_mode_for_analysis(analysis_mode)},
        {"參數": "market", "值": market},
        {"參數": "symbol", "值": symbol or "2330"},
        {"參數": "top_n", "值": int(top_n)},
        {"參數": "feature_set", "值": CORE_MODEL_SETTINGS["feature_set"]},
        {"參數": "model_type", "值": CORE_MODEL_SETTINGS["model_type"]},
        {"參數": "model_profile", "值": CORE_MODEL_SETTINGS["model_profile"]},
        {"參數": "target_col", "值": CORE_MODEL_SETTINGS["target_col"]},
        {"參數": "diagnostic_scope", "值": CORE_MODEL_SETTINGS["diagnostic_scope"]},
        {"參數": "display_mode", "值": CORE_MODEL_SETTINGS["display_mode"]},
        {"參數": "backtest profile", "值": BACKTEST_SETTINGS["backtest_profile"]},
        {"參數": "backtest entry mode", "值": BACKTEST_SETTINGS["backtest_entry_mode"]},
        {"參數": "max_positions", "值": BACKTEST_SETTINGS["max_positions"]},
        {"參數": "initial / add1 / add2", "值": "8% / 6% / 6%"},
        {"參數": "max_single_position_pct", "值": format_pct(BACKTEST_SETTINGS["max_single_position_pct"])},
        {"參數": "max_total_exposure", "值": format_pct(BACKTEST_SETTINGS["max_total_exposure"])},
        {"參數": "min_cash_reserve", "值": format_pct(BACKTEST_SETTINGS["min_cash_reserve"])},
    ]
    st.dataframe(pd.DataFrame(settings_rows), use_container_width=True, hide_index=True, height=360)

    st.markdown("#### 最新 output 概況")
    overview_files = [
        ("ranking", ranking_path),
        ("single_diagnostic", single_path),
        ("validation", validation_path),
        ("backtest_equity_curve", equity_path),
        ("backtest_trade_log", trade_path),
        ("portfolio_metrics", metrics_path),
        ("entry_diagnostic", entry_path),
        ("sensitivity_summary", sensitivity_path),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {"資料類型": label, "檔名": path.name if path else "-", "修改時間": file_time(path)}
                for label, path in overview_files
            ]
        ),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

    if ranking_error:
        st.warning(f"ranking CSV 讀取失敗：{ranking_error}")
    elif ranking_path is None or ranking_df is None or ranking_df.empty:
        st.info("尚未讀取到最新 ranking CSV。")
    else:
        st.caption(f"最新 ranking CSV：`{ranking_path.name}`，更新時間：{file_time(ranking_path)}")
        ranking_work = prepare_final_decision_ranking(prepare_common_model_ranking(ranking_df))
        stock_count = int(ranking_work["symbol"].nunique()) if "symbol" in ranking_work.columns else len(ranking_work)
        avg_final = numeric_series(ranking_work, "final_decision_score").mean() if "final_decision_score" in ranking_work.columns else np.nan
        avg_combined = numeric_series(ranking_work, "combined_decision_score").mean() if "combined_decision_score" in ranking_work.columns else np.nan
        veto_count = int(ranking_work["trade_veto_flag"].map(truthy_flag).sum()) if "trade_veto_flag" in ranking_work.columns else 0
        top_row = ranking_work.iloc[0] if not ranking_work.empty else None
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("股票數量", f"{stock_count:,}")
        k2.metric("平均 final_decision_score", format_value(avg_final, 4))
        k3.metric("平均 combined_decision_score", format_value(avg_combined, 4))
        k4.metric("Veto 數量", f"{veto_count:,}")
        k5.metric("Top 1 股票", row_stock_label(top_row))

    if equity_error:
        st.warning(f"equity curve CSV 讀取失敗：{equity_error}")
    elif equity_df is not None and not equity_df.empty:
        st.markdown("#### 回測績效摘要")
        st.caption(f"最新 equity curve：`{equity_path.name}`，更新時間：{file_time(equity_path)}")
        render_backtest_kpis(equity_df, metrics_df)
        render_equity_curve_chart(equity_df, key_prefix="overview_equity_curve")
    else:
        st.info("尚未讀取到 backtest equity curve CSV。")

    if trade_error:
        st.warning(f"trade log CSV 讀取失敗：{trade_error}")
    elif trade_df is not None and not trade_df.empty:
        st.markdown("#### 交易摘要")
        st.caption(f"最新 trade log：`{trade_path.name}`，更新時間：{file_time(trade_path)}")
        render_trade_kpis(trade_df)
    else:
        st.info("尚未讀取到 backtest trade log CSV。")

with tab_ranking:
    st.title("多股票排名分析")
    if ranking_path is None:
        st.info("尚未讀取到最新 ranking CSV，請先執行模型。")
    elif ranking_error:
        st.error(f"ranking CSV 讀取失敗：{ranking_error}")
    elif ranking_df is None or ranking_df.empty:
        st.warning("ranking CSV 沒有資料。")
    else:
        common_work = prepare_common_model_ranking(ranking_df)
        final_work = prepare_final_decision_ranking(common_work)
        st.caption(f"讀取檔案：`{ranking_path.name}`，更新時間：{file_time(ranking_path)}")

        stock_count = int(common_work["symbol"].nunique()) if "symbol" in common_work.columns else len(common_work)
        veto_count = int(common_work["trade_veto_flag"].map(truthy_flag).sum()) if "trade_veto_flag" in common_work.columns else 0
        metrics = [("股票數量", f"{stock_count:,}")]
        if "combined_decision_score" in common_work.columns:
            avg_combined = common_work["combined_decision_score"].mean()
            metrics.append(("平均 combined_decision_score", "-" if pd.isna(avg_combined) else f"{avg_combined:.4f}"))
        if "final_decision_score" in final_work.columns:
            avg_final = final_work["final_decision_score"].mean()
            metrics.append(("平均 final_decision_score", "-" if pd.isna(avg_final) else f"{avg_final:.4f}"))
        metrics.append(("trade_veto_flag=True 數量", f"{veto_count:,}"))

        metric_cols = st.columns(len(metrics))
        for col, (label, value) in zip(metric_cols, metrics):
            col.metric(label, value)

        b1, b2 = st.columns(2)
        best_common = common_work.iloc[0] if not common_work.empty else None
        best_final = final_work.iloc[0] if "final_decision_score" in final_work.columns and not final_work.empty else None
        common_score = format_value(best_common.get("combined_decision_score"), 4) if best_common is not None else "-"
        final_score = format_value(best_final.get("final_decision_score"), 4) if best_final is not None else "-"
        b1.metric("最高共同模型分數股票", row_stock_label(best_common), common_score)
        b2.metric("最高最終決策分數股票", row_stock_label(best_final), final_score)

        st.markdown("### 共同模型排名（與終端機 Rank 表一致）")
        st.caption(
            "此排名反映第一層 shared common model 的 safety_score、return_score 與 combined_decision_score，"
            "對應終端機輸出的 Rank / Safety / Return / Combined。"
        )
        render_common_model_bar(common_work, key_prefix="ranking_common_model_bar")
        common_cols = existing_columns(common_work, COMMON_MODEL_COLUMNS)
        st.dataframe(common_work[common_cols] if common_cols else common_work, use_container_width=True, height=420, hide_index=True)

        st.markdown("### 最終決策排名（加入規則過濾與風控後）")
        st.caption(
            "此排名使用 final_decision_score，已納入 rule_filter_score、risk_reward_score 與 trade_veto_flag，"
            "因此可能與共同模型排名不同。"
        )
        if "final_decision_score" in final_work.columns:
            render_top15_bar(final_work, key_prefix="ranking_final_score_top15")
            final_cols = existing_columns(final_work, FINAL_DECISION_COLUMNS)
            st.dataframe(final_work[final_cols] if final_cols else final_work, use_container_width=True, height=520, hide_index=True)
        else:
            st.warning("ranking CSV 缺少 final_decision_score，無法顯示最終決策排名。")

        st.markdown("### Final Score vs R/R 交易可行性")
        st.caption(
            "x 軸是 final_decision_score，不是 combined_decision_score；"
            "y 軸是 risk_reward_ratio 或 practical_risk_reward_ratio。此圖用於判斷交易可行性，不是第一層模型排名。"
        )
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            render_score_rr_scatter(final_work, key_prefix="ranking_score_rr_scatter")
        with col_b:
            render_veto_reason_chart(final_work, key_prefix="ranking_veto_reason")

        st.markdown("### 分數、標籤與風控分布")
        r1, r2 = st.columns(2)
        with r1:
            render_safety_return_scatter(final_work, key_prefix="ranking_safety_return_scatter")
            render_value_count_bar(final_work, "final_decision_label", "final_decision_label 分布", key_prefix="ranking_final_label_distribution")
        with r2:
            render_trade_veto_distribution(final_work, key_prefix="ranking_trade_veto_distribution")
            render_numeric_histogram(final_work, "final_decision_score", "final_decision_score 分布", key_prefix="ranking_final_score_histogram")

        st.markdown("### Risk/Reward 與倉位建議")
        rr1, rr2 = st.columns(2)
        with rr1:
            render_numeric_histogram(final_work, "risk_reward_ratio", "risk_reward_ratio 分布", key_prefix="ranking_rr_histogram")
            render_top_numeric_bar(final_work, "practical_risk_reward_ratio", "Top 15 practical_risk_reward_ratio", key_prefix="ranking_practical_rr_top15")
        with rr2:
            render_numeric_histogram(final_work, "practical_risk_reward_ratio", "practical_risk_reward_ratio 分布", key_prefix="ranking_practical_rr_histogram")
            render_top_numeric_bar(final_work, "suggested_position_pct", "Top 15 suggested_position_pct", key_prefix="ranking_suggested_position_top15")

        st.markdown("### 支撐與壓力距離")
        render_support_resistance_scatter(final_work, key_prefix="ranking_support_resistance_scatter")

        st.download_button(
            "下載 ranking CSV",
            data=ranking_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=ranking_path.name,
            mime="text/csv",
        )

with tab_single:
    st.title("個股深度診斷")
    if single_path is None:
        st.info("尚未讀取到最新 common_single_diagnostic / single_diagnostic CSV，請先執行個股深度分析。")
    elif single_error:
        st.error(f"single diagnostic CSV 讀取失敗：{single_error}")
    elif single_df is None or single_df.empty:
        st.warning("single diagnostic CSV 沒有資料。")
    else:
        row = get_latest_row(single_df)
        if row is None:
            st.warning("single diagnostic CSV 沒有可顯示資料。")
        else:
            st.caption(f"讀取檔案：`{single_path.name}`，更新時間：{file_time(single_path)}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("股票代碼", row.get("symbol", symbol))
            c2.metric("最新日期", row.get("date", "-"))
            c3.metric("收盤價", format_value(row.get("close"), 2))
            c4.metric("final_decision_label", row.get("final_decision_label", "-"))

            action_text = row.get("decision_action", row.get("suggested_action_text", ""))
            veto_reason = row.get("trade_veto_reason", "")
            if action_text:
                st.info(str(action_text))
            if veto_reason:
                st.warning(f"trade_veto_reason：{veto_reason}")

            for metric_names in SINGLE_METRIC_ROWS:
                cols = st.columns(3)
                for idx, metric_name in enumerate(metric_names):
                    if metric_name in row.index:
                        value = row.get(metric_name)
                        if metric_name in {"stop_loss_pct", "suggested_position_pct"}:
                            display = format_pct(value)
                        elif "ratio" in metric_name:
                            display = format_value(value, 2)
                        else:
                            display = format_value(value, 4)
                        cols[idx].metric(metric_name, display)

            left, right = st.columns(2)
            with left:
                render_single_radar(row, key_prefix="single_score_radar")
            with right:
                render_price_level_chart(row, key_prefix="single_price_level")

            st.markdown("#### 交易計畫")
            render_trade_plan(row)

            st.markdown("#### 個股歷史與技術指標")
            s1, s2 = st.columns(2)
            with s1:
                render_single_metric_timeseries(single_df, key_prefix="single_score_timeseries")
                render_single_indicator_chart(single_df, ["RSI14"], "RSI14 時序", reference_lines=[(70, "70"), (30, "30")], key_prefix="single_rsi14_timeseries")
                render_single_indicator_chart(single_df, ["K", "D", "J"], "K / D / J 時序", key_prefix="single_kdj_timeseries")
                render_single_indicator_chart(single_df, ["support_distance", "resistance_distance"], "支撐距離 / 壓力距離時序", key_prefix="single_support_resistance_timeseries")
            with s2:
                render_single_close_ma_chart(single_df, key_prefix="single_close_ma_timeseries")
                render_single_indicator_chart(single_df, ["MACD", "DIF", "DEA"], "MACD / DIF / DEA 時序", key_prefix="single_macd_timeseries")
                render_single_indicator_chart(single_df, ["atr_pct"], "atr_pct 時序", key_prefix="single_atr_pct_timeseries")
                render_single_indicator_chart(single_df, ["stop_loss_pct", "practical_risk_reward_ratio"], "stop_loss_pct / practical_risk_reward_ratio 時序", key_prefix="single_stop_rr_timeseries")
            render_single_price_plan_lines(single_df, key_prefix="single_price_plan_lines")

            if "enhanced_decision_report_text" in row.index and pd.notna(row.get("enhanced_decision_report_text")):
                with st.expander("查看完整決策報告", expanded=False):
                    st.write(str(row.get("enhanced_decision_report_text")))

            guide_cols = ["decision_guide_text", "suggested_action_text", "decision_summary_text", "decision_reason_text"]
            guide_text = "\n\n".join(str(row[col]) for col in guide_cols if col in row.index and pd.notna(row[col]))
            if guide_text:
                with st.expander("查看決策說明與建議", expanded=False):
                    st.write(guide_text)

            with st.expander("查看完整診斷資料", expanded=False):
                st.dataframe(single_df, use_container_width=True, height=520, hide_index=True)

with tab_backtest:
    st.title("回測績效")
    if equity_path is None:
        st.info("尚未讀取到 backtest equity curve CSV。請先執行回測分析。")
    elif equity_error:
        st.error(f"backtest equity curve CSV 讀取失敗：{equity_error}")
    elif equity_df is None or equity_df.empty:
        st.warning("backtest equity curve CSV 沒有資料。")
    else:
        st.caption(f"讀取檔案：`{equity_path.name}`，更新時間：{file_time(equity_path)}")
        if metrics_error:
            st.warning(f"portfolio metrics CSV 讀取失敗：{metrics_error}")
        render_backtest_kpis(equity_df, metrics_df)

        b1, b2 = st.columns([1.2, 1])
        with b1:
            render_equity_curve_chart(equity_df, key_prefix="backtest_equity_curve")
        with b2:
            render_drawdown_chart(equity_df, key_prefix="backtest_drawdown")

        b3, b4 = st.columns(2)
        with b3:
            render_cash_market_equity_chart(equity_df, key_prefix="backtest_cash_market_equity")
            render_daily_return_histogram(equity_df, key_prefix="backtest_daily_return_histogram")
        with b4:
            render_exposure_chart(equity_df, key_prefix="backtest_exposure")

        if metrics_path is not None and metrics_df is not None and not metrics_df.empty:
            with st.expander("查看 portfolio metrics / performance CSV", expanded=False):
                st.caption(f"讀取檔案：`{metrics_path.name}`，更新時間：{file_time(metrics_path)}")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True, height=420)
        else:
            st.info("尚未讀取到 portfolio_metrics / backtest_metrics / metrics CSV；CAGR、MDD 等指標會由 equity_curve 估算。")

        st.download_button(
            "下載 equity curve CSV",
            data=equity_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=equity_path.name,
            mime="text/csv",
        )

with tab_trades:
    st.title("交易明細")
    if trade_path is None:
        st.info("尚未讀取到 backtest trade log CSV。請先執行回測分析。")
    elif trade_error:
        st.error(f"backtest trade log CSV 讀取失敗：{trade_error}")
    elif trade_df is None or trade_df.empty:
        st.warning("backtest trade log CSV 沒有資料。")
    else:
        st.caption(f"讀取檔案：`{trade_path.name}`，更新時間：{file_time(trade_path)}")
        trade_work = prepare_trade_log(trade_df)
        render_trade_kpis(trade_work)
        st.dataframe(trade_work, use_container_width=True, hide_index=True, height=430)

        t1, t2 = st.columns(2)
        with t1:
            render_numeric_histogram(trade_work, "return_pct", "return_pct 分布", key_prefix="trade_return_histogram")
            render_symbol_avg_return_bar(trade_work, key_prefix="trade_symbol_return_chart")
            render_value_count_bar(trade_work, "exit_reason", "exit_reason 分布", key_prefix="trade_exit_reason_chart")
        with t2:
            render_numeric_histogram(trade_work, "holding_days", "holding_days 分布", key_prefix="trade_holding_days_histogram")
            render_symbol_trade_count_bar(trade_work, key_prefix="trade_symbol_count_chart")
            render_trade_date_distribution(trade_work, key_prefix="trade_date_distribution")
        render_cumulative_pnl_curve(trade_work, key_prefix="trade_cumulative_pnl_curve")

        with st.expander("查看以 trade_id 聚合後的真實交易", expanded=False):
            real_trade_df = aggregate_real_trades(trade_work)
            st.dataframe(real_trade_df, use_container_width=True, hide_index=True, height=420)

        st.download_button(
            "下載 trade log CSV",
            data=trade_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=trade_path.name,
            mime="text/csv",
        )

with tab_entry:
    st.title("進場診斷")
    if entry_path is None:
        st.info("尚未讀取到 entry diagnostic / rejection summary CSV。請先執行回測分析。")
    elif entry_error:
        st.error(f"entry diagnostic CSV 讀取失敗：{entry_error}")
    elif entry_df is None or entry_df.empty:
        st.warning("entry diagnostic CSV 沒有資料。")
    else:
        st.caption(f"讀取檔案：`{entry_path.name}`，更新時間：{file_time(entry_path)}")
        render_entry_kpis(entry_df)
        render_entry_daily_line(entry_df, key_prefix="entry_candidate_executed_chart")

        rejection_cols = [
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
        entry_type_cols = [
            "executed_by_strict_support",
            "executed_by_near_support_relaxed",
            "executed_by_strong_momentum",
            "executed_by_quiet_pullback_support",
            "executed_by_volume_rebound_support",
            "executed_by_volume_breakout_momentum",
        ]
        e1, e2 = st.columns(2)
        with e1:
            render_entry_reason_bar(entry_df, rejection_cols, "Rejection reason 分布", key_prefix="entry_rejection_reason_chart")
        with e2:
            render_entry_reason_bar(entry_df, entry_type_cols, "進場類型分布", key_prefix="entry_execution_type_chart")
        render_volume_confirmation_rate(entry_df, key_prefix="entry_volume_confirmation_rate")

        st.dataframe(entry_df, use_container_width=True, hide_index=True, height=430)
        st.download_button(
            "下載 entry diagnostic CSV",
            data=entry_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=entry_path.name,
            mime="text/csv",
        )

with tab_validation:
    st.title("模型驗證與表現")
    if validation_error:
        st.warning(f"validation CSV 讀取失敗，暫用內建摘要指標：{validation_error}")
    elif validation_path is not None and validation_df is not None and not validation_df.empty:
        st.caption(f"讀取檔案：`{validation_path.name}`，更新時間：{file_time(validation_path)}")
    else:
        st.caption("尚未讀取到 validation CSV，暫用系統內建的 shared common model 摘要指標。")

    validation_metrics = prepare_validation_metrics(validation_df if not validation_error else None)
    safety = validation_metrics.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Safety Spearman", format_value(safety.get("Spearman"), 4))
    mae_improvement = None
    if pd.notna(safety.get("Baseline MAE")) and pd.notna(safety.get("MAE")):
        mae_improvement = safety.get("Baseline MAE") - safety.get("MAE")
    k2.metric("Safety MAE improvement", format_value(mae_improvement, 4))
    k3.metric("Top5 Lift", format_value(safety.get("Top5 Lift"), 4))
    k4.metric("Long-Short Spread", format_value(safety.get("Long-Short Spread"), 4))

    render_validation_charts(validation_metrics, key_prefix="validation")
    render_validation_fold_chart(validation_df if not validation_error else None, key_prefix="validation_fold_metrics")
    st.info(
        "Safety model 是主要模型，Return model 是輔助模型。"
        "最終決策不是單純看模型分數，而是模型 + rule_filter + risk_reward 的組合。"
    )
    with st.expander("查看完整驗證資料", expanded=False):
        if validation_path is not None and validation_df is not None and not validation_df.empty:
            st.dataframe(validation_df, use_container_width=True, hide_index=True, height=420)
        else:
            st.dataframe(validation_metrics, use_container_width=True, hide_index=True)

with tab_features:
    st.title("特徵解釋與決策依據")
    st.dataframe(feature_dataframe(), use_container_width=True, hide_index=True, height=520)
    st.markdown("#### 最終決策分數公式")
    try:
        import pipeline_risk as _risk_mod
        _profile = _risk_mod.get_final_score_profile_info()
        st.code(_profile["formula"].replace(" + ", "\n+ ").replace("= ", "=\n"), language="text")
        st.caption(f"權重組合：{_profile['profile']}")
        st.markdown(_profile["explanation"])
    except Exception:
        st.code("final_decision_score =\n0.30 * safety_score\n+ 0.20 * return_score\n+ 0.25 * rule_filter_score\n+ 0.25 * risk_reward_score",
                language="text")

with tab_logs:
    st.title("執行紀錄與除錯資訊")
    st.markdown("#### 執行狀態")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("狀態", run_status_label())
    s2.metric("PID", st.session_state.last_pid if st.session_state.last_pid else "-")
    s3.metric("開始時間", st.session_state.last_started_at or "-")
    s4.metric("結束時間", st.session_state.last_finished_at or "-")

    st.markdown("#### 實際 command")
    display_command = st.session_state.last_command or command_text(command)
    st.code(display_command, language="powershell")

    if st.session_state.model_running:
        st.info("模型仍在背景執行中；請按側邊欄的「重新整理狀態」更新 stdout / stderr。")
    elif st.session_state.last_status == "尚未執行":
        st.info("尚未執行。")
    elif st.session_state.last_status == "執行成功":
        st.success("執行成功。")
    elif st.session_state.last_status == "已停止":
        st.warning("已停止模型執行。")
    else:
        st.error("執行失敗。")

    st.markdown("#### 目前讀取的 CSV")
    loaded_files = [
        {
            "類型": "ranking CSV",
            "檔名": ranking_path.name if ranking_path else "-",
            "修改時間": file_time(ranking_path),
            "完整路徑": str(ranking_path) if ranking_path else "-",
            "讀取狀態": ranking_error or "OK" if ranking_path else "-",
        },
        {
            "類型": "single diagnostic CSV",
            "檔名": single_path.name if single_path else "-",
            "修改時間": file_time(single_path),
            "完整路徑": str(single_path) if single_path else "-",
            "讀取狀態": single_error or "OK" if single_path else "-",
        },
        {
            "類型": "validation CSV",
            "檔名": validation_path.name if validation_path else "-",
            "修改時間": file_time(validation_path),
            "完整路徑": str(validation_path) if validation_path else "-",
            "讀取狀態": validation_error or "OK" if validation_path else "-",
        },
        {
            "類型": "backtest equity curve CSV",
            "檔名": equity_path.name if equity_path else "-",
            "修改時間": file_time(equity_path),
            "完整路徑": str(equity_path) if equity_path else "-",
            "讀取狀態": equity_error or "OK" if equity_path else "-",
        },
        {
            "類型": "backtest trade log CSV",
            "檔名": trade_path.name if trade_path else "-",
            "修改時間": file_time(trade_path),
            "完整路徑": str(trade_path) if trade_path else "-",
            "讀取狀態": trade_error or "OK" if trade_path else "-",
        },
        {
            "類型": "portfolio metrics CSV",
            "檔名": metrics_path.name if metrics_path else "-",
            "修改時間": file_time(metrics_path),
            "完整路徑": str(metrics_path) if metrics_path else "-",
            "讀取狀態": metrics_error or "OK" if metrics_path else "-",
        },
        {
            "類型": "entry diagnostic CSV",
            "檔名": entry_path.name if entry_path else "-",
            "修改時間": file_time(entry_path),
            "完整路徑": str(entry_path) if entry_path else "-",
            "讀取狀態": entry_error or "OK" if entry_path else "-",
        },
        {
            "類型": "sensitivity summary CSV",
            "檔名": sensitivity_path.name if sensitivity_path else "-",
            "修改時間": file_time(sensitivity_path),
            "完整路徑": str(sensitivity_path) if sensitivity_path else "-",
            "讀取狀態": sensitivity_error or "OK" if sensitivity_path else "-",
        },
    ]
    st.dataframe(pd.DataFrame(loaded_files), use_container_width=True, hide_index=True)

    with st.expander("查看模型輸出 stdout", expanded=False):
        st.text_area("stdout", value=str(st.session_state.last_stdout), height=400, label_visibility="collapsed")
    with st.expander("查看錯誤訊息 stderr", expanded=False):
        st.text_area("stderr", value=str(st.session_state.last_stderr), height=400, label_visibility="collapsed")
    with st.expander("除錯資訊", expanded=False):
        csv_list = output_csv_files(20)
        debug_rows = [
            {"項目": "BASE_DIR", "值": str(BASE_DIR)},
            {"項目": "PIPELINE_SCRIPT", "值": str(PIPELINE_SCRIPT)},
            {"項目": "OUTPUT_DIR", "值": str(OUTPUT_DIR)},
            {"項目": "PIPELINE_SCRIPT.exists()", "值": str(PIPELINE_SCRIPT.exists())},
            {"項目": "最近 output CSV 前 20 筆", "值": "\n".join(path.name for path in csv_list)},
        ]
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
