from __future__ import annotations

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


def find_latest_file(patterns: Iterable[str], prefer_after: float | None = None) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in OUTPUT_DIR.glob(pattern) if path.is_file())
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
    return pd.to_numeric(df[column], errors="coerce")


def to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
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


def render_common_model_bar(common_df: pd.DataFrame) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


def make_command(analysis_mode: str, market: str, symbol: str, top_n: int) -> list[str]:
    is_single = analysis_mode == "個股深度分析"
    return [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--mode",
        "single_diagnostic" if is_single else "train",
        "--market",
        market,
        "--symbol",
        symbol if is_single else "ALL",
        "--top-n",
        str(30 if is_single else max(int(top_n), 5)),
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


def render_top15_bar(ranking_df: pd.DataFrame) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


def render_score_rr_scatter(ranking_df: pd.DataFrame) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


def render_veto_reason_chart(ranking_df: pd.DataFrame) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


def render_single_radar(row: pd.Series) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


def render_price_level_chart(row: pd.Series) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


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


def render_validation_charts(metrics_df: pd.DataFrame) -> None:
    metric_cols = [col for col in ["MAE", "Baseline MAE", "RMSE", "Baseline RMSE"] if col in metrics_df.columns]
    if metric_cols:
        long_df = metrics_df.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value")
        fig_mae = px.bar(long_df, x="model", y="value", color="metric", barmode="group", title="誤差指標比較")
        fig_mae.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), font=dict(size=13))
        st.plotly_chart(fig_mae, use_container_width=True)

    perf_cols = [col for col in ["Spearman", "Top5 Lift", "Top10 Lift", "Long-Short Spread"] if col in metrics_df.columns]
    if perf_cols:
        long_df = metrics_df.melt(id_vars=["model"], value_vars=perf_cols, var_name="metric", value_name="value")
        fig_perf = px.bar(long_df, x="metric", y="value", color="model", barmode="group", title="排序與投組效果")
        fig_perf.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), font=dict(size=13))
        st.plotly_chart(fig_perf, use_container_width=True)


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
        options=["個股深度分析", "多股票排名分析"],
        index=0,
        horizontal=False,
    )
    market = st.selectbox(label="市場", options=["TWSE", "TPEX"], index=0)

    if analysis_mode == "個股深度分析":
        symbol = normalize_symbol(st.text_input(label="股票代碼", value="2330"))
        top_n = 30
    else:
        symbol = "ALL"
        top_n = st.number_input(label="Top N", min_value=5, value=30, step=1)

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

if st.session_state.model_running:
    st.info("模型正在背景執行中。你可以切換分頁查看目前已存在的圖表與 CSV；需要更新狀態時請按側邊欄的「重新整理狀態」。")
elif st.session_state.last_status == "執行成功":
    st.success("上一輪模型已執行成功，Dashboard 已讀取 output 資料夾中的最新 CSV。")
elif st.session_state.last_status == "執行失敗":
    st.error("上一輪模型執行失敗，請到「執行紀錄」查看 stderr。")
elif st.session_state.last_status == "已停止":
    st.warning("上一輪模型已由使用者停止。")

tab_overview, tab_ranking, tab_single, tab_validation, tab_features, tab_logs = st.tabs(
    ["系統總覽", "多股票排名", "個股診斷", "模型驗證", "特徵解釋", "執行紀錄"]
)

with tab_overview:
    st.title("台股機器學習投資決策支援系統")
    st.caption("Machine Learning Based Stock Investment Decision Support Dashboard")
    st.info("本系統整合機器學習模型、技術指標、價格結構、風險控管與 Risk/Reward，輸出可解釋的投資決策建議。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型核心", "Extra Trees")
    c2.metric("特徵組合", CORE_MODEL_SETTINGS["feature_set"])
    c3.metric("驗證方式", "Walk-forward 8 folds")
    c4.metric("決策架構", "模型 + 規則 + 風控")

    st.markdown("#### 目前模型設定")
    settings_rows = [
        {"參數": "mode", "值": "single_diagnostic" if analysis_mode == "個股深度分析" else "train"},
        {"參數": "market", "值": market},
        {"參數": "symbol", "值": symbol or "2330"},
        {"參數": "top_n", "值": 30 if analysis_mode == "個股深度分析" else int(top_n)},
        {"參數": "feature_set", "值": CORE_MODEL_SETTINGS["feature_set"]},
        {"參數": "model_type", "值": CORE_MODEL_SETTINGS["model_type"]},
        {"參數": "model_profile", "值": CORE_MODEL_SETTINGS["model_profile"]},
        {"參數": "target_col", "值": CORE_MODEL_SETTINGS["target_col"]},
    ]
    st.dataframe(pd.DataFrame(settings_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 最新資料概況")
    if ranking_path is None or ranking_df is None:
        st.info("尚未讀取到最新 ranking CSV，請先執行模型。")
    else:
        st.caption(f"最新 ranking CSV：`{ranking_path.name}`，更新時間：{file_time(ranking_path)}")
        safety = FALLBACK_VALIDATION_METRICS.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE", f"{safety['MAE']:.4f}")
        k2.metric("Baseline MAE", f"{safety['Baseline MAE']:.4f}")
        k3.metric("Spearman", f"{safety['Spearman']:.4f}")
        k4.metric("Top5 Lift", f"{safety['Top5 Lift']:.4f}")

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
        render_common_model_bar(common_work)
        common_cols = existing_columns(common_work, COMMON_MODEL_COLUMNS)
        st.dataframe(common_work[common_cols] if common_cols else common_work, use_container_width=True, height=420, hide_index=True)

        st.markdown("### 最終決策排名（加入規則過濾與風控後）")
        st.caption(
            "此排名使用 final_decision_score，已納入 rule_filter_score、risk_reward_score 與 trade_veto_flag，"
            "因此可能與共同模型排名不同。"
        )
        if "final_decision_score" in final_work.columns:
            render_top15_bar(final_work)
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
            render_score_rr_scatter(final_work)
        with col_b:
            render_veto_reason_chart(final_work)

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
                render_single_radar(row)
            with right:
                render_price_level_chart(row)

            st.markdown("#### 交易計畫")
            render_trade_plan(row)

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

with tab_validation:
    st.title("模型驗證與表現")
    if validation_error:
        st.warning(f"validation CSV 讀取失敗，暫用內建摘要指標：{validation_error}")
    elif validation_path is not None and validation_df is not None and not validation_df.empty:
        st.caption(f"讀取檔案：`{validation_path.name}`，更新時間：{file_time(validation_path)}")
    else:
        st.caption("尚未讀取到 validation CSV，暫用系統內建的 shared common model 摘要指標。")

    safety = FALLBACK_VALIDATION_METRICS.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Safety Spearman", f"{safety['Spearman']:.4f}")
    k2.metric("Safety MAE improvement", f"{(safety['Baseline MAE'] - safety['MAE']):.4f}")
    k3.metric("Top5 Lift", f"{safety['Top5 Lift']:.4f}")
    k4.metric("Long-Short Spread", f"{safety['Long-Short Spread']:.4f}")

    render_validation_charts(FALLBACK_VALIDATION_METRICS)
    st.info(
        "Safety model 在 MAE、RMSE、Spearman 與 Top-N lift 上優於 baseline；"
        "Return model 的穩定性較弱，因此最終決策提高規則過濾與 Risk/Reward 權重。"
    )
    with st.expander("查看完整驗證資料", expanded=False):
        if validation_path is not None and validation_df is not None and not validation_df.empty:
            st.dataframe(validation_df, use_container_width=True, hide_index=True, height=420)
        else:
            st.dataframe(FALLBACK_VALIDATION_METRICS, use_container_width=True, hide_index=True)

with tab_features:
    st.title("特徵解釋與決策依據")
    st.dataframe(feature_dataframe(), use_container_width=True, hide_index=True, height=520)
    st.markdown("#### 最終決策分數公式")
    st.code(
        """final_decision_score =
0.30 * safety_score
+ 0.20 * return_score
+ 0.25 * rule_filter_score
+ 0.25 * risk_reward_score""",
        language="text",
    )
    st.markdown(
        "由於第一層模型在 walk-forward 驗證中未穩定優於 baseline，"
        "本系統降低模型分數權重，並提高規則式過濾與 Risk/Reward 權重，"
        "使最終決策更偏向風險控管與交易可行性。"
    )

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
        },
        {
            "類型": "single diagnostic CSV",
            "檔名": single_path.name if single_path else "-",
            "修改時間": file_time(single_path),
            "完整路徑": str(single_path) if single_path else "-",
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
