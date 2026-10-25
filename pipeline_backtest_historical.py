"""
Historical Walk-Forward Backtest
=================================

模擬「如果這套系統 2018 年就存在」的歷史區間回測。
使用 balanced_v2_combined 策略（0.30 safety + 0.20 return + 0.25 rule_filter + 0.25 risk_reward）。

執行順序：
  1. 先跑 2022_rate_hike_bear 驗證邏輯正確
  2. 跑通後再跑其他事件與池化統計

[!] 重要限制揭露
-----------------
build_shared_common_stock_pool() 目前是以「今天」（程式執行當日）的成交金額排行去選定
股票池，並非以 2018-2022 年的歷史成交金額排行。這意味著本回測存在「選股池前視偏差
（look-ahead bias）」：我們是用「現在才知道的高流動性贏家股票」去回測「過去」。
本版本先維持此做法，但必須在解讀回測結果時明確考量此限制。回測績效可能因此被高估。
此問題將於後續版本改以歷史期間內實際成交量排行選股後修正。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_backtest_report import (
    run_portfolio_backtest,
    calculate_portfolio_metrics,
    print_portfolio_report,
)
from pipeline_backtest_engine import generate_oos_predictions_for_backtest
from pipeline_model_output import (
    fetch_and_prepare_multi_stock_data_historical,
    build_shared_common_stock_pool,
    build_shared_common_model_config,
)
from pipeline_data import (
    PipelineConfig,
    get_historical_market_index_data,
    CONFIG as DEFAULT_CONFIG,
)


# ============================================================
# Step 4：歷史事件視窗定義
# ============================================================

HISTORICAL_EVENT_WINDOWS = [
    {
        "event_name": "2022_rate_hike_bear",
        "test_start":  "2022-01-01",
        "test_end":    "2022-10-31",
        "burn_in_months": 12,
        "is_bear": True,
        "description": "2022 年升息熊市",
    },
    {
        "event_name": "2018_trade_war",
        "test_start":  "2018-10-01",
        "test_end":    "2018-12-31",
        "burn_in_months": 12,
        "is_bear": True,
        "description": "2018 年中美貿易戰急跌",
    },
    {
        "event_name": "2020_covid_crash",
        "test_start":  "2020-01-01",
        "test_end":    "2020-03-31",
        "burn_in_months": 12,
        "is_bear": True,
        "description": "2020 年 COVID-19 崩跌",
    },
    {
        "event_name": "2021_bull_market",
        "test_start":  "2021-01-01",
        "test_end":    "2021-12-31",
        "burn_in_months": 12,
        "is_bear": False,
        "description": "2021 年多頭對照組",
    },
    {
        "event_name": "2018_2022_full",
        "test_start":  "2018-01-01",
        "test_end":    "2022-12-31",
        "burn_in_months": 12,
        "is_bear": False,
        "description": "2018-2022 完整長窗口 walk-forward",
    },
]

BEAR_EVENT_NAMES = {"2018_trade_war", "2020_covid_crash", "2022_rate_hike_bear"}


# ============================================================
# 工具函式
# ============================================================

def _count_burn_in_trading_days(
    combined_data: pd.DataFrame,
    test_start: str,
) -> int:
    """計算 burn-in 期間（< test_start）的交易日數，作為 min_train_dates。"""
    if combined_data.empty or "date" not in combined_data.columns:
        return 126
    dates = pd.to_datetime(combined_data["date"]).dropna()
    cutoff = pd.Timestamp(test_start)
    burn_dates = dates[dates < cutoff].unique()
    return max(int(len(burn_dates)), 20)


def _window_duration_months(test_start: str, test_end: str) -> float:
    s = pd.Timestamp(test_start)
    e = pd.Timestamp(test_end)
    return (e.year - s.year) * 12 + (e.month - s.month) + 1


def _is_short_window(test_start: str, test_end: str) -> bool:
    return _window_duration_months(test_start, test_end) < 6


def _calc_mdd(equity_vals: np.ndarray) -> float:
    if len(equity_vals) == 0:
        return float("nan")
    rolling_max = np.maximum.accumulate(equity_vals)
    drawdowns = (equity_vals - rolling_max) / rolling_max
    return float(drawdowns.min())


# ============================================================
# Step 5：單一事件視窗回測
# ============================================================

def run_historical_event_backtest(
    event: dict,
    base_config: PipelineConfig,
    symbols: list[str],
) -> dict:
    """
    對單一歷史事件視窗執行完整回測流程。
    Returns: {
        "event_name": str,
        "metrics": dict,           # calculate_portfolio_metrics 的輸出
        "equity_curve": DataFrame, # 淨值曲線（僅事件視窗內，不跨事件接續）
        "trade_log": DataFrame,    # 交易明細
        "market_df": DataFrame,    # 用於 run_portfolio_backtest 的大盤資料
        "test_start": str,
        "test_end": str,
    }
    """
    event_name    = event["event_name"]
    test_start    = event["test_start"]
    test_end      = event["test_end"]
    burn_in_months = event["burn_in_months"]

    print("\n" + "=" * 70)
    print(f"[HIST] 開始事件回測：{event_name}")
    print(f"       測試區間：{test_start} ~ {test_end}  burn-in：{burn_in_months} 個月")
    print("=" * 70)

    # ── Step 5-1：抓取歷史資料（含 burn-in）─────────────────────────
    combined_data = fetch_and_prepare_multi_stock_data_historical(
        symbols=symbols,
        base_config=base_config,
        start_date=test_start,
        end_date=test_end,
        burn_in_months=burn_in_months,
    )

    # ── 大盤指數（事件視窗用）────────────────────────────────────────
    market_df = get_historical_market_index_data(
        config=base_config,
        start_date=test_start,
        end_date=test_end,
        burn_in_months=burn_in_months,
    )

    # ── Step 5-2：walk-forward OOS 預測（對合併後資料整體呼叫一次）────
    # generate_oos_predictions_for_backtest 設計為接受多股票合併 DataFrame，
    # 在 walk-forward 每個 fold 內做跨股票橫斷面排名，不能逐股票分開呼叫。
    # min_train_date_cutoff=test_start：讓函式自行在 model_df（valid_mask 後）
    # 數 test_start 之前的有效交易日數，精確決定 min_train_size，
    # 避免用 combined_data raw 日期數（含 indicator warm-up 期）高估。
    print(f"[HIST] 開始 walk-forward OOS 預測（全部 {combined_data['symbol'].nunique() if 'symbol' in combined_data.columns else '?'} 支股票合併呼叫）")
    oos_full = generate_oos_predictions_for_backtest(
        df=combined_data,
        config=base_config,
        n_splits=12,
        min_train_date_cutoff=test_start,
    )

    # 只保留 test_start ~ test_end 的 OOS 資料進入 Portfolio 回測
    # （burn-in 期間的資料已用來訓練，不進入績效統計）
    oos_full["date"] = pd.to_datetime(oos_full["date"], errors="coerce")
    oos_in_window = oos_full[
        (oos_full["date"] >= pd.Timestamp(test_start))
        & (oos_full["date"] <= pd.Timestamp(test_end))
    ].copy()

    # 拆分為逐股 dict 供 run_portfolio_backtest 使用
    all_stock_signals: dict[str, pd.DataFrame] = {}
    if "symbol" in oos_in_window.columns:
        for sym, grp in oos_in_window.groupby("symbol"):
            if not grp.empty:
                all_stock_signals[str(sym)] = grp.copy()

    print(f"[HIST] OOS 視窗內有效股票數：{len(all_stock_signals)}，資料筆數：{len(oos_in_window)}")

    if not all_stock_signals:
        print(f"[HIST] {event_name}：無有效 OOS 訊號，略過。")
        return {
            "event_name": event_name,
            "metrics": {},
            "equity_curve": pd.DataFrame(),
            "trade_log": pd.DataFrame(),
            "market_df": market_df,
            "test_start": test_start,
            "test_end": test_end,
        }

    # ── Step 5-3：Portfolio 回測（淨值曲線、MDD 只在此事件視窗內計算）──
    equity_curve, trade_log, _ = run_portfolio_backtest(
        all_stock_signals=all_stock_signals,
        config=base_config,
        market_df=market_df,
        expected_symbols=symbols,
        investor_mode=True,
    )

    # ── Step 5-4：計算績效指標────────────────────────────────────────
    metrics = calculate_portfolio_metrics(
        equity_curve=equity_curve,
        trade_log=trade_log,
        initial_capital=base_config.backtest_initial_capital,
    )

    # ── 印出 equity curve 實際起訖，並以完整測試窗口天數覆寫 CAGR / Calmar ──
    # calculate_portfolio_metrics() 預設用 equity_curve 自身的日期範圍（OOS 第一筆訊號~最後一筆）
    # 但熊市回測中空手期也是策略績效的一部分，必須用完整 test_start~test_end 年化才正確。
    if metrics and not equity_curve.empty:
        nav_first = pd.to_datetime(equity_curve["date"].iloc[0])
        nav_last  = pd.to_datetime(equity_curve["date"].iloc[-1])
        print(f"[HIST] equity curve 起訖：{nav_first.date()} ~ {nav_last.date()}"
              f"（{(nav_last - nav_first).days} 曆日）")

        test_start_dt  = pd.Timestamp(test_start)
        test_end_dt    = pd.Timestamp(test_end)
        n_days_window  = max((test_end_dt - test_start_dt).days, 1)
        print(f"[HIST] 完整測試窗口：{test_start_dt.date()} ~ {test_end_dt.date()}"
              f"（{n_days_window} 曆日，用於 CAGR 年化分母）")

        total_return = metrics.get("total_return", 0.0)
        corrected_cagr = (1.0 + total_return) ** (365.0 / n_days_window) - 1.0
        mdd = metrics.get("max_drawdown", 0.0)
        corrected_calmar = (corrected_cagr / abs(mdd)) if (mdd and mdd != 0) else float("nan")
        metrics["cagr"]         = corrected_cagr
        metrics["calmar_ratio"] = corrected_calmar
        print(f"[HIST] CAGR（修正後）：{corrected_cagr:.2%}  Calmar（修正後）：{corrected_calmar:.2f}")

    print(f"[HIST] {event_name} 完成，交易筆數：{len(trade_log)}")
    if metrics:
        print(f"       累積報酬：{metrics.get('total_return', float('nan')):.2%}")
        print(f"       MDD：{metrics.get('max_drawdown', float('nan')):.2%}")

    return {
        "event_name": event_name,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "market_df": market_df,
        "test_start": test_start,
        "test_end": test_end,
    }


# ============================================================
# Step 6：三次熊市 trade-level 池化統計
# ============================================================

def compute_pooled_bear_stats(event_results: list[dict]) -> dict:
    """
    將三次熊市事件的 trade log 池化，計算跨事件統計量。
    注意：MDD 不合併，只列出各自的 MDD 做對照。
    """
    bear_results = [r for r in event_results if r["event_name"] in BEAR_EVENT_NAMES]

    pooled_trades: list[pd.DataFrame] = []
    per_event_stats: list[dict] = {}

    for r in bear_results:
        tl = r.get("trade_log", pd.DataFrame())
        ev = r["event_name"]
        n_trades = len(tl) if not tl.empty else 0
        mdd = r.get("metrics", {}).get("max_drawdown", float("nan"))

        per_event_stats[ev] = {
            "n_trades": n_trades,
            "mdd": mdd,
            "small_sample": n_trades < 30,
        }

        if not tl.empty and "return_pct" in tl.columns:
            tl = tl.copy()
            tl["_event_name"] = ev
            pooled_trades.append(tl)

    if not pooled_trades:
        return {
            "per_event": per_event_stats,
            "pooled_win_rate": float("nan"),
            "pooled_avg_return": float("nan"),
            "pooled_raw_sharpe": float("nan"),
            "pooled_n_trades": 0,
            "deepest_mdd_event": None,
            "deepest_mdd_value": float("nan"),
        }

    pooled_df = pd.concat(pooled_trades, ignore_index=True)
    returns = pd.to_numeric(pooled_df["return_pct"], errors="coerce").dropna()

    pooled_win_rate  = float((returns > 0).mean()) if len(returns) > 0 else float("nan")
    pooled_avg_return = float(returns.mean())       if len(returns) > 0 else float("nan")
    # 未年化 Sharpe：跨事件池化，不乘 sqrt(252)
    pooled_raw_sharpe = (
        float(returns.mean() / returns.std())
        if len(returns) > 1 and returns.std() > 0
        else float("nan")
    )

    # 最深 MDD 事件
    valid_mdds = {
        ev: s["mdd"]
        for ev, s in per_event_stats.items()
        if pd.notna(s["mdd"])
    }
    if valid_mdds:
        deepest_event = min(valid_mdds, key=lambda e: valid_mdds[e])
        deepest_value = valid_mdds[deepest_event]
    else:
        deepest_event = None
        deepest_value = float("nan")

    return {
        "per_event": per_event_stats,
        "pooled_win_rate":   pooled_win_rate,
        "pooled_avg_return": pooled_avg_return,
        "pooled_raw_sharpe": pooled_raw_sharpe,
        "pooled_n_trades":   len(returns),
        "deepest_mdd_event": deepest_event,
        "deepest_mdd_value": deepest_value,
    }


# ============================================================
# Step 7：輸出報告
# ============================================================

_BASELINE_CAGR = 0.7175   # 現有主線基準（71.75% CAGR）


def _fmt_pct(v) -> str:
    return f"{float(v):.2%}" if pd.notna(v) and not isinstance(v, str) else "N/A"


def _fmt_f(v, d: int = 2) -> str:
    return f"{float(v):.{d}f}" if pd.notna(v) and not isinstance(v, str) else "N/A"


def _fmt_i(v) -> str:
    try:
        return str(int(v)) if pd.notna(v) else "N/A"
    except Exception:
        return "N/A"


def print_historical_summary_report(
    event_results: list[dict],
    pooled_stats: dict,
    events: list[dict],
) -> None:
    print("\n" + "=" * 90)
    print("歷史區間回測總表  balanced_v2_combined 策略")
    print("=" * 90)
    print()
    print("[!]  選股池前視偏差（Look-Ahead Bias）揭露")
    print("-" * 90)
    print("  本回測使用「今日（程式執行當日）成交金額排行前 N 檔」作為股票池，")
    print("  而非 2018-2022 年間各時間點的真實高成交量股票。")
    print("  這代表我們是以『現在才知道的高流動性贏家』回測『過去』，")
    print("  存在選股池層面的 look-ahead bias，歷史回測績效可能因此被高估。")
    print("  此問題將於後續版本改以歷史期間成交量排行選股後修正。")
    print()

    # ── 現有主線基準 ────────────────────────────────────────────────
    print("-" * 90)
    print(f"【現有主線基準（近 3 年樣本內）】")
    print(f"  CAGR：{_fmt_pct(_BASELINE_CAGR)}")
    print()

    # ── 各事件結果 ──────────────────────────────────────────────────
    print("=" * 90)
    hdr = (
        f"{'事件':<28} {'區間':<25} {'累積報酬':>10} {'CAGR':>9} "
        f"{'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'交易筆數':>8}"
    )
    print(hdr)
    print("-" * 90)

    results_by_name = {r["event_name"]: r for r in event_results}
    events_by_name  = {e["event_name"]: e for e in events}

    for ev in events:
        ev_name  = ev["event_name"]
        t_start  = ev["test_start"]
        t_end    = ev["test_end"]
        r        = results_by_name.get(ev_name, {})
        m        = r.get("metrics", {})
        tl       = r.get("trade_log", pd.DataFrame())
        n_trades = len(tl) if not tl.empty else 0

        is_short = _is_short_window(t_start, t_end)
        total_return = m.get("total_return", float("nan"))
        cagr         = m.get("cagr",         float("nan"))
        mdd          = m.get("max_drawdown",  float("nan"))
        sharpe       = m.get("sharpe_ratio",  float("nan"))
        calmar       = m.get("calmar_ratio",  float("nan"))

        cagr_str   = "(短窗口，見累積報酬)" if is_short else _fmt_pct(cagr)
        calmar_str = "N/A" if is_short else _fmt_f(calmar)

        range_str = f"{t_start[:7]}~{t_end[:7]}"
        print(
            f"{ev_name:<28} {range_str:<25} {_fmt_pct(total_return):>10} "
            f"{cagr_str:>9} {_fmt_pct(mdd):>8} {_fmt_f(sharpe):>7} "
            f"{calmar_str:>7} {_fmt_i(n_trades):>8}"
        )

        if n_trades < 30:
            print(f"  [!]  {ev_name}：樣本數有限（{n_trades} 筆），指標僅供參考。")

    print()
    print("  * 短窗口事件（< 6 個月）不報 CAGR，只報累積報酬")
    print("  * 2018_2022_full 才報年化 CAGR（完整長窗口）")

    # ── 池化熊市統計 ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("熊市事件池化統計（2018_trade_war / 2020_covid_crash / 2022_rate_hike_bear）")
    print("=" * 90)

    ps = pooled_stats
    per = ps.get("per_event", {})

    for ev_name in ["2018_trade_war", "2020_covid_crash", "2022_rate_hike_bear"]:
        es = per.get(ev_name, {})
        n  = es.get("n_trades", 0)
        mdd_val = es.get("mdd", float("nan"))
        small   = es.get("small_sample", True)
        note    = "（樣本數有限，指標僅供參考）" if small else ""
        print(f"  {ev_name}：MDD = {_fmt_pct(mdd_val)}，交易筆數 = {n}{note}")

    deepest = ps.get("deepest_mdd_event")
    deepest_val = ps.get("deepest_mdd_value", float("nan"))
    if deepest:
        print(f"  最深 MDD：{deepest}（{_fmt_pct(deepest_val)}）")

    print()
    print(f"  池化交易總筆數：{ps.get('pooled_n_trades', 0)}")
    print(f"  池化勝率：{_fmt_pct(ps.get('pooled_win_rate'))}")
    print(f"  池化平均報酬率：{_fmt_pct(ps.get('pooled_avg_return'))}")
    print(
        f"  池化未年化 Sharpe：{_fmt_f(ps.get('pooled_raw_sharpe'))}  "
        "（跨事件池化統計量，非傳統年化 Sharpe，未乘以 √252）"
    )

    print()
    print("=" * 90)


def print_single_event_detail(result: dict, event: dict) -> None:
    """輸出單一事件的詳細回測報告。"""
    ev_name  = result["event_name"]
    m        = result.get("metrics", {})
    ec       = result.get("equity_curve", pd.DataFrame())
    tl       = result.get("trade_log", pd.DataFrame())
    t_start  = result["test_start"]
    t_end    = result["test_end"]
    is_short = _is_short_window(t_start, t_end)

    print("\n" + "=" * 70)
    print(f"歷史回測詳細報告：{ev_name}")
    print(f"描述：{event.get('description', '')}")
    print(f"區間：{t_start} ~ {t_end}")
    print("=" * 70)

    if not m:
        print("無有效回測結果。")
        return

    n_trades = len(tl) if not tl.empty else 0
    total_return = m.get("total_return", float("nan"))
    cagr         = m.get("cagr",         float("nan"))
    mdd          = m.get("max_drawdown",  float("nan"))
    sharpe       = m.get("sharpe_ratio",  float("nan"))
    calmar       = m.get("calmar_ratio",  float("nan"))
    win_rate     = m.get("win_rate",      float("nan"))

    print(f"累積報酬率：{_fmt_pct(total_return)}")
    if not is_short:
        print(f"年化報酬（CAGR）：{_fmt_pct(cagr)}")
    else:
        print("年化報酬（CAGR）：短窗口事件，不報 CAGR")
    print(f"最大回撤（MDD）：{_fmt_pct(mdd)}")
    print(f"Sharpe Ratio：{_fmt_f(sharpe)}")
    if not is_short:
        print(f"Calmar Ratio：{_fmt_f(calmar)}")
    print(f"勝率：{_fmt_pct(win_rate)}")
    print(f"交易筆數：{n_trades}")

    if n_trades < 30:
        print(f"[!]  樣本數有限（{n_trades} 筆），指標僅供參考。")

    if not ec.empty and "date" in ec.columns:
        start_d = pd.to_datetime(ec["date"].min()).date()
        end_d   = pd.to_datetime(ec["date"].max()).date()
        print(f"淨值曲線期間：{start_d} ~ {end_d}")


# ============================================================
# 股票池快取（可重現性保障）
# ============================================================

def _load_or_build_stock_pool(
    run_config: "PipelineConfig",
    top_n: int,
    force_refresh: bool = False,
) -> tuple["pd.DataFrame", list[str]]:
    """
    首次執行時呼叫 TWSE API 抓取股票池並寫入本地快取；
    後續執行直接讀取快取，確保同一組股票、結果可重現。

    重新選股方法（二擇一）：
      - 刪除快取檔 historical_stock_pool_top{N}.csv
      - 執行時加 --force-refresh-pool 參數
    """
    cache_path = Path(__file__).parent / f"historical_stock_pool_top{top_n}.csv"

    if not force_refresh and cache_path.exists():
        stock_list = pd.read_csv(cache_path, dtype={"symbol": str})
        symbols = stock_list["symbol"].astype(str).tolist()
        frozen_at = (
            stock_list["frozen_at"].iloc[0]
            if "frozen_at" in stock_list.columns
            else "unknown"
        )
        print(f"[HIST] 股票池從本地快取載入（可重現模式）")
        print(f"       快取檔：{cache_path.name}，凍結時間：{frozen_at}")
        print(f"[HIST] 股票池大小：{len(symbols)}  股票：{symbols}")
        return stock_list, symbols

    if force_refresh:
        print("[HIST] 強制重新抓取股票池（--force-refresh-pool）...")
    else:
        print(f"[HIST] 快取不存在，首次抓取股票池並寫入本地快取...")

    stock_list, symbols = build_shared_common_stock_pool(base_config=run_config, top_n=top_n)

    frozen_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stock_list = stock_list.copy()
    stock_list["frozen_at"] = frozen_at
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stock_list.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[HIST] 股票池已凍結並寫入快取：{cache_path.name}（凍結時間：{frozen_at}）")
    print(f"[HIST] 股票池大小：{len(symbols)}  股票：{symbols}")
    return stock_list, symbols


# ============================================================
# 主流程
# ============================================================

def run_historical_backtest(
    event_names: Optional[list[str]] = None,
    top_n: int = 30,
    base_config: Optional[PipelineConfig] = None,
    force_refresh_pool: bool = False,
) -> None:
    """
    主流程入口。

    Args:
        event_names: 指定要跑的事件名稱清單；None 表示只跑第一個（驗證模式）。
        top_n: 股票池大小。
        base_config: 若為 None 則使用 DEFAULT_CONFIG。
        force_refresh_pool: True 則強制重抓股票池並覆蓋快取；
                            False（預設）則優先讀取本地快取確保可重現性。
    """
    if base_config is None:
        base_config = DEFAULT_CONFIG

    # 使用 balanced_v2_combined 策略設定
    run_config = replace(
        base_config,
        stock_no="ALL",
        backtest_profile_name="balanced_v2_combined",
        backtest_portfolio_config_version="balanced_v2_combined",
        backtest_entry_mode="balanced_support",
        backtest_max_positions=8,
    )

    # ── 建立／載入股票池（快取優先，確保可重現性）───────────────────
    stock_list, symbols = _load_or_build_stock_pool(run_config, top_n, force_refresh_pool)

    model_config = build_shared_common_model_config(run_config)

    # ── 決定要跑的事件 ────────────────────────────────────────────
    if event_names is None:
        # 預設只跑 2022_rate_hike_bear 驗證
        target_events = [e for e in HISTORICAL_EVENT_WINDOWS if e["event_name"] == "2022_rate_hike_bear"]
        print("[HIST] 驗證模式：只跑 2022_rate_hike_bear")
    else:
        target_events = [e for e in HISTORICAL_EVENT_WINDOWS if e["event_name"] in event_names]

    if not target_events:
        print("[HIST] 找不到指定的事件視窗，結束。")
        return

    # ── 逐事件回測 ───────────────────────────────────────────────
    event_results: list[dict] = []
    for event in target_events:
        try:
            result = run_historical_event_backtest(
                event=event,
                base_config=model_config,
                symbols=symbols,
            )
            event_results.append(result)
            print_single_event_detail(result, event)
        except Exception as e:
            print(f"[HIST] 事件 {event['event_name']} 失敗：{e}")
            import traceback
            traceback.print_exc()
            event_results.append({
                "event_name": event["event_name"],
                "metrics": {},
                "equity_curve": pd.DataFrame(),
                "trade_log": pd.DataFrame(),
                "market_df": pd.DataFrame(),
                "test_start": event["test_start"],
                "test_end": event["test_end"],
            })

    # ── 池化熊市統計（只在有足夠熊市事件時才跑）────────────────────
    ran_event_names = {r["event_name"] for r in event_results}
    bear_ran = ran_event_names & BEAR_EVENT_NAMES

    pooled_stats: dict = {}
    if len(bear_ran) >= 2:
        pooled_stats = compute_pooled_bear_stats(event_results)
    elif len(bear_ran) == 1:
        pooled_stats = compute_pooled_bear_stats(event_results)
        print(f"[HIST] 只有 {len(bear_ran)} 個熊市事件跑完，池化統計樣本有限。")
    else:
        print("[HIST] 尚無熊市事件完成，跳過池化統計。")

    # ── 總表輸出 ─────────────────────────────────────────────────
    if event_results and pooled_stats is not None:
        print_historical_summary_report(
            event_results=event_results,
            pooled_stats=pooled_stats or {},
            events=target_events,
        )

    # ── 可重現性憑證：印出凍結股票池清單 ────────────────────────────
    cache_path = Path(__file__).parent / f"historical_stock_pool_top{top_n}.csv"
    frozen_at_str = "unknown"
    if cache_path.exists():
        try:
            _cached = pd.read_csv(cache_path, dtype={"symbol": str})
            if "frozen_at" in _cached.columns:
                frozen_at_str = _cached["frozen_at"].iloc[0]
        except Exception:
            pass
    print("\n" + "=" * 70)
    print("可重現性憑證（Reproducibility Snapshot）")
    print("=" * 70)
    print(f"  快取檔：{cache_path.name}")
    print(f"  凍結時間：{frozen_at_str}")
    print(f"  股票數量：{len(symbols)}")
    print(f"  股票清單：{symbols}")
    print("  提示：刪除上述快取檔或加 --force-refresh-pool 可重新選股")
    print("=" * 70)

    print("\n[HIST] 歷史回測完成。")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="歷史區間回測（walk-forward simulation）")
    parser.add_argument(
        "--events",
        nargs="*",
        default=None,
        help=(
            "指定要跑的事件名稱（空白分隔）。"
            "不填則只跑 2022_rate_hike_bear（驗證模式）。"
            "填 all 跑全部事件。"
            "可選：2018_trade_war, 2020_covid_crash, 2021_bull_market, "
            "2022_rate_hike_bear, 2018_2022_full"
        ),
    )
    parser.add_argument("--top_n", type=int, default=30, help="股票池大小（預設 30）")
    parser.add_argument(
        "--force-refresh-pool",
        action="store_true",
        default=False,
        help=(
            "強制重新從 TWSE API 抓取股票池並覆蓋本地快取。"
            "預設行為：優先讀取本地快取 historical_stock_pool_top{N}.csv，"
            "確保每次執行使用相同股票、結果可重現。"
        ),
    )
    args = parser.parse_args()

    if args.events is not None and "all" in args.events:
        event_names = None  # run_historical_backtest 內部 None = 只跑第一個，所以改為全部
        event_names = [e["event_name"] for e in HISTORICAL_EVENT_WINDOWS]
    elif args.events:
        event_names = args.events
    else:
        event_names = None  # 驗證模式：只跑 2022_rate_hike_bear

    run_historical_backtest(
        event_names=event_names,
        top_n=args.top_n,
        force_refresh_pool=args.force_refresh_pool,
    )
