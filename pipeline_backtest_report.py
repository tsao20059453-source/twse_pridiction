from __future__ import annotations

from pipeline_model_output import *
from pipeline_backtest_engine import *
from pipeline_historical_universe import (
    append_unscored_execution_rows,
    build_daily_top_n_membership,
    load_daily_snapshot_prices,
)

def aggregate_trades_by_entry(trade_log: pd.DataFrame) -> pd.DataFrame:
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
        # 依出場順序排序（第一批出場=1，第二批=2，…）
        if "exit_sequence" in g.columns:
            g_sorted = g.sort_values("exit_sequence")
        else:
            g_sorted = g

        n_exits = len(g_sorted)
        entry_date  = g_sorted["entry_date"].iloc[0]
        entry_price = float(g_sorted["entry_price"].iloc[0])

        # 使用實際出場股數加權，才能正確反映 30% / 30% / trailing final 的分批出場。
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


def calculate_real_trade_metrics(trade_log: pd.DataFrame) -> dict:
    """
    以 aggregate_trades_by_entry() 的結果計算「真實」（以進場為單位，而非出場記錄數）
    的績效指標：真實進場次數 / 真實勝率 / 真實平均單筆報酬 / 真實獲利因子。
    """
    agg = aggregate_trades_by_entry(trade_log)
    if agg.empty:
        return {
            "real_trades":        0,
            "real_win_rate":      np.nan,
            "real_avg_return":    np.nan,
            "real_profit_factor": np.nan,
            "agg_trades":         agg,
        }

    rets   = pd.to_numeric(agg["total_return_pct"], errors="coerce").dropna()
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]

    real_trades     = len(agg)
    real_win_rate   = float(agg["is_win"].mean())
    real_avg_return = float(rets.mean()) if not rets.empty else np.nan

    if (len(wins) > 0 and len(losses) > 0
            and pd.notna(wins.mean()) and pd.notna(losses.mean())
            and losses.mean() != 0):
        real_profit_factor = (wins.mean() * len(wins)) / (abs(losses.mean()) * len(losses))
    else:
        real_profit_factor = np.nan

    return {
        "real_trades":        real_trades,
        "real_win_rate":      real_win_rate,
        "real_avg_return":    real_avg_return,
        "real_profit_factor": real_profit_factor,
        "agg_trades":         agg,
    }


def calculate_backtest_metrics(
    trade_log: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
) -> dict:
    if equity_curve.empty:
        return {}

    final_equity  = float(equity_curve["total_equity"].iloc[-1])
    total_return  = (final_equity - initial_capital) / initial_capital

    first_date = pd.to_datetime(equity_curve["date"].iloc[0])
    last_date  = pd.to_datetime(equity_curve["date"].iloc[-1])
    n_days     = max((last_date - first_date).days, 1)
    annual_return = (1 + total_return) ** (365 / n_days) - 1

    equity_vals  = equity_curve["total_equity"].to_numpy(dtype=float)
    rolling_max  = np.maximum.accumulate(equity_vals)
    drawdowns    = (equity_vals - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    daily_returns = pd.Series(equity_vals).pct_change().dropna()
    std_daily     = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0
    risk_free_daily = 0.015 / 252
    sharpe_ratio  = (
        float((daily_returns.mean() - risk_free_daily) / std_daily * np.sqrt(252))
        if std_daily > 0 else 0.0
    )

    if trade_log.empty:
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "total_trades": 0,
            "avg_holding_days": np.nan,
            "max_consecutive_losses": 0,
            "real_trades": 0,
            "real_win_rate": np.nan,
            "real_avg_return": np.nan,
            "real_profit_factor": np.nan,
        }

    returns = pd.to_numeric(trade_log["return_pct"], errors="coerce").dropna()
    wins    = returns[returns > 0]
    losses  = returns[returns <= 0]

    win_rate  = len(wins) / len(returns) if len(returns) > 0 else np.nan
    avg_win   = float(wins.mean())   if len(wins)   > 0 else np.nan
    avg_loss  = float(losses.mean()) if len(losses) > 0 else np.nan

    total_profit = float(trade_log.loc[trade_log["pnl"] > 0, "pnl"].sum()) if "pnl" in trade_log.columns else 0.0
    total_loss   = abs(float(trade_log.loc[trade_log["pnl"] <= 0, "pnl"].sum())) if "pnl" in trade_log.columns else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else np.nan

    avg_holding = float(
        pd.to_numeric(trade_log["holding_days"], errors="coerce").mean()
    ) if "holding_days" in trade_log.columns else np.nan

    # max consecutive losses
    is_loss   = (pd.to_numeric(trade_log.get("pnl", pd.Series()), errors="coerce") <= 0).tolist()
    max_consec = cur = 0
    for v in is_loss:
        cur = cur + 1 if v else 0
        max_consec = max(max_consec, cur)

    # 真實交易指標（以進場為單位，合併分批出場記錄）
    real_metrics = calculate_real_trade_metrics(trade_log)

    return {
        "total_return":            total_return,
        "annual_return":           annual_return,
        "max_drawdown":            max_drawdown,
        "sharpe_ratio":            sharpe_ratio,
        "win_rate":                win_rate,
        "avg_win":                 avg_win,
        "avg_loss":                avg_loss,
        "profit_factor":           profit_factor,
        "total_trades":            len(trade_log),
        "avg_holding_days":        avg_holding,
        "max_consecutive_losses":  max_consec,
        "real_trades":             real_metrics["real_trades"],
        "real_win_rate":           real_metrics["real_win_rate"],
        "real_avg_return":         real_metrics["real_avg_return"],
        "real_profit_factor":      real_metrics["real_profit_factor"],
    }


def print_backtest_report(
    metrics: dict,
    trade_log: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    initial_capital = config.backtest_initial_capital
    final_equity = float(equity_curve["total_equity"].iloc[-1]) if not equity_curve.empty else initial_capital

    print("=" * 70)
    print("回測績效報告")
    print("=" * 70)
    print("回測方法：Rolling Walk-forward OOS 回測（無資料洩漏）")
    print("Walk-forward 折數：12 折（每折約 3 個月）")
    print("每折預測均為 out-of-sample，訓練資料不包含驗證期資料")
    print("-" * 70)
    if not equity_curve.empty:
        print(f"回測期間：{pd.to_datetime(equity_curve['date'].min()).date()} ~ {pd.to_datetime(equity_curve['date'].max()).date()}")
    print(f"起始資金：{initial_capital:,.0f}")
    print(f"最終資產：{final_equity:,.0f}")
    print(f"總報酬率：{metrics.get('total_return', 0):.2%}")
    print(f"年化報酬率：{metrics.get('annual_return', 0):.2%}")
    print(f"最大回撤：{metrics.get('max_drawdown', 0):.2%}")
    print(f"Sharpe Ratio：{metrics.get('sharpe_ratio', 0):.2f}")

    wr = metrics.get("win_rate")
    print(f"勝率：{wr:.2%}" if pd.notna(wr) else "勝率：N/A")

    aw = metrics.get("avg_win")
    print(f"平均獲利：{aw:.2%}" if pd.notna(aw) else "平均獲利：N/A")

    al = metrics.get("avg_loss")
    print(f"平均虧損：{al:.2%}" if pd.notna(al) else "平均虧損：N/A")

    pf = metrics.get("profit_factor")
    print(f"獲利因子：{pf:.2f}" if pd.notna(pf) else "獲利因子：N/A")

    print(f"總交易筆數：{metrics.get('total_trades', 0)}（出場記錄數，含分批出場）")

    ahd = metrics.get("avg_holding_days")
    print(f"平均持有天數：{ahd:.1f}" if pd.notna(ahd) else "平均持有天數：N/A")

    print(f"最大連續虧損：{metrics.get('max_consecutive_losses', 0)} 筆")

    # ── 真實交易指標（以進場為單位，合併分批出場記錄）────────────────
    print(f"真實進場次數：{metrics.get('real_trades', 0)}")
    rwr = metrics.get("real_win_rate")
    print(f"真實勝率：{rwr:.2%}" if pd.notna(rwr) else "真實勝率：N/A")
    rar = metrics.get("real_avg_return")
    print(f"真實平均單筆報酬：{rar:.2%}" if pd.notna(rar) else "真實平均單筆報酬：N/A")
    rpf = metrics.get("real_profit_factor")
    print(f"真實獲利因子：{rpf:.2f}" if pd.notna(rpf) else "真實獲利因子：N/A")

    if not trade_log.empty:
        print("\n交易明細（個別報酬率，與總資金大小無關）：")
        display_cols = [c for c in [
            "symbol", "entry_date", "entry_price",
            "actual_stop_loss_price", "actual_stop_loss_pct", "stop_adjusted",
            "exit_date", "exit_price", "exit_reason",
            "return_pct", "holding_days", "stage", "trailing_stop_price",
            "support_entry_zone_low", "support_entry_zone_high",
            "entry_decision_score",
        ] if c in trade_log.columns]
        recent = trade_log.tail(10)[display_cols].copy()
        if "return_pct" in recent.columns:
            recent["return_pct"] = recent["return_pct"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        if "entry_date" in recent.columns:
            recent["entry_date"] = pd.to_datetime(recent["entry_date"]).dt.date
        if "exit_date" in recent.columns:
            recent["exit_date"]  = pd.to_datetime(recent["exit_date"]).dt.date
        print(recent.to_string(index=False))

        # 每筆報酬率摘要
        _all_r = pd.to_numeric(trade_log["return_pct"], errors="coerce").dropna()
        _wins_r = _all_r[_all_r > 0]
        _loss_r = _all_r[_all_r <= 0]
        _avg_all = float(_all_r.mean())  if not _all_r.empty  else float("nan")
        _avg_win = float(_wins_r.mean()) if not _wins_r.empty else float("nan")
        _avg_los = float(_loss_r.mean()) if not _loss_r.empty else float("nan")
        _win_s = f"{_avg_win:.2%}" if pd.notna(_avg_win) else "N/A"
        _los_s = f"{_avg_los:.2%}" if pd.notna(_avg_los) else "N/A"
        if pd.notna(_avg_all):
            print(f"平均單筆報酬率：{_avg_all:.2%}（獲利 {_win_s} / 虧損 {_los_s}）")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    ts = datetime.today().strftime("%Y%m%d_%H%M%S")
    prefix = output_dir / f"{config.market.upper()}_{config.stock_no}_backtest_{ts}"

    if not trade_log.empty:
        tl_csv  = Path(f"{prefix}_trade_log.csv")
        tl_xlsx = Path(f"{prefix}_trade_log.xlsx")
        trade_log.to_csv(tl_csv, index=False, encoding="utf-8-sig")
        trade_log.to_excel(tl_xlsx, index=False)
        print(f"\n交易明細已輸出：{tl_csv}")

    if not equity_curve.empty:
        ec_csv  = Path(f"{prefix}_equity_curve.csv")
        ec_xlsx = Path(f"{prefix}_equity_curve.xlsx")
        equity_curve.to_csv(ec_csv, index=False, encoding="utf-8-sig")
        equity_curve.to_excel(ec_xlsx, index=False)
        print(f"資產曲線已輸出：{ec_csv}")


def _symbol_eligibility_cutoff_date(
    oos_df: pd.DataFrame,
    config: PipelineConfig,
    n_folds_for_screening: int = 2,
) -> Optional[pd.Timestamp]:
    """Return the last date used by the symbol eligibility screening window."""
    if oos_df is None or oos_df.empty or "date" not in oos_df.columns:
        return None

    dates = pd.to_datetime(oos_df["date"], errors="coerce")
    if dates.dropna().empty:
        return None

    if "oos_fold" in oos_df.columns:
        fold_no = pd.to_numeric(oos_df["oos_fold"], errors="coerce")
        screening_dates = dates[fold_no <= n_folds_for_screening].dropna()
        if not screening_dates.empty:
            return pd.Timestamp(screening_dates.max())

    months = int(getattr(config, "eligibility_window_months", 6) or 6)
    return pd.Timestamp(dates.dropna().min()) + pd.DateOffset(months=months)


def filter_suitable_symbols(
    oos_df: pd.DataFrame,
    config: PipelineConfig,
    investor_mode: bool = False,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    股票適用性過濾：使用全部 12 折 OOS 驗證資料，以既有單股回測引擎（run_backtest）
    模擬交易，評估每支股票是否明顯不適用本策略。

    排除條件（符合任一條件即排除）：
      - 整體勝率 < 30% 且 profit_factor < 0.80
      - 整體平均單筆報酬 < -2.0%
      - 最差單筆虧損 < -10%（代表停損邏輯在這支股票上失效）

    豁免條件（符合則不排除）：
      - 後半段（第 7-12 折）勝率 >= 55%

    只在「決定哪些股票進入 Portfolio 回測」這一步使用；被排除的股票仍可在
    既有逐股回測診斷中正常輸出，只是不計入 Portfolio 層級績效。

    Returns:
        (passed_symbols, excluded_symbols, exclusion_reasons)
        exclusion_reasons: {symbol: 排除原因說明}
    """
    if oos_df is None or oos_df.empty or "symbol" not in oos_df.columns:
        return [], [], {}

    symbols = sorted(oos_df["symbol"].astype(str).unique().tolist())

    # 後半段（第 7-12 折）資料，用於豁免判斷
    has_fold_col = "oos_fold" in oos_df.columns
    if has_fold_col:
        late_folds_df = oos_df[pd.to_numeric(oos_df["oos_fold"], errors="coerce") >= 7].copy()
    else:
        late_folds_df = pd.DataFrame()

    passed_symbols: list[str] = []
    excluded_symbols: list[str] = []
    exclusion_reasons: dict[str, str] = {}

    for sym in symbols:
        sym_all = oos_df[oos_df["symbol"].astype(str) == sym].copy()
        if sym_all.empty:
            passed_symbols.append(sym)  # 無資料時不主動排除
            continue

        sym_config = replace(config, stock_no=sym)

        # ── 全折回測 ──────────────────────────────────────────────────
        try:
            tl_all, _ec = run_backtest(sym_all, sym_config, mode="single", investor_mode=investor_mode)
        except Exception:
            passed_symbols.append(sym)
            continue

        if tl_all is None or tl_all.empty or "return_pct" not in tl_all.columns:
            passed_symbols.append(sym)
            continue

        rets = pd.to_numeric(tl_all["return_pct"], errors="coerce").dropna()
        if rets.empty:
            passed_symbols.append(sym)
            continue

        wins   = rets[rets > 0]
        losses = rets[rets <= 0]
        win_rate         = len(wins) / len(rets)
        avg_trade_return = float(rets.mean())
        worst_trade      = float(rets.min())

        if len(wins) > 0 and len(losses) > 0 and losses.mean() != 0:
            profit_factor = (wins.mean() * len(wins)) / (abs(losses.mean()) * len(losses))
        else:
            profit_factor = np.nan

        # ── 排除條件判斷 ──────────────────────────────────────────────
        reasons: list[str] = []
        if win_rate < 0.30 and pd.notna(profit_factor) and profit_factor < 0.80:
            reasons.append(f"整體勝率 {win_rate:.1%} 且 profit_factor {profit_factor:.2f} 偏低")
        if avg_trade_return < -0.02:
            reasons.append(f"整體平均單筆報酬 {avg_trade_return:.2%}")
        if worst_trade < -0.10:
            reasons.append(f"最差單筆虧損 {worst_trade:.2%}（停損邏輯可能失效）")

        if not reasons:
            passed_symbols.append(sym)
            continue

        # ── 豁免：後半段（第 7-12 折）勝率 >= 55% ───────────────────
        late_win_rate: float | None = None
        if has_fold_col and not late_folds_df.empty:
            sym_late = late_folds_df[late_folds_df["symbol"].astype(str) == sym].copy()
            if not sym_late.empty:
                try:
                    tl_late, _ec2 = run_backtest(sym_late, sym_config, mode="single", investor_mode=investor_mode)
                    if tl_late is not None and not tl_late.empty and "return_pct" in tl_late.columns:
                        rets_late = pd.to_numeric(tl_late["return_pct"], errors="coerce").dropna()
                        if not rets_late.empty:
                            late_win_rate = len(rets_late[rets_late > 0]) / len(rets_late)
                except Exception:
                    pass

        if late_win_rate is not None and late_win_rate >= 0.55:
            passed_symbols.append(sym)
        else:
            excluded_symbols.append(sym)
            exclusion_reasons[sym] = "；".join(reasons)

    passed_symbols   = sorted(passed_symbols)
    excluded_symbols = sorted(excluded_symbols)

    if not investor_mode:
        print("股票適用性過濾結果：")
        print(f"通過篩選：{passed_symbols}")
        print(f"被排除：{excluded_symbols}（原因：{exclusion_reasons}）")

    return passed_symbols, excluded_symbols, exclusion_reasons


def print_portfolio_report(
    metrics: dict,
    equity_curve: pd.DataFrame,
    config: PipelineConfig,
    investor_mode: bool = False,
) -> None:
    """輸出 Portfolio 層級績效報告，並將每日淨值曲線輸出為 CSV。"""
    initial_capital = config.backtest_initial_capital

    if equity_curve.empty or not metrics:
        print("Portfolio 回測無有效資料，無法輸出報告。")
        return

    start_date   = pd.to_datetime(equity_curve["date"].min()).date()
    end_date     = pd.to_datetime(equity_curve["date"].max()).date()
    final_equity = float(metrics.get("final_equity", equity_curve["total_equity"].iloc[-1]))
    config_version = getattr(
        config,
        "backtest_profile_name",
        getattr(config, "backtest_portfolio_config_version", "balanced_v2_combined"),
    )
    entry_mode = getattr(config, "backtest_entry_mode", "balanced_support")

    print("=" * 70)
    print("Portfolio 層級績效報告（真實資金限制回測）")
    print("=" * 70)
    print(f"配置版本：{config_version}")
    print(f"進場模式：{entry_mode}")
    print(f"最大持股數：{config.backtest_max_positions} 檔")
    print(f"初始試倉比例：{config.backtest_initial_position_pct:.1%}")
    print(f"第一次加碼比例：{config.backtest_add1_position_pct:.1%}")
    print(f"第二次加碼比例：{config.backtest_add2_position_pct:.1%}")
    print(f"單檔最大倉位：{config.backtest_max_single_position_pct:.1%}")
    print(f"最大總曝險：{config.backtest_max_total_exposure:.1%}")
    print(f"最低現金保留：{config.backtest_min_cash_reserve:.1%}")
    print(f"回測期間：{start_date} ~ {end_date}")
    print(f"起始資金：{initial_capital:,.0f} 元")
    print(f"最終資產：{final_equity:,.0f} 元")
    print(f"總報酬率：{metrics.get('total_return', float('nan')):.2%}")
    print(f"年化報酬率（CAGR）：{metrics.get('cagr', float('nan')):.2%}")
    print(f"最大回撤（MDD）：{metrics.get('max_drawdown', float('nan')):.2%}")
    print(f"Sharpe Ratio：{metrics.get('sharpe_ratio', float('nan')):.2f}")
    cr = metrics.get("calmar_ratio")
    print(f"Calmar Ratio：{cr:.2f}" if pd.notna(cr) else "Calmar Ratio：N/A")
    print(f"平均持倉數量：{metrics.get('avg_positions', float('nan')):.1f} 支")
    print(f"資金使用率：{metrics.get('capital_utilization', float('nan')):.1%}")

    # ── 加碼統計（金字塔式分層進場）────────────────────────────────
    total_entries = metrics.get("total_entries", 0)
    add1_count    = metrics.get("add1_count", 0)
    add2_count    = metrics.get("add2_count", 0)
    avg_final_position = metrics.get("avg_final_position", float("nan"))
    avg_exposure        = metrics.get("avg_exposure", float("nan"))

    print(f"試倉次數：{total_entries}")
    if total_entries > 0:
        print(f"完成第一次加碼次數：{add1_count}（{add1_count / total_entries:.1%}）")
        print(f"完成第二次加碼次數：{add2_count}（{add2_count / total_entries:.1%}）")
    else:
        print("完成第一次加碼次數：N/A")
        print("完成第二次加碼次數：N/A")
    print(f"平均每筆最終倉位：{avg_final_position:.1%}" if pd.notna(avg_final_position) else "平均每筆最終倉位：N/A")
    print(f"平均持倉曝險：{avg_exposure:.1%}" if pd.notna(avg_exposure) else "平均持倉曝險：N/A")


    # ── 大盤過濾統計 ───────────────────────────────────────────────
    total_days       = metrics.get("total_days", 0)
    filter_days      = metrics.get("filter_days", 0)
    filtered_entries = metrics.get("filtered_entries", 0)

    if total_days > 0:
        print(f"大盤過濾啟動天數：{filter_days}（佔回測期間 {filter_days / total_days:.1%}）")
    else:
        print("大盤過濾啟動天數：N/A")
    print(f"大盤過濾期間避免的潛在進場次數：{filtered_entries}")

    if not investor_mode:
        # ── return_20d_score 分組績效統計 ─────────────────────────────────
        high_r20d_count = metrics.get("high_return_20d_count", 0)
        high_r20d_avg   = metrics.get("high_return_20d_avg_return", float("nan"))
        low_r20d_avg    = metrics.get("low_return_20d_avg_return", float("nan"))
        if total_entries > 0:
            print(f"return_20d_score daily rank 前 30% 的進場次數：{high_r20d_count}（{high_r20d_count / total_entries:.1%}）")
        else:
            print("return_20d_score daily rank 前 30% 的進場次數：N/A")
        print(f"高 return_20d_score 交易的平均報酬：{high_r20d_avg:.2%}" if pd.notna(high_r20d_avg) else "高 return_20d_score 交易的平均報酬：N/A")
        print(f"低 return_20d_score 交易的平均報酬：{low_r20d_avg:.2%}" if pd.notna(low_r20d_avg) else "低 return_20d_score 交易的平均報酬：N/A")

        r20d_stats = metrics.get("return_20d_score_stats", {})
        print()
        print("return_20d_score 驗收診斷：")
        print(f"return_20d_score 欄位存在：{bool(r20d_stats.get('column_present', False))}")
        _r20d_nonzero = r20d_stats.get("nonzero_ratio", float("nan"))
        print(f"return_20d_score 非零值比例：{_r20d_nonzero:.1%}" if pd.notna(_r20d_nonzero) else "return_20d_score 非零值比例：N/A")
        _r20d_mean = r20d_stats.get("mean", float("nan"))
        print(f"return_20d_score 平均值：{_r20d_mean:.3f}" if pd.notna(_r20d_mean) else "return_20d_score 平均值：N/A")
        print(f"OOS/Portfolio dataframe 欄位存在：{bool(r20d_stats.get('column_present', False))}")
        print(f"count：{int(r20d_stats.get('count', 0) or 0)}")
        print(f"return_20d_score_daily_rank 欄位存在：{bool(r20d_stats.get('daily_rank_column_present', False))}")
        print(f"daily_rank_top30_count：{int(r20d_stats.get('daily_rank_top30_count', 0) or 0)}")
        print(f"daily_rank_top20_count：{int(r20d_stats.get('daily_rank_top20_count', 0) or 0)}")
        for key, label in [
            ("mean", "mean"),
            ("min", "min"),
            ("max", "max"),
            ("q25", "25%"),
            ("q50", "50%"),
            ("q75", "75%"),
            ("nan_ratio", "NaN 比例"),
        ]:
            value = r20d_stats.get(key, np.nan)
            if key == "nan_ratio":
                print(f"{label}：{value:.1%}" if pd.notna(value) else f"{label}：N/A")
            else:
                print(f"{label}：{value:.4f}" if pd.notna(value) else f"{label}：N/A")
        print(f"進場缺值改以 0.5 處理次數：{int(r20d_stats.get('entry_missing_filled_count', 0) or 0)}")
        print(f"缺值處理原因：{r20d_stats.get('missing_fill_reason', 'N/A')}")

        missing_cols = metrics.get("missing_trade_log_columns", [])
        print()
        print("交易紀錄欄位驗收：")
        if missing_cols:
            print(f"缺少欄位：{missing_cols}")
        else:
            print("必要欄位：全部存在")

        print()
        print("Portfolio 執行事件摘要：")
        print(f"同股加碼次數（獲利中）：{int(metrics.get('same_stock_add_count', 0) or 0)}")
        print(f"冷卻期觸發次數（停損事件數）：{int(metrics.get('cooldown_triggered_count', 0) or 0)}")
        print(f"冷卻期擋下次數：{int(metrics.get('cooldown_rejected_count', 0) or 0)}")
        print(f"冷卻期後重新進場次數：{int(metrics.get('post_cooldown_entries', 0) or 0)}")
        print(f"分批出場後新試倉次數：{int(metrics.get('post_partial_exit_entries', 0) or 0)}")
        print(f"假跌破次數：{int(metrics.get('false_breakdown_count', 0) or 0)}")
        print(f"跳空停損次數：{int(metrics.get('gap_down_stop_count', 0) or 0)}")
        print(f"收盤確認停損次數：{int(metrics.get('close_confirm_stop_count', 0) or 0)}")
        print(f"trailing stop 出場次數：{int(metrics.get('trailing_stop_exit_count', 0) or 0)}")
        print(f"獲利 30% 分批出場次數：{int(metrics.get('profit_30_count', 0) or 0)}")
        print(f"獲利 50% 全出次數：{int(metrics.get('profit_50_count', 0) or 0)}")
        print(f"趨勢結構破壞出場次數：{int(metrics.get('trend_break_count', 0) or 0)}")
        print(f"60日期限到期次數：{int(metrics.get('expired_count', 0) or 0)}")
        print(f"獲利出場後重新進場次數：{int(metrics.get('re_entry_count', 0) or 0)}")
        _avg_pp = metrics.get("avg_peak_profit", float("nan"))
        print(f"持倉期間平均最高獲利率：{_avg_pp:.2%}" if pd.notna(_avg_pp) else "持倉期間平均最高獲利率：N/A")

        add2_diag = metrics.get("add2_diagnostics", {})
        print()
        print("第二次加碼診斷：")
        for key in [
            "add2_candidate_count",
            "rejected_add2_by_score",
            "rejected_add2_by_support_quality",
            "rejected_add2_by_cash",
            "rejected_add2_by_exposure",
            "rejected_add2_by_price_condition",
            "rejected_add2_by_trend_condition",
            "rejected_add2_by_defense_line",
            "rejected_add2_by_existing_flag",
        ]:
            print(f"{key}：{int(add2_diag.get(key, 0) or 0)}")

    entry_diag = metrics.get("entry_diagnostics", {})
    primary_counts = metrics.get("entry_primary_rejection_counts", {})
    entry_diag_report_order = [
        "candidate_entry_signals",
        "executed_entries",
        "entry_execution_rate",
        "executed_by_support_entry",
        "executed_by_relaxed_support_entry",
        "accepted_strong_continuation_entries",
        "volume_confirmed_support_entry",
        "volume_confirmed_strong_entry",
        "rejected_by_market_filter",
        "rejected_by_support_zone",
        "rejected_by_momentum_condition",
        "rejected_by_volume_confirmation",
        "rejected_by_stop_loss_pct",
        "rejected_by_rr",
        "rejected_by_cash",
        "rejected_by_position_limit",
        "rejected_by_total_exposure",
        "rejected_by_single_position_limit",
        "rejected_by_existing_position",
        "support_entry_candidates",
        "strong_continuation_candidates",
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
    ]
    entry_signal_display_rows = [
        ("candidate_entry_signals", "\u5019\u9078\u9032\u5834\u8a0a\u865f\u6578", "int"),
        ("executed_entries", "\u5be6\u969b\u9032\u5834\u6b21\u6578", "int"),
        ("entry_execution_rate", "\u9032\u5834\u57f7\u884c\u7387", "pct"),
        ("executed_by_support_entry", "support_entry \u9032\u5834\u6b21\u6578", "int"),
        ("executed_by_relaxed_support_entry", "relaxed_support_entry \u9032\u5834\u6b21\u6578", "int"),
        ("accepted_strong_continuation_entries", "strong_continuation_entry \u9032\u5834\u6b21\u6578", "int"),
        ("volume_confirmed_support_entry", "volume_confirmed_support_entry \u9032\u5834\u6b21\u6578", "int"),
        ("volume_confirmed_strong_entry", "volume_confirmed_strong_entry \u9032\u5834\u6b21\u6578", "int"),
    ]
    strong_diag_rows = [
        ("strong_continuation_candidates", "strong_continuation_candidates"),
        ("accepted_strong_continuation_entries", "accepted_strong_continuation_entries"),
        ("rejected_strong_by_trend", "rejected_strong_by_trend"),
        ("rejected_strong_by_return_rank", "rejected_strong_by_return_rank"),
        ("rejected_strong_by_high_position", "rejected_strong_by_high_position"),
        ("rejected_strong_by_atr_risk", "rejected_strong_by_atr_risk"),
        ("rejected_strong_by_market_filter", "rejected_strong_by_market_filter"),
        ("rejected_strong_by_cash", "rejected_strong_by_cash"),
        ("rejected_strong_by_position_limit", "rejected_strong_by_position_limit"),
        ("rejected_strong_by_total_exposure", "rejected_strong_by_total_exposure"),
        ("rejected_strong_by_existing_position", "rejected_strong_by_existing_position"),
        ("rejected_strong_by_rr", "rejected_strong_by_rr"),
        ("rejected_strong_by_volume_soft_condition", "rejected_strong_by_volume_soft_condition"),
    ]
    primary_sorted = sorted(
        ((key, int(value)) for key, value in primary_counts.items() if int(value) > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )

    if entry_diag and not investor_mode:
        print()
        print("\u9032\u5834\u8a0a\u865f\u8a3a\u65b7\uff1a")
        for key, label, value_type in entry_signal_display_rows:
            value = entry_diag.get(key, np.nan)
            if value_type == "pct":
                print(f"{label}: {value:.1%}" if pd.notna(value) else f"{label}: N/A")
            else:
                print(f"{label}: {int(value)}" if pd.notna(value) else f"{label}: N/A")

        print()
        print("Strong Continuation Entry Diagnostics:")
        for key, label in strong_diag_rows:
            value = entry_diag.get(key, np.nan)
            print(f"{label}: {int(value)}" if pd.notna(value) else f"{label}: N/A")

        if int(entry_diag.get("strong_continuation_candidates", 0) or 0) == 0:
            print("Strong continuation available column check:")
            for col, info in metrics.get("strong_continuation_column_check", {}).items():
                present = bool(info.get("present", False)) if isinstance(info, dict) else False
                non_null = int(info.get("non_null_count", 0) or 0) if isinstance(info, dict) else 0
                print(f"{col}: present={present}, non_null_count={non_null}")

        print()
        print("\u4e3b\u8981\u62d2\u7d55\u539f\u56e0\uff1a")
        if primary_sorted:
            for i, (key, value) in enumerate(primary_sorted[:3], start=1):
                label = ENTRY_REJECTION_LABELS.get(key, key)
                print(f"{i}. {label}: {value}")
        else:
            print("1. N/A")

    if False and entry_diag:
        print()
        print("進場訊號診斷：")
        for key in entry_diag_report_order:
            label = ENTRY_DIAGNOSTIC_LABELS.get(key, key)
            value = entry_diag.get(key, np.nan)
            if key in ("entry_execution_rate", "volume_confirmation_rate"):
                print(f"{label}：{value:.1%}" if pd.notna(value) else f"{label}：N/A")
            else:
                print(f"{label}：{int(value)}" if pd.notna(value) else f"{label}：N/A")

        print()
        print("主要拒絕原因排序：")
        if primary_sorted:
            for i, (key, value) in enumerate(primary_sorted[:3], start=1):
                label = ENTRY_REJECTION_LABELS.get(key, key)
                print(f"{i}. {label}：{value}")
        else:
            print("1. N/A")

    print()
    print("每月績效：")

    rejection_future_summary = metrics.get("rejection_future_return_summary")
    if not investor_mode and isinstance(rejection_future_summary, pd.DataFrame) and not rejection_future_summary.empty:
        print()
        print("Rejected Signal Forward Return Diagnostics:")
        display_rej = rejection_future_summary.copy()
        for col in ["future_return_10d", "future_return_20d"]:
            if col in display_rej.columns:
                display_rej[col] = display_rej[col].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
        print(display_rej.to_string(index=False))

    trailing_future_summary = metrics.get("trailing_stop_future_return_summary")
    if not investor_mode and isinstance(trailing_future_summary, pd.DataFrame) and not trailing_future_summary.empty:
        print()
        print("Trailing Stop Forward Return Diagnostics:")
        display_ts = trailing_future_summary.copy()
        for col in ["future_return_10d", "future_return_20d"]:
            if col in display_ts.columns:
                display_ts[col] = display_ts[col].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
        print(display_ts.to_string(index=False))

    monthly_df = equity_curve.copy()
    monthly_df["date"]  = pd.to_datetime(monthly_df["date"])
    monthly_df["month"] = monthly_df["date"].dt.strftime("%Y-%m")
    month_end_equity = monthly_df.groupby("month")["total_equity"].last()
    prev_month_equity = month_end_equity.shift(1)
    if len(prev_month_equity) > 0:
        prev_month_equity.iloc[0] = initial_capital
    monthly_returns = month_end_equity / prev_month_equity - 1.0

    for month, ret in monthly_returns.items():
        print(f"{month}：{ret:+.2%}" if pd.notna(ret) else f"{month}：N/A")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    ts = datetime.today().strftime("%Y%m%d_%H%M%S")
    ec_out = equity_curve.copy()
    ec_out["profile_name"] = config_version
    ec_out["entry_mode"] = entry_mode
    if "positions" in ec_out.columns:
        ec_out["positions"] = ec_out["positions"].apply(
            lambda lst: ",".join(lst) if isinstance(lst, (list, tuple)) else lst
        )
    csv_path = output_dir / f"portfolio_equity_curve_{config_version}_{ts}.csv"
    ec_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if entry_diag:
        diag_rows = [
            {"metric": "profile_name", "value": config_version},
            {"metric": "entry_mode", "value": entry_mode},
        ] + [
            {"metric": key, "value": entry_diag.get(key, np.nan)}
            for key in entry_diag_report_order
        ]
        for key, value in primary_sorted:
            diag_rows.append({
                "metric": f"primary_{key}",
                "value": value,
            })
        for i, (key, value) in enumerate(primary_sorted[:3], start=1):
            diag_rows.append({
                "metric": f"primary_rejection_reason_{i}",
                "value": ENTRY_REJECTION_LABELS.get(key, key),
            })
            diag_rows.append({
                "metric": f"primary_rejection_count_{i}",
                "value": value,
            })
        for key, value in metrics.get("return_20d_score_stats", {}).items():
            diag_rows.append({
                "metric": f"return_20d_score_{key}",
                "value": value,
            })
        for col, info in metrics.get("strong_continuation_column_check", {}).items():
            if isinstance(info, dict):
                diag_rows.append({
                    "metric": f"strong_continuation_column_{col}_present",
                    "value": info.get("present", False),
                })
                diag_rows.append({
                    "metric": f"strong_continuation_column_{col}_non_null_count",
                    "value": info.get("non_null_count", 0),
                })
        diag_rows.append({
            "metric": "missing_trade_log_columns",
            "value": ",".join(metrics.get("missing_trade_log_columns", [])),
        })
        for key in [
            "same_stock_add_count",
            "cooldown_triggered_count",
            "cooldown_rejected_count",
            "post_cooldown_entries",
            "post_partial_exit_entries",
            "false_breakdown_count",
            "gap_down_stop_count",
            "close_confirm_stop_count",
            "trailing_stop_exit_count",
        ]:
            diag_rows.append({
                "metric": key,
                "value": metrics.get(key, np.nan),
            })
        for key, value in metrics.get("add2_diagnostics", {}).items():
            diag_rows.append({
                "metric": key,
                "value": value,
            })
        diag_path = output_dir / f"portfolio_entry_diagnostics_{config_version}_{ts}.csv"
        pd.DataFrame(diag_rows).to_csv(diag_path, index=False, encoding="utf-8-sig")

        daily_diag = metrics.get("entry_diagnostics_daily")
        if isinstance(daily_diag, pd.DataFrame) and not daily_diag.empty:
            daily_out = daily_diag.copy()
            daily_out["profile_name"] = config_version
            daily_out["entry_mode"] = entry_mode
            daily_cols = ["date"] + ENTRY_DIAGNOSTIC_DAILY_COLUMNS
            daily_cols = ["profile_name", "entry_mode"] + daily_cols
            daily_out = daily_out[[c for c in daily_cols if c in daily_out.columns]]
            daily_path = output_dir / f"portfolio_entry_diagnostics_daily_{config_version}_{ts}.csv"
            daily_out.to_csv(daily_path, index=False, encoding="utf-8-sig")
        else:
            daily_path = None

        rejection_future_summary = metrics.get("rejection_future_return_summary")
        if isinstance(rejection_future_summary, pd.DataFrame) and not rejection_future_summary.empty:
            rejection_future_path = output_dir / f"portfolio_rejection_future_returns_{config_version}_{ts}.csv"
            rejection_future_summary.to_csv(rejection_future_path, index=False, encoding="utf-8-sig")
        else:
            rejection_future_path = None

        trailing_future_detail = metrics.get("trailing_stop_future_return_detail")
        trailing_future_summary = metrics.get("trailing_stop_future_return_summary")
        if isinstance(trailing_future_detail, pd.DataFrame) and not trailing_future_detail.empty:
            trailing_future_path = output_dir / f"portfolio_trailing_stop_future_returns_{config_version}_{ts}.csv"
            trailing_future_detail.to_csv(trailing_future_path, index=False, encoding="utf-8-sig")
            if isinstance(trailing_future_summary, pd.DataFrame) and not trailing_future_summary.empty:
                trailing_summary_path = output_dir / f"portfolio_trailing_stop_future_summary_{config_version}_{ts}.csv"
                trailing_future_summary.to_csv(trailing_summary_path, index=False, encoding="utf-8-sig")
            else:
                trailing_summary_path = None
        else:
            trailing_future_path = None
            trailing_summary_path = None

    print()
    print(f"每日淨值曲線已輸出：{csv_path}")
    if entry_diag:
        print(f"進場診斷總表已輸出：{diag_path}")
        if daily_path is not None:
            print(f"每日進場診斷明細已輸出：{daily_path}")
        if rejection_future_path is not None:
            print(f"Rejected signal forward-return diagnostics saved: {rejection_future_path}")
        if trailing_future_path is not None:
            print(f"Trailing stop forward-return diagnostics saved: {trailing_future_path}")
        if trailing_summary_path is not None:
            print(f"Trailing stop forward-return summary saved: {trailing_summary_path}")
    print("=" * 70)


def _run_daily_top_n_portfolio_backtest(
    oos_df: pd.DataFrame,
    config: PipelineConfig,
    top_n: int,
    rank_metric: str,
    investor_mode: bool,
) -> None:
    """Run the portfolio directly from the point-in-time daily universe.

    The normal batch path creates a static eligibility whitelist after
    per-symbol diagnostics. That is not valid for a daily Top-N universe:
    securities may enter or leave it, while unscored rows remain only for exits.
    """
    if oos_df.empty:
        print("每日 Top-N OOS 資料不足，無法執行 Portfolio 回測。")
        return

    prediction_mask = oos_df.get(
        "is_oos_prediction", pd.Series(True, index=oos_df.index)
    ).fillna(False).astype(bool)
    prediction_rows = oos_df.loc[prediction_mask].copy()
    if prediction_rows.empty:
        print("每日 Top-N 沒有有效的 OOS 模型預測列，無法執行 Portfolio 回測。")
        return

    cutoff_date = _symbol_eligibility_cutoff_date(prediction_rows, config)
    portfolio_rows = oos_df.copy()
    portfolio_rows["date"] = pd.to_datetime(portfolio_rows["date"], errors="coerce")
    if cutoff_date is not None:
        portfolio_rows = portfolio_rows[
            portfolio_rows["date"] > pd.Timestamp(cutoff_date)
        ].copy()
        print(
            "每日 Top-N 正式 Portfolio 回測排除前兩折 warm-up 期，"
            f"績效自 {pd.Timestamp(cutoff_date).date()} 之後開始計算。"
        )

    symbols = sorted(portfolio_rows["symbol"].dropna().astype(str).unique().tolist())
    all_stock_signals = {
        symbol: portfolio_rows[portfolio_rows["symbol"].astype(str) == symbol].copy()
        for symbol in symbols
    }
    print(
        f"Step 3/3: 以每日 {rank_metric} Top {top_n} 的 OOS 訊號執行 Portfolio 回測"
        f"（期間聯集 {len(symbols)} 檔；不使用事後靜態股票白名單）。"
    )

    print("取得 TAIEX 指數歷史資料（大盤過濾用）...")
    try:
        market_df = get_market_index_data(config)
    except Exception as exc:
        print(f"WARNING: TAIEX 指數資料取得失敗：{exc}")
        market_df = pd.DataFrame()

    try:
        equity_curve, trade_log, metrics = run_portfolio_backtest(
            all_stock_signals=all_stock_signals,
            config=config,
            market_df=market_df,
            expected_symbols=symbols,
            investor_mode=investor_mode,
        )
    except Exception as exc:
        print(f"每日 Top-N Portfolio 回測失敗：{exc}")
        return

    print()
    print_portfolio_report(metrics, equity_curve, config, investor_mode=investor_mode)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    prefix = output_dir / (
        f"{config.market.upper()}_daily_{rank_metric}_top{top_n}_portfolio_{timestamp}"
    )
    if not trade_log.empty:
        trade_path = Path(f"{prefix}_trade_log.csv")
        trade_log.to_csv(trade_path, index=False, encoding="utf-8-sig")
        print(f"每日 Top-N 交易明細已輸出：{trade_path}")
    if not equity_curve.empty:
        equity_path = Path(f"{prefix}_equity_curve.csv")
        equity_curve.to_csv(equity_path, index=False, encoding="utf-8-sig")
        print(f"每日 Top-N 資產曲線已輸出：{equity_path}")


def _buy_hold_metrics_from_equity(equity: pd.Series, dates: pd.Series) -> dict:
    """
    依買進持有淨值序列計算 total_return / cagr / max_drawdown / sharpe_ratio / calmar_ratio，
    計算方式與 calculate_portfolio_metrics() 一致，確保可與策略績效直接比較。
    """
    eq = pd.to_numeric(pd.Series(equity).reset_index(drop=True), errors="coerce")
    dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    valid = eq.notna() & dt.notna()
    eq = eq[valid].reset_index(drop=True)
    dt = dt[valid].reset_index(drop=True)

    empty_result = {
        "total_return": np.nan, "cagr": np.nan, "max_drawdown": np.nan,
        "sharpe_ratio": np.nan, "calmar_ratio": np.nan,
    }
    if eq.empty:
        return empty_result

    initial = float(eq.iloc[0])
    final   = float(eq.iloc[-1])
    if initial <= 0:
        return empty_result
    total_return = (final - initial) / initial

    n_days = max((dt.iloc[-1] - dt.iloc[0]).days, 1)
    cagr = (final / initial) ** (365.0 / n_days) - 1.0

    vals         = eq.to_numpy(dtype=float)
    rolling_max  = np.maximum.accumulate(vals)
    drawdowns    = (vals - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    daily_returns = pd.Series(vals).pct_change().dropna()
    mean_daily = float(daily_returns.mean()) if not daily_returns.empty else 0.0
    std_daily  = float(daily_returns.std())  if len(daily_returns) > 1 else 0.0
    annualized_mean_return = mean_daily * 252
    sharpe_ratio = (
        (annualized_mean_return - 0.015) / (std_daily * np.sqrt(252))
        if std_daily > 0 else 0.0
    )
    calmar_ratio = (cagr / abs(max_drawdown)) if max_drawdown != 0 else np.nan

    return {
        "total_return":  total_return,
        "cagr":          cagr,
        "max_drawdown":  max_drawdown,
        "sharpe_ratio":  sharpe_ratio,
        "calmar_ratio":  calmar_ratio,
    }


def build_benchmark_equity_curves(
    config: PipelineConfig,
    all_stock_signals: dict[str, pd.DataFrame],
    target_dates: pd.Series,
    initial_capital: float,
    tsmc_symbol: str = "2330",
) -> tuple[pd.DataFrame, dict]:
    """
    建立三種 Benchmark 的每日淨值曲線（對齊至 target_dates，與 Portfolio 回測使用同一組交易日）：
      1. TAIEX 買進持有
      2. Top N 等權買進持有（all_stock_signals 內所有股票，回測起始日等權買入）
      3. 2330 買進持有

    Returns:
        (benchmark_df[date, taiex_equity, top30_equity, tsmc_equity], metrics_dict)
        metrics_dict = {"taiex": {...}, "top30": {...}, "tsmc": {...}}
    """
    target_dates = (
        pd.to_datetime(pd.Series(target_dates), errors="coerce")
        .dropna().drop_duplicates().sort_values().reset_index(drop=True)
    )
    empty_df = pd.DataFrame(columns=["date", "taiex_equity", "top30_equity", "tsmc_equity"])
    if target_dates.empty:
        return empty_df, {}

    def _reindex_ffill(df_with_date: pd.DataFrame, value_col: str) -> pd.Series:
        """將 date-indexed 序列對齊到 target_dates；缺值以前值（必要時後值）填補。"""
        s = df_with_date.set_index("date")[value_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        aligned = s.reindex(pd.Index(target_dates)).ffill().bfill()
        return aligned

    # ── Benchmark 1：TAIEX 買進持有 ──────────────────────────────────
    taiex_equity = pd.Series(np.nan, index=range(len(target_dates)))
    try:
        taiex_df = get_market_index_data(config)
    except Exception as exc:
        print(f"WARNING: 取得 TAIEX 指數資料失敗：{exc}")
        taiex_df = pd.DataFrame()

    if taiex_df is not None and not taiex_df.empty and "market_close" in taiex_df.columns:
        t_df = taiex_df[["date", "market_close"]].copy()
        t_df["date"] = pd.to_datetime(t_df["date"], errors="coerce")
        t_df = t_df.dropna(subset=["date"]).rename(columns={"market_close": "close"})
        t_series = _reindex_ffill(t_df, "close").reset_index(drop=True)
        base_price = t_series.iloc[0] if t_series.notna().any() else np.nan
        if pd.notna(base_price) and base_price:
            taiex_equity = initial_capital * (t_series / base_price)

    # ── Benchmark 2：Top N 等權買進持有 ──────────────────────────────
    symbol_series: dict[str, pd.Series] = {}
    for sym, sym_df in (all_stock_signals or {}).items():
        if sym_df is None or sym_df.empty or "close" not in sym_df.columns:
            continue
        d = sym_df[["date", "close"]].copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        if d.empty:
            continue
        symbol_series[str(sym)] = _reindex_ffill(d, "close").reset_index(drop=True)

    top30_equity = pd.Series(0.0, index=range(len(target_dates)))
    n_syms = len(symbol_series)
    if n_syms > 0:
        per_symbol_capital = initial_capital / n_syms
        any_valid = False
        for s in symbol_series.values():
            entry_price = s.iloc[0] if s.notna().any() else np.nan
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            shares = per_symbol_capital / entry_price
            top30_equity = top30_equity.add(shares * s.fillna(entry_price), fill_value=0.0)
            any_valid = True
        if not any_valid:
            top30_equity = pd.Series(np.nan, index=range(len(target_dates)))
    else:
        top30_equity = pd.Series(np.nan, index=range(len(target_dates)))

    # ── Benchmark 3：2330 買進持有 ────────────────────────────────────
    tsmc_df = None
    sig_tsmc = (all_stock_signals or {}).get(tsmc_symbol)
    if sig_tsmc is not None and not sig_tsmc.empty and "close" in sig_tsmc.columns:
        tsmc_df = sig_tsmc[["date", "close"]].copy()
    else:
        try:
            tsmc_config = replace(config, stock_no=tsmc_symbol)
            raw_tsmc = get_last_n_years_trading_data(tsmc_config)
            if raw_tsmc is not None and not raw_tsmc.empty and "close" in raw_tsmc.columns:
                tsmc_df = raw_tsmc[["date", "close"]].copy()
        except Exception as exc:
            print(f"WARNING: 無法取得 {tsmc_symbol} 價格資料作為 Benchmark：{exc}")

    tsmc_equity = pd.Series(np.nan, index=range(len(target_dates)))
    if tsmc_df is not None and not tsmc_df.empty:
        tsmc_df["date"] = pd.to_datetime(tsmc_df["date"], errors="coerce")
        tsmc_df = tsmc_df.dropna(subset=["date"])
        tsmc_series = _reindex_ffill(tsmc_df, "close").reset_index(drop=True)
        base_price = tsmc_series.iloc[0] if tsmc_series.notna().any() else np.nan
        if pd.notna(base_price) and base_price:
            tsmc_equity = initial_capital * (tsmc_series / base_price)

    benchmark_df = pd.DataFrame({
        "date":          target_dates.to_numpy(),
        "taiex_equity":  taiex_equity.to_numpy(),
        "top30_equity":  top30_equity.to_numpy(),
        "tsmc_equity":   tsmc_equity.to_numpy(),
    })

    metrics = {
        "taiex": _buy_hold_metrics_from_equity(taiex_equity, target_dates),
        "top30": _buy_hold_metrics_from_equity(top30_equity, target_dates),
        "tsmc":  _buy_hold_metrics_from_equity(tsmc_equity, target_dates),
    }
    return benchmark_df, metrics


def print_strategy_vs_benchmark_report(
    strategy_metrics: dict,
    benchmark_metrics: dict,
) -> None:
    """在 Portfolio 層級績效報告之後，輸出策略 vs 三種 Benchmark 的比較表格。"""
    s     = strategy_metrics or {}
    taiex = (benchmark_metrics or {}).get("taiex", {}) or {}
    top30 = (benchmark_metrics or {}).get("top30", {}) or {}
    tsmc  = (benchmark_metrics or {}).get("tsmc", {}) or {}

    def _g(d: dict, key: str) -> float:
        v = d.get(key, np.nan)
        return float(v) if pd.notna(v) else float("nan")

    strategy_return = _g(s, "total_return")
    strategy_cagr    = _g(s, "cagr")
    strategy_mdd      = _g(s, "max_drawdown")
    strategy_sharpe = _g(s, "sharpe_ratio")
    strategy_calmar = _g(s, "calmar_ratio")

    taiex_return = _g(taiex, "total_return")
    taiex_cagr   = _g(taiex, "cagr")
    taiex_mdd    = _g(taiex, "max_drawdown")
    taiex_sharpe = _g(taiex, "sharpe_ratio")
    taiex_calmar = _g(taiex, "calmar_ratio")

    top30_return = _g(top30, "total_return")
    top30_cagr   = _g(top30, "cagr")
    top30_mdd    = _g(top30, "max_drawdown")
    top30_sharpe = _g(top30, "sharpe_ratio")
    top30_calmar = _g(top30, "calmar_ratio")

    tsmc_return = _g(tsmc, "total_return")
    tsmc_cagr   = _g(tsmc, "cagr")
    tsmc_mdd    = _g(tsmc, "max_drawdown")
    tsmc_sharpe = _g(tsmc, "sharpe_ratio")
    tsmc_calmar = _g(tsmc, "calmar_ratio")

    print("=" * 70)
    print("策略 vs Benchmark 比較")
    print("=" * 70)
    print(f"{'指標':<20} {'本策略':>12} {'TAIEX':>12} {'Top30等權':>12} {'2330':>12}")
    print("-" * 70)
    print(f"{'總報酬率':<20} {strategy_return:>11.2%} {taiex_return:>11.2%} {top30_return:>11.2%} {tsmc_return:>11.2%}")
    print(f"{'年化報酬(CAGR)':<20} {strategy_cagr:>11.2%} {taiex_cagr:>11.2%} {top30_cagr:>11.2%} {tsmc_cagr:>11.2%}")
    print(f"{'最大回撤(MDD)':<20} {strategy_mdd:>11.2%} {taiex_mdd:>11.2%} {top30_mdd:>11.2%} {tsmc_mdd:>11.2%}")
    print(f"{'Sharpe Ratio':<20} {strategy_sharpe:>11.2f} {taiex_sharpe:>11.2f} {top30_sharpe:>11.2f} {tsmc_sharpe:>11.2f}")
    print(f"{'Calmar Ratio':<20} {strategy_calmar:>11.2f} {taiex_calmar:>11.2f} {top30_calmar:>11.2f} {tsmc_calmar:>11.2f}")
    print("-" * 70)
    print(f"{'超額報酬 vs TAIEX':<20} {strategy_return - taiex_return:>11.2%}")
    print(f"{'超額報酬 vs Top30':<20} {strategy_return - top30_return:>11.2%}")
    print("=" * 70)


def run_portfolio_profile_experiments(
    all_stock_signals: dict[str, pd.DataFrame],
    base_config: PipelineConfig,
    market_df: Optional[pd.DataFrame] = None,
    expected_symbols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run fixed Portfolio A/B profiles on the same OOS signals and export a comparison table."""
    rows: list[dict] = []

    for profile in PORTFOLIO_EXPERIMENT_PROFILES:
        profile_name = profile["profile_name"]
        entry_mode = profile["entry_mode"]
        max_positions = profile["backtest_max_positions"]
        profile_config = replace(
            base_config,
            backtest_profile_name=profile_name,
            backtest_portfolio_config_version=profile_name,
            backtest_entry_mode=entry_mode,
            backtest_max_positions=max_positions,
            backtest_initial_position_pct=0.08,
            backtest_add1_position_pct=0.06,
            backtest_add2_position_pct=0.06,
            backtest_max_single_position_pct=0.20,
            backtest_max_total_exposure=0.85,
            backtest_min_cash_reserve=0.15,
        )

        try:
            _, _, metrics = run_portfolio_backtest(
                all_stock_signals=all_stock_signals,
                config=profile_config,
                market_df=market_df,
                expected_symbols=expected_symbols,
            )
            entry_diag = metrics.get("entry_diagnostics", {}) if metrics else {}
            row = {
                "profile_name": profile_name,
                "entry_mode": entry_mode,
                "max_positions": max_positions,
                "final_equity": metrics.get("final_equity", np.nan),
                "total_return": metrics.get("total_return", np.nan),
                "cagr": metrics.get("cagr", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                "calmar_ratio": metrics.get("calmar_ratio", np.nan),
                "avg_positions": metrics.get("avg_positions", np.nan),
                "capital_utilization": metrics.get("capital_utilization", np.nan),
                "avg_exposure": metrics.get("avg_exposure", np.nan),
                "candidate_entry_signals": entry_diag.get("candidate_entry_signals", np.nan),
                "executed_entries": entry_diag.get("executed_entries", np.nan),
                "entry_execution_rate": entry_diag.get("entry_execution_rate", np.nan),
                "rejected_by_support_zone": entry_diag.get("rejected_by_support_zone", np.nan),
                "rejected_by_momentum_condition": entry_diag.get("rejected_by_momentum_condition", np.nan),
                "rejected_by_volume_confirmation": entry_diag.get("rejected_by_volume_confirmation", np.nan),
                "rejected_by_position_limit": entry_diag.get("rejected_by_position_limit", np.nan),
                "rejected_by_existing_position": entry_diag.get("rejected_by_existing_position", np.nan),
                "executed_by_strict_support": entry_diag.get("executed_by_strict_support", np.nan),
                "executed_by_near_support_relaxed": entry_diag.get("executed_by_near_support_relaxed", np.nan),
                "executed_by_strong_momentum": entry_diag.get("executed_by_strong_momentum", np.nan),
                "executed_by_strong_continuation": entry_diag.get("executed_by_strong_continuation", np.nan),
                "executed_by_quiet_pullback_support": entry_diag.get("executed_by_quiet_pullback_support", np.nan),
                "executed_by_volume_rebound_support": entry_diag.get("executed_by_volume_rebound_support", np.nan),
                "executed_by_volume_breakout_momentum": entry_diag.get("executed_by_volume_breakout_momentum", np.nan),
                "candidate_with_volume_confirmation": entry_diag.get("candidate_with_volume_confirmation", np.nan),
                "volume_confirmation_rate": entry_diag.get("volume_confirmation_rate", np.nan),
            }
        except Exception as exc:
            row = {
                "profile_name": profile_name,
                "entry_mode": entry_mode,
                "max_positions": max_positions,
                "final_equity": np.nan,
                "total_return": np.nan,
                "cagr": np.nan,
                "max_drawdown": np.nan,
                "sharpe_ratio": np.nan,
                "calmar_ratio": np.nan,
                "avg_positions": np.nan,
                "capital_utilization": np.nan,
                "avg_exposure": np.nan,
                "candidate_entry_signals": np.nan,
                "executed_entries": np.nan,
                "entry_execution_rate": np.nan,
                "rejected_by_support_zone": np.nan,
                "rejected_by_momentum_condition": np.nan,
                "rejected_by_volume_confirmation": np.nan,
                "rejected_by_position_limit": np.nan,
                "rejected_by_existing_position": np.nan,
                "executed_by_strict_support": np.nan,
                "executed_by_near_support_relaxed": np.nan,
                "executed_by_strong_momentum": np.nan,
                "executed_by_strong_continuation": np.nan,
                "executed_by_quiet_pullback_support": np.nan,
                "executed_by_volume_rebound_support": np.nan,
                "executed_by_volume_breakout_momentum": np.nan,
                "candidate_with_volume_confirmation": np.nan,
                "volume_confirmation_rate": np.nan,
                "error": str(exc),
            }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(exist_ok=True)
    ts = datetime.today().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"portfolio_profile_comparison_{ts}.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("=" * 240)
    print("Portfolio Profile Comparison")
    print("=" * 240)
    print(
        f"{'Profile':<30} {'Total Return':>12} {'CAGR':>10} {'MDD':>10} "
        f"{'Sharpe':>8} {'Calmar':>8} {'Avg Positions':>14} "
        f"{'Capital Utilization':>20} {'Avg Exposure':>14} "
        f"{'Entry Execution Rate':>22} {'Support Rejections':>20} "
        f"{'Position Limit Rejections':>26} {'Executed Strict':>17} "
        f"{'Executed Near Support':>23} {'Executed Momentum':>19} "
        f"{'Executed Continuation':>23} "
        f"{'Executed Quiet Pullback':>25} {'Executed Volume Rebound':>25} "
        f"{'Executed Volume Momentum':>26} {'Volume Confirmation Rate':>26} "
        f"{'Rejected By Volume':>20}"
    )
    print("-" * 240)

    def _fmt_pct(value) -> str:
        return f"{float(value):.2%}" if pd.notna(value) else "N/A"

    def _fmt_float(value, digits: int = 2) -> str:
        return f"{float(value):.{digits}f}" if pd.notna(value) else "N/A"

    def _fmt_int(value) -> str:
        return f"{int(value)}" if pd.notna(value) else "N/A"

    for _, row in comparison_df.iterrows():
        print(
            f"{str(row.get('profile_name', '')):<30} "
            f"{_fmt_pct(row.get('total_return')):>12} "
            f"{_fmt_pct(row.get('cagr')):>10} "
            f"{_fmt_pct(row.get('max_drawdown')):>10} "
            f"{_fmt_float(row.get('sharpe_ratio')):>8} "
            f"{_fmt_float(row.get('calmar_ratio')):>8} "
            f"{_fmt_float(row.get('avg_positions'), 1):>14} "
            f"{_fmt_pct(row.get('capital_utilization')):>20} "
            f"{_fmt_pct(row.get('avg_exposure')):>14} "
            f"{_fmt_pct(row.get('entry_execution_rate')):>22} "
            f"{_fmt_int(row.get('rejected_by_support_zone')):>20} "
            f"{_fmt_int(row.get('rejected_by_position_limit')):>26} "
            f"{_fmt_int(row.get('executed_by_strict_support')):>17} "
            f"{_fmt_int(row.get('executed_by_near_support_relaxed')):>23} "
            f"{_fmt_int(row.get('executed_by_strong_momentum')):>19} "
            f"{_fmt_int(row.get('executed_by_strong_continuation')):>23} "
            f"{_fmt_int(row.get('executed_by_quiet_pullback_support')):>25} "
            f"{_fmt_int(row.get('executed_by_volume_rebound_support')):>25} "
            f"{_fmt_int(row.get('executed_by_volume_breakout_momentum')):>26} "
            f"{_fmt_pct(row.get('volume_confirmation_rate')):>26} "
            f"{_fmt_int(row.get('rejected_by_volume_confirmation')):>20}"
        )
    print("-" * 240)
    print(f"Portfolio profile comparison 已輸出：{comparison_path}")
    print("=" * 240)

    return comparison_df


def run_backtest_mode(
    config: PipelineConfig,
    top_n: int = 30,
    target_symbol: Optional[str] = None,
    mode: str = "single",
    run_profile_experiments: bool = False,
    investor_mode: bool = False,
    daily_top30_universe: bool = False,
    daily_top30_rank_metric: str = "volume",
) -> None:
    """
    Top-level coordinator for Mode 3 backtest analysis.
    Fetches data, trains model, generates historical predictions,
    runs backtest engine, prints report.
    """
    import contextlib, io
    from pathlib import Path

    if investor_mode:
        print("正在分析市場資料...")
    else:
        print_presentation_process_header()

    shared_config = build_shared_common_model_config(config)
    if target_symbol:
        shared_config = replace(shared_config, stock_no=target_symbol)

    universe_membership = None
    execution_source = None
    if daily_top30_universe:
        if target_symbol:
            raise ValueError("每日 Top-N 股票池只支援批次 Portfolio 回測，不支援單股模式。")
        if daily_top30_rank_metric not in {"volume", "turnover"}:
            raise ValueError("daily_top30_rank_metric must be 'volume' or 'turnover'")
        market_df = get_market_index_data(shared_config)
        if market_df.empty or "date" not in market_df.columns:
            raise ValueError("每日 Top-N 股票池需要完整 TAIEX 交易日曆。")
        start_date = get_reference_today() - pd.DateOffset(years=int(shared_config.years))
        end_date = get_reference_today()
        trading_dates = pd.to_datetime(market_df["date"], errors="coerce")
        trading_dates = trading_dates[trading_dates.between(start_date, end_date)]
        universe_membership = build_daily_top_n_membership(
            trading_dates=trading_dates,
            top_n=top_n,
            rank_metric=daily_top30_rank_metric,
            sleep_sec=0,
        )
        symbols = sorted(universe_membership["symbol"].astype(str).unique().tolist())
        stock_list = pd.DataFrame({"symbol": symbols})
        print(
            f"每日 {daily_top30_rank_metric} Top {top_n} 股票池："
            f"{len(trading_dates)} 個交易日、期間聯集 {len(symbols)} 檔。"
        )
    else:
        stock_list, symbols = build_shared_common_stock_pool(
            base_config=config,
            top_n=top_n,
            target_symbol=target_symbol,
            quiet=True,
        )

    print(f"Step 1/3: 抓取 {len(symbols)} 支股票的歷史資料...")
    if not investor_mode:
        print("[DEBUG Backtest Config]")
        print("mode =", getattr(shared_config, "mode", None))
        print("analysis_mode =", getattr(shared_config, "analysis_mode", None))
        print("feature_set =", getattr(shared_config, "feature_set", None))
        print("target_mode =", getattr(shared_config, "target_mode", None))
        print("target_col =", getattr(shared_config, "target_col", None))
        print("use_sensitivity =", getattr(shared_config, "use_sensitivity", None))
        print("require_sensitivity_data =", getattr(shared_config, "require_sensitivity_data", None))
        print("use_return_20d =", getattr(shared_config, "use_return_20d", None))
    with contextlib.redirect_stdout(io.StringIO()):
        if daily_top30_universe:
            snapshot_prices = load_daily_snapshot_prices(
                trading_dates=trading_dates,
                symbols=symbols,
                sleep_sec=0,
            )
            combined_data = fetch_and_prepare_multi_stock_data_historical(
                symbols=symbols,
                base_config=shared_config,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                burn_in_months=0,
                universe_membership=universe_membership,
                historical_price_data=snapshot_prices,
                market_index_data=market_df,
            )
            execution_source = combined_data.copy()
        else:
            combined_data = fetch_and_prepare_multi_stock_data_with_config(
                symbols=symbols,
                base_config=shared_config,
            )

    print(f"Step 2/3: 執行 12 折 rolling walk-forward OOS 預測生成（無資料洩漏）...")
    oos_df = generate_oos_predictions_for_backtest(
        df=combined_data,
        config=shared_config,
        n_splits=12,
    )

    if daily_top30_universe and not oos_df.empty and execution_source is not None:
        oos_dates = pd.to_datetime(oos_df["date"], errors="coerce")
        execution_dates = pd.to_datetime(execution_source["date"], errors="coerce")
        execution_source = execution_source.loc[
            execution_dates.between(oos_dates.min(), oos_dates.max())
        ].copy()
        oos_df = append_unscored_execution_rows(oos_df, execution_source)

    if oos_df.empty:
        print("OOS 預測資料不足，無法執行回測。")
        return

    # ── 候選股排名：最後一個交易日，依 final_decision_score 降序 Top 10 ──
    if "final_decision_score" in oos_df.columns and "date" in oos_df.columns:
        _cv = getattr(
            config,
            "backtest_profile_name",
            getattr(config, "backtest_portfolio_config_version", "portfolio"),
        )
        _ts_cand = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        _last_date = oos_df["date"].max()
        _last_day = oos_df[oos_df["date"] == _last_date].copy()
        _last_day["final_decision_score"] = pd.to_numeric(
            _last_day["final_decision_score"], errors="coerce"
        )
        _last_day = (
            _last_day
            .sort_values("final_decision_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        _want_cols = [
            "symbol", "date",
            "final_decision_score", "safety_score", "return_score",
            "rule_filter_score", "risk_reward_score",
            "score_band", "decision_action", "technical_filter_result",
            "entry_rr_1", "support_entry_zone_low", "support_entry_zone_high",
        ]
        _out_cols = [c for c in _want_cols if c in _last_day.columns]
        _cand_dir = Path(getattr(config, "output_dir", "output"))
        _cand_dir.mkdir(exist_ok=True)
        _cand_path = _cand_dir / f"top_candidates_{_cv}_{_ts_cand}.csv"
        _last_day[_out_cols].to_csv(_cand_path, index=False, encoding="utf-8-sig")
        print(f"候選股排名已輸出（最後交易日：{_last_date}）：{_cand_path}")

    if mode == "single":
        # 單股模式：過濾至目標股票
        if target_symbol:
            oos_df = oos_df[oos_df["symbol"].astype(str) == str(target_symbol)].copy()
            if oos_df.empty:
                print(f"目標股票 {target_symbol} 在 OOS 預測中無資料，請確認股票代碼。")
                return

        bt_config = replace(config, stock_no=target_symbol or "SINGLE")
        print(f"Step 3/3: 使用 Portfolio execution engine 執行指定個股回測（{len(oos_df)} 筆 OOS 歷史信號）...")
        print(f"配置版本：{bt_config.backtest_profile_name}")
        print(f"進場模式：{bt_config.backtest_entry_mode}")

        print("取得 TAIEX 指數歷史資料（大盤過濾用）...")
        try:
            single_market_df = get_market_index_data(bt_config)
        except Exception as exc:
            print(f"WARNING: TAIEX 指數資料取得失敗：{exc}")
            single_market_df = pd.DataFrame()

        single_symbol = str(target_symbol or bt_config.stock_no or "SINGLE")
        equity_curve, trade_log, metrics = run_portfolio_backtest(
            all_stock_signals={single_symbol: oos_df.copy()},
            config=bt_config,
            market_df=single_market_df,
            expected_symbols=[single_symbol],
            investor_mode=investor_mode,
        )
        print()
        print_portfolio_report(metrics, equity_curve, bt_config, investor_mode=investor_mode)

        # 回測最後一日的系統決策 → AI 分析師解讀（2026-10-27）
        try:
            from pipeline_ai import AI_SETTINGS, maybe_run_ai_analyst
            if AI_SETTINGS.get("enabled") and not oos_df.empty:
                _last = oos_df.sort_values("date").iloc[-1]
                maybe_run_ai_analyst(_last, bt_config, None,
                                     title=f"十二、AI 分析師解讀（{single_symbol} 回測最後一日 {str(_last.get('date'))[:10]} 的系統決策）")
        except Exception as exc:
            print(f"[AI] 分析師報告產生失敗（不影響回測結果）：{exc}")

        if not trade_log.empty:
            print("\n交易明細（Portfolio execution engine）：")
            display_cols = [c for c in [
                "symbol", "entry_date", "entry_price",
                "exit_date", "exit_price", "exit_reason", "exit_trigger",
                "layer_id", "entry_layer_id", "partial_exit_flag", "exit_sequence",
                "false_breakdown_count", "holding_days_at_exit",
                "trailing_stop_activated", "return_20d_score",
                "return_pct",
            ] if c in trade_log.columns]
            recent = trade_log.tail(20)[display_cols].copy()
            if "return_pct" in recent.columns:
                recent["return_pct"] = recent["return_pct"].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
            if "entry_date" in recent.columns:
                recent["entry_date"] = pd.to_datetime(recent["entry_date"]).dt.date
            if "exit_date" in recent.columns:
                recent["exit_date"] = pd.to_datetime(recent["exit_date"]).dt.date
            print(recent.to_string(index=False))

        output_dir = Path(bt_config.output_dir)
        output_dir.mkdir(exist_ok=True)
        ts = datetime.today().strftime("%Y%m%d_%H%M%S")
        prefix = output_dir / f"{lookup_symbol_market(single_symbol, bt_config.market)}_{single_symbol}_portfolio_backtest_{bt_config.backtest_profile_name}_{ts}"
        if not trade_log.empty:
            tl_csv = Path(f"{prefix}_trade_log.csv")
            tl_xlsx = Path(f"{prefix}_trade_log.xlsx")
            trade_log.to_csv(tl_csv, index=False, encoding="utf-8-sig")
            trade_log.to_excel(tl_xlsx, index=False)
            print(f"\n交易明細已輸出：{tl_csv}")
        if not equity_curve.empty:
            ec_csv = Path(f"{prefix}_equity_curve.csv")
            ec_xlsx = Path(f"{prefix}_equity_curve.xlsx")
            equity_curve.to_csv(ec_csv, index=False, encoding="utf-8-sig")
            equity_curve.to_excel(ec_xlsx, index=False)
            print(f"資產曲線已輸出：{ec_csv}")

    else:
        if daily_top30_universe:
            _run_daily_top_n_portfolio_backtest(
                oos_df=oos_df,
                config=config,
                top_n=top_n,
                rank_metric=daily_top30_rank_metric,
                investor_mode=investor_mode,
            )
            return

        # 批次模式：逐股回測，彙整績效表
        bt_symbols = sorted(oos_df["symbol"].astype(str).unique().tolist())
        print(f"Step 3/3: 批次回測 {len(bt_symbols)} 支股票（每股獨立 OOS 回測）...")

        all_trade_logs: list[pd.DataFrame] = []
        per_symbol_rows: list[dict] = []
        raw_metrics_list: list[dict] = []   # 保留原始 float 值供排名用

        def _fp(v, digits=1) -> str:
            return f"{v:.{digits}%}" if pd.notna(v) else "N/A"

        def _ff(v, digits=1) -> str:
            return f"{v:.{digits}f}" if pd.notna(v) else "N/A"

        for i, sym in enumerate(bt_symbols, start=1):
            sym_oos = oos_df[oos_df["symbol"].astype(str) == sym].copy()
            if sym_oos.empty:
                continue
            sym_config = replace(config, stock_no=sym)
            try:
                tl, ec = run_backtest(sym_oos, sym_config, mode="single", investor_mode=investor_mode)
            except Exception as exc:
                print(f"  [{i}/{len(bt_symbols)}] {stock_label(sym)} 回測失敗：{exc}")
                continue

            if not tl.empty:
                tl["symbol"] = sym
                all_trade_logs.append(tl)

            sym_metrics = calculate_backtest_metrics(
                trade_log=tl,
                equity_curve=ec,
                initial_capital=config.backtest_initial_capital,
            )

            # 逐筆計算分組統計（純報酬率基礎，與資金無關）
            if not tl.empty:
                tl_w = tl.copy()
                tl_w["_ret"]  = pd.to_numeric(tl_w["return_pct"],  errors="coerce")
                tl_w["_hday"] = pd.to_numeric(tl_w.get("holding_days", np.nan), errors="coerce")
                tl_w_valid = tl_w[tl_w["_ret"].notna()]
                wins   = tl_w_valid[tl_w_valid["_ret"] > 0]
                losses = tl_w_valid[tl_w_valid["_ret"] <= 0]
                avg_trade_return  = float(tl_w_valid["_ret"].mean()) if not tl_w_valid.empty else np.nan
                avg_win_return    = float(wins["_ret"].mean())        if not wins.empty        else np.nan
                avg_loss_return   = float(losses["_ret"].mean())      if not losses.empty      else np.nan
                avg_hdays_win     = float(wins["_hday"].mean())       if not wins.empty        else np.nan
                avg_hdays_loss    = float(losses["_hday"].mean())     if not losses.empty      else np.nan
                best_trade_ret    = float(tl_w_valid["_ret"].max())   if not tl_w_valid.empty else np.nan
                worst_trade_ret   = float(tl_w_valid["_ret"].min())   if not tl_w_valid.empty else np.nan
                # 報酬率基礎獲利因子
                n_wins = len(wins); n_losses = len(losses)
                if n_wins > 0 and n_losses > 0 and pd.notna(avg_win_return) and pd.notna(avg_loss_return) and avg_loss_return != 0:
                    ret_profit_factor = (avg_win_return * n_wins) / (abs(avg_loss_return) * n_losses)
                else:
                    ret_profit_factor = np.nan
            else:
                avg_trade_return = avg_win_return = avg_loss_return = np.nan
                avg_hdays_win = avg_hdays_loss = np.nan
                best_trade_ret = worst_trade_ret = ret_profit_factor = np.nan

            m = sym_metrics or {}
            raw_metrics_list.append({
                "symbol":            sym,
                "total_return":      m.get("total_return", np.nan),
                "win_rate":          m.get("win_rate", np.nan),
                "total_trades":      m.get("total_trades", 0),
                "avg_holding_days":  m.get("avg_holding_days", np.nan),
                "max_drawdown":      m.get("max_drawdown", np.nan),
                "sharpe_ratio":      m.get("sharpe_ratio", np.nan),
                "avg_trade_return":  avg_trade_return,
                "avg_win_return":    avg_win_return,
                "avg_loss_return":   avg_loss_return,
                "profit_factor":     ret_profit_factor,
                "best_trade":        best_trade_ret,
                "worst_trade":       worst_trade_ret,
                "avg_hdays_win":     avg_hdays_win,
                "avg_hdays_loss":    avg_hdays_loss,
                "real_trades":       m.get("real_trades", 0),
                "real_win_rate":     m.get("real_win_rate", np.nan),
                "real_avg_return":   m.get("real_avg_return", np.nan),
                "real_profit_factor": m.get("real_profit_factor", np.nan),
            })
            per_symbol_rows.append({
                "股票代碼":          sym,
                "名稱":              get_stock_name(sym),
                "總報酬":            _fp(m.get("total_return")),
                "勝率":              _fp(m.get("win_rate")),
                "交易筆數":          m.get("total_trades", 0),
                "平均持有天數":      _ff(m.get("avg_holding_days")),
                "最大回撤":          _fp(m.get("max_drawdown")),
                "avg_trade_return":  _fp(avg_trade_return, digits=2),
                "avg_win_return":    _fp(avg_win_return,   digits=2),
                "avg_loss_return":   _fp(avg_loss_return,  digits=2),
                "profit_factor":     _ff(ret_profit_factor, digits=2),
                "best_trade":        _fp(best_trade_ret,   digits=2),
                "worst_trade":       _fp(worst_trade_ret,  digits=2),
                "Sharpe":            _ff(m.get("sharpe_ratio"), digits=2),
                "獲利持有天":        _ff(avg_hdays_win),
                "虧損持有天":        _ff(avg_hdays_loss),
                "real_trades":       m.get("real_trades", 0),
                "real_win_rate":     _fp(m.get("real_win_rate"), digits=2),
                "real_avg_return":   _fp(m.get("real_avg_return"), digits=2),
                "real_profit_factor": _ff(m.get("real_profit_factor"), digits=2),
            })
            n_trades  = len(tl)
            total_ret = m.get("total_return", 0)
            if not investor_mode or i % 10 == 0 or i == len(bt_symbols):
                print(f"  [{i}/{len(bt_symbols)}] {stock_label(sym)}：{n_trades} 筆交易，總報酬 {total_ret:.1%}")

        # 合併交易紀錄
        combined_log = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()

        # ── 逐股績效表 ──────────────────────────────────────────────────────
        perf_df = pd.DataFrame(per_symbol_rows) if per_symbol_rows else pd.DataFrame()
        if not investor_mode:
            print()
            print("=" * 90)
            print("批次回測 — 逐股績效彙整")
            print("=" * 90)
            if per_symbol_rows:
                print(perf_df.to_string(index=False))
            else:
                print("無有效回測結果。")
            print("=" * 90)

        # ── 策略適用性評估 ───────────────────────────────────────────────────
        if raw_metrics_list and not investor_mode:
            high_quality: list[str] = []
            not_suitable: list[str] = []
            for rm in raw_metrics_list:
                wr = rm.get("win_rate", float("nan"))
                pf = rm.get("profit_factor", float("nan"))
                sym_label = rm["symbol"]
                if pd.notna(wr) and pd.notna(pf):
                    if wr >= 0.60 and pf >= 3.0:
                        high_quality.append(sym_label)
                    elif wr < 0.35 and pf < 1.2:
                        not_suitable.append(sym_label)
            print()
            print("策略適用性評估：")
            hq_str = "、".join(high_quality) if high_quality else "（無）"
            ns_str = "、".join(not_suitable) if not_suitable else "（無）"
            print(f"高品質標的：{hq_str}")
            print(f"策略不適用（建議排除）：{ns_str}")

        # ── 整體合併績效報告 ─────────────────────────────────────────────────
        if not combined_log.empty and raw_metrics_list and not investor_mode:
            print()
            print("=" * 70)
            print("整體合併績效報告（所有股票合併計算）")
            print("=" * 70)

            all_rets   = pd.to_numeric(combined_log["return_pct"], errors="coerce").dropna()
            all_wins   = all_rets[all_rets > 0]
            all_losses = all_rets[all_rets <= 0]

            total_trades      = len(combined_log)
            overall_win_rate  = len(all_wins) / len(all_rets) if len(all_rets) > 0 else float("nan")
            avg_trade_ret_all = float(all_rets.mean())        if not all_rets.empty   else float("nan")
            avg_win_ret_all   = float(all_wins.mean())        if not all_wins.empty   else float("nan")
            avg_loss_ret_all  = float(all_losses.mean())      if not all_losses.empty else float("nan")

            n_wins_all = len(all_wins); n_losses_all = len(all_losses)
            if (n_wins_all > 0 and n_losses_all > 0
                    and pd.notna(avg_win_ret_all) and pd.notna(avg_loss_ret_all)
                    and avg_loss_ret_all != 0):
                overall_profit_factor = (avg_win_ret_all * n_wins_all) / (abs(avg_loss_ret_all) * n_losses_all)
            else:
                overall_profit_factor = float("nan")

            # 最佳/最差單筆交易
            best_idx  = all_rets.idxmax()  if not all_rets.empty else None
            worst_idx = all_rets.idxmin()  if not all_rets.empty else None
            def _trade_info(idx):
                if idx is None:
                    return "N/A", "N/A", float("nan")
                row = combined_log.loc[idx]
                sym  = str(row.get("symbol", "N/A"))
                date = str(pd.to_datetime(row.get("exit_date", row.get("entry_date", "N/A"))).date()
                           if pd.notna(row.get("exit_date", row.get("entry_date"))) else "N/A")
                ret  = float(row.get("return_pct", float("nan")))
                return sym, date, ret

            best_sym,  best_date,  best_ret  = _trade_info(best_idx)
            worst_sym, worst_date, worst_ret = _trade_info(worst_idx)

            # 真實交易指標（以進場為單位，合併分批出場記錄，跨股票彙整）
            overall_real_metrics = calculate_real_trade_metrics(combined_log)

            print(f"出場記錄數（含分批）：{total_trades}")
            if pd.notna(overall_win_rate):
                print(f"整體勝率：{overall_win_rate:.1%}")
            else:
                print("整體勝率：N/A")
            if pd.notna(avg_trade_ret_all):
                print(f"平均單筆報酬率：{avg_trade_ret_all:.2%}")
            else:
                print("平均單筆報酬率：N/A")
            if pd.notna(avg_win_ret_all):
                print(f"獲利交易平均報酬：{avg_win_ret_all:.2%}")
            else:
                print("獲利交易平均報酬：N/A")
            if pd.notna(avg_loss_ret_all):
                print(f"虧損交易平均報酬：{avg_loss_ret_all:.2%}")
            else:
                print("虧損交易平均報酬：N/A")
            if pd.notna(overall_profit_factor):
                print(f"整體獲利因子：{overall_profit_factor:.2f}")
            else:
                print("整體獲利因子：N/A")
            if pd.notna(best_ret):
                print(f"最佳單筆交易：{best_sym} {best_date} +{best_ret:.2%}")
            else:
                print("最佳單筆交易：N/A")
            if pd.notna(worst_ret):
                print(f"最差單筆交易：{worst_sym} {worst_date} {worst_ret:.2%}")
            else:
                print("最差單筆交易：N/A")

            # ── 真實交易指標（以進場為單位，合併分批出場記錄）──────────
            print(f"真實進場次數：{overall_real_metrics['real_trades']}")
            _r_wr = overall_real_metrics["real_win_rate"]
            print(f"真實勝率：{_r_wr:.1%}" if pd.notna(_r_wr) else "真實勝率：N/A")
            _r_ar = overall_real_metrics["real_avg_return"]
            print(f"真實平均單筆報酬：{_r_ar:.2%}" if pd.notna(_r_ar) else "真實平均單筆報酬：N/A")
            _r_pf = overall_real_metrics["real_profit_factor"]
            print(f"真實獲利因子：{_r_pf:.2f}" if pd.notna(_r_pf) else "真實獲利因子：N/A")

            # ── 交易成本分析（台股手續費 / 證交稅 / 滑價）──────────────
            buy_fee_rate  = config.backtest_buy_fee_rate
            sell_fee_rate = config.backtest_sell_fee_rate
            tax_rate      = config.backtest_tax_rate
            slippage_rate = config.backtest_slippage_rate
            fee_discount  = config.backtest_fee_discount

            gross_rets_all = pd.to_numeric(combined_log.get("gross_return_pct", pd.Series(dtype=float)), errors="coerce").dropna()
            net_rets_all   = pd.to_numeric(combined_log.get("net_return_pct",   pd.Series(dtype=float)), errors="coerce").dropna()

            avg_gross_return = float(gross_rets_all.mean()) if not gross_rets_all.empty else float("nan")
            avg_net_return   = float(net_rets_all.mean())   if not net_rets_all.empty   else float("nan")

            net_wins   = net_rets_all[net_rets_all > 0]
            net_losses = net_rets_all[net_rets_all <= 0]
            if (len(net_wins) > 0 and len(net_losses) > 0
                    and pd.notna(net_wins.mean()) and pd.notna(net_losses.mean())
                    and net_losses.mean() != 0):
                net_profit_factor = (net_wins.mean() * len(net_wins)) / (abs(net_losses.mean()) * len(net_losses))
            else:
                net_profit_factor = float("nan")

            print()
            print("交易成本分析：")
            print(f"手續費率（折扣後）：{buy_fee_rate * fee_discount * 100:.4f}%（買）/ {sell_fee_rate * fee_discount * 100:.4f}%（賣）")
            print(f"證交稅率：{tax_rate * 100:.2f}%")
            print(f"滑價假設：{slippage_rate * 100:.2f}%")
            print(f"單筆交易總成本約：{(buy_fee_rate * fee_discount + sell_fee_rate * fee_discount + tax_rate + slippage_rate) * 100:.3f}%")
            print(f"扣成本前平均單筆報酬：{avg_gross_return:.2%}" if pd.notna(avg_gross_return) else "扣成本前平均單筆報酬：N/A")
            print(f"扣成本後平均單筆報酬：{avg_net_return:.2%}" if pd.notna(avg_net_return) else "扣成本後平均單筆報酬：N/A")
            if pd.notna(avg_gross_return) and pd.notna(avg_net_return):
                print(f"成本對報酬的影響：{(avg_gross_return - avg_net_return):.2%}")
            else:
                print("成本對報酬的影響：N/A")
            print(f"扣成本後整體獲利因子：{net_profit_factor:.2f}" if pd.notna(net_profit_factor) else "扣成本後整體獲利因子：N/A")

            rm_df = pd.DataFrame(raw_metrics_list)

            def _show_rank(rows_df: pd.DataFrame, extra_col: str, extra_label: str) -> None:
                for _, r in rows_df.iterrows():
                    tr  = r.get("total_return", float("nan"))
                    wr  = r.get("win_rate",     float("nan"))
                    nt  = int(r.get("total_trades", 0))
                    ev  = r.get(extra_col, float("nan"))
                    tr_s = f"{tr:.1%}" if pd.notna(tr) else "N/A"
                    wr_s = f"{wr:.1%}" if pd.notna(wr) else "N/A"
                    ev_s = f"{ev:.2%}" if pd.notna(ev) else "N/A"
                    print(f"  {stock_label(r['symbol'])}  總報酬 {tr_s}  勝率 {wr_s}  {nt} 筆  {extra_label} {ev_s}")

            print()
            print("表現最好前 5 支（依總報酬率）：")
            top5 = rm_df[rm_df["total_return"].notna()].sort_values("total_return", ascending=False).head(5)
            _show_rank(top5, "avg_trade_return", "平均單筆")

            print()
            print("表現最差後 5 支（依總報酬率）：")
            bot5 = rm_df[rm_df["total_return"].notna()].sort_values("total_return", ascending=True).head(5)
            _show_rank(bot5, "avg_trade_return", "平均單筆")

            print()
            print("勝率最高前 5 支：")
            wr5 = rm_df[rm_df["win_rate"].notna()].sort_values("win_rate", ascending=False).head(5)
            _show_rank(wr5, "avg_trade_return", "平均單筆")

            # 等權重複利模擬績效（每筆投入當前資金 10%，複利滾動）
            ew_initial  = 1_000_000.0
            ew_capital  = ew_initial
            ew_eq_hist  = [ew_capital]
            ew_log_sorted = combined_log.copy()
            if "entry_date" in ew_log_sorted.columns:
                ew_log_sorted = ew_log_sorted.sort_values("entry_date").reset_index(drop=True)
            ew_rets_sorted = pd.to_numeric(ew_log_sorted["return_pct"], errors="coerce").fillna(0.0)
            for _ret in ew_rets_sorted:
                invested   = ew_capital * 0.10
                ew_capital = ew_capital - invested + invested * (1.0 + _ret)
                ew_eq_hist.append(ew_capital)

            ew_final        = ew_capital
            ew_total_return = (ew_final - ew_initial) / ew_initial

            # 年化報酬率（依實際回測日期區間）
            try:
                _start_d = pd.to_datetime(combined_log["entry_date"]).min()
                _end_d   = pd.to_datetime(combined_log.get("exit_date", combined_log["entry_date"])).dropna().max()
                _n_days  = max(int((_end_d - _start_d).days), 1)
            except Exception:
                _n_days = 252
            ew_annual_return = (1.0 + ew_total_return) ** (365.0 / _n_days) - 1.0

            # 最大回撤
            ew_eq_arr       = np.array(ew_eq_hist, dtype=float)
            ew_running_max  = np.maximum.accumulate(ew_eq_arr)
            ew_max_drawdown = float(((ew_eq_arr - ew_running_max) / ew_running_max).min())

            print()
            print("等權重複利模擬績效（每筆投入當前資金 10%，複利滾動）：")
            print(f"起始資金：{ew_initial:,.0f} 元")
            print(f"最終資產：{ew_final:,.0f} 元")
            print(f"等權重複利總報酬率：{ew_total_return:.2%}")
            print(f"等權重年化報酬率：{ew_annual_return:.2%}（{_n_days} 天）")
            print(f"等權重最大回撤：{ew_max_drawdown:.2%}")

            print("=" * 70)

        # ── 股票適用性過濾：只讓通過篩選的股票進入 Portfolio 回測 ─────────────
        print()
        use_symbol_filter = bool(getattr(config, "use_symbol_eligibility_filter", True))
        eligibility_cutoff_date = None
        if use_symbol_filter:
            try:
                passed_symbols, excluded_symbols, exclusion_reasons = filter_suitable_symbols(oos_df, config, investor_mode=investor_mode)
            except Exception as exc:
                print(f"股票適用性過濾失敗，改用全部 Top N 股票：{exc}")
                passed_symbols, excluded_symbols, exclusion_reasons = list(bt_symbols), [], {}
            _ = exclusion_reasons  # 已於 filter_suitable_symbols() 內印出
            portfolio_symbols = [s for s in bt_symbols if s in set(passed_symbols)] or list(bt_symbols)
            if bool(getattr(config, "exclude_eligibility_window_from_official_backtest", True)):
                eligibility_cutoff_date = _symbol_eligibility_cutoff_date(oos_df, config)
        else:
            print("股票適用性過濾：已停用，Portfolio 使用全部 Top N 股票。")
            passed_symbols, excluded_symbols, exclusion_reasons = list(bt_symbols), [], {}
            portfolio_symbols = list(bt_symbols)

        # ── Portfolio 層級回測（真實資金限制，跨股票共用資金池）───────────────
        # 注意：被適用性過濾排除的股票仍保留在上方逐股回測診斷／績效表中，
        # 只是不計入下方 Portfolio 層級績效。
        all_stock_signals = {}
        for sym in portfolio_symbols:
            sym_signals = oos_df[oos_df["symbol"].astype(str) == sym].copy()
            if eligibility_cutoff_date is not None and "date" in sym_signals.columns:
                sym_dates = pd.to_datetime(sym_signals["date"], errors="coerce")
                sym_signals = sym_signals[sym_dates > eligibility_cutoff_date].copy()
            all_stock_signals[sym] = sym_signals
        if eligibility_cutoff_date is not None:
            print(
                "正式 Portfolio 回測已排除股票適用性篩選期，"
                f"績效自 {pd.Timestamp(eligibility_cutoff_date).date()} 之後開始計算。"
            )

        print()
        print("取得 TAIEX 指數歷史資料（Benchmark 比較／大盤過濾用）...")
        try:
            taiex_index_df = get_market_index_data(config)
        except Exception as exc:
            print(f"WARNING: TAIEX 指數資料預先取得失敗：{exc}")
            taiex_index_df = pd.DataFrame()

        try:
            pf_equity_curve, pf_trade_log, pf_metrics = run_portfolio_backtest(
                all_stock_signals=all_stock_signals,
                config=config,
                market_df=taiex_index_df,
                expected_symbols=bt_symbols,
                investor_mode=investor_mode,
            )
        except Exception as exc:
            print(f"Portfolio 層級回測失敗：{exc}")
            pf_equity_curve, pf_trade_log, pf_metrics = pd.DataFrame(), pd.DataFrame(), {}

        # ── Benchmark 比較（TAIEX 買進持有 / Top N 等權買進持有 / 2330 買進持有）──
        benchmark_metrics: dict = {}
        if not pf_equity_curve.empty:
            try:
                benchmark_df, benchmark_metrics = build_benchmark_equity_curves(
                    config=config,
                    all_stock_signals=all_stock_signals,
                    target_dates=pf_equity_curve["date"],
                    initial_capital=config.backtest_initial_capital,
                    tsmc_symbol="2330",
                )
            except Exception as exc:
                print(f"Benchmark 計算失敗：{exc}")
                benchmark_df, benchmark_metrics = pd.DataFrame(), {}

            if not benchmark_df.empty:
                pf_equity_curve = pf_equity_curve.merge(benchmark_df, on="date", how="left")

        print()
        print_portfolio_report(pf_metrics, pf_equity_curve, config, investor_mode=investor_mode)

        if benchmark_metrics:
            print()
            print_strategy_vs_benchmark_report(pf_metrics, benchmark_metrics)

        # 回測最後一日的候選股決策總表 → AI 分析師解讀（2026-10-27）
        try:
            if "date" in oos_df.columns and "final_decision_score" in oos_df.columns:
                _last_date = pd.to_datetime(oos_df["date"]).max()
                _last_rows = oos_df[pd.to_datetime(oos_df["date"]) == _last_date].copy()
                print_full_decision_table(
                    _last_rows, f"十一、回測最後一日（{_last_date.date()}）各股系統決策（依最終分數排序）")
                from pipeline_ai import run_ai_for_rows
                run_ai_for_rows(_last_rows, config, title_prefix="十二、AI 分析師解讀（回測最後一日候選股）", max_n=3)
        except Exception as exc:
            print(f"[AI] 最後一日決策解讀產生失敗（不影響回測結果）：{exc}")

        if run_profile_experiments:
            try:
                print()
                run_portfolio_profile_experiments(
                    all_stock_signals=all_stock_signals,
                    base_config=config,
                    market_df=taiex_index_df,
                    expected_symbols=bt_symbols,
                )
            except Exception as exc:
                print(f"Portfolio profile experiments 失敗：{exc}")

        # ── 輸出檔案 ─────────────────────────────────────────────────────────
        if not combined_log.empty:
            from pathlib import Path
            prefix = f"backtest_batch_{config.market}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            tl_csv    = Path(f"{prefix}_trades.csv")
            tl_xlsx   = Path(f"{prefix}_trades.xlsx")
            perf_csv  = Path(f"{prefix}_perf.csv")
            perf_xlsx = Path(f"{prefix}_perf.xlsx")

            combined_log.to_csv(tl_csv,   index=False, encoding="utf-8-sig")
            combined_log.to_excel(tl_xlsx, index=False)
            print(f"合併交易紀錄已輸出：{tl_csv}")

            if per_symbol_rows:
                perf_df.to_csv(perf_csv,   index=False, encoding="utf-8-sig")
                perf_df.to_excel(perf_xlsx, index=False)
                print(f"逐股績效表已輸出：{perf_csv}")




__all__ = [name for name in globals() if not name.startswith('__')]
