from __future__ import annotations

from pipeline_model_core import *

def _add_return_20d_score_daily_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional daily rank for return_20d_score; higher is stronger."""
    out = df.copy()
    if "return_20d_score" not in out.columns or "date" not in out.columns:
        out["return_20d_score_daily_rank"] = np.nan
        return out

    scores = pd.to_numeric(out["return_20d_score"], errors="coerce")
    out["return_20d_score_daily_rank"] = scores.groupby(out["date"]).rank(pct=True)
    return out


def _future_return_lookup(df: pd.DataFrame, horizons: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
    if df is None or df.empty or not {"symbol", "date", "close"}.issubset(df.columns):
        columns = ["symbol", "date", "close"]
        for horizon in horizons:
            columns.extend([f"future_close_{horizon}d", f"future_return_{horizon}d_from_close"])
        return pd.DataFrame(columns=columns)

    px = df[["symbol", "date", "close"]].copy()
    px["symbol"] = px["symbol"].astype(str)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["date"]).sort_values(["symbol", "date"])
    for horizon in horizons:
        future_close = px.groupby("symbol", sort=False)["close"].shift(-horizon)
        px[f"future_close_{horizon}d"] = future_close
        px[f"future_return_{horizon}d_from_close"] = future_close / px["close"] - 1.0
    return px


def _build_rejection_future_return_summary(
    rows: list[dict],
    df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "rejection_reason",
        "candidate_count",
        "future_return_10d",
        "future_return_20d",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    rejected = pd.DataFrame(rows)
    rejected["date"] = pd.to_datetime(rejected["date"], errors="coerce")
    rejected["symbol"] = rejected["symbol"].astype(str)
    lookup = _future_return_lookup(df)
    merged = rejected.merge(
        lookup[[
            "symbol",
            "date",
            "future_return_10d_from_close",
            "future_return_20d_from_close",
        ]],
        on=["symbol", "date"],
        how="left",
    )
    summary = (
        merged.groupby("rejection_reason", dropna=False)
        .agg(
            candidate_count=("symbol", "size"),
            future_return_10d=("future_return_10d_from_close", "mean"),
            future_return_20d=("future_return_20d_from_close", "mean"),
        )
        .reset_index()
        .sort_values("candidate_count", ascending=False)
    )
    return summary[columns]


def _build_trailing_stop_future_return_diagnostics(
    trade_log: pd.DataFrame,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_columns = [
        "symbol",
        "exit_date",
        "exit_price",
        "future_return_10d",
        "future_return_20d",
        "peak_profit_pct",
        "entry_return_20d_score",
        "entry_return_20d_score_daily_rank",
    ]
    summary_columns = [
        "group",
        "exit_count",
        "future_return_10d",
        "future_return_20d",
    ]
    if trade_log is None or trade_log.empty or "symbol" not in trade_log.columns:
        return pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=summary_columns)

    tl = trade_log.copy()
    trigger = tl.get("exit_trigger", pd.Series("", index=tl.index)).astype(str)
    reason = tl.get("exit_reason", pd.Series("", index=tl.index)).astype(str)
    trailing = tl[(trigger == "trailing") | reason.str.contains("Trailing", case=False, na=False)].copy()
    if trailing.empty:
        return pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=summary_columns)

    trailing["symbol"] = trailing["symbol"].astype(str)
    trailing["exit_date"] = pd.to_datetime(trailing["exit_date"], errors="coerce")
    trailing["exit_price"] = pd.to_numeric(trailing["exit_price"], errors="coerce")
    lookup = _future_return_lookup(df)
    merged = trailing.merge(
        lookup[[
            "symbol",
            "date",
            "close",
            "future_close_10d",
            "future_close_20d",
        ]].rename(columns={"date": "exit_date", "close": "close_at_exit"}),
        on=["symbol", "exit_date"],
        how="left",
    )
    merged["future_return_10d"] = (
        pd.to_numeric(merged["future_close_10d"], errors="coerce") / merged["exit_price"] - 1.0
    )
    merged["future_return_20d"] = (
        pd.to_numeric(merged["future_close_20d"], errors="coerce") / merged["exit_price"] - 1.0
    )
    for col in ["peak_profit_pct", "entry_return_20d_score", "entry_return_20d_score_daily_rank"]:
        if col not in merged.columns:
            merged[col] = np.nan
    detail = merged[detail_columns].copy()

    strong_profit = pd.to_numeric(detail["peak_profit_pct"], errors="coerce") >= 0.20
    summary_rows = []
    for group_name, mask in [
        ("all_trailing_stop_exits", pd.Series(True, index=detail.index)),
        ("peak_profit_over_20pct", strong_profit),
    ]:
        subset = detail[mask]
        summary_rows.append({
            "group": group_name,
            "exit_count": int(len(subset)),
            "future_return_10d": float(pd.to_numeric(subset["future_return_10d"], errors="coerce").mean()) if len(subset) else np.nan,
            "future_return_20d": float(pd.to_numeric(subset["future_return_20d"], errors="coerce").mean()) if len(subset) else np.nan,
        })
    return detail, pd.DataFrame(summary_rows, columns=summary_columns)


# ============================================================
# 回測模組
# ============================================================

def generate_backtest_history(
    combined_data: pd.DataFrame,
    shared_config: PipelineConfig,
    target_symbol: Optional[str] = None,
    n_splits: int = 8,
) -> pd.DataFrame:
    """
    Walk-forward OOS predictions on combined_data. Returns DataFrame with
    OHLC + all feature columns + predicted_score / safety_score / return_score +
    full apply_enhanced_decision_layers() output.
    """
    import contextlib, io

    safety_target_col = f"target_safety_{shared_config.horizon}d_cs_rank"
    return_target_col  = f"target_return_{shared_config.horizon}d_cs_rank"

    # ── 準備 model_df（保留 OHLC 和所有 feature 欄位）──────────────
    work = clean_duplicate_columns(combined_data.copy())
    work = work.sort_values(
        ["date", "symbol"] if "symbol" in work.columns else ["date"]
    ).reset_index(drop=True)

    with contextlib.redirect_stdout(io.StringIO()):
        numeric_features, categorical_features = get_feature_columns(
            work, feature_set=shared_config.feature_set
        )

    for col in numeric_features:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in categorical_features:
        if col in work.columns:
            work[col] = work[col].astype("object").where(work[col].notna(), "none").astype(str)

    feature_cols = list(dict.fromkeys(numeric_features + categorical_features))

    # 保留 OHLC 供後續合併
    ohlc_extra = ["open", "high", "low"]
    keep_cols = list(dict.fromkeys(
        ["date", "symbol"]
        + [c for c in ohlc_extra if c in work.columns]
        + [c for c in [safety_target_col, return_target_col] if c in work.columns]
        + feature_cols
    ))
    # 也保留所有特徵欄位供 decision layer 使用
    for col in work.columns:
        if col not in keep_cols:
            keep_cols.append(col)

    model_df = work[keep_cols].copy()
    if safety_target_col not in model_df.columns or return_target_col not in model_df.columns:
        raise ValueError(f"回測需要 {safety_target_col} 和 {return_target_col}")

    valid_mask = model_df[feature_cols + [safety_target_col]].notna().all(axis=1)
    model_df = model_df[valid_mask].reset_index(drop=True)

    X          = model_df[feature_cols]
    y_safety   = pd.to_numeric(model_df[safety_target_col], errors="coerce")
    y_return   = pd.to_numeric(model_df[return_target_col], errors="coerce")

    splits = build_walk_forward_splits_by_date(model_df=model_df, n_splits=n_splits)

    oos_frames: list[pd.DataFrame] = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ys_train = y_safety.iloc[train_idx]
        yr_train = y_return.iloc[train_idx]

        train_dates   = pd.to_datetime(model_df.iloc[train_idx]["date"], errors="coerce")
        latest_td     = train_dates.max()
        days_back     = (latest_td - train_dates).dt.days.clip(lower=0)
        sample_weight = np.exp(-days_back / 365).to_numpy(dtype=float)

        safety_pipe = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=shared_config.random_state + fold_no,
            model_profile=shared_config.model_profile,
            model_type=shared_config.model_type,
        )
        safety_pipe.fit(X_train, ys_train, model__sample_weight=sample_weight)

        return_pipe = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=shared_config.random_state + fold_no + 1000,
            model_profile=shared_config.model_profile,
            model_type=shared_config.model_type,
        )
        return_pipe.fit(X_train, yr_train, model__sample_weight=sample_weight)

        fold_df = model_df.iloc[test_idx].copy()
        fold_df["predicted_score"] = np.clip(safety_pipe.predict(X_test), 0, 1)
        fold_df["safety_score"]    = fold_df["predicted_score"]
        fold_df["return_score"]    = np.clip(return_pipe.predict(X_test), 0, 1)

        if target_symbol:
            fold_df = fold_df[fold_df["symbol"].astype(str) == str(target_symbol)]

        oos_frames.append(fold_df)

    if not oos_frames:
        return pd.DataFrame()

    oos_df = (
        pd.concat(oos_frames, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    oos_df = _add_return_20d_score_daily_rank(oos_df)
    oos_df["combined_decision_score"] = (
        0.65 * oos_df["safety_score"] + 0.35 * oos_df["return_score"]
    )

    # ── 套用決策層（包含 entry_plan / override / final_decision）──────
    with contextlib.redirect_stdout(io.StringIO()):
        oos_df = apply_enhanced_decision_layers(oos_df)

    return oos_df


RETURN_20D_PURGE_DATES = 20


def generate_oos_predictions_for_backtest(
    df: pd.DataFrame,
    config: PipelineConfig,
    n_splits: int = 12,
    min_train_dates: Optional[int] = None,
    min_train_date_cutoff: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rolling walk-forward OOS predictions for backtesting.
    Each test-fold row receives only out-of-sample predictions.
    Rule filter, entry plan, and override are computed per-fold on
    the test set only — no leakage from training data.
    Returns a DataFrame containing OHLC + all decision columns.
    """
    import contextlib, io

    safety_target_col = f"target_safety_{config.horizon}d_cs_rank"
    return_target_col  = f"target_return_{config.horizon}d_cs_rank"

    work = clean_duplicate_columns(df.copy())
    sort_keys = ["date", "symbol"] if "symbol" in work.columns else ["date"]
    work = work.sort_values(sort_keys).reset_index(drop=True)

    with contextlib.redirect_stdout(io.StringIO()):
        numeric_features, categorical_features = get_feature_columns(
            work, feature_set=config.feature_set
        )

    for col in numeric_features:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in categorical_features:
        if col in work.columns:
            work[col] = work[col].astype("object").where(work[col].notna(), "none").astype(str)

    feature_cols = list(dict.fromkeys(numeric_features + categorical_features))

    if safety_target_col not in work.columns or return_target_col not in work.columns:
        raise ValueError(
            f"OOS 回測需要 {safety_target_col} 和 {return_target_col}，"
            "請確認資料已包含對應 target 欄位。"
        )

    # ── 20 日報酬輔助分數（return_20d_score）：額外訓練的中期方向模型 ──────
    use_return_20d = bool(getattr(config, "use_return_20d", False))
    return20d_target_col = "target_return_20d_cs_rank"
    numeric_features_20d: list[str] = []
    feature_cols_20d: list[str] = []
    if use_return_20d:
        if "symbol" in work.columns:
            work["future_return_20d"] = (
                work.groupby("symbol")["close"].shift(-20) / work["close"] - 1
            )
        else:
            work["future_return_20d"] = work["close"].shift(-20) / work["close"] - 1
        work[return20d_target_col] = work.groupby("date")["future_return_20d"].rank(pct=True)

        midterm_features = [
            c for c in ["ma60_distance", "volatility_20d", "return_20d", "market_return_20d"]
            if c in work.columns
        ]
        for col in midterm_features:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        numeric_features_20d = list(dict.fromkeys(numeric_features + midterm_features))
        feature_cols_20d      = list(dict.fromkeys(numeric_features_20d + categorical_features))

    valid_mask = work[feature_cols + [safety_target_col]].notna().all(axis=1)
    model_df = work[valid_mask].reset_index(drop=True)

    X        = model_df[feature_cols]
    y_safety = pd.to_numeric(model_df[safety_target_col], errors="coerce")
    y_return = pd.to_numeric(model_df[return_target_col],  errors="coerce")

    if use_return_20d and feature_cols_20d:
        valid_mask_20d = model_df[feature_cols_20d + [return20d_target_col]].notna().all(axis=1)
        y_return20d    = pd.to_numeric(model_df[return20d_target_col], errors="coerce")
    else:
        valid_mask_20d = pd.Series(False, index=model_df.index)
        y_return20d    = pd.Series(np.nan, index=model_df.index)

    unique_dates_bt = pd.Series(pd.to_datetime(model_df["date"]).dropna().unique()).sort_values()
    model_df_valid_start = unique_dates_bt.min().date() if not unique_dates_bt.empty else "N/A"
    print(f"[回測資料生成] model_df 有效起始日期（valid_mask後）：{model_df_valid_start}，"
          f"總有效交易日數：{len(unique_dates_bt)}")
    if min_train_date_cutoff is not None:
        # Count valid model_df dates (post valid_mask) before the cutoff — exact, no indicator-warmup guessing
        cutoff_ts = pd.Timestamp(min_train_date_cutoff)
        valid_before_cutoff = int((unique_dates_bt < cutoff_ts).sum())
        min_train_size = max(valid_before_cutoff, 20)
        print(f"[回測資料生成] min_train_size 自 model_df 計算：cutoff={min_train_date_cutoff}，"
              f"有效交易日數={valid_before_cutoff}（= 動態 min_train_dates），min_train_size={min_train_size}")
    elif min_train_dates is not None:
        min_train_size = min_train_dates
    else:
        min_train_size = max(126, len(unique_dates_bt) // 4)
    splits = build_walk_forward_splits_by_date(model_df=model_df, n_splits=n_splits, min_train_dates=min_train_size)

    oos_frames: list[pd.DataFrame] = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        test_dates  = pd.to_datetime(model_df.iloc[test_idx]["date"], errors="coerce")
        test_start  = test_dates.min().date() if not test_dates.empty else "N/A"
        test_end    = test_dates.max().date() if not test_dates.empty else "N/A"

        train_dates   = pd.to_datetime(model_df.iloc[train_idx]["date"], errors="coerce")
        latest_td     = train_dates.max()
        days_back     = (latest_td - train_dates).dt.days.clip(lower=0)
        sample_weight = np.exp(-days_back / 365).to_numpy(dtype=float)

        safety_pipe = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=config.random_state + fold_no,
            model_profile=config.model_profile,
            model_type=config.model_type,
        )
        safety_pipe.fit(X_train, y_safety.iloc[train_idx], model__sample_weight=sample_weight)

        return_pipe = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=config.random_state + fold_no + 1000,
            model_profile=config.model_profile,
            model_type=config.model_type,
        )
        return_pipe.fit(X_train, y_return.iloc[train_idx], model__sample_weight=sample_weight)

        fold_df = model_df.iloc[test_idx].copy()
        fold_df["predicted_score"]        = np.clip(safety_pipe.predict(X_test), 0, 1)
        fold_df["safety_score"]           = fold_df["predicted_score"]
        fold_df["return_score"]           = np.clip(return_pipe.predict(X_test), 0, 1)
        fold_df["combined_decision_score"] = (
            0.65 * fold_df["safety_score"] + 0.35 * fold_df["return_score"]
        )
        fold_df["oos_fold"] = fold_no  # 供 filter_suitable_symbols() 取「前兩折（約前六個月）」驗證資料

        # 額外訓練 20 日 return 輔助模型（return_20d_score），供 Portfolio 加碼/停利/trailing 邏輯使用
        if use_return_20d and feature_cols_20d:
            train_mask_20d = valid_mask_20d.iloc[train_idx]
            # 20 日標籤要往後看 20 個交易日：清洗期擴大為 20（舊做法 purge=0 時不額外清洗）
            if get_walk_forward_purge() > 0:
                test_start_ts = pd.Timestamp(test_dates.min())
                pos_20 = int(unique_dates_bt.searchsorted(test_start_ts)) - 1 - RETURN_20D_PURGE_DATES
                if pos_20 >= 0:
                    cutoff_20 = unique_dates_bt.iloc[pos_20]
                    train_mask_20d = train_mask_20d & (train_dates <= cutoff_20).to_numpy()
                else:
                    train_mask_20d = train_mask_20d & False
            if int(train_mask_20d.sum()) >= 30:
                X20_train = model_df.iloc[train_idx][feature_cols_20d][train_mask_20d.values]
                y20_train = y_return20d.iloc[train_idx][train_mask_20d.values]
                w20_train = sample_weight[train_mask_20d.values]
                return20d_pipe = build_model_pipeline(
                    numeric_features=numeric_features_20d,
                    categorical_features=categorical_features,
                    random_state=config.random_state + fold_no + 2000,
                    model_profile=config.model_profile,
                    model_type=config.model_type,
                )
                return20d_pipe.fit(X20_train, y20_train, model__sample_weight=w20_train)
                X20_test = model_df.iloc[test_idx][feature_cols_20d]
                fold_df["return_20d_score"] = np.clip(return20d_pipe.predict(X20_test), 0, 1)
            else:
                fold_df["return_20d_score"] = np.nan
        else:
            fold_df["return_20d_score"] = np.nan

        # Rule filter + entry plan + override + final decision — all OOS
        with contextlib.redirect_stdout(io.StringIO()):
            fold_df = apply_enhanced_decision_layers(fold_df)

        if fold_no == 1:
            train_end_date = latest_td.date() if pd.notna(latest_td) else "N/A"
            train_start_date = train_dates.min().date() if not train_dates.empty else "N/A"
            print(f"[回測資料生成] Fold 1 train：{train_start_date} ~ {train_end_date}  "
                  f"→ test_fold_1 起始：{test_start}（清洗期 {get_walk_forward_purge()} 個交易日）")
        print(
            f"[回測資料生成] Fold {fold_no}/{n_splits}："
            f"{test_start} ~ {test_end}，{len(test_idx)} 筆"
        )
        oos_frames.append(fold_df)

    if not oos_frames:
        return pd.DataFrame()

    oos_df = (
        pd.concat(oos_frames, ignore_index=True)
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )
    oos_df = _add_return_20d_score_daily_rank(oos_df)
    return oos_df


def run_backtest(
    df: pd.DataFrame,
    config: PipelineConfig,
    mode: str = "single",
    investor_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper around run_portfolio_backtest().

    Older call sites still call run_backtest(), but execution is delegated to the
    Portfolio engine so single-symbol, per-symbol, and portfolio paths share the
    same balanced_v2_combined stop-loss, partial-take-profit, and trailing logic.
    Returns (trade_log_df, equity_curve_df) to preserve the legacy API shape.
    """
    # Keep the legacy public function name for callers, but execute the current
    # Portfolio engine so single-stock and portfolio backtests share stop/TP logic.
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    exec_df = df.copy()
    if "symbol" not in exec_df.columns:
        exec_df["symbol"] = str(getattr(config, "stock_no", "SINGLE"))

    all_stock_signals = {
        str(sym): sym_df.copy()
        for sym, sym_df in exec_df.groupby(exec_df["symbol"].astype(str), sort=False)
    }
    equity_curve, trade_log_df, _metrics = run_portfolio_backtest(
        all_stock_signals=all_stock_signals,
        config=config,
        market_df=None,
        expected_symbols=list(all_stock_signals.keys()),
        investor_mode=investor_mode,
        save_snapshot=False,
    )
    return trade_log_df, equity_curve

    initial_capital = config.backtest_initial_capital
    max_positions   = config.backtest_max_positions
    atr_mult        = config.backtest_atr_trailing_multiplier
    partial_ratio   = config.backtest_partial_exit_ratio
    max_hold_days   = 30

    # ── 台股交易成本假設 ────────────────────────────────────────────
    buy_fee_rate   = config.backtest_buy_fee_rate
    sell_fee_rate  = config.backtest_sell_fee_rate
    tax_rate       = config.backtest_tax_rate
    slippage_rate  = config.backtest_slippage_rate
    fee_discount   = config.backtest_fee_discount

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)

    # ── 回測前診斷 ──────────────────────────────────────────────────
    print("回測資料檢查：")
    if "close" in df.columns:
        print(f"價格範圍：{df['close'].min():.2f} ~ {df['close'].max():.2f}")
    print(f"日期範圍：{df['date'].min().date()} ~ {df['date'].max().date()}")

    override_sum = "N/A"
    if "entry_plan_override" in df.columns:
        try:
            override_sum = int(pd.to_numeric(df["entry_plan_override"], errors="coerce").fillna(0).sum())
        except Exception:
            override_sum = "N/A"
    print(f"進場訊號天數（entry_plan_override=True）：{override_sum}")

    in_zone_count = 0
    if {"support_entry_zone_low", "support_entry_zone_high", "close"}.issubset(df.columns):
        zl = pd.to_numeric(df["support_entry_zone_low"],  errors="coerce")
        zh = pd.to_numeric(df["support_entry_zone_high"], errors="coerce")
        cl = pd.to_numeric(df["close"], errors="coerce")
        in_zone_count = int(((cl >= zl) & (cl <= zh)).sum())
    print(f"支撐區間進場天數：符合 support_entry_zone_low <= close <= support_entry_zone_high 的天數：{in_zone_count}")

    trigger_count = 0
    req_cols = {"final_decision_score", "support_entry_zone_low", "support_entry_zone_high",
                "close", "entry_stop_loss_price", "entry_stop_loss_pct", "entry_rr_1"}
    if req_cols.issubset(df.columns):
        scores    = pd.to_numeric(df["final_decision_score"], errors="coerce").fillna(0.0)
        overrides = pd.to_numeric(df.get("entry_plan_override", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).astype(bool)
        zl2  = pd.to_numeric(df["support_entry_zone_low"],  errors="coerce")
        zh2  = pd.to_numeric(df["support_entry_zone_high"], errors="coerce")
        cl2  = pd.to_numeric(df["close"], errors="coerce")
        slp  = pd.to_numeric(df["entry_stop_loss_price"], errors="coerce")
        spct = pd.to_numeric(df["entry_stop_loss_pct"],   errors="coerce")
        rr1  = pd.to_numeric(df["entry_rr_1"],            errors="coerce")
        trig  = overrides | (scores >= 0.55)
        in_z  = (cl2 >= zl2) & (cl2 <= zh2) & cl2.notna() & zl2.notna() & zh2.notna()
        sl_ok = slp.notna() & (slp > 0)
        sp_ok = spct.notna() & (spct >= 0.015) & (spct <= 0.08)   # dynamic floor ≈ 0.015
        rr_ok = rr1.notna() & (rr1 >= 1.5)
        trigger_count = int((trig & in_z & sl_ok & sp_ok & rr_ok).sum())
    print(f"實際觸發進場天數：{trigger_count}")

    dates = sorted(df["date"].unique())

    cash: float = initial_capital
    positions: list[dict] = []
    trade_log: list[dict] = []
    equity_rows: list[dict] = []
    entry_counter = 0  # 以進場為單位的流水號（第1筆進場=1，第2筆=2，…）

    def _f(row, col, default=np.nan):
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    def _log(pos: dict, partial_exit: bool = False, **kw) -> dict:
        """Assemble a trade-log row from position base fields + keyword overrides.

        同一次進場（同一個 pos）的所有出場記錄共用 trade_id / entry_id；
        exit_sequence 依序遞增（第一批出場=1，第二批=2，…）。
        partial_exit=True 代表本次出場為分批出場（部位尚未全部出清）。

        同時依台股交易成本假設（買進手續費、賣出手續費、證交稅、滑價）
        計算 actual_entry_price / actual_exit_price 與成本後報酬率欄位。
        """
        pos["exit_seq"] = pos.get("exit_seq", 0) + 1

        entry_price        = pos["entry_price"]
        actual_entry_price = pos["actual_entry_price"]
        exit_price          = kw.get("exit_price", np.nan)

        slippage_sell      = exit_price * slippage_rate / 2 if pd.notna(exit_price) else np.nan
        sell_cost_per_share = exit_price * sell_fee_rate * fee_discount if pd.notna(exit_price) else np.nan
        tax_per_share       = exit_price * tax_rate if pd.notna(exit_price) else np.nan
        actual_exit_price   = (
            exit_price - slippage_sell - sell_cost_per_share - tax_per_share
            if pd.notna(exit_price) else np.nan
        )

        gross_return_pct = (
            (exit_price - entry_price) / entry_price
            if pd.notna(exit_price) and entry_price else np.nan
        )
        net_return_pct = (
            (actual_exit_price - actual_entry_price) / actual_entry_price
            if pd.notna(actual_exit_price) and actual_entry_price else np.nan
        )
        total_cost_pct = (
            ((actual_entry_price - entry_price) + (exit_price - actual_exit_price)) / entry_price
            if pd.notna(exit_price) and entry_price else np.nan
        )
        exit_reason_text = str(kw.get("exit_reason", ""))
        if "止盈" in exit_reason_text:
            default_exit_trigger = "take_profit"
        elif "期限" in exit_reason_text:
            default_exit_trigger = "time_exit"
        elif "均線" in exit_reason_text:
            default_exit_trigger = "signal_exit"
        elif "Trailing" in exit_reason_text or "保本" in exit_reason_text:
            default_exit_trigger = "trailing"
        elif "停損" in exit_reason_text:
            default_exit_trigger = "close_confirm"
        else:
            default_exit_trigger = np.nan

        return {
            "trade_id":                 pos["trade_id"],
            "entry_id":                 pos["entry_id"],
            "exit_sequence":            pos["exit_seq"],
            "partial_exit_flag":        bool(partial_exit),
            "is_final_exit":            not bool(partial_exit),
            "symbol":                   pos["symbol"],
            "entry_date":               pos["entry_date"],
            "entry_price":              entry_price,
            "actual_entry_price":       actual_entry_price,
            "buy_cost_per_share":       pos["buy_cost_per_share"],
            "actual_stop_loss_price":   pos["actual_stop_loss_price"],
            "actual_stop_loss_pct":     pos["actual_stop_loss_pct"],
            "stop_adjusted":            pos["stop_adjusted"],
            "support_entry_zone_low":   pos.get("support_entry_zone_low"),
            "support_entry_zone_high":  pos.get("support_entry_zone_high"),
            "entry_decision_score":     pos.get("entry_decision_score"),
            "actual_exit_price":        actual_exit_price,
            "sell_cost_per_share":      sell_cost_per_share,
            "tax_per_share":            tax_per_share,
            "gross_return_pct":         gross_return_pct,
            "net_return_pct":           net_return_pct,
            "total_cost_pct":           total_cost_pct,
            "return_pct":               net_return_pct,
            "layer_id":                 pos.get("entry_layer_id", pos.get("layer_id", 1)),
            "entry_layer_id":           pos.get("entry_layer_id", pos.get("layer_id", 1)),
            "exit_trigger":             kw.get("exit_trigger", default_exit_trigger),
            "false_breakdown_count":    pos.get("false_breakdown_count", 0),
            "holding_days_at_exit":     pos.get("holding_days", kw.get("holding_days", np.nan)),
            "trailing_stop_activated":  pos.get("trailing_stop_activated", False),
            "return_20d_score":         pos.get("entry_return_20d_score", np.nan),
            "entry_return_20d_score":   pos.get("entry_return_20d_score", np.nan),
            **kw,
        }

    for current_date in dates:
        day = df[df["date"] == current_date].set_index("symbol")

        # ── 持倉處理 ────────────────────────────────────────────────
        remaining: list[dict] = []
        for pos in positions:
            sym = pos["symbol"]
            if sym not in day.index:
                pos["holding_days"] += 1
                remaining.append(pos)
                continue

            row         = day.loc[sym]
            today_low   = _f(row, "low",    pos["entry_price"])
            today_high  = _f(row, "high",   pos["entry_price"])
            today_close = _f(row, "close",  pos["entry_price"])
            today_atr   = _f(row, "atr_14", pos["atr_at_entry"])
            today_ma20  = _f(row, "ma20")

            pos["holding_days"] += 1
            exited = False

            if not pos["phase1_done"]:
                # ── 階段一：第一止盈前 ────────────────────────────────
                # P1 停損
                if today_low <= pos["stop_loss"]:
                    ep = pos["stop_loss"]
                    cash += pos["shares"] * ep
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=ep, exit_reason="停損",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (ep - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * ep,
                        stage=1, trailing_stop_price=np.nan,
                    ))
                    exited = True

                # P2 跌破均線（僅階段一）
                elif (pd.notna(today_ma20)
                      and today_close < today_ma20
                      and pos["holding_days"] > 5):
                    cash += pos["shares"] * today_close
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=today_close, exit_reason="跌破均線",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (today_close - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * today_close,
                        stage=1, trailing_stop_price=np.nan,
                    ))
                    exited = True

                # P3 第一止盈（分批，不完全出場）
                elif pd.notna(pos["take_profit_1"]) and today_high >= pos["take_profit_1"]:
                    tp1         = pos["take_profit_1"]
                    exit_shares = pos["shares"] * partial_ratio
                    proceeds    = exit_shares * tp1
                    cash       += proceeds
                    trade_log.append(_log(pos, partial_exit=True,
                        exit_date=current_date, exit_price=tp1, exit_reason="第一止盈（分批）",
                        shares=exit_shares,
                        pnl=proceeds - exit_shares * pos["entry_price"],
                        holding_days=pos["holding_days"],
                        position_value=proceeds,
                        stage=2, trailing_stop_price=np.nan,
                    ))
                    pos["shares"]     -= exit_shares
                    pos["cost"]        = pos["shares"] * pos["entry_price"]
                    pos["phase1_done"] = True
                    pos["max_close"]   = today_close
                    # 停損移至進場價（保本底線）
                    pos["stop_loss"]   = pos["entry_price"]

                # 時限（階段一到期）
                if not exited and not pos["phase1_done"] and pos["holding_days"] >= max_hold_days:
                    cash += pos["shares"] * today_close
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=today_close, exit_reason="持有期限到期",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (today_close - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * today_close,
                        stage=1, trailing_stop_price=np.nan,
                    ))
                    exited = True

            else:
                # ── 階段三：第一止盈後（剩餘部位）────────────────────
                # 更新 trailing stop（只升不降，底線為進場價）
                pos["max_close"] = max(pos["max_close"], today_close)
                if pd.notna(today_atr) and today_atr > 0:
                    trailing = pos["max_close"] - atr_mult * today_atr
                    new_stop = max(trailing, pos["entry_price"])
                else:
                    new_stop = pos["entry_price"]
                pos["stop_loss"] = max(pos["stop_loss"], new_stop)
                cur_stop = pos["stop_loss"]

                # P1 trailing stop 或保本停損
                if today_low <= cur_stop:
                    exit_reason = "Trailing Stop" if cur_stop > pos["entry_price"] else "保本停損"
                    cash += pos["shares"] * cur_stop
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=cur_stop, exit_reason=exit_reason,
                        shares=pos["shares"],
                        pnl=pos["shares"] * (cur_stop - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * cur_stop,
                        stage=3, trailing_stop_price=cur_stop,
                    ))
                    exited = True

                # P2 第二止盈
                elif pd.notna(pos["take_profit_2"]) and today_high >= pos["take_profit_2"]:
                    tp2 = pos["take_profit_2"]
                    cash += pos["shares"] * tp2
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=tp2, exit_reason="第二止盈（全出）",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (tp2 - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * tp2,
                        stage=3, trailing_stop_price=cur_stop,
                    ))
                    exited = True

                # P3 持有期限（30 日）
                elif pos["holding_days"] >= max_hold_days:
                    cash += pos["shares"] * today_close
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=today_close, exit_reason="持有期限到期",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (today_close - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * today_close,
                        stage=3, trailing_stop_price=cur_stop,
                    ))
                    exited = True

            if not exited:
                remaining.append(pos)

        positions = remaining

        # ── 新進場訊號掃描 ───────────────────────────────────────────
        single_mode = (mode == "single")
        position_ok = single_mode or (len(positions) < max_positions)
        if position_ok:
            held_symbols = {p["symbol"] for p in positions}
            candidates   = day[~day.index.isin(held_symbols)]

            for sym, row in candidates.iterrows():
                if not single_mode and len(positions) >= max_positions:
                    break

                score        = _f(row, "final_decision_score", 0.0)
                override_raw = row.get("entry_plan_override", False)
                try:
                    override = bool(override_raw) if pd.notna(override_raw) else False
                except (TypeError, ValueError):
                    override = False

                zone_low  = _f(row, "support_entry_zone_low")
                zone_high = _f(row, "support_entry_zone_high")
                close     = _f(row, "close")

                if not (override or score >= 0.55):
                    continue
                if not (pd.notna(zone_low) and pd.notna(zone_high)
                        and pd.notna(close) and zone_low <= close <= zone_high):
                    continue

                # 動態最小停損距離
                atr_pct_val = _f(row, "atr_pct")
                if pd.notna(atr_pct_val) and atr_pct_val > 0:
                    min_stop_pct = max(1.5 * atr_pct_val, 0.015)
                else:
                    min_stop_pct = 0.02

                stop_loss_price = _f(row, "entry_stop_loss_price")
                entry_rr_1      = _f(row, "entry_rr_1")

                if pd.isna(stop_loss_price) or stop_loss_price <= 0:
                    continue
                if pd.isna(close) or close <= 0:
                    continue

                raw_stop_pct = (close - stop_loss_price) / close
                # 上限守門
                if raw_stop_pct > 0.08:
                    continue
                if pd.isna(entry_rr_1) or entry_rr_1 < 1.5:
                    continue

                # 動態調整：若原始距離不足 min_stop_pct 則拉大
                if raw_stop_pct < min_stop_pct:
                    actual_stop_price = close * (1 - min_stop_pct)
                    actual_stop_pct   = min_stop_pct
                    stop_adjusted     = True
                else:
                    actual_stop_price = stop_loss_price
                    actual_stop_pct   = raw_stop_pct
                    stop_adjusted     = False

                tp1     = _f(row, "entry_take_profit_1")
                tp2     = _f(row, "entry_take_profit_2")
                pos_pct = _f(row, "suggested_position_pct")
                atr_val = _f(row, "atr_14", 0.0)

                if pd.isna(pos_pct) or pos_pct <= 0:
                    pos_pct = 0.05

                total_equity = cash + sum(
                    p["shares"] * _f(
                        day.loc[p["symbol"]] if p["symbol"] in day.index else pd.Series(),
                        "close", p["entry_price"],
                    )
                    for p in positions
                )
                position_value = pos_pct * total_equity
                if position_value <= 0:
                    continue

                shares = position_value / close
                cash  -= position_value

                entry_counter += 1
                entry_date_str = pd.Timestamp(current_date).strftime("%Y-%m-%d")
                trade_id = f"{sym}_{entry_date_str}_{round(float(close), 4)}"

                # 台股交易成本（買進）：滑價（買賣各一半）+ 手續費（含折扣）
                slippage_buy       = close * slippage_rate / 2
                buy_cost_per_share = close * buy_fee_rate * fee_discount
                actual_entry_price = close + slippage_buy + buy_cost_per_share

                positions.append({
                    "trade_id":                 trade_id,
                    "entry_id":                 entry_counter,
                    "entry_layer_id":           1,
                    "layer_id":                 1,
                    "symbol":                   str(sym),
                    "entry_date":               current_date,
                    "entry_price":              close,
                    "actual_entry_price":       actual_entry_price,
                    "buy_cost_per_share":       buy_cost_per_share,
                    "shares":                   shares,
                    "cost":                     position_value,
                    "stop_loss":                actual_stop_price,
                    "actual_stop_loss_price":   actual_stop_price,
                    "actual_stop_loss_pct":     actual_stop_pct,
                    "stop_adjusted":            stop_adjusted,
                    "take_profit_1":            tp1,
                    "take_profit_2":            tp2,
                    "phase1_done":              False,
                    "max_close":                close,
                    "holding_days":             0,
                    "atr_at_entry":             atr_val,
                    "support_entry_zone_low":   zone_low,
                    "support_entry_zone_high":  zone_high,
                    "entry_decision_score":     score,
                    "entry_return_20d_score":   _f(row, "return_20d_score"),
                    "false_breakdown_count":    0,
                    "trailing_stop_activated":  False,
                })

        # ── 當日資產計算 ─────────────────────────────────────────────
        pos_value = 0.0
        for pos in positions:
            sym = pos["symbol"]
            px = _f(
                day.loc[sym] if sym in day.index else pd.Series(),
                "close", pos["entry_price"],
            )
            pos_value += pos["shares"] * px

        equity_rows.append({
            "date":           current_date,
            "cash":           cash,
            "position_value": pos_value,
            "total_equity":   cash + pos_value,
            "open_positions": len(positions),
        })

    trade_log_df = pd.DataFrame(trade_log)
    equity_curve = pd.DataFrame(equity_rows)
    return trade_log_df, equity_curve


def run_portfolio_backtest(
    all_stock_signals: dict[str, pd.DataFrame],
    config: PipelineConfig,
    market_df: Optional[pd.DataFrame] = None,
    expected_symbols: Optional[list[str]] = None,
    investor_mode: bool = False,
    save_snapshot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Portfolio 層級回測：所有股票共用同一筆資金池與最大持倉數限制，
    每日先處理所有持倉出場、釋放資金，再依 final_decision_score 由高到低
    決定新進場優先順序，直到達到最大持倉數量或現金不足為止。

    Args:
        all_stock_signals: 每支股票的 OOS 預測資料，key 為 symbol，value 為該股 DataFrame。
        config: PipelineConfig，提供資金與成本假設。
        market_df: TAIEX 歷史資料（含 date / market_close），用於大盤 20MA 過濾；
                   若為 None 則不啟用大盤過濾（視為隨時允許進場）。
        expected_symbols: 原始 Top N 股票池清單，用於驗證 all_stock_signals 的股票組成；
                   若為 None 則跳過驗證。

    Returns:
        (每日淨值曲線 DataFrame, 交易明細 DataFrame, 績效指標 dict)
    """
    # ── 股票池組成驗證 ───────────────────────────────────────────────
    pool_symbols = sorted(str(s) for s in (all_stock_signals or {}).keys())
    if not investor_mode:
        print(f"回測股票池：{[stock_label(s) for s in pool_symbols]}")
        print(f"股票數量：{len(pool_symbols)}")
    if expected_symbols is not None:
        expected_set = {str(s) for s in expected_symbols}
        unexpected_symbols = sorted(s for s in pool_symbols if s not in expected_set)
        if unexpected_symbols and not investor_mode:
            print(f"WARNING：以下股票不在 Top N 清單中：{unexpected_symbols}")

    initial_capital = config.backtest_initial_capital
    max_positions   = config.backtest_max_positions
    max_hold_days   = 60   # 60日期限到期（原 30 日強制出場已移除）

    # ── 倉位管理參數（金字塔式分層加碼：試倉 + 加碼1 + 加碼2）──────────
    initial_pct             = config.backtest_initial_position_pct
    add1_pct                = config.backtest_add1_position_pct
    add2_pct                = config.backtest_add2_position_pct
    max_single_position_pct = config.backtest_max_single_position_pct
    max_total_exposure      = config.backtest_max_total_exposure
    min_cash_reserve        = config.backtest_min_cash_reserve
    allocation_eps          = 1e-12
    entry_mode              = getattr(config, "backtest_entry_mode", "strict_support")
    if entry_mode not in (
        "strict_support",
        "balanced_support",
        "balanced_support_v2",
        "volume_confirmed_entry_v1",
    ):
        raise ValueError(f"Unsupported portfolio entry_mode: {entry_mode}")

    # ── 台股交易成本假設（與 run_backtest 一致）──────────────────────
    buy_fee_rate   = config.backtest_buy_fee_rate
    sell_fee_rate  = config.backtest_sell_fee_rate
    tax_rate       = config.backtest_tax_rate
    slippage_rate  = config.backtest_slippage_rate
    fee_discount   = config.backtest_fee_discount

    # ── 20 日報酬輔助分數（return_20d_score）：trailing 倍數 / 第一止盈比例 /
    #    加碼1 條件鬆緊，依進場當下的 return_20d_score 決定 ─────────────
    use_return_20d           = bool(getattr(config, "use_return_20d", False))
    base_trailing_multiplier = 3.0   # 一般情況下的 ATR trailing 倍數（原 2.0 → 3.0）
    high_trailing_multiplier = 3.5   # return_20d daily rank 前段（缺 rank 時 fallback 分數）給更寬鬆空間
    low_trailing_multiplier  = 2.5   # return_20d_score <  0.4（中期偏弱，較保守）
    base_tp1_ratio           = 0.30  # 第一止盈出場比例 30%；剩餘 70% 繼續持有
    high_tp1_ratio            = 0.30  # return_20d daily rank 前段時同為 30%
    tp2_ratio                  = 0.30  # 第二止盈固定再出場 30%（相對於 phase1 完成時的基準股數）
    trailing_min_holding_days = 5     # 持有未滿 5 個交易日，trailing stop 不啟動

    def _entry_return_20d_score(row) -> float:
        if not use_return_20d:
            return np.nan
        v = row.get("return_20d_score", np.nan)
        try:
            return float(v) if pd.notna(v) else 0.5
        except (TypeError, ValueError):
            return 0.5

    def _entry_return_20d_score_missing(row) -> bool:
        if not use_return_20d or "return_20d_score" not in row.index:
            return True
        v = row.get("return_20d_score", np.nan)
        try:
            return bool(pd.isna(v) or pd.isna(float(v)))
        except (TypeError, ValueError):
            return True

    return_20d_top_entry_pct = float(getattr(config, "backtest_return_20d_top_entry_pct", 0.30) or 0.30)
    return_20d_top_add_pct = float(getattr(config, "backtest_return_20d_top_add_pct", 0.30) or 0.30)
    return_20d_top_strict_pct = float(getattr(config, "backtest_return_20d_top_strict_pct", 0.20) or 0.20)

    def _entry_return_20d_daily_rank(row) -> float:
        v = row.get("return_20d_score_daily_rank", np.nan)
        try:
            return float(v) if pd.notna(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def _return_20d_rank_top(row, top_pct: float = 0.30, fallback_score: float = 0.60) -> bool:
        rank_value = _entry_return_20d_daily_rank(row)
        if pd.notna(rank_value):
            return bool(rank_value >= max(0.0, 1.0 - float(top_pct)))
        score_value = _entry_return_20d_score(row)
        return bool(pd.notna(score_value) and score_value >= fallback_score)

    def _close_above_ma(row, close: float, ma_col: str, distance_col: str) -> bool:
        ma_value = _f(row, ma_col)
        if pd.notna(close) and close > 0 and pd.notna(ma_value) and ma_value > 0:
            return bool(close > ma_value)
        ma_distance = _f(row, distance_col)
        return bool(pd.notna(ma_distance) and ma_distance > 0)

    def _atr_pct_for_entry(row, close: float) -> float:
        atr_pct = _f(row, "atr_pct")
        if pd.notna(atr_pct) and atr_pct > 0:
            return atr_pct
        atr_14 = _f(row, "atr_14")
        if pd.notna(atr_14) and pd.notna(close) and close > 0:
            return atr_14 / close
        return np.nan

    def _strong_volume_soft_condition(row) -> bool:
        volume_ratio_20 = _f(row, "volume_ratio_20")
        return bool(pd.notna(volume_ratio_20) and volume_ratio_20 >= 1.20)

    def _strong_continuation_base_checks(row, score: float, close: float) -> dict[str, bool]:
        rank_value = _entry_return_20d_daily_rank(row)
        close_to_10d_high = _f(row, "close_to_10d_high")
        close_to_20d_high = _f(row, "close_to_20d_high")
        atr_pct = _atr_pct_for_entry(row, close)

        trend_ok = bool(
            pd.notna(score) and score >= 0.55
            and pd.notna(close) and close > 0
            and _close_above_ma(row, close, "ma20", "ma20_distance")
            and _close_above_ma(row, close, "ma60", "ma60_distance")
        )
        return_rank_ok = bool(pd.notna(rank_value) and rank_value >= 0.70)
        high_position_ok = bool(
            (pd.notna(close_to_10d_high) and close_to_10d_high >= -0.03)
            or (pd.notna(close_to_20d_high) and close_to_20d_high >= -0.03)
        )
        atr_risk_ok = bool(pd.notna(atr_pct) and atr_pct <= 0.08)
        candidate = bool(trend_ok and return_rank_ok and high_position_ok and atr_risk_ok)

        return {
            "trend_ok": trend_ok,
            "return_rank_ok": return_rank_ok,
            "high_position_ok": high_position_ok,
            "atr_risk_ok": atr_risk_ok,
            "candidate": candidate,
            "volume_soft_ok": _strong_volume_soft_condition(row),
        }

    def _strong_continuation_entry_reason(row, score: float, close: float) -> Optional[str]:
        checks = _strong_continuation_base_checks(row, score, close)
        return "strong_trend_return_rank_top30" if checks["candidate"] else None

    def _trend_still_strong(row, close: float) -> bool:
        ma20_distance = _f(row, "ma20_distance")
        ma60_distance = _f(row, "ma60_distance")
        ma20_slope = _f(row, "ma20_slope")
        close_to_20d_high = _f(row, "close_to_20d_high")
        return bool(
            pd.notna(close) and close > 0
            and pd.notna(ma20_distance) and ma20_distance > 0
            and pd.notna(ma60_distance) and ma60_distance > 0
            and (pd.isna(ma20_slope) or ma20_slope >= 0)
            and (pd.isna(close_to_20d_high) or close_to_20d_high >= -0.05)
            and _return_20d_rank_top(row, return_20d_top_add_pct, fallback_score=0.60)
        )

    # ── 大盤過濾（TAIEX 跌破 20MA 時暫停所有新進場與加碼）─────────────
    market_lookup = pd.DataFrame()
    if market_df is not None and not market_df.empty and "market_close" in market_df.columns:
        m = market_df.copy()
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        m = m.dropna(subset=["date"]).sort_values("date")
        m["market_close"] = pd.to_numeric(m["market_close"], errors="coerce")
        m["market_ma20"]  = m["market_close"].rolling(window=20, min_periods=1).mean()
        market_lookup = m.set_index("date")[["market_close", "market_ma20"]]

    def _market_state(date) -> tuple[float, float, bool]:
        """回傳 (market_close, market_ma20, market_filter_active)；以 asof 對齊最近一個有效交易日。"""
        if market_lookup.empty:
            return np.nan, np.nan, False
        snap = market_lookup.asof(date)
        if snap is None or (isinstance(snap, pd.Series) and snap.isna().all()):
            return np.nan, np.nan, False
        mc  = snap.get("market_close", np.nan)
        mma = snap.get("market_ma20", np.nan)
        mc  = float(mc)  if pd.notna(mc)  else np.nan
        mma = float(mma) if pd.notna(mma) else np.nan
        active = bool(pd.notna(mc) and pd.notna(mma) and mc < mma)
        return mc, mma, active

    filtered_entries_count = 0  # 大盤過濾期間，原本符合條件但被暫停的進場/加碼次數

    # ── 合併所有股票訊號為單一時間軸 ───────────────────────────────
    frames: list[pd.DataFrame] = []
    for sym, sym_df in (all_stock_signals or {}).items():
        if sym_df is None or sym_df.empty:
            continue
        d = sym_df.copy()
        d["symbol"] = str(sym)
        frames.append(d)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(), {}

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    df = _add_return_20d_score_daily_rank(df)

    return_20d_score_column_present = "return_20d_score" in df.columns
    if return_20d_score_column_present:
        r20d_series = pd.to_numeric(df["return_20d_score"], errors="coerce")
        r20d_count = int(r20d_series.notna().sum())
        _r20d_nonzero = (r20d_series.notna() & (r20d_series != 0))
        return_20d_score_stats = {
            "column_present": True,
            "count": r20d_count,
            "mean": float(r20d_series.mean()) if r20d_count else np.nan,
            "min": float(r20d_series.min()) if r20d_count else np.nan,
            "max": float(r20d_series.max()) if r20d_count else np.nan,
            "q25": float(r20d_series.quantile(0.25)) if r20d_count else np.nan,
            "q50": float(r20d_series.quantile(0.50)) if r20d_count else np.nan,
            "q75": float(r20d_series.quantile(0.75)) if r20d_count else np.nan,
            "nan_ratio": float(r20d_series.isna().mean()) if len(r20d_series) else np.nan,
            "nonzero_ratio": float(_r20d_nonzero.sum() / len(r20d_series)) if len(r20d_series) else np.nan,
        }
    else:
        return_20d_score_stats = {
            "column_present": False,
            "count": 0,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "nan_ratio": 1.0,
            "nonzero_ratio": np.nan,
        }
    rank_series = pd.to_numeric(
        df.get("return_20d_score_daily_rank", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    rank_count = int(rank_series.notna().sum())
    return_20d_score_stats["daily_rank_column_present"] = "return_20d_score_daily_rank" in df.columns
    return_20d_score_stats["daily_rank_count"] = rank_count
    return_20d_score_stats["daily_rank_top30_count"] = int((rank_series >= 0.70).sum()) if rank_count else 0
    return_20d_score_stats["daily_rank_top20_count"] = int((rank_series >= 0.80).sum()) if rank_count else 0

    strong_continuation_column_check = {}
    for col in [
        "ma20",
        "ma60",
        "return_20d_score_daily_rank",
        "close_to_10d_high",
        "close_to_20d_high",
        "atr_pct",
    ]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            strong_continuation_column_check[col] = {
                "present": True,
                "non_null_count": int(series.notna().sum()),
            }
        else:
            strong_continuation_column_check[col] = {
                "present": False,
                "non_null_count": 0,
            }

    if "open" not in df.columns:
        print("WARNING：合併後的回測資料缺少 open 欄位，跳空停損保護將退化為一般停損（以停損價出場）。")

    dates = sorted(df["date"].unique())
    dates_list = list(dates)  # for O(n) index lookup in profit cooldown calculation

    cash: float = initial_capital
    positions: list[dict] = []
    trade_log: list[dict] = []
    equity_rows: list[dict] = []
    entry_counter = 0
    prev_equity = initial_capital
    entry_diagnostics = {key: 0 for key in ENTRY_DIAGNOSTIC_DAILY_COLUMNS}
    entry_primary_rejection_counts = {key: 0 for key in ENTRY_REJECTION_KEYS}
    entry_diagnostics_daily_rows: list[dict] = []
    rejection_future_return_rows: list[dict] = []
    rejection_future_return_reasons = {
        "rejected_by_support_zone",
        "rejected_by_market_filter",
        "rejected_by_rr",
        "rejected_by_position_limit",
        "rejected_by_total_exposure",
    }
    strong_rejection_future_return_reasons = {
        "rejected_strong_by_trend",
        "rejected_strong_by_return_rank",
        "rejected_strong_by_high_position",
        "rejected_strong_by_atr_risk",
        "rejected_strong_by_rr",
    }
    add2_diagnostics = {
        "add2_candidate_count": 0,
        "rejected_add2_by_score": 0,
        "rejected_add2_by_support_quality": 0,
        "rejected_add2_by_cash": 0,
        "rejected_add2_by_exposure": 0,
        "rejected_add2_by_price_condition": 0,
        "rejected_add2_by_trend_condition": 0,
        "rejected_add2_by_defense_line": 0,
        "rejected_add2_by_existing_flag": 0,
        "add2_score_bypassed": 0,
        "add2_score_required": 0,
    }
    # ── 獲利出場冷卻期與統計計數器 ───────────────────────────────────
    profit_exit_cooldowns: dict[str, object] = {}   # sym -> cooldown_end_date
    profit_exited_symbols: set[str] = set()          # 曾觸發獲利出場的股票
    profit_30_count   = 0
    profit_50_count   = 0
    trend_break_count = 0
    expired_count     = 0
    re_entry_count    = 0
    peak_profit_list: list[float] = []
    trailing_stop_stage1_count = 0
    trailing_stop_stage2_count = 0

    def _f(row, col, default=np.nan):
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    def _new_daily_entry_diagnostics(date) -> dict:
        row = {"date": date}
        for key in ENTRY_DIAGNOSTIC_DAILY_COLUMNS:
            row[key] = 0
        return row

    def _bump_entry_diagnostic(daily_diag: dict, key: str, amount: int = 1) -> None:
        entry_diagnostics[key] += amount
        daily_diag[key] += amount

    def _entry_signal(row) -> tuple[bool, float, bool]:
        score = _f(row, "final_decision_score", 0.0)
        override_raw = row.get("entry_plan_override", False)
        try:
            override = bool(override_raw) if pd.notna(override_raw) else False
        except (TypeError, ValueError):
            override = False
        return bool(override or score >= 0.55), score, override

    def _entry_support_reason(
        row,
        score: float,
        close: float,
        zone_low: float,
        zone_high: float,
    ) -> Optional[str]:
        if entry_mode == "volume_confirmed_entry_v1":
            volume_reason = _volume_entry_reason(
                row=row,
                score=score,
                close=close,
                zone_low=zone_low,
                zone_high=zone_high,
            )
            if volume_reason in ("quiet_pullback_support", "volume_rebound_support"):
                support_reason = "strict_support"
            else:
                support_reason = None
        else:
            support_reason = _balanced_support_v2_reason(
                row=row,
                score=score,
                close=close,
                zone_low=zone_low,
                zone_high=zone_high,
            )

        if support_reason is not None:
            return support_reason
        prox = _support_proximity_reason(row, close)
        if prox is not None:
            return prox
        brk = _breakout_entry_reason(row, close)
        if brk is not None:
            return brk
        return None

    def _balanced_support_v2_reason(
        row,
        score: float,
        close: float,
        zone_low: float,
        zone_high: float,
    ) -> Optional[str]:
        in_support_zone = bool(
            pd.notna(zone_low) and pd.notna(zone_high)
            and pd.notna(close) and zone_low <= close <= zone_high
        )
        if in_support_zone:
            return "strict_support"
        if entry_mode == "strict_support":
            return None

        if entry_mode == "balanced_support":
            high_score_ok = bool(pd.notna(score) and score >= 0.70)
            near_support_ok = bool(
                pd.notna(close) and pd.notna(zone_high)
                and close <= zone_high * 1.03
            )
            return "near_support_relaxed" if high_score_ok and near_support_ok else None

        score_support_ok = bool(pd.notna(score) and score >= 0.65)
        near_support_relaxed_ok = bool(
            pd.notna(close) and pd.notna(zone_high)
            and close <= zone_high * 1.05
        )
        if score_support_ok and near_support_relaxed_ok:
            return "near_support_relaxed"

        return None

    def _support_proximity_reason(row, close: float) -> Optional[str]:
        nearest_support = _f(row, "nearest_support_level")
        if pd.isna(close) or close <= 0 or pd.isna(nearest_support) or nearest_support <= 0:
            return None
        if nearest_support * 0.97 <= close <= nearest_support * 1.03:
            return "support_proximity"
        return None

    def _breakout_entry_reason(row, close: float) -> Optional[str]:
        nearest_resistance = _f(row, "nearest_resistance_level")
        volume_ratio_20 = _f(row, "volume_ratio_20")
        rsi14 = _f(row, "RSI14")
        if pd.isna(close) or close <= 0:
            return None
        if (pd.notna(nearest_resistance) and nearest_resistance > 0
                and close > nearest_resistance * 1.005
                and pd.notna(volume_ratio_20) and volume_ratio_20 >= 1.3
                and pd.notna(rsi14) and 50 <= rsi14 <= 75):
            return "breakout_entry"
        return None

    def _volume_entry_reason(
        row,
        score: float,
        close: float,
        zone_low: float,
        zone_high: float,
    ) -> str:
        in_support_zone = bool(
            pd.notna(zone_low) and pd.notna(zone_high)
            and pd.notna(close) and zone_low <= close <= zone_high
        )
        volume_ratio_20 = _f(row, "volume_ratio_20")
        volume_ratio_5 = _f(row, "volume_ratio_5")  # reserved for volume-confirmation diagnostics/extensions
        volume_ratio_20_change_3d = _f(row, "volume_ratio_20_change_3d")  # reserved
        return_1d = _f(row, "return_1d")
        close_to_10d_high = _f(row, "close_to_10d_high")
        ma20_distance = _f(row, "ma20_distance")
        _ = volume_ratio_5, volume_ratio_20_change_3d

        quiet_pullback_support_ok = bool(
            in_support_zone
            and pd.notna(volume_ratio_20) and volume_ratio_20 <= 1.20
            and pd.notna(return_1d) and return_1d >= -0.035
        )
        volume_rebound_support_ok = bool(
            in_support_zone
            and pd.notna(volume_ratio_20) and volume_ratio_20 >= 1.30
            and pd.notna(return_1d) and return_1d >= -0.025
        )
        volume_breakout_momentum_ok = bool(
            pd.notna(score) and score >= 0.75
            and pd.notna(close_to_10d_high) and close_to_10d_high >= -0.03
            and pd.notna(ma20_distance) and ma20_distance > 0
            and pd.notna(volume_ratio_20) and volume_ratio_20 >= 1.30
        )

        if volume_breakout_momentum_ok:
            return "volume_breakout_momentum"
        if volume_rebound_support_ok:
            return "volume_rebound_support"
        if quiet_pullback_support_ok:
            return "quiet_pullback_support"
        return "none"

    def _entry_support_ok(
        row,
        score: float,
        close: float,
        zone_low: float,
        zone_high: float,
    ) -> bool:
        return _entry_support_reason(
            row=row,
            score=score,
            close=close,
            zone_low=zone_low,
            zone_high=zone_high,
        ) is not None

    def _momentum_condition_rejected(row, score: float) -> bool:
        if entry_mode != "balanced_support_v2" or pd.isna(score) or score < 0.75:
            return False
        close_to_10d_high = _f(row, "close_to_10d_high")
        ma20_distance = _f(row, "ma20_distance")
        return not (
            pd.notna(close_to_10d_high) and close_to_10d_high >= -0.03
            and pd.notna(ma20_distance) and ma20_distance > 0
        )

    def _strong_continuation_rejection_flags(
        sym,
        row,
        held_symbols: set[str],
        positions_count: int,
        cash_value: float,
        portfolio_value_now: float,
        total_equity: float,
        market_filter_active: bool,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        _, score, _ = _entry_signal(row)
        close = _f(row, "close")
        checks = _strong_continuation_base_checks(row, score, close)
        flags = {
            "rejected_strong_by_trend": not checks["trend_ok"],
            "rejected_strong_by_return_rank": not checks["return_rank_ok"],
            "rejected_strong_by_high_position": not checks["high_position_ok"],
            "rejected_strong_by_atr_risk": not checks["atr_risk_ok"],
            "rejected_strong_by_market_filter": False,
            "rejected_strong_by_cash": False,
            "rejected_strong_by_position_limit": False,
            "rejected_strong_by_total_exposure": False,
            "rejected_strong_by_existing_position": False,
            "rejected_strong_by_rr": False,
            "rejected_strong_by_volume_soft_condition": False,
        }

        if checks["candidate"]:
            flags["rejected_strong_by_volume_soft_condition"] = not checks["volume_soft_ok"]
            flags["rejected_strong_by_existing_position"] = str(sym) in held_symbols
            flags["rejected_strong_by_position_limit"] = positions_count >= max_positions
            flags["rejected_strong_by_market_filter"] = bool(market_filter_active)

            entry_rr_1 = _f(row, "entry_rr_1")
            flags["rejected_strong_by_rr"] = bool(pd.isna(entry_rr_1) or entry_rr_1 < 1.5)

            position_value = np.nan
            if total_equity <= 0:
                flags["rejected_strong_by_cash"] = True
            else:
                current_exposure_pct = portfolio_value_now / total_equity
                position_value = initial_pct * total_equity
                proposed_cash_pct = (cash_value - position_value) / total_equity
                flags["rejected_strong_by_total_exposure"] = (
                    current_exposure_pct + initial_pct > max_total_exposure + allocation_eps
                )
                flags["rejected_strong_by_cash"] = bool(
                    position_value <= 0
                    or proposed_cash_pct + allocation_eps < min_cash_reserve
                )

        return flags, checks

    def _entry_rejection_flags(
        sym,
        row,
        held_symbols: set[str],
        positions_count: int,
        cash_value: float,
        portfolio_value_now: float,
        total_equity: float,
        market_filter_active: bool,
        strong_checks: Optional[dict[str, bool]] = None,
    ) -> tuple[dict[str, bool], Optional[str]]:
        flags = {key: False for key in ENTRY_REJECTION_KEYS}

        flags["rejected_by_existing_position"] = str(sym) in held_symbols
        flags["rejected_by_position_limit"] = positions_count >= max_positions

        _, score, _ = _entry_signal(row)
        zone_low  = _f(row, "support_entry_zone_low")
        zone_high = _f(row, "support_entry_zone_high")
        close     = _f(row, "close")
        entry_support_reason = _entry_support_reason(
            row=row,
            score=score,
            close=close,
            zone_low=zone_low,
            zone_high=zone_high,
        )
        strong_candidate = bool(strong_checks and strong_checks.get("candidate", False))
        balanced_support_v2_reason = _balanced_support_v2_reason(
            row=row,
            score=score,
            close=close,
            zone_low=zone_low,
            zone_high=zone_high,
        )
        volume_reason = _volume_entry_reason(
            row=row,
            score=score,
            close=close,
            zone_low=zone_low,
            zone_high=zone_high,
        )
        flags["rejected_by_support_zone"] = entry_support_reason is None and not strong_candidate
        flags["rejected_by_momentum_condition"] = (
            entry_support_reason is None
            and not strong_candidate
            and _momentum_condition_rejected(row, score)
        )
        flags["rejected_by_volume_confirmation"] = bool(
            entry_mode == "volume_confirmed_entry_v1"
            and balanced_support_v2_reason is not None
            and volume_reason == "none"
        )

        entry_rr_1 = _f(row, "entry_rr_1")
        flags["rejected_by_rr"] = bool(
            not strong_candidate
            and (pd.isna(entry_rr_1) or entry_rr_1 < 1.5)
        )

        stop_loss_price = _f(row, "entry_stop_loss_price")
        stop_loss_pct   = _f(row, "entry_stop_loss_pct")
        flags["rejected_by_stop_loss_pct"] = bool(
            pd.isna(stop_loss_price) or stop_loss_price <= 0
            or pd.isna(stop_loss_pct) or stop_loss_pct < 0.012 or stop_loss_pct > 0.08
        )

        position_value = np.nan
        if total_equity <= 0:
            flags["rejected_by_cash"] = True
        else:
            current_exposure_pct = portfolio_value_now / total_equity
            position_value = initial_pct * total_equity
            proposed_cash_pct = (cash_value - position_value) / total_equity
            flags["rejected_by_single_position_limit"] = (
                initial_pct > max_single_position_pct + allocation_eps
            )
            flags["rejected_by_total_exposure"] = (
                current_exposure_pct + initial_pct > max_total_exposure + allocation_eps
            )
            flags["rejected_by_cash"] = bool(
                position_value <= 0
                or proposed_cash_pct + allocation_eps < min_cash_reserve
            )

        flags["rejected_by_market_filter"] = bool(market_filter_active)

        # 主要拒絕原因依照目前實際新進場檢查順序紀錄第一個擋下原因。
        primary_reason = None
        if flags["rejected_by_existing_position"]:
            primary_reason = "rejected_by_existing_position"
        elif flags["rejected_by_position_limit"]:
            primary_reason = "rejected_by_position_limit"
        elif flags["rejected_by_volume_confirmation"]:
            primary_reason = "rejected_by_volume_confirmation"
        elif flags["rejected_by_support_zone"]:
            primary_reason = "rejected_by_support_zone"
        elif flags["rejected_by_rr"]:
            primary_reason = "rejected_by_rr"
        elif flags["rejected_by_stop_loss_pct"]:
            primary_reason = "rejected_by_stop_loss_pct"
        elif total_equity <= 0 or (pd.notna(position_value) and position_value <= 0):
            primary_reason = "rejected_by_cash"
        elif flags["rejected_by_single_position_limit"]:
            primary_reason = "rejected_by_single_position_limit"
        elif flags["rejected_by_total_exposure"]:
            primary_reason = "rejected_by_total_exposure"
        elif flags["rejected_by_cash"]:
            primary_reason = "rejected_by_cash"
        elif flags["rejected_by_market_filter"]:
            primary_reason = "rejected_by_market_filter"

        return flags, primary_reason

    def _record_entry_candidate(
        daily_diag: dict,
        sym,
        row,
        held_symbols: set[str],
        positions_count: int,
        cash_value: float,
        portfolio_value_now: float,
        total_equity: float,
        market_filter_active: bool,
    ) -> Optional[str]:
        _bump_entry_diagnostic(daily_diag, "candidate_entry_signals")
        _, score, _ = _entry_signal(row)
        zone_low  = _f(row, "support_entry_zone_low")
        zone_high = _f(row, "support_entry_zone_high")
        close     = _f(row, "close")
        support_reason = _entry_support_reason(
            row=row,
            score=score,
            close=close,
            zone_low=zone_low,
            zone_high=zone_high,
        )
        if support_reason is not None:
            _bump_entry_diagnostic(daily_diag, "support_entry_candidates")

        strong_flags, strong_checks = _strong_continuation_rejection_flags(
            sym=sym,
            row=row,
            held_symbols=held_symbols,
            positions_count=positions_count,
            cash_value=cash_value,
            portfolio_value_now=portfolio_value_now,
            total_equity=total_equity,
            market_filter_active=market_filter_active,
        )
        if strong_checks["candidate"]:
            _bump_entry_diagnostic(daily_diag, "strong_continuation_candidates")
        for key, value in strong_flags.items():
            if value:
                _bump_entry_diagnostic(daily_diag, key)
        for reason in strong_rejection_future_return_reasons:
            if strong_flags.get(reason, False):
                rejection_future_return_rows.append({
                    "date": current_date,
                    "symbol": str(sym),
                    "rejection_reason": reason,
                    "signal_close": close,
                })

        flags, primary_reason = _entry_rejection_flags(
            sym=sym,
            row=row,
            held_symbols=held_symbols,
            positions_count=positions_count,
            cash_value=cash_value,
            portfolio_value_now=portfolio_value_now,
            total_equity=total_equity,
            market_filter_active=market_filter_active,
            strong_checks=strong_checks,
        )
        for key in ENTRY_REJECTION_KEYS:
            if flags[key]:
                _bump_entry_diagnostic(daily_diag, key)
        for reason in rejection_future_return_reasons:
            if flags.get(reason, False):
                rejection_future_return_rows.append({
                    "date": current_date,
                    "symbol": str(sym),
                    "rejection_reason": reason,
                    "signal_close": close,
                })
        if _volume_entry_reason(row, score, close, zone_low, zone_high) != "none":
            _bump_entry_diagnostic(daily_diag, "candidate_with_volume_confirmation")
        if primary_reason is not None:
            entry_primary_rejection_counts[primary_reason] += 1
        return primary_reason

    def _recompute_position_aggregates(pos: dict) -> None:
        """依 position_layers 重新計算總股數/總成本/加權平均進場價（含成本後價）。

        為相容既有出場邏輯（停損/止盈/trailing stop 等皆以 pos["entry_price"] /
        pos["shares"] 計算 pnl 與 reward_per_share），這裡同步把這兩個欄位
        設成 avg_entry_price / total_shares 的別名，讓既有程式碼自動套用
        「reward_per_share 改用 avg_entry_price 計算」。
        """
        layers = pos["position_layers"]
        total_shares = sum(l["shares"] for l in layers)
        total_cost   = sum(l["cost"]   for l in layers)
        pos["total_shares"] = total_shares
        pos["total_cost"]   = total_cost
        pos["avg_entry_price"] = (total_cost / total_shares) if total_shares else np.nan

        actual_cost_sum = 0.0
        fee_cost_sum    = 0.0
        for l in layers:
            p    = l["entry_price"]
            slip = p * slippage_rate / 2
            fee  = p * buy_fee_rate * fee_discount
            actual_cost_sum += l["shares"] * (p + slip + fee)
            fee_cost_sum    += l["shares"] * fee
        pos["actual_entry_price"] = (actual_cost_sum / total_shares) if total_shares else np.nan
        pos["buy_cost_per_share"] = (fee_cost_sum / total_shares) if total_shares else np.nan

        # legacy 別名：既有出場 / _log 邏輯沿用 entry_price / shares 即可自動運作
        pos["entry_price"] = pos["avg_entry_price"]
        pos["shares"]      = pos["total_shares"]

    def _add_layer(pos: dict, layer_id: int, entry_date, entry_price: float,
                    shares: float, cost: float) -> None:
        pos["position_layers"].append({
            "layer_id":    layer_id,
            "entry_date":  entry_date,
            "entry_price": entry_price,
            "shares":      shares,
            "cost":        cost,
        })
        pos["current_layer"] = layer_id
        _recompute_position_aggregates(pos)

    def _set_profit_cooldown(sym_key: str, exit_date) -> None:
        """設定獲利出場後的 5 個交易日冷卻期，並標記該股票曾觸發獲利出場。"""
        profit_exited_symbols.add(sym_key)
        try:
            idx = dates_list.index(exit_date)
            end_idx = min(idx + 5, len(dates_list) - 1)
            profit_exit_cooldowns[sym_key] = dates_list[end_idx]
        except (ValueError, IndexError):
            profit_exit_cooldowns[sym_key] = exit_date

    def _layer_stop_loss(pos: dict) -> float:
        """依目前加碼層數計算有效停損價：
        layer1=原始停損；layer2=max(原始停損, avg_entry_price×0.98)；
        layer3=max(原始停損, avg_entry_price)（保本）。
        """
        base  = pos["entry_stop_loss_price_original"]
        layer = pos["current_layer"]
        avg   = pos["avg_entry_price"]
        if layer <= 1 or pd.isna(avg):
            return base
        if layer == 2:
            return max(base, avg * 0.98)
        return max(base, avg)

    def _log(pos: dict, partial_exit: bool = False, **kw) -> dict:
        """同 run_backtest._log：組裝出場記錄並計算成本後報酬率欄位，並附加加碼層級資訊。"""
        pos["exit_seq"] = pos.get("exit_seq", 0) + 1

        entry_price        = pos["entry_price"]          # = avg_entry_price（別名）
        actual_entry_price = pos["actual_entry_price"]
        exit_price          = kw.get("exit_price", np.nan)

        slippage_sell       = exit_price * slippage_rate / 2 if pd.notna(exit_price) else np.nan
        sell_cost_per_share = exit_price * sell_fee_rate * fee_discount if pd.notna(exit_price) else np.nan
        tax_per_share        = exit_price * tax_rate if pd.notna(exit_price) else np.nan
        actual_exit_price    = (
            exit_price - slippage_sell - sell_cost_per_share - tax_per_share
            if pd.notna(exit_price) else np.nan
        )

        gross_return_pct = (
            (exit_price - entry_price) / entry_price
            if pd.notna(exit_price) and entry_price else np.nan
        )
        net_return_pct = (
            (actual_exit_price - actual_entry_price) / actual_entry_price
            if pd.notna(actual_exit_price) and actual_entry_price else np.nan
        )
        total_cost_pct = (
            ((actual_entry_price - entry_price) + (exit_price - actual_exit_price)) / entry_price
            if pd.notna(exit_price) and entry_price else np.nan
        )
        total_invested_pct = (pos["total_cost"] / prev_equity) if prev_equity else np.nan

        return {
            "trade_id":                 pos["trade_id"],
            "entry_id":                 pos["entry_id"],
            "exit_sequence":            pos["exit_seq"],
            "partial_exit_flag":        bool(partial_exit),
            "is_final_exit":            not bool(partial_exit),
            "symbol":                   pos["symbol"],
            "entry_date":               pos["entry_date"],
            "entry_price":              entry_price,
            "actual_entry_price":       actual_entry_price,
            "buy_cost_per_share":       pos["buy_cost_per_share"],
            "actual_stop_loss_price":   pos["actual_stop_loss_price"],
            "actual_stop_loss_pct":     pos["actual_stop_loss_pct"],
            "stop_adjusted":            pos["stop_adjusted"],
            "support_entry_zone_low":   pos.get("support_entry_zone_low"),
            "support_entry_zone_high":  pos.get("support_entry_zone_high"),
            "entry_decision_score":     pos.get("entry_decision_score"),
            "actual_exit_price":        actual_exit_price,
            "sell_cost_per_share":      sell_cost_per_share,
            "tax_per_share":            tax_per_share,
            "gross_return_pct":         gross_return_pct,
            "net_return_pct":           net_return_pct,
            "total_cost_pct":           total_cost_pct,
            "return_pct":               net_return_pct,
            "entry_type":               pos.get("entry_type"),
            "entry_mode_used":          pos.get("entry_mode_used"),
            "entry_reason":             pos.get("entry_reason"),
            "volume_entry_reason":      pos.get("volume_entry_reason", "none"),
            "layer_id":                 pos.get("entry_layer_id", pos["current_layer"]),
            "entry_layer_id":           pos.get("entry_layer_id", pos["current_layer"]),
            "layer_at_exit":            pos["current_layer"],
            "avg_entry_price":          pos["avg_entry_price"],
            "add1_triggered":           pos["add1_triggered"],
            "add2_triggered":           pos["add2_triggered"],
            "total_invested_pct":       total_invested_pct,
            "exit_trigger":             kw.get("exit_trigger"),
            "false_breakdown_count":    pos.get("false_breakdown_count", 0),
            "holding_days_at_exit":     pos["holding_days"],
            "trailing_stop_activated":  pos.get("trailing_stop_activated", False),
            "trailing_multiplier_used": pos.get("trailing_multiplier", np.nan),
            "trailing_method":          pos.get("trailing_method", "none"),
            "highest_close_at_exit":    pos.get("max_close", np.nan),
            "return_20d_score":         pos.get("entry_return_20d_score", np.nan),
            "entry_return_20d_score":   pos.get("entry_return_20d_score", np.nan),
            "return_20d_score_daily_rank": pos.get("entry_return_20d_score_daily_rank", np.nan),
            "entry_return_20d_score_daily_rank": pos.get("entry_return_20d_score_daily_rank", np.nan),
            "return_20d_score_missing_filled": pos.get("return_20d_score_missing_filled", False),
            "post_cooldown_entry":      pos.get("post_cooldown_entry", False),
            "same_stock_add_flag":      pos.get("same_stock_add_flag", False),
            "post_partial_exit_entry":  pos.get("post_partial_exit_entry", False),
            "profit_exit_flag":         kw.get("profit_exit_flag", False),
            "re_entry_after_profit":    bool(pos.get("re_entry_after_profit", False)),
            "peak_profit_pct":          pos.get("peak_profit_pct", np.nan),
            **kw,
        }

    for current_date in dates:
        day = df[df["date"] == current_date].set_index("symbol")

        market_close_today, market_ma20_today, market_filter_active = _market_state(current_date)
        daily_entry_diag = _new_daily_entry_diagnostics(current_date)

        # ── 1. 先處理所有持倉的出場 ──────────────────────────────────
        remaining: list[dict] = []
        for pos in positions:
            sym = pos["symbol"]
            if sym not in day.index:
                pos["holding_days"] += 1
                pos["days_above_tp1"] = 0  # 無資料日中斷連續計數
                remaining.append(pos)
                continue

            row         = day.loc[sym]
            today_low   = _f(row, "low",    pos["entry_price"])
            today_high  = _f(row, "high",   pos["entry_price"])
            today_close = _f(row, "close",  pos["entry_price"])
            today_atr   = _f(row, "atr_14", pos["atr_at_entry"])
            today_ma20  = _f(row, "ma20")
            today_open  = _f(row, "open",   pos["entry_price"])

            pos["holding_days"] += 1
            exited = False

            # ── 追蹤欄位：days_above_tp1 / peak_profit_pct ──────────────
            _tp1_ref = pos.get("take_profit_1")
            if pd.notna(_tp1_ref) and pd.notna(today_close) and today_close > _tp1_ref:
                pos["days_above_tp1"] = pos.get("days_above_tp1", 0) + 1
            else:
                pos["days_above_tp1"] = 0
            _avg_ep_pk = pos.get("avg_entry_price")
            if pd.notna(_avg_ep_pk) and _avg_ep_pk > 0:
                pos["peak_profit_pct"] = max(
                    pos.get("peak_profit_pct", 0.0),
                    (today_close - _avg_ep_pk) / _avg_ep_pk,
                )

            # ── Trailing stop：持有滿 5 個交易日後才啟動（只升不降）──────
            # 動態下限：獲利 >= 20% 時 floor = avg_entry_price × 1.10；
            #           獲利 >= 30% 時 floor = avg_entry_price × 1.15
            # 分段觸發：獲利 < 15% 使用 max_close - ATR×2（寬鬆緩衝）；
            #           15% <= 獲利 < 30% 維持原邏輯；30% 以上維持原分批出場
            pos["max_close"] = max(pos["max_close"], today_close)
            if (pos["holding_days"] >= trailing_min_holding_days):
                _avg_ep_ts = pos.get("avg_entry_price")
                _atr_val = today_atr if pd.notna(today_atr) and today_atr > 0 else (
                    today_close * 0.03 if pd.notna(today_close) and today_close > 0 else None
                )
                if _atr_val is not None:
                    if pd.notna(_avg_ep_ts) and _avg_ep_ts > 0:
                        _profit_ts = (today_close - _avg_ep_ts) / _avg_ep_ts
                    else:
                        _profit_ts = 0.0
                    if _profit_ts < 0.20:
                        # 階段一：寬鬆緩衝，避免正常波動洗出（門檻 0.20）
                        trailing_price = pos["max_close"] - 2.0 * _atr_val
                        pos["trailing_method"] = "atr_stage1_wide"
                        pos["trailing_stage"] = 1
                    elif _profit_ts >= 0.20 and _trend_still_strong(row, today_close):
                        # 階段二-A：趨勢仍強，使用 ma20_atr_relaxed
                        trailing_price = pos["max_close"] - pos["trailing_multiplier"] * _atr_val
                        wider_atr_stop = pos["max_close"] - max(pos["trailing_multiplier"] + 1.0, 4.0) * _atr_val
                        trend_stops = [wider_atr_stop]
                        if pd.notna(today_ma20) and today_ma20 > 0 and today_ma20 < today_close:
                            trend_stops.append(today_ma20)
                        trailing_price = max(min(trend_stops), _avg_ep_ts * 1.05)
                        pos["trailing_method"] = "ma20_atr_relaxed"
                        pos["trailing_stage"] = 2
                    elif _profit_ts >= 0.30:
                        # 階段二-B：趨勢非強，獲利 >= 30%，floor = entry × 1.15
                        trailing_price = pos["max_close"] - pos["trailing_multiplier"] * _atr_val
                        trailing_price = max(trailing_price, _avg_ep_ts * 1.15)
                        pos["trailing_method"] = "atr_profit_floor"
                        pos["trailing_stage"] = 2
                    else:
                        # 階段二-C：趨勢非強，20% <= profit < 30%，floor = entry × 1.10
                        trailing_price = pos["max_close"] - pos["trailing_multiplier"] * _atr_val
                        trailing_price = max(trailing_price, _avg_ep_ts * 1.10)
                        pos["trailing_method"] = "atr_profit_floor"
                        pos["trailing_stage"] = 2
                    pos["stop_loss"] = max(pos["stop_loss"], trailing_price)
                    pos["trailing_stop_activated"] = True

            layer_floor    = _layer_stop_loss(pos)
            effective_stop = max(layer_floor, pos["stop_loss"])

            # ── 停損三態判定：跳空停損 / 收盤確認停損 / 假跌破（不出場）──
            stop_ep, stop_kind = None, None
            if pd.notna(today_open) and today_open <= effective_stop * 0.99:
                stop_ep, stop_kind = today_open, "gap_down"
            elif (pd.notna(today_open) and today_open > effective_stop
                  and pd.notna(today_close) and today_close < effective_stop * 0.99):
                stop_ep, stop_kind = today_close, "close_confirm"
            elif (today_low <= effective_stop
                  and pd.notna(today_close) and today_close >= effective_stop * 0.99):
                pos["false_breakdown_count"] = pos.get("false_breakdown_count", 0) + 1

            if stop_ep is not None:
                if pos["phase1_done"]:
                    exit_trigger  = "trailing"
                    exit_reason   = "Trailing Stop（最終部位）"
                    stage_no = 3
                else:
                    layer_label  = f"層數 {pos['current_layer']}"
                    exit_trigger = stop_kind
                    exit_reason  = ("跳空停損" if stop_kind == "gap_down" else "收盤確認停損") + f"（{layer_label}）"
                    stage_no = 1

                if pos.get("trailing_stop_activated"):
                    _ts_stage = pos.get("trailing_stage", 0)
                    if _ts_stage == 1:
                        trailing_stop_stage1_count += 1
                    elif _ts_stage == 2:
                        trailing_stop_stage2_count += 1
                cash += pos["shares"] * stop_ep
                trade_log.append(_log(pos,
                    exit_date=current_date, exit_price=stop_ep, exit_reason=exit_reason,
                    exit_trigger=exit_trigger,
                    shares=pos["shares"],
                    pnl=pos["shares"] * (stop_ep - pos["entry_price"]),
                    holding_days=pos["holding_days"],
                    position_value=pos["shares"] * stop_ep,
                    stage=stage_no,
                    trailing_stop_price=(pos["stop_loss"] if pos["trailing_stop_activated"] else np.nan),
                    gap_down_exit=(stop_kind == "gap_down"),
                ))
                exited = True

            if not exited:
                avg_ep     = pos.get("avg_entry_price", pos["entry_price"])
                profit_now = (
                    (today_close - avg_ep) / avg_ep
                    if pd.notna(avg_ep) and avg_ep > 0 else 0.0
                )
                _ts_price = pos["stop_loss"] if pos.get("trailing_stop_activated") else np.nan

                # ── 動態獲利保護 50% → 全部剩餘部位出場 ─────────────
                if profit_now >= 0.50:
                    cash += pos["shares"] * today_close
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=today_close,
                        exit_reason="獲利 50% 全出",
                        exit_trigger="take_profit",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (today_close - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * today_close,
                        stage=(3 if pos["phase1_done"] else 1),
                        trailing_stop_price=_ts_price,
                        gap_down_exit=False,
                        profit_exit_flag=True,
                    ))
                    peak_profit_list.append(pos.get("peak_profit_pct", 0.0))
                    profit_50_count += 1
                    _set_profit_cooldown(sym, current_date)
                    exited = True

                # ── 條件二：動態獲利保護 30% → 出場 50% 剩餘部位（若尚未執行）
                elif profit_now >= 0.30 and not pos.get("profit_30_done", False):
                    exit_shares = pos["shares"] * 0.50
                    proceeds    = exit_shares * today_close
                    cash       += proceeds
                    trade_log.append(_log(pos, partial_exit=True,
                        exit_date=current_date, exit_price=today_close,
                        exit_reason="獲利 30% 分批出場",
                        exit_trigger="take_profit",
                        shares=exit_shares,
                        pnl=proceeds - exit_shares * pos["entry_price"],
                        holding_days=pos["holding_days"],
                        position_value=proceeds,
                        stage=(3 if pos["phase1_done"] else 2),
                        trailing_stop_price=_ts_price,
                        gap_down_exit=False,
                        profit_exit_flag=True,
                    ))
                    pos["shares"]        -= exit_shares
                    pos["cost"]           = pos["shares"] * pos["entry_price"]
                    pos["profit_30_done"] = True
                    if pd.notna(avg_ep) and avg_ep > 0:
                        pos["stop_loss"] = max(pos["stop_loss"], avg_ep * 1.15)
                    profit_30_count += 1
                    _set_profit_cooldown(sym, current_date)

                # ── TP1 觸發（第一止盈，出場 tp1_ratio；階段一）─────────────
                elif not pos["phase1_done"] and pd.notna(pos["take_profit_1"]) and today_high >= pos["take_profit_1"]:
                    tp1         = pos["take_profit_1"]
                    ratio       = pos["tp1_ratio"]
                    base_shares = pos["shares"]
                    exit_shares = base_shares * ratio
                    proceeds    = exit_shares * tp1
                    cash       += proceeds
                    trade_log.append(_log(pos, partial_exit=True,
                        exit_date=current_date, exit_price=tp1,
                        exit_reason=f"第一止盈（{round(ratio * 100)}% 分批）",
                        exit_trigger="take_profit",
                        shares=exit_shares,
                        pnl=proceeds - exit_shares * pos["entry_price"],
                        holding_days=pos["holding_days"],
                        position_value=proceeds,
                        stage=2, trailing_stop_price=np.nan,
                        gap_down_exit=False,
                        profit_exit_flag=True,
                    ))
                    pos["shares"]      -= exit_shares
                    pos["cost"]          = pos["shares"] * pos["entry_price"]
                    pos["shares_at_tp1"] = base_shares
                    pos["phase1_done"]   = True
                    pos["stop_loss"]     = max(pos["stop_loss"], pos["entry_price"])  # 保本

                # ── TP2 觸發（第二止盈，30% 分批；階段三）──────────────────
                elif (pos["phase1_done"]
                        and not pos.get("phase2_done", False)
                        and pd.notna(pos["take_profit_2"]) and today_high >= pos["take_profit_2"]):
                    tp2         = pos["take_profit_2"]
                    base_shares = pos.get("shares_at_tp1", pos["shares"])
                    exit_shares = min(base_shares * tp2_ratio, pos["shares"])
                    proceeds    = exit_shares * tp2
                    cash       += proceeds
                    trade_log.append(_log(pos, partial_exit=True,
                        exit_date=current_date, exit_price=tp2,
                        exit_reason="第二止盈（30% 分批）",
                        exit_trigger="take_profit",
                        shares=exit_shares,
                        pnl=proceeds - exit_shares * pos["entry_price"],
                        holding_days=pos["holding_days"],
                        position_value=proceeds,
                        stage=3,
                        trailing_stop_price=_ts_price,
                        gap_down_exit=False,
                        profit_exit_flag=True,
                    ))
                    pos["shares"]     -= exit_shares
                    pos["cost"]         = pos["shares"] * pos["entry_price"]
                    pos["phase2_done"] = True

                # ── 條件四：60 日期限到期 ─────────────────────────────────────
                elif pos["holding_days"] >= max_hold_days:
                    cash += pos["shares"] * today_close
                    trade_log.append(_log(pos,
                        exit_date=current_date, exit_price=today_close,
                        exit_reason="60日期限到期",
                        exit_trigger="time_exit",
                        shares=pos["shares"],
                        pnl=pos["shares"] * (today_close - pos["entry_price"]),
                        holding_days=pos["holding_days"],
                        position_value=pos["shares"] * today_close,
                        stage=(3 if pos["phase1_done"] else 1),
                        trailing_stop_price=_ts_price,
                        gap_down_exit=False,
                        profit_exit_flag=False,
                    ))
                    peak_profit_list.append(pos.get("peak_profit_pct", 0.0))
                    expired_count += 1
                    exited = True

            pos["prev_macd"] = _f(row, "MACD")

            if not exited:
                remaining.append(pos)

        positions = remaining
        # （cash 已因上述出場而更新 → 完成「釋放資金回到可用現金」）

        # ── 2. 加碼檢查（已持倉、尚未進入第一止盈分批階段的股票）──────
        def _exposure_snapshot():
            """目前（含已執行的加碼）持倉市值與總資產，供曝險/現金門檻判斷用。"""
            pv = sum(
                p["shares"] * _f(
                    day.loc[p["symbol"]] if p["symbol"] in day.index else pd.Series(),
                    "close", p["entry_price"],
                )
                for p in positions
            )
            return pv, cash + pv

        for pos in positions:
            sym = pos["symbol"]
            if sym not in day.index:
                continue
            row = day.loc[sym]
            today_close = _f(row, "close")
            if pd.isna(today_close) or today_close <= 0:
                continue

            portfolio_value_now, total_equity = _exposure_snapshot()
            if total_equity <= 0:
                continue
            current_exposure_pct = portfolio_value_now / total_equity

            if pos["current_layer"] == 1 and not pos["phase1_done"] and not pos["add1_triggered"]:
                zone_high     = pos.get("support_entry_zone_high")
                sqs           = _f(row, "support_quality_score")
                support_now   = _f(row, "nearest_support_level")
                support_entry = pos.get("entry_nearest_support_level")
                avg_price     = pos["avg_entry_price"]

                # return_20d daily rank 前段（add1_relaxed）時，第一次加碼不需要 support_quality_score >= 0.4
                quality_ok = bool(pos.get("add1_relaxed")) or bool(pd.notna(sqs) and sqs >= 0.4)
                cond_a = bool(
                    pd.notna(today_close) and pd.notna(avg_price) and avg_price
                    and today_close > avg_price * 0.98
                    and pd.notna(zone_high) and today_close <= zone_high * 1.01
                    and quality_ok
                )
                cond_b = bool(
                    pd.notna(support_now) and pd.notna(support_entry)
                    and support_now < support_entry
                    and today_close > support_now
                )

                add_value = add1_pct * total_equity
                position_value_now = pos["shares"] * today_close
                proposed_single_position_pct = (
                    (position_value_now + add_value) / total_equity
                    if total_equity else np.nan
                )
                proposed_single_cost_pct = (
                    (pos["total_cost"] + add_value) / total_equity
                    if total_equity else np.nan
                )
                proposed_cash_pct = (
                    (cash - add_value) / total_equity if total_equity else np.nan
                )

                if ((cond_a or cond_b)
                        and add_value > 0
                        and proposed_single_position_pct <= max_single_position_pct + allocation_eps
                        and proposed_single_cost_pct <= max_single_position_pct + allocation_eps
                        and current_exposure_pct + add1_pct <= max_total_exposure + allocation_eps
                        and proposed_cash_pct + allocation_eps >= min_cash_reserve):
                    if market_filter_active:
                        filtered_entries_count += 1
                    else:
                        add_shares = add_value / today_close
                        cash -= add_value
                        _add_layer(pos, layer_id=2, entry_date=current_date,
                                   entry_price=today_close, shares=add_shares, cost=add_value)
                        pos["add1_triggered"] = True

            elif pos["current_layer"] == 2 and not pos.get("phase2_done", False) and not pos["add2_triggered"]:
                tp1   = pos.get("take_profit_1")
                macd  = _f(row, "MACD")
                rsi14 = _f(row, "RSI14")
                ma20_distance = _f(row, "ma20_distance")
                ma60_distance = _f(row, "ma60_distance")
                ma20_slope = _f(row, "ma20_slope")
                close_to_20d_high = _f(row, "close_to_20d_high")
                add2_diagnostics["add2_candidate_count"] += 1

                avg_price = pos.get("avg_entry_price")
                profit_now = (
                    (today_close - avg_price) / avg_price
                    if pd.notna(avg_price) and avg_price > 0 else np.nan
                )
                defense_line = max(_layer_stop_loss(pos), pos.get("stop_loss", np.nan))
                price_ok = bool(pd.notna(profit_now) and profit_now > 0)
                holding_days_now = pos.get("holding_days", 0)
                score_bypass = bool(
                    pd.notna(profit_now) and profit_now >= 0.05
                    and holding_days_now >= 3
                )
                if score_bypass:
                    score_ok = True
                    add2_diagnostics["add2_score_bypassed"] += 1
                else:
                    score_ok = bool(
                        _return_20d_rank_top(row, return_20d_top_add_pct, fallback_score=0.60)
                        or _return_20d_rank_top(row, return_20d_top_strict_pct, fallback_score=0.65)
                    )
                    add2_diagnostics["add2_score_required"] += 1
                trend_ok = bool(
                    pd.notna(ma20_distance) and ma20_distance > 0
                    and pd.notna(ma60_distance) and ma60_distance > 0
                    and (pd.isna(ma20_slope) or ma20_slope >= 0)
                    and (pd.isna(close_to_20d_high) or close_to_20d_high >= -0.05)
                    and pd.notna(macd) and macd > 0
                    and (pd.isna(rsi14) or 45 <= rsi14 <= 78)
                )
                defense_ok = bool(
                    pd.notna(today_close)
                    and pd.notna(defense_line)
                    and today_close > defense_line
                )
                add_value = add2_pct * total_equity
                position_value_now = pos["shares"] * today_close
                proposed_single_position_pct = (
                    (position_value_now + add_value) / total_equity
                    if total_equity else np.nan
                )
                proposed_single_cost_pct = (
                    (pos["total_cost"] + add_value) / total_equity
                    if total_equity else np.nan
                )
                proposed_cash_pct = (
                    (cash - add_value) / total_equity if total_equity else np.nan
                )
                exposure_ok = bool(
                    add_value > 0
                    and proposed_single_position_pct <= max_single_position_pct + allocation_eps
                    and proposed_single_cost_pct <= max_single_position_pct + allocation_eps
                    and current_exposure_pct + add2_pct <= max_total_exposure + allocation_eps
                )
                cash_ok = bool(proposed_cash_pct + allocation_eps >= min_cash_reserve)

                if not price_ok:
                    add2_diagnostics["rejected_add2_by_price_condition"] += 1
                if price_ok and not score_ok:
                    add2_diagnostics["rejected_add2_by_score"] += 1
                if price_ok and score_ok and not trend_ok:
                    add2_diagnostics["rejected_add2_by_trend_condition"] += 1
                if price_ok and score_ok and trend_ok and not defense_ok:
                    add2_diagnostics["rejected_add2_by_defense_line"] += 1
                # 第二次加碼目前沒有 support_quality 條件；保留欄位供驗收時明確看到為 0。
                if price_ok and score_ok and trend_ok and defense_ok and not exposure_ok:
                    add2_diagnostics["rejected_add2_by_exposure"] += 1
                if price_ok and score_ok and trend_ok and defense_ok and exposure_ok and not cash_ok:
                    add2_diagnostics["rejected_add2_by_cash"] += 1

                if (price_ok and score_ok and trend_ok and defense_ok and exposure_ok and cash_ok):
                    if market_filter_active:
                        filtered_entries_count += 1
                    else:
                        add_shares = add_value / today_close
                        cash -= add_value
                        _add_layer(pos, layer_id=3, entry_date=current_date,
                                   entry_price=today_close, shares=add_shares, cost=add_value)
                        pos["add2_triggered"] = True
                        # 第二次加碼後停損改為保本（avg_entry_price）：
                        # 已由 _layer_stop_loss() 依 current_layer 動態反映，無需另外設定。

            elif pos["current_layer"] == 2 and pos["add2_triggered"]:
                add2_diagnostics["rejected_add2_by_existing_flag"] += 1

        # ── 3. 處理新進場（試倉）：依 final_decision_score 由高到低排序 ──
        held_symbols = {p["symbol"] for p in positions}

        if held_symbols:
            portfolio_value_now, total_equity = _exposure_snapshot()
            held_signal_rows = day[day.index.isin(held_symbols)]
            for sym, row in held_signal_rows.iterrows():
                is_entry_signal, _, _ = _entry_signal(row)
                if is_entry_signal:
                    _record_entry_candidate(
                        daily_diag=daily_entry_diag,
                        sym=sym,
                        row=row,
                        held_symbols=held_symbols,
                        positions_count=len(positions),
                        cash_value=cash,
                        portfolio_value_now=portfolio_value_now,
                        total_equity=total_equity,
                        market_filter_active=market_filter_active,
                    )

        candidates = day[~day.index.isin(held_symbols)].copy()

        if not candidates.empty:
            candidates["_score"] = pd.to_numeric(
                candidates.get("final_decision_score", 0.0), errors="coerce"
            ).fillna(0.0)
            candidates = candidates.sort_values("_score", ascending=False)

            for candidate_idx, (sym, row) in enumerate(candidates.iterrows()):
                if len(positions) >= max_positions:
                    portfolio_value_now, total_equity = _exposure_snapshot()
                    for rem_sym, rem_row in candidates.iloc[candidate_idx:].iterrows():
                        is_entry_signal, _, _ = _entry_signal(rem_row)
                        if is_entry_signal:
                            _record_entry_candidate(
                                daily_diag=daily_entry_diag,
                                sym=rem_sym,
                                row=rem_row,
                                held_symbols=held_symbols,
                                positions_count=len(positions),
                                cash_value=cash,
                                portfolio_value_now=portfolio_value_now,
                                total_equity=total_equity,
                                market_filter_active=market_filter_active,
                            )
                    break

                is_entry_signal, _, _ = _entry_signal(row)
                if is_entry_signal:
                    portfolio_value_now, total_equity = _exposure_snapshot()
                    _record_entry_candidate(
                        daily_diag=daily_entry_diag,
                        sym=sym,
                        row=row,
                        held_symbols=held_symbols,
                        positions_count=len(positions),
                        cash_value=cash,
                        portfolio_value_now=portfolio_value_now,
                        total_equity=total_equity,
                        market_filter_active=market_filter_active,
                    )

                score        = _f(row, "final_decision_score", 0.0)
                override_raw = row.get("entry_plan_override", False)
                try:
                    override = bool(override_raw) if pd.notna(override_raw) else False
                except (TypeError, ValueError):
                    override = False

                zone_low  = _f(row, "support_entry_zone_low")
                zone_high = _f(row, "support_entry_zone_high")
                close     = _f(row, "close")

                if not (override or score >= 0.55):
                    continue
                # 獲利出場冷卻期：期間內不接受同標的新進場
                if sym in profit_exit_cooldowns and current_date <= profit_exit_cooldowns[sym]:
                    continue
                volume_entry_reason = _volume_entry_reason(
                    row=row,
                    score=score,
                    close=close,
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
                support_entry_reason = _entry_support_reason(
                    row=row,
                    score=score,
                    close=close,
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
                strong_entry_reason = _strong_continuation_entry_reason(
                    row=row,
                    score=score,
                    close=close,
                )
                if strong_entry_reason is not None:
                    entry_type = "strong_continuation_entry"
                    entry_mode_used = "strong_continuation"
                    entry_reason = strong_entry_reason
                elif support_entry_reason == "near_support_relaxed":
                    entry_type = "relaxed_support_entry"
                    entry_mode_used = support_entry_reason
                    entry_reason = "relaxed_near_support"
                elif support_entry_reason == "strict_support":
                    entry_type = "support_entry"
                    entry_mode_used = support_entry_reason
                    entry_reason = "near_support"
                elif support_entry_reason == "support_proximity":
                    entry_type = "support_proximity_entry"
                    entry_mode_used = "support_proximity"
                    entry_reason = "support_proximity"
                elif support_entry_reason == "breakout_entry":
                    entry_type = "breakout_entry"
                    entry_mode_used = "breakout_entry"
                    entry_reason = "breakout_entry"
                else:
                    continue

                entry_rr_1      = _f(row, "entry_rr_1")
                stop_loss_price = _f(row, "entry_stop_loss_price")
                stop_loss_pct   = _f(row, "entry_stop_loss_pct")

                if pd.isna(entry_rr_1) or entry_rr_1 < 1.5:
                    continue
                if pd.isna(stop_loss_price) or stop_loss_price <= 0:
                    continue
                if pd.isna(stop_loss_pct) or stop_loss_pct < 0.012 or stop_loss_pct > 0.08:
                    continue
                if pd.isna(close) or close <= 0:
                    continue

                tp1           = _f(row, "entry_take_profit_1")
                tp2           = _f(row, "entry_take_profit_2")
                atr_val       = _f(row, "atr_14", 0.0)
                support_level = _f(row, "nearest_support_level")

                portfolio_value_now = sum(
                    p["shares"] * _f(
                        day.loc[p["symbol"]] if p["symbol"] in day.index else pd.Series(),
                        "close", p["entry_price"],
                    )
                    for p in positions
                )
                total_equity = cash + portfolio_value_now
                if total_equity <= 0:
                    continue

                current_exposure_pct = portfolio_value_now / total_equity
                position_value = initial_pct * total_equity
                proposed_cash_pct = (
                    (cash - position_value) / total_equity if total_equity else np.nan
                )

                if initial_pct > max_single_position_pct + allocation_eps:
                    continue
                if current_exposure_pct + initial_pct > max_total_exposure + allocation_eps:
                    continue
                if proposed_cash_pct + allocation_eps < min_cash_reserve:
                    continue

                if position_value <= 0:
                    continue

                if market_filter_active:
                    # 大盤跌破 20MA：暫停所有新進場（試倉），僅計入潛在進場次數
                    filtered_entries_count += 1
                    continue

                shares = position_value / close
                cash  -= position_value

                entry_counter += 1
                entry_date_str = pd.Timestamp(current_date).strftime("%Y-%m-%d")
                trade_id = f"{sym}_{entry_date_str}_{round(float(close), 4)}"
                _is_re_entry = sym in profit_exited_symbols
                if _is_re_entry:
                    re_entry_count += 1

                # ── return_20d_score（item5）：決定 trailing 倍數 / 第一止盈比例 / 加碼1 寬鬆度
                entry_r20d = _entry_return_20d_score(row)
                entry_r20d_missing_filled = _entry_return_20d_score_missing(row)
                entry_r20d_daily_rank = _entry_return_20d_daily_rank(row)
                if _return_20d_rank_top(row, return_20d_top_entry_pct, fallback_score=0.60):
                    trailing_multiplier = high_trailing_multiplier
                    tp1_ratio            = high_tp1_ratio
                    add1_relaxed          = True
                elif (
                    (pd.notna(entry_r20d_daily_rank) and entry_r20d_daily_rank < 0.40)
                    or (pd.isna(entry_r20d_daily_rank) and pd.notna(entry_r20d) and entry_r20d < 0.40)
                ):
                    trailing_multiplier = low_trailing_multiplier
                    tp1_ratio            = base_tp1_ratio
                    add1_relaxed          = False
                else:
                    trailing_multiplier = base_trailing_multiplier
                    tp1_ratio            = base_tp1_ratio
                    add1_relaxed          = False

                pos = {
                    "trade_id":                        trade_id,
                    "entry_id":                         entry_counter,
                    "entry_layer_id":                   1,
                    "symbol":                           str(sym),
                    "entry_date":                       current_date,
                    "entry_stop_loss_price_original":   stop_loss_price,
                    "actual_stop_loss_price":           stop_loss_price,
                    "actual_stop_loss_pct":              stop_loss_pct,
                    "stop_adjusted":                     False,
                    "take_profit_1":                     tp1,
                    "take_profit_2":                     tp2,
                    "phase1_done":                        False,
                    "phase2_done":                        False,
                    "stop_loss":                          stop_loss_price,
                    "max_close":                          close,
                    "holding_days":                       0,
                    "atr_at_entry":                       atr_val,
                    "support_entry_zone_low":             zone_low,
                    "support_entry_zone_high":            zone_high,
                    "entry_decision_score":               score,
                    "entry_type":                         entry_type,
                    "entry_mode_used":                    entry_mode_used,
                    "entry_reason":                       entry_reason,
                    "volume_entry_reason":                volume_entry_reason,
                    "entry_nearest_support_level":        support_level,
                    "position_layers":                    [],
                    "current_layer":                      0,
                    "total_shares":                       0.0,
                    "total_cost":                          0.0,
                    "avg_entry_price":                    np.nan,
                    "actual_entry_price":                 np.nan,
                    "buy_cost_per_share":                 np.nan,
                    "add1_triggered":                     False,
                    "add2_triggered":                     False,
                    "entry_return_20d_score":             entry_r20d,
                    "entry_return_20d_score_daily_rank":  entry_r20d_daily_rank,
                    "return_20d_score_missing_filled":     entry_r20d_missing_filled,
                    "trailing_multiplier":                trailing_multiplier,
                    "trailing_method":                    "none",
                    "tp1_ratio":                           tp1_ratio,
                    "add1_relaxed":                        add1_relaxed,
                    "false_breakdown_count":               0,
                    "trailing_stop_activated":             False,
                    "post_cooldown_entry":                 False,
                    "same_stock_add_flag":                 False,
                    "post_partial_exit_entry":             False,
                    "shares_at_tp1":                       np.nan,
                    "days_above_tp1":                      0,
                    "profit_30_done":                      False,
                    "peak_profit_pct":                     0.0,
                    "prev_macd":                           np.nan,
                    "re_entry_after_profit":               _is_re_entry,
                }
                _add_layer(pos, layer_id=1, entry_date=current_date,
                           entry_price=close, shares=shares, cost=position_value)

                positions.append(pos)
                _bump_entry_diagnostic(daily_entry_diag, "executed_entries")
                if entry_type == "support_entry":
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_support_entry")
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_strict_support")
                    _bump_entry_diagnostic(daily_entry_diag, "zone_entry_count")
                elif entry_type == "relaxed_support_entry":
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_relaxed_support_entry")
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_near_support_relaxed")
                elif entry_type == "strong_continuation_entry":
                    _bump_entry_diagnostic(daily_entry_diag, "accepted_strong_continuation_entries")
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_strong_continuation")
                elif entry_type == "support_proximity_entry":
                    _bump_entry_diagnostic(daily_entry_diag, "support_proximity_count")
                elif entry_type == "breakout_entry":
                    _bump_entry_diagnostic(daily_entry_diag, "breakout_entry_count")

                if volume_entry_reason != "none":
                    if entry_type in ("support_entry", "relaxed_support_entry"):
                        _bump_entry_diagnostic(daily_entry_diag, "volume_confirmed_support_entry")
                    elif entry_type == "strong_continuation_entry":
                        _bump_entry_diagnostic(daily_entry_diag, "volume_confirmed_strong_entry")

                if volume_entry_reason == "quiet_pullback_support":
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_quiet_pullback_support")
                elif volume_entry_reason == "volume_rebound_support":
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_volume_rebound_support")
                elif volume_entry_reason == "volume_breakout_momentum":
                    _bump_entry_diagnostic(daily_entry_diag, "executed_by_volume_breakout_momentum")

        # ── 4. 計算當日總資產 ─────────────────────────────────────────
        pos_value = 0.0
        for pos in positions:
            sym = pos["symbol"]
            px = _f(
                day.loc[sym] if sym in day.index else pd.Series(),
                "close", pos["entry_price"],
            )
            pos_value += pos["shares"] * px

        total_equity_today = cash + pos_value
        daily_return = (
            (total_equity_today - prev_equity) / prev_equity if prev_equity else np.nan
        )
        prev_equity = total_equity_today

        equity_rows.append({
            "date":             current_date,
            "total_equity":     total_equity_today,
            "cash":             cash,
            "portfolio_value":  pos_value,
            "daily_return":     daily_return,
            "num_positions":    len(positions),
            "positions":        [p["symbol"] for p in positions],
            "market_filter_active": market_filter_active,
            "market_close":          market_close_today,
            "market_ma20":           market_ma20_today,
        })
        daily_candidates = daily_entry_diag["candidate_entry_signals"]
        daily_volume_confirmed = daily_entry_diag["candidate_with_volume_confirmation"]
        daily_entry_diag["volume_confirmation_rate"] = (
            daily_volume_confirmed / daily_candidates if daily_candidates else np.nan
        )
        entry_diagnostics_daily_rows.append(daily_entry_diag)

    equity_curve = pd.DataFrame(equity_rows)
    trade_log_df = pd.DataFrame(trade_log)
    if "return_20d_score_missing_filled" in trade_log_df.columns and "trade_id" in trade_log_df.columns:
        filled_by_trade = (
            trade_log_df.groupby("trade_id", sort=False)["return_20d_score_missing_filled"]
            .first()
            .fillna(False)
            .astype(bool)
        )
        return_20d_score_stats["entry_missing_filled_count"] = int(filled_by_trade.sum())
    else:
        return_20d_score_stats["entry_missing_filled_count"] = 0
    return_20d_score_stats["missing_fill_value"] = 0.5
    return_20d_score_stats["missing_fill_reason"] = (
        "return_20d_score 欄位缺失、值為 NaN、無法轉成數值，或 use_return_20d=False 時，"
        "進場端以中性值 0.5 處理。"
    )
    metrics = calculate_portfolio_metrics(equity_curve, trade_log_df, initial_capital)
    metrics["filtered_entries"] = filtered_entries_count
    metrics["return_20d_score_stats"] = return_20d_score_stats
    metrics["strong_continuation_column_check"] = strong_continuation_column_check
    metrics["add2_diagnostics"] = dict(add2_diagnostics)
    metrics["rejection_future_return_summary"] = _build_rejection_future_return_summary(
        rejection_future_return_rows,
        df,
    )
    trailing_detail, trailing_summary = _build_trailing_stop_future_return_diagnostics(
        trade_log_df,
        df,
    )
    metrics["trailing_stop_future_return_detail"] = trailing_detail
    metrics["trailing_stop_future_return_summary"] = trailing_summary
    metrics["profit_30_count"]            = profit_30_count
    metrics["profit_50_count"]            = profit_50_count
    metrics["trend_break_count"]          = trend_break_count
    metrics["expired_count"]              = expired_count
    metrics["re_entry_count"]             = re_entry_count
    metrics["avg_peak_profit"]            = float(np.mean(peak_profit_list)) if peak_profit_list else np.nan
    metrics["trailing_stop_stage1_count"] = trailing_stop_stage1_count
    metrics["trailing_stop_stage2_count"] = trailing_stop_stage2_count
    required_trade_log_columns = [
        "layer_id",
        "entry_layer_id",
        "exit_trigger",
        "false_breakdown_count",
        "holding_days_at_exit",
        "trailing_stop_activated",
        "trailing_multiplier_used",
        "trailing_method",
        "entry_type",
        "entry_reason",
        "return_20d_score",
        "return_20d_score_daily_rank",
        "post_cooldown_entry",
        "same_stock_add_flag",
        "post_partial_exit_entry",
        "profit_exit_flag",
        "re_entry_after_profit",
        "peak_profit_pct",
    ]
    metrics["required_trade_log_columns"] = required_trade_log_columns
    metrics["missing_trade_log_columns"] = [
        col for col in required_trade_log_columns if col not in trade_log_df.columns
    ]
    candidate_entries = entry_diagnostics["candidate_entry_signals"]
    executed_entries = entry_diagnostics["executed_entries"]
    entry_diagnostics_summary = dict(entry_diagnostics)
    entry_diagnostics_summary["entry_execution_rate"] = (
        executed_entries / candidate_entries if candidate_entries else np.nan
    )
    entry_diagnostics_summary["volume_confirmation_rate"] = (
        entry_diagnostics["candidate_with_volume_confirmation"] / candidate_entries
        if candidate_entries else np.nan
    )
    metrics["entry_diagnostics"] = entry_diagnostics_summary
    metrics["entry_primary_rejection_counts"] = dict(entry_primary_rejection_counts)
    metrics["primary_rejection_reason"] = dict(entry_primary_rejection_counts)
    metrics["entry_diagnostics_daily"] = pd.DataFrame(entry_diagnostics_daily_rows)

    # ── 持倉快照（回測結束時仍未平倉的持倉）─────────────────────────
    if save_snapshot and positions:
        from pathlib import Path
        _cv = getattr(config, "backtest_profile_name",
                      getattr(config, "backtest_portfolio_config_version", "portfolio"))
        _ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        _rows = []
        for pos in positions:
            _ep = pos.get("avg_entry_price", pos.get("entry_price", np.nan))
            _mc = pos.get("max_close", np.nan)
            _upct = (_mc - _ep) / _ep if (pd.notna(_ep) and _ep > 0 and pd.notna(_mc)) else np.nan
            _rows.append({
                "symbol":          pos.get("symbol"),
                "entry_date":      pos.get("entry_date"),
                "avg_entry_price": _ep,
                "max_close":       _mc,
                "unrealized_pct":  _upct,
                "holding_days":    pos.get("holding_days", 0),
                "stop_loss":       pos.get("stop_loss"),
                "trailing_method": pos.get("trailing_method", "none"),
                "trailing_stage":  pos.get("trailing_stage", 0),
            })
        _snap_dir = Path(getattr(config, "output_dir", "output"))
        _snap_dir.mkdir(exist_ok=True)
        _holdings_path = _snap_dir / f"current_holdings_{_cv}_{_ts}.csv"
        pd.DataFrame(_rows).to_csv(_holdings_path, index=False, encoding="utf-8-sig")
        print(f"持倉快照已輸出：{_holdings_path}")

    return equity_curve, trade_log_df, metrics


def calculate_portfolio_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """
    計算 Portfolio 層級績效指標（以每日總資產為基礎，反映真實資金限制下的整體表現）。
    """
    if equity_curve.empty:
        return {}

    final_equity = float(equity_curve["total_equity"].iloc[-1])
    total_return = (final_equity - initial_capital) / initial_capital

    first_date = pd.to_datetime(equity_curve["date"].iloc[0])
    last_date  = pd.to_datetime(equity_curve["date"].iloc[-1])
    n_days     = max((last_date - first_date).days, 1)
    cagr = (final_equity / initial_capital) ** (365.0 / n_days) - 1.0

    equity_vals  = equity_curve["total_equity"].to_numpy(dtype=float)
    rolling_max  = np.maximum.accumulate(equity_vals)
    drawdowns    = (equity_vals - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    daily_returns = pd.to_numeric(
        equity_curve.get("daily_return", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if daily_returns.empty:
        daily_returns = pd.Series(equity_vals).pct_change().dropna()

    mean_daily = float(daily_returns.mean()) if not daily_returns.empty else 0.0
    std_daily  = float(daily_returns.std())  if len(daily_returns) > 1 else 0.0
    annualized_mean_return = mean_daily * 252
    sharpe_ratio = (
        (annualized_mean_return - 0.015) / (std_daily * np.sqrt(252))
        if std_daily > 0 else 0.0
    )

    calmar_ratio = (cagr / abs(max_drawdown)) if max_drawdown != 0 else np.nan

    win_days  = int((daily_returns > 0).sum())
    loss_days = int((daily_returns < 0).sum())

    avg_positions = float(
        pd.to_numeric(equity_curve.get("num_positions", pd.Series(dtype=float)), errors="coerce").mean()
    )
    avg_portfolio_value = float(
        pd.to_numeric(equity_curve.get("portfolio_value", pd.Series(dtype=float)), errors="coerce").mean()
    )
    capital_utilization = (avg_portfolio_value / initial_capital) if initial_capital else np.nan

    # ── 加碼統計（以 trade_id 為單位，取每筆交易最後一筆出場記錄的最終狀態）──
    total_entries = 0
    add1_count = 0
    add2_count = 0
    avg_final_position = np.nan
    if trade_log is not None and not trade_log.empty and "trade_id" in trade_log.columns:
        tl = trade_log.copy()
        if "exit_sequence" in tl.columns:
            tl = tl.sort_values("exit_sequence")
        last_rows = tl.groupby("trade_id", sort=False).tail(1)

        total_entries = int(last_rows["trade_id"].nunique())
        if total_entries > 0:
            add1_count = int(
                last_rows.get("add1_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
            )
            add2_count = int(
                last_rows.get("add2_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
            )
            avg_final_position = float(
                pd.to_numeric(last_rows.get("total_invested_pct", pd.Series(dtype=float)), errors="coerce").mean()
            )

    # ── return_20d_score 分組統計（優先使用每日橫斷面排名）────
    high_return_20d_count      = 0
    high_return_20d_avg_return = np.nan
    low_return_20d_avg_return  = np.nan
    if (trade_log is not None and not trade_log.empty
            and "trade_id" in trade_log.columns and "entry_return_20d_score" in trade_log.columns):
        tl2 = trade_log.copy()
        if "exit_sequence" in tl2.columns:
            tl2 = tl2.sort_values("exit_sequence")
        last_rows_20d = tl2.groupby("trade_id", sort=False).tail(1)

        scores_20d = pd.to_numeric(last_rows_20d["entry_return_20d_score"], errors="coerce")
        ranks_20d = pd.to_numeric(
            last_rows_20d.get("entry_return_20d_score_daily_rank", pd.Series(np.nan, index=last_rows_20d.index)),
            errors="coerce",
        )
        rets_20d   = pd.to_numeric(last_rows_20d.get("return_pct", pd.Series(dtype=float)), errors="coerce")
        high_mask  = ranks_20d >= 0.70
        low_mask   = ranks_20d < 0.40
        high_mask = high_mask.where(ranks_20d.notna(), scores_20d >= 0.60)
        low_mask = low_mask.where(ranks_20d.notna(), scores_20d < 0.40)

        high_return_20d_count = int(high_mask.sum())
        if high_mask.any():
            high_return_20d_avg_return = float(rets_20d[high_mask].mean())
        if low_mask.any():
            low_return_20d_avg_return = float(rets_20d[low_mask].mean())

    if "portfolio_value" in equity_curve.columns and "total_equity" in equity_curve.columns:
        exposure_series = (
            pd.to_numeric(equity_curve["portfolio_value"], errors="coerce")
            / pd.to_numeric(equity_curve["total_equity"], errors="coerce")
        )
        avg_exposure = float(exposure_series.mean()) if not exposure_series.empty else np.nan
    else:
        avg_exposure = np.nan

    # ── 大盤過濾統計 ───────────────────────────────────────────────
    total_days = len(equity_curve)
    if "market_filter_active" in equity_curve.columns:
        filter_days = int(
            equity_curve["market_filter_active"].fillna(False).astype(bool).sum()
        )
    else:
        filter_days = 0

    same_stock_add_count = 0
    post_cooldown_entries = 0
    post_partial_exit_entries = 0
    false_breakdown_count = 0
    gap_down_stop_count = 0
    close_confirm_stop_count = 0
    trailing_stop_exit_count = 0
    cooldown_triggered_count = 0
    cooldown_rejected_count = 0
    if trade_log is not None and not trade_log.empty:
        tl3 = trade_log.copy()
        if "same_stock_add_flag" in tl3.columns:
            same_stock_add_count = int(tl3["same_stock_add_flag"].fillna(False).astype(bool).sum())
        if "post_cooldown_entry" in tl3.columns:
            post_cooldown_entries = int(tl3["post_cooldown_entry"].fillna(False).astype(bool).sum())
        if "post_partial_exit_entry" in tl3.columns:
            post_partial_exit_entries = int(tl3["post_partial_exit_entry"].fillna(False).astype(bool).sum())
        if "trade_id" in tl3.columns and "false_breakdown_count" in tl3.columns:
            if "exit_sequence" in tl3.columns:
                tl3 = tl3.sort_values("exit_sequence")
            final_by_trade = tl3.groupby("trade_id", sort=False).tail(1)
            false_breakdown_count = int(
                pd.to_numeric(final_by_trade["false_breakdown_count"], errors="coerce").fillna(0).sum()
            )
        if "exit_trigger" in trade_log.columns:
            triggers = trade_log["exit_trigger"].astype(str)
            gap_down_stop_count = int((triggers == "gap_down").sum())
            close_confirm_stop_count = int((triggers == "close_confirm").sum())
            trailing_stop_exit_count = int((triggers == "trailing").sum())
            # Cooldown state machine is not yet wired. This value counts stop exits that would trigger cooldown.
            cooldown_triggered_count = gap_down_stop_count + close_confirm_stop_count

    return {
        "final_equity":         final_equity,
        "total_return":         total_return,
        "cagr":                 cagr,
        "max_drawdown":         max_drawdown,
        "sharpe_ratio":         sharpe_ratio,
        "calmar_ratio":         calmar_ratio,
        "win_days":             win_days,
        "loss_days":            loss_days,
        "avg_positions":        avg_positions,
        "capital_utilization":  capital_utilization,
        "total_entries":        total_entries,
        "add1_count":           add1_count,
        "add2_count":           add2_count,
        "avg_final_position":   avg_final_position,
        "avg_exposure":         avg_exposure,
        "total_days":           total_days,
        "filter_days":          filter_days,
        "high_return_20d_count":      high_return_20d_count,
        "high_return_20d_avg_return": high_return_20d_avg_return,
        "low_return_20d_avg_return":  low_return_20d_avg_return,
        "same_stock_add_count":       same_stock_add_count,
        "cooldown_triggered_count":   cooldown_triggered_count,
        "cooldown_rejected_count":    cooldown_rejected_count,
        "post_cooldown_entries":      post_cooldown_entries,
        "post_partial_exit_entries":  post_partial_exit_entries,
        "false_breakdown_count":      false_breakdown_count,
        "gap_down_stop_count":        gap_down_stop_count,
        "close_confirm_stop_count":   close_confirm_stop_count,
        "trailing_stop_exit_count":   trailing_stop_exit_count,
    }




__all__ = [name for name in globals() if not name.startswith('__')]
