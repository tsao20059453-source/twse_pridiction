from __future__ import annotations

from pipeline_risk import *

# ============================================================
# 5. Target 建立與模型訓練
# ============================================================

def add_risk_reward_target_score(
    df: pd.DataFrame,
    horizon: int = 10,
    risk_penalty: float = 1.0,
    opportunity_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    建立 target_score、target_score_v2 與 target_score_v3。

    target_score 並非未來報酬率，也不是上漲機率。
    本研究將 target_score 定義為未來 10 個交易日內的風險報酬結構分數。
    future_upside_10d 代表從當前收盤價到未來 10 日最高價的潛在上行空間。
    future_downside_10d 代表從當前收盤價到未來 10 日最低價的潛在下行風險。
    target_score 越接近 1，代表上行空間相對大於下行風險；
    target_score 越接近 0，代表下行風險相對較高。

    target_score_v2 是在原本風險報酬比例的基礎上，加入 opportunity_score。
    其目的是避免 1% upside / 1% downside 與 10% upside / 10% downside
    被視為相同投資吸引力。因此 target_score_v2 同時考慮：
    1. 上行空間相對於下行風險的比例
    2. 上行空間本身是否具有足夠投資機會

    target_score_v3 是舊版相對排名對照 target：
    future_return_10d + future_drawdown_10d 後，再轉成 percentile rank。
    """
    out = df.copy().sort_values("date").reset_index(drop=True)

    out[f"future_max_high_{horizon}d"] = (
        out["high"]
        .shift(-1)
        .rolling(window=horizon, min_periods=horizon)
        .max()
        .shift(-(horizon - 1))
    )

    out[f"future_min_low_{horizon}d"] = (
        out["low"]
        .shift(-1)
        .rolling(window=horizon, min_periods=horizon)
        .min()
        .shift(-(horizon - 1))
    )

    # 下列欄位只作為歷史訓練 target，不可當作當下特徵。
    out[f"future_upside_{horizon}d"] = (
        out[f"future_max_high_{horizon}d"] / out["close"] - 1
    ).clip(lower=0)
    out[f"future_downside_{horizon}d"] = (
        1 - out[f"future_min_low_{horizon}d"] / out["close"]
    ).clip(lower=0)

    denominator = out[f"future_upside_{horizon}d"] + risk_penalty * out[f"future_downside_{horizon}d"]
    reward_risk_score = out[f"future_upside_{horizon}d"] / denominator

    out["target_score"] = reward_risk_score
    out.loc[denominator <= 0, "target_score"] = np.nan
    out["target_score"] = out["target_score"].clip(0, 1)

    opportunity_score = (out[f"future_upside_{horizon}d"] / opportunity_threshold).clip(0, 1)
    out["target_score_v2"] = (out["target_score"] * opportunity_score).clip(0, 1)

    out[f"future_close_{horizon}d"] = out["close"].shift(-horizon)
    out[f"future_return_{horizon}d"] = out[f"future_close_{horizon}d"] / out["close"] - 1
    out[f"future_drawdown_{horizon}d"] = out[f"future_min_low_{horizon}d"] / out["close"] - 1
    out[f"risk_adjusted_return_{horizon}d"] = (
        out[f"future_return_{horizon}d"] + out[f"future_drawdown_{horizon}d"]
    )
    out[f"target_score_raw_{horizon}d"] = (
        out[f"future_return_{horizon}d"] + out[f"future_drawdown_{horizon}d"]
    )

    if "symbol" in out.columns and out["symbol"].nunique(dropna=True) > 1:
        out["target_score_v3"] = (
            out.groupby("date")[f"risk_adjusted_return_{horizon}d"]
            .rank(pct=True)
        )
    else:
        out["target_score_v3"] = out[f"risk_adjusted_return_{horizon}d"].rank(pct=True)

    out["target_score_v3"] = out["target_score_v3"].clip(0, 1)

    # 保留舊欄位名稱，讓既有輸出與報表流程可延續；正式主模型 target 改為 target_score_v2。
    out["investment_score"] = out["target_score_v3"]

    out = add_low_subjective_targets(out, horizon=horizon)
    return out.replace([np.inf, -np.inf], np.nan)


def percentile_safety_rank(s: pd.Series) -> pd.Series:
    valid = pd.to_numeric(s, errors="coerce")
    n = valid.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=s.index)
    rank = valid.rank(method="average", ascending=True)
    pct = (rank - 1) / (n - 1)
    return 1 - pct


def add_low_subjective_targets(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    out = df.copy()
    future_return_col = f"future_return_{horizon}d"
    future_drawdown_col = f"future_drawdown_{horizon}d"
    required_cols = [future_return_col, future_drawdown_col]
    missing_sources = [col for col in required_cols if col not in out.columns]
    if missing_sources:
        print(f"WARNING: cannot build low-subjective targets; missing columns: {missing_sources}")
        return out.replace([np.inf, -np.inf], np.nan)

    future_return = pd.to_numeric(out[future_return_col], errors="coerce")
    future_drawdown = pd.to_numeric(out[future_drawdown_col], errors="coerce")
    downside_risk = future_drawdown.abs()

    out[f"target_return_{horizon}d"] = future_return
    out[f"target_downside_risk_{horizon}d"] = downside_risk
    target_downside_risk_col = f"target_downside_risk_{horizon}d"
    target_abs_drawdown_safety_col = f"target_abs_drawdown_safety_{horizon}d"
    if "symbol" in out.columns:
        out[target_abs_drawdown_safety_col] = (
            out.groupby("symbol")[target_downside_risk_col].transform(percentile_safety_rank)
        )
    else:
        print("WARNING: symbol column missing; absolute drawdown safety target uses global percentile fallback.")
        out[target_abs_drawdown_safety_col] = percentile_safety_rank(out[target_downside_risk_col])

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        eligible = None
        if "historical_universe_eligible" in out.columns:
            eligible = out["historical_universe_eligible"].fillna(False).astype(bool)
        out[f"target_return_{horizon}d_cs_rank"] = (
            out[f"target_return_{horizon}d"].where(eligible).groupby(out["date"]).rank(pct=True)
            if eligible is not None
            else out.groupby("date")[f"target_return_{horizon}d"].rank(pct=True)
        )
        out[f"target_downside_risk_{horizon}d_cs_rank"] = (
            out[f"target_downside_risk_{horizon}d"].where(eligible).groupby(out["date"]).rank(pct=True)
            if eligible is not None
            else out.groupby("date")[f"target_downside_risk_{horizon}d"].rank(pct=True)
        )
    else:
        print("WARNING: date column missing; low-subjective cross-sectional target ranks use global rank fallback.")
        out[f"target_return_{horizon}d_cs_rank"] = out[f"target_return_{horizon}d"].rank(pct=True)
        out[f"target_downside_risk_{horizon}d_cs_rank"] = out[f"target_downside_risk_{horizon}d"].rank(pct=True)

    out[f"target_safety_{horizon}d_cs_rank"] = 1 - out[f"target_downside_risk_{horizon}d_cs_rank"]

    rar = future_return / (downside_risk + 1e-6)
    rar = rar.replace([np.inf, -np.inf], np.nan)
    valid_rar = rar.dropna()
    if not valid_rar.empty:
        lower = valid_rar.quantile(0.01)
        upper = valid_rar.quantile(0.99)
        rar = rar.clip(lower=lower, upper=upper)
    out[f"target_rar_{horizon}d"] = rar

    if "date" in out.columns:
        out[f"target_rar_{horizon}d_cs_rank"] = (
            out[f"target_rar_{horizon}d"].where(eligible).groupby(out["date"]).rank(pct=True)
            if "historical_universe_eligible" in out.columns
            else out.groupby("date")[f"target_rar_{horizon}d"].rank(pct=True)
        )
    else:
        out[f"target_rar_{horizon}d_cs_rank"] = out[f"target_rar_{horizon}d"].rank(pct=True)

    return out.replace([np.inf, -np.inf], np.nan)


def add_investment_score_target(
    df: pd.DataFrame,
    horizon: int = 10,
    risk_penalty: float = 1.0,
    opportunity_threshold: float = 0.05,
) -> pd.DataFrame:
    return add_risk_reward_target_score(
        df=df,
        horizon=horizon,
        risk_penalty=risk_penalty,
        opportunity_threshold=opportunity_threshold,
    )


def add_realized_history_features(
    df: pd.DataFrame,
    horizon: int = 10,
    lag: int = 10,
    report: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    sort_cols = ["symbol", "date"] if "symbol" in out.columns else ["date"]
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    group_key = "symbol"
    temp_group_col = None
    if "symbol" not in out.columns:
        print("WARNING: realized history features cannot group by symbol because symbol column is missing; using single-series lag.")
        temp_group_col = "__realized_history_group__"
        out[temp_group_col] = "single"
        group_key = temp_group_col

    source_cols = {
        "target_score_v2": "target_score_v2",
        "future_return": f"future_return_{horizon}d",
        "future_drawdown": f"future_drawdown_{horizon}d",
        "risk_adjusted_return": f"risk_adjusted_return_{horizon}d",
    }
    missing_sources = [col for col in source_cols.values() if col not in out.columns]

    def grouped_transform(source_col: str, func):
        values = pd.to_numeric(out[source_col], errors="coerce")
        return values.groupby(out[group_key], sort=False).transform(func)

    if source_cols["target_score_v2"] in out.columns:
        score_col = source_cols["target_score_v2"]
        out["realized_score_v2_lag10"] = grouped_transform(score_col, lambda s: s.shift(lag))
        out["realized_score_v2_lag20"] = grouped_transform(score_col, lambda s: s.shift(20))
        out["realized_score_v2_roll20_lag10"] = grouped_transform(
            score_col,
            lambda s: s.shift(lag).rolling(20, min_periods=20).mean(),
        )
        out["realized_score_v2_roll60_lag10"] = grouped_transform(
            score_col,
            lambda s: s.shift(lag).rolling(60, min_periods=60).mean(),
        )
        out["realized_score_v2_std20_lag10"] = grouped_transform(
            score_col,
            lambda s: s.shift(lag).rolling(20, min_periods=20).std(),
        )

    if source_cols["future_return"] in out.columns:
        return_col = source_cols["future_return"]
        out["realized_return_10d_lag10"] = grouped_transform(return_col, lambda s: s.shift(lag))
        out["realized_return_10d_roll20_lag10"] = grouped_transform(
            return_col,
            lambda s: s.shift(lag).rolling(20, min_periods=20).mean(),
        )
        out["realized_win_rate_20_lag10"] = (
            pd.to_numeric(out[return_col], errors="coerce")
            .groupby(out[group_key], sort=False)
            .transform(
                lambda s: pd.Series(
                    np.where(s.notna(), (s > 0).astype(float), np.nan),
                    index=s.index,
                ).shift(lag).rolling(20, min_periods=20).mean()
            )
        )

    if source_cols["future_drawdown"] in out.columns:
        drawdown_col = source_cols["future_drawdown"]
        out["realized_drawdown_10d_roll20_lag10"] = grouped_transform(
            drawdown_col,
            lambda s: s.shift(lag).rolling(20, min_periods=20).mean(),
        )

    if source_cols["risk_adjusted_return"] in out.columns:
        risk_adjusted_col = source_cols["risk_adjusted_return"]
        out["realized_risk_adjusted_return_roll20_lag10"] = grouped_transform(
            risk_adjusted_col,
            lambda s: s.shift(lag).rolling(20, min_periods=20).mean(),
        )

    if temp_group_col is not None:
        out = out.drop(columns=[temp_group_col])

    out = out.replace([np.inf, -np.inf], np.nan)

    if report:
        present_features = [col for col in REALIZED_HISTORY_FEATURES if col in out.columns]
        missing_features = [col for col in REALIZED_HISTORY_FEATURES if col not in out.columns]
        print("\n" + "=" * 70)
        print("realized history feature availability check")
        print("=" * 70)
        if missing_sources:
            print(f"WARNING: missing realized history source columns: {missing_sources}")
        print(f"realized history feature count: {len(present_features)} / {len(REALIZED_HISTORY_FEATURES)}")
        print(f"missing realized history feature list: {missing_features}")

    return out


def get_feature_columns(df: pd.DataFrame, feature_set: str = "full") -> tuple[list[str], list[str]]:
    cs_rank_features = [
        "return_5d_cs_rank",
        "return_10d_cs_rank",
        "ma20_distance_cs_rank",
        "ma60_distance_cs_rank",
        "volume_ratio_20_cs_rank",
        "VR_14_cs_rank",
        "VR_26_cs_rank",
        "VR_14_change_cs_rank",
        "RSI14_cs_rank",
        "MACD_cs_rank",
        "K_cs_rank",
        "BIAS24_cs_rank",
        "atr_pct_cs_rank",
        "volatility_10d_cs_rank",
        "fib_position_cs_rank",
        "support_distance_cs_rank",
        "resistance_distance_cs_rank",
    ]

    basic_numeric_features = [
        "return_1d", "return_3d", "return_5d", "return_10d",
        "ma5_distance", "ma20_distance", "ma60_distance",
        "volatility_5d", "volatility_10d", "high_low_range_pct",
        "volume_ratio_5", "volume_ratio_20",
        "DIF", "DEA", "MACD", "K", "D", "J", "RSI14", "BIAS6", "BIAS12", "BIAS24",
        "atr_pct",
    ]

    structure_extra_numeric_features = [
        "fib_position", "fib_score",
        "support_distance", "resistance_distance",
        "risk_stoploss_pct", "risk_atr_stop_distance_pct",
    ]

    if feature_set == "basic":
        numeric_features = basic_numeric_features
        categorical_features = []
    elif feature_set == "structure":
        numeric_features = basic_numeric_features + structure_extra_numeric_features
        categorical_features = ["fib_zone"]
    elif feature_set == "technical_core":
        numeric_features = [
            "K", "D", "J",
            "DIF", "DEA", "MACD",
            "volume", "volume_ratio_5", "volume_ratio_20", "vol_ma5", "vol_ma20",
            "RSI14",
            "BIAS6", "BIAS12", "BIAS24",
            "ma5", "ma20", "ma60", "ma5_distance", "ma20_distance", "ma60_distance",
            "nearest_support_level", "nearest_support_zone_low", "nearest_support_zone_high",
            "support_distance",
            "nearest_resistance_level", "nearest_resistance_zone_low", "nearest_resistance_zone_high",
            "resistance_distance",
            "atr_14", "atr_pct",
            "risk_entry_price", "risk_atr_14", "risk_ob_bottom", "risk_atr_stop",
            "risk_stoploss_price", "risk_per_share", "risk_stoploss_pct",
            "risk_atr_stop_distance_pct", "risk_ob_bottom_distance_pct",
            "risk_stoploss_distance_pct", "risk_valid",
        ]
        categorical_features = []
    elif feature_set == "structural":
        numeric_features = [
            col for col in STRUCTURAL_MODEL_FEATURES
            if col not in STRUCTURAL_CATEGORICAL_FEATURES and col not in EXCLUDED_RAW_FEATURES
        ]
        categorical_features = [
            col for col in STRUCTURAL_CATEGORICAL_FEATURES
            if col not in EXCLUDED_RAW_FEATURES
        ]
    elif feature_set == "selected":
        numeric_features = [
            "atr_pct",
            "risk_atr_stop_distance_pct",
            "volatility_10d",
            "atr_pct_cs_rank",
            "recent_active_gap_size_pct",
            "ma60_distance",
            "return_5d",
            "support_distance",
            "resistance_distance",
            "DEA",
            "MACD",
            "RSI14",
            "K",
            "BIAS24",
            "VR_26",
            "recent_2w_ob_volume_ratio",
            "close_to_10d_high",
            "fib_position",
        ]
        categorical_features = [
            "fib_zone",
            "recent_2w_ob_type",
            "recent_active_gap_type",
        ]
    elif feature_set == "selected_v2":
        numeric_features = [
            "atr_pct",
            "risk_atr_stop_distance_pct",
            "volatility_10d",
            "atr_pct_cs_rank",
            "recent_active_gap_size_pct",
            "ma60_distance",
            "return_5d",
            "support_distance",
            "resistance_distance",
            "DEA",
            "MACD",
            "RSI14",
            "K",
            "BIAS24",
            "VR_26",
            "recent_2w_ob_volume_ratio",
            "close_to_10d_high",
            "fib_position",
            "volatility_10d_cs_rank",
            "volatility_5d",
            "J",
            "DIF",
            "recent_2w_ob_strength_score",
            "close_to_10d_low",
            "VR_14",
            "return_10d",
            "ma20_distance",
            "fib_score",
        ]
        _extra_cs_ranks = [
            "RSI14_cs_rank",
            "MACD_cs_rank",
            "K_cs_rank",
            "BIAS24_cs_rank",
            "VR_14_cs_rank",
            "VR_26_cs_rank",
            "ma20_distance_cs_rank",
            "ma60_distance_cs_rank",
        ]
        numeric_features += [col for col in _extra_cs_ranks if col in df.columns]
        # 籌碼與營收特徵（--chip-features；2026-10-26）
        from pipeline_chip import CHIP_SETTINGS, chip_feature_columns
        if CHIP_SETTINGS["enabled"]:
            numeric_features += [col for col in chip_feature_columns(df) if col not in numeric_features]
        categorical_features = [
            "fib_zone",
            "recent_2w_ob_type",
            "recent_active_gap_type",
            "market_trend_regime",
        ]
    elif feature_set == "selected_risk_v1":
        selected_v2_numeric_features, selected_v2_categorical_features = get_feature_columns(
            df,
            feature_set="selected_v2",
        )
        risk_overheat_features = [
            "close_to_20d_high",
            "close_to_20d_low",
            "volatility_20d",
            "atr_pct_change_5d",
            "ma20_slope",
            "ma60_slope",
            "volume_ratio_20_change_3d",
            "recent_5d_return_from_high",
            "recent_10d_return_from_high",
        ]
        numeric_features = list(dict.fromkeys(selected_v2_numeric_features + risk_overheat_features))
        categorical_features = selected_v2_categorical_features
    elif feature_set == "full":
        numeric_features = [
            # 報酬與趨勢
            "return_1d", "return_3d", "return_5d", "return_10d",
            "ma5_distance", "ma20_distance", "ma60_distance",

            # 波動與價格位置
            "volatility_5d", "volatility_10d", "high_low_range_pct",
            "close_to_5d_high", "close_to_5d_low", "close_to_10d_high", "close_to_10d_low",
            "volume_ratio_5", "volume_ratio_20",
            "VR_14", "VR_26", "VR_14_change", "VR_26_change",

            # 技術指標
            "DIF", "DEA", "MACD", "K", "D", "J", "RSI14", "BIAS6", "BIAS12", "BIAS24",

            # Fib / Gap / 支撐壓力
            "fib_position", "fib_score",
            "recent_active_gap_size_pct", "recent_active_gap_distance_pct",
            "support_distance", "resistance_distance",

            # OB
            "recent_2w_ob_volume_ratio",
            "recent_2w_ob_days_since",
            "recent_2w_ob_strength_score",
            "recent_2w_ob_is_fallback",

            # 風險
            "atr_pct", "risk_stoploss_pct", "risk_atr_stop_distance_pct",
            "risk_ob_bottom_distance_pct", "risk_stoploss_distance_pct", "risk_valid",
        ] + (
            cs_rank_features
            + MARKET_NUMERIC_FEATURES
            + AMOUNT_FEATURES
            + CROSS_SECTIONAL_AMOUNT_FEATURES
            + RELATIVE_STRENGTH_FEATURES
            + CROSS_SECTIONAL_RELATIVE_STRENGTH_FEATURES
        )

        categorical_features = [
            "fib_zone",
            "recent_2w_ob_type",
            "recent_active_gap_type",
        ] + MARKET_CATEGORICAL_FEATURES
    elif feature_set in {"full_no_market", "full_amount_relative"}:
        full_numeric_features, full_categorical_features = get_feature_columns(df, feature_set="full")
        numeric_features = [
            col for col in full_numeric_features
            if col not in MARKET_NUMERIC_FEATURES
        ]
        categorical_features = [
            col for col in full_categorical_features
            if col not in MARKET_CATEGORICAL_FEATURES
        ]
    elif feature_set == "full_amount_relative_selected":
        base_numeric_features, base_categorical_features = get_feature_columns(df, feature_set="full_no_market")
        selected_amount_features = [
            "amount_ratio_20",
            "amount_change_10d",
            "amount_zscore_20",
            "amount_rank_pct",
            "amount_rank_change_10d",
            "amount_share_in_pool",
        ]
        selected_relative_features = [
            "relative_return_10d",
            "relative_return_20d",
            "relative_strength_rank_10d",
            "stock_vs_market_ma20_distance",
            "stock_vs_market_ma60_distance",
        ]
        context_feature_pool = set(
            AMOUNT_FEATURES
            + CROSS_SECTIONAL_AMOUNT_FEATURES
            + RELATIVE_STRENGTH_FEATURES
            + CROSS_SECTIONAL_RELATIVE_STRENGTH_FEATURES
            + MARKET_NUMERIC_FEATURES
        )
        numeric_features = list(dict.fromkeys(
            [
                col for col in base_numeric_features
                if col not in context_feature_pool
            ]
            + selected_amount_features
            + selected_relative_features
        ))
        categorical_features = [
            col for col in base_categorical_features
            if col not in MARKET_CATEGORICAL_FEATURES
        ]
    elif feature_set in {"full_error_v1", "full_error_v1_symbol"}:
        base_numeric_features, base_categorical_features = get_feature_columns(
            df,
            feature_set="full_amount_relative_selected",
        )
        numeric_features = list(dict.fromkeys(base_numeric_features + REALIZED_HISTORY_FEATURES))
        categorical_features = list(base_categorical_features)
        if feature_set == "full_error_v1_symbol":
            if "symbol" in df.columns:
                categorical_features = list(dict.fromkeys(categorical_features + ["symbol"]))
            else:
                print("WARNING: feature_set=full_error_v1_symbol requested symbol categorical feature, but symbol column is missing.")
    elif feature_set == "full_market_light":
        full_numeric_features, full_categorical_features = get_feature_columns(df, feature_set="full")
        light_market_features = {
            "market_return_10d",
            "market_ma20_distance",
            "market_ma60_distance",
            "market_volatility_10d",
        }
        numeric_features = [
            col for col in full_numeric_features
            if col not in MARKET_NUMERIC_FEATURES or col in light_market_features
        ]
        categorical_features = full_categorical_features
    else:
        raise ValueError(f"feature_set must be one of: {FEATURE_SET_CHOICES}")

    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    return numeric_features, categorical_features


def feature_set_uses_realized_history(feature_set: str) -> bool:
    return feature_set in {"full_error_v1", "full_error_v1_symbol"}


def audit_md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def audit_md5_items(items) -> str:
    normalized = [str(item) for item in items if pd.notna(item)]
    return audit_md5_text("|".join(normalized))


def audit_date_range(data: Optional[pd.DataFrame]) -> tuple[str, str]:
    if data is None or data.empty or "date" not in data.columns:
        return "-", "-"
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    if dates.empty:
        return "-", "-"
    return str(dates.min().date()), str(dates.max().date())


def audit_sorted_symbols(data: Optional[pd.DataFrame]) -> list[str]:
    if data is None or data.empty or "symbol" not in data.columns:
        return []
    return sorted(data["symbol"].dropna().astype(str).unique().tolist())


def audit_row_count_by_symbol_hash(data: Optional[pd.DataFrame]) -> str:
    if data is None or data.empty or "symbol" not in data.columns:
        return "-"
    counts = data["symbol"].astype(str).value_counts().sort_index()
    items = [f"{symbol}:{int(count)}" for symbol, count in counts.items()]
    return audit_md5_items(items)


def audit_feature_non_null_summary_hash(data: Optional[pd.DataFrame], feature_cols: Optional[list[str]]) -> str:
    if data is None or data.empty or not feature_cols:
        return "-"
    items = []
    for col in feature_cols:
        if col in data.columns:
            items.append(f"{col}:{int(data[col].notna().sum())}")
        else:
            items.append(f"{col}:MISSING")
    return audit_md5_items(items)


def audit_metric(metrics: Optional[dict], *keys, default="-"):
    if not metrics:
        return default
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return default


def collect_model_flow_audit(
    context_name: str,
    config: PipelineConfig,
    data: Optional[pd.DataFrame] = None,
    feature_cols: Optional[list[str]] = None,
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    removed_features: Optional[list[str]] = None,
    metrics: Optional[dict] = None,
    validation_scope: str = "-",
    model_flow: str = "-",
) -> dict:
    symbols = audit_sorted_symbols(data)
    date_min, date_max = audit_date_range(data)
    feature_cols = list(feature_cols or audit_metric(metrics, "feature_cols", default=[]) or [])
    numeric_features = list(numeric_features or audit_metric(metrics, "numeric_features", default=[]) or [])
    categorical_features = list(categorical_features or audit_metric(metrics, "categorical_features", default=[]) or [])
    removed_features = list(removed_features or audit_metric(metrics, "removed_features", default=[]) or [])
    model_params = get_regression_model_profile_params(config.model_profile)
    target_symbol = str(config.stock_no).strip()

    audit = {
        "context_name": context_name,
        "mode": "single_diagnostic" if context_name == "MODE_1_SINGLE_DIAGNOSTIC" else "train/predict",
        "stock_no": config.stock_no,
        "market": config.market,
        "top_n": audit_metric(metrics, "top_n", default="-"),
        "target_col": config.target_col,
        "feature_set": config.feature_set,
        "model_type": config.model_type,
        "model_profile": config.model_profile,
        "diagnostic_scope": config.diagnostic_scope,
        "display_mode": config.display_mode,
        "horizon": config.horizon,
        "random_state": config.random_state,
        "run_feature_importance": config.run_feature_importance,
        "skip_classification": config.skip_classification,
        "data_rows": 0 if data is None else int(len(data)),
        "unique_symbols": len(symbols),
        "date_min": date_min,
        "date_max": date_max,
        "target_symbol_rows": (
            int((data["symbol"].astype(str) == target_symbol).sum())
            if data is not None and "symbol" in data.columns and target_symbol
            else "-"
        ),
        "symbols_sample": symbols[:10],
        "sorted_symbols_hash": audit_md5_items(symbols),
        "row_count_by_symbol_hash": audit_row_count_by_symbol_hash(data),
        "target_col_non_null_count": (
            int(pd.to_numeric(data[config.target_col], errors="coerce").notna().sum())
            if data is not None and config.target_col in data.columns
            else "-"
        ),
        "feature_count": len(feature_cols),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "feature_cols_hash": audit_md5_items(feature_cols),
        "removed_features_hash": audit_md5_items(removed_features),
        "feature_non_null_summary_hash": audit_feature_non_null_summary_hash(data, feature_cols),
        "validation_method": audit_metric(metrics, "validation_method", default=validation_scope),
        "validation_scope": validation_scope,
        "model_flow": model_flow,
        "walk_forward_folds": audit_metric(metrics, "walk_forward_folds", default="-"),
        "train_rows": audit_metric(metrics, "train_rows", "training_rows", default="-"),
        "test_rows": audit_metric(metrics, "test_rows", "oos_rows", default="-"),
        "mae": audit_metric(metrics, "mae", "MAE", default="-"),
        "rmse": audit_metric(metrics, "rmse", "RMSE", default="-"),
        "r2": audit_metric(metrics, "r2", "R2", default="-"),
        "baseline_mae": audit_metric(metrics, "baseline_mae", default="-"),
        "baseline_rmse": audit_metric(metrics, "baseline_rmse", default="-"),
        "baseline_r2": audit_metric(metrics, "baseline_r2", default="-"),
        "spearman_corr": audit_metric(metrics, "spearman_corr", default="-"),
        "top5_lift": audit_metric(metrics, "top5_lift", default="-"),
        "top10_lift": audit_metric(metrics, "top10_lift", default="-"),
        "long_short_spread": audit_metric(metrics, "long_short_spread", default="-"),
        "model_class": {
            "random_forest": "RandomForestRegressor",
            "extra_trees": "ExtraTreesRegressor",
            "hist_gradient_boosting": "HistGradientBoostingRegressor",
        }.get(config.model_type, config.model_type),
        "n_estimators": model_params.get("n_estimators", "-"),
        "max_depth": model_params.get("max_depth", "-"),
        "min_samples_leaf": model_params.get("min_samples_leaf", "-"),
        "max_features": model_params.get("max_features", "-"),
    }
    return audit


def print_model_flow_audit(audit: dict) -> None:
    print("\n" + "=" * 70)
    print(f"MODEL FLOW AUDIT: {audit['context_name']}")
    print("=" * 70)
    for key in [
        "mode", "stock_no", "market", "top_n", "target_col", "feature_set",
        "model_type", "model_profile", "diagnostic_scope", "display_mode",
        "horizon", "random_state", "run_feature_importance", "skip_classification",
        "data_rows", "unique_symbols", "date_min", "date_max",
        "target_symbol_rows", "feature_count", "numeric_feature_count",
        "categorical_feature_count", "validation_method", "validation_scope",
        "model_flow", "walk_forward_folds", "train_rows", "test_rows",
        "mae", "rmse", "r2", "baseline_mae", "baseline_rmse", "baseline_r2",
        "spearman_corr", "top5_lift", "top10_lift", "long_short_spread",
        "model_class", "n_estimators", "max_depth", "min_samples_leaf",
        "max_features", "sorted_symbols_hash", "row_count_by_symbol_hash",
        "feature_cols_hash", "removed_features_hash", "feature_non_null_summary_hash",
    ]:
        print(f"{key}={audit.get(key)}")
    print(f"symbols_sample={audit.get('symbols_sample')}")


def print_model_flow_consistency_check() -> None:
    mode_1 = MODEL_FLOW_AUDITS.get("MODE_1_SINGLE_DIAGNOSTIC")
    mode_2 = MODEL_FLOW_AUDITS.get("MODE_2_MULTI_STOCK_RANKING")

    print("\n" + "=" * 70)
    print("MODE 1 vs MODE 2 MODEL CONSISTENCY CHECK")
    print("=" * 70)

    if not mode_1 or not mode_2:
        missing = []
        if not mode_1:
            missing.append("MODE_1_SINGLE_DIAGNOSTIC")
        if not mode_2:
            missing.append("MODE_2_MULTI_STOCK_RANKING")
        print(f"comparison_available=False")
        print(f"missing_contexts={missing}")
        print("explanation=This process has only executed one mode so far; compare fingerprints from both runs.")
        return

    checks = {
        "same_model_type": mode_1["model_type"] == mode_2["model_type"],
        "same_model_profile": mode_1["model_profile"] == mode_2["model_profile"],
        "same_feature_set": mode_1["feature_set"] == mode_2["feature_set"],
        "same_target_col": mode_1["target_col"] == mode_2["target_col"],
        "same_feature_cols": mode_1["feature_cols_hash"] == mode_2["feature_cols_hash"],
        "same_training_symbols": mode_1["sorted_symbols_hash"] == mode_2["sorted_symbols_hash"],
        "same_training_rows": mode_1["data_rows"] == mode_2["data_rows"],
        "same_date_range": (mode_1["date_min"], mode_1["date_max"]) == (mode_2["date_min"], mode_2["date_max"]),
        "same_walk_forward_folds": mode_1["walk_forward_folds"] == mode_2["walk_forward_folds"],
        "same_validation_scope": mode_1["validation_scope"] == mode_2["validation_scope"],
    }
    for key, value in checks.items():
        print(f"{key}: {value}")

    metrics_directly_comparable = all([
        checks["same_target_col"],
        checks["same_feature_cols"],
        checks["same_training_symbols"],
        checks["same_date_range"],
        checks["same_validation_scope"],
    ])
    print(f"metrics_directly_comparable: {metrics_directly_comparable}")
    if metrics_directly_comparable:
        print("explanation=Model settings, features, data scope, date range, and validation scope match.")
    else:
        print("explanation=At least one of data scope, feature columns, target, or validation scope differs; metrics are not directly comparable.")


def register_model_flow_audit(audit: dict) -> None:
    MODEL_FLOW_AUDITS[audit["context_name"]] = audit
    print_model_flow_audit(audit)
    print_model_flow_consistency_check()


def warn_if_deprecated_feature_set(feature_set: str) -> None:
    if feature_set in DEPRECATED_FEATURE_SETS:
        print(f"WARNING: feature_set={feature_set} is {DEPRECATED_FEATURE_SETS[feature_set]}")


def get_leakage_columns(horizon: int = 10) -> list[str]:
    return [
        "target_score",
        "target_score_v2",
        "target_score_v3",
        "investment_score",
        f"target_return_{horizon}d",
        f"target_return_{horizon}d_cs_rank",
        f"target_downside_risk_{horizon}d",
        f"target_downside_risk_{horizon}d_cs_rank",
        f"target_safety_{horizon}d_cs_rank",
        f"target_abs_drawdown_safety_{horizon}d",
        f"target_rar_{horizon}d",
        f"target_rar_{horizon}d_cs_rank",
        f"target_score_raw_{horizon}d",
        f"future_max_high_{horizon}d",
        f"future_min_low_{horizon}d",
        f"future_upside_{horizon}d",
        f"future_downside_{horizon}d",
        f"future_close_{horizon}d",
        f"future_return_{horizon}d",
        f"future_drawdown_{horizon}d",
        f"future_risk_reward_{horizon}d",
        f"risk_adjusted_return_{horizon}d",
    ]


def remove_leakage_features(
    numeric_features: list[str],
    categorical_features: list[str],
    horizon: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    leakage_cols = set(get_leakage_columns(horizon=horizon))
    def is_leakage_col(col: str) -> bool:
        return col in leakage_cols or col.startswith("future_")

    removed_features = [
        col for col in numeric_features + categorical_features
        if is_leakage_col(col)
    ]
    numeric_features = [col for col in numeric_features if not is_leakage_col(col)]
    categorical_features = [col for col in categorical_features if not is_leakage_col(col)]
    return numeric_features, categorical_features, removed_features


def remove_high_missing_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    threshold: float = 0.3,
) -> tuple[list[str], list[str], list[str]]:
    feature_cols = numeric_features + categorical_features
    missing_rate = df[feature_cols].isna().mean()
    valid_features = missing_rate[missing_rate < threshold].index.tolist()
    removed_features = [col for col in feature_cols if col not in valid_features]

    numeric_features = [col for col in numeric_features if col in valid_features]
    categorical_features = [col for col in categorical_features if col in valid_features]

    return numeric_features, categorical_features, removed_features


def remove_low_variance_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str], list[str]]:
    removed_features = []
    kept_numeric = []
    kept_categorical = []

    for col in numeric_features:
        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        series = pd.to_numeric(raw, errors="coerce")
        if series.dropna().nunique() <= 1:
            removed_features.append(col)
        else:
            kept_numeric.append(col)

    for col in categorical_features:
        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        series = raw.astype("object").where(raw.notna(), "none").astype(str)
        if series.nunique(dropna=False) <= 1:
            removed_features.append(col)
        else:
            kept_categorical.append(col)

    return kept_numeric, kept_categorical, removed_features


def print_target_distribution(data: pd.DataFrame, target_col: str) -> None:
    target_data = data[target_col]
    if isinstance(target_data, pd.DataFrame):
        target_data = target_data.iloc[:, 0]
    target_valid = pd.to_numeric(target_data, errors="coerce").dropna()

    print("\n" + "=" * 70)
    print(f"Target Distribution: {target_col}")
    print("=" * 70)

    print(target_valid.describe())

    print("\nQuantiles:")
    print(target_valid.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

    print("\nTarget bins:")
    bounded_rank_like = (
        target_col in {"target_score", "target_score_v2", "target_score_v3"}
        or target_col.endswith("_cs_rank")
        or target_col.startswith("target_abs_drawdown_safety_")
    )
    if bounded_rank_like:
        bins = [0, 0.3, 0.45, 0.55, 0.7, 1.0]
        print(pd.cut(
            target_valid,
            bins=bins,
            include_lowest=True
        ).value_counts().sort_index())
    elif target_col.startswith("target_score_raw_") or target_col.startswith("target_return_"):
        bins = [-np.inf, -0.10, -0.05, 0, 0.03, 0.05, 0.10, np.inf]
        print(pd.cut(
            target_valid,
            bins=bins,
            include_lowest=True
        ).value_counts().sort_index())
    else:
        try:
            print(pd.qcut(target_valid, q=5, duplicates="drop").value_counts().sort_index())
        except ValueError:
            print(target_valid.value_counts(dropna=False).sort_index())


def get_original_feature_name(transformed_name: str, categorical_features: list[str]) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)

    if transformed_name.startswith("cat__"):
        cat_name = transformed_name.replace("cat__", "", 1)
        for feature in sorted(categorical_features, key=len, reverse=True):
            if cat_name == feature or cat_name.startswith(f"{feature}_"):
                return feature
        return cat_name

    return transformed_name


def select_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    model_profile: str = "conservative",
    cumulative_importance: float = 0.95,
    min_features: int = 12,
    max_features: int = 24,
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    feature_cols = numeric_features + categorical_features
    if len(feature_cols) <= min_features:
        report = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.nan,
            "cumulative_importance": np.nan,
            "selected": True,
            "feature_type": ["numeric" if col in numeric_features else "categorical" for col in feature_cols],
            "removed_reason": "",
        })
        return numeric_features, categorical_features, [], report

    selector = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        model_profile=model_profile,
    )
    selector.fit(X, y)

    transformed_names = selector.named_steps["preprocessor"].get_feature_names_out()
    importances = selector.named_steps["model"].feature_importances_

    rows = []
    for transformed_name, importance in zip(transformed_names, importances):
        rows.append({
            "feature": get_original_feature_name(str(transformed_name), categorical_features),
            "importance": float(importance),
        })

    importance_df = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
    )

    all_features_df = pd.DataFrame({"feature": feature_cols})
    importance_df = all_features_df.merge(importance_df, on="feature", how="left")
    importance_df["importance"] = importance_df["importance"].fillna(0.0)
    importance_df["feature_type"] = np.where(
        importance_df["feature"].isin(numeric_features),
        "numeric",
        "categorical",
    )
    importance_df = importance_df.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    total_importance = importance_df["importance"].sum()
    if total_importance <= 0:
        report = importance_df.copy()
        report["cumulative_importance"] = np.nan
        report["selected"] = True
        report["removed_reason"] = ""
        return numeric_features, categorical_features, [], report

    importance_df["importance_share"] = importance_df["importance"] / total_importance
    importance_df["cumulative_importance"] = importance_df["importance_share"].cumsum()

    selected_mask = pd.Series(
        (importance_df.index < min_features)
        | (
            (importance_df["cumulative_importance"] <= cumulative_importance)
            & (importance_df.index < max_features)
        ),
        index=importance_df.index,
    )

    if selected_mask.sum() < len(importance_df):
        first_removed_idx = int(selected_mask.sum())
        if first_removed_idx < len(importance_df) and first_removed_idx < max_features:
            selected_mask.iloc[first_removed_idx] = True

    importance_df["selected"] = selected_mask
    importance_df["removed_reason"] = np.where(
        importance_df["selected"],
        "",
        "low_model_importance",
    )

    selected_features = importance_df.loc[importance_df["selected"], "feature"].tolist()
    removed_by_importance = [
        col for col in feature_cols
        if col not in selected_features
    ]

    selected_numeric = [col for col in numeric_features if col in selected_features]
    selected_categorical = [col for col in categorical_features if col in selected_features]

    return selected_numeric, selected_categorical, removed_by_importance, importance_df


def build_feature_importance_report(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    model_profile: str = "conservative",
    model_type: str = "random_forest",
) -> pd.DataFrame:
    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        return pd.DataFrame()

    pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        model_profile=model_profile,
        model_type=model_type,
    )
    pipeline.fit(X, y)

    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.nan,
        })
        importance_df["feature_type"] = np.where(
            importance_df["feature"].isin(numeric_features),
            "numeric",
            "categorical",
        )
        importance_df["importance_share"] = np.nan
        importance_df["cumulative_importance"] = np.nan
        importance_df["used_for_removal"] = False
        importance_df["note"] = f"{model_type} does not expose feature_importances_"
        return importance_df

    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model.feature_importances_

    rows = []
    for transformed_name, importance in zip(transformed_names, importances):
        rows.append({
            "feature": get_original_feature_name(str(transformed_name), categorical_features),
            "importance": float(importance),
        })

    importance_df = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
    )
    importance_df = pd.DataFrame({"feature": feature_cols}).merge(
        importance_df,
        on="feature",
        how="left",
    )
    importance_df["importance"] = importance_df["importance"].fillna(0.0)
    importance_df["feature_type"] = np.where(
        importance_df["feature"].isin(numeric_features),
        "numeric",
        "categorical",
    )
    importance_df = importance_df.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    total_importance = importance_df["importance"].sum()
    if total_importance > 0:
        importance_df["importance_share"] = importance_df["importance"] / total_importance
        importance_df["cumulative_importance"] = importance_df["importance_share"].cumsum()
    else:
        importance_df["importance_share"] = np.nan
        importance_df["cumulative_importance"] = np.nan

    importance_df["used_for_removal"] = ENABLE_IMPORTANCE_FEATURE_SELECTION
    return importance_df


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_regression_model_profile_params(model_profile: str) -> dict:
    profile_params = {
        "quick": {
            "n_estimators": 150,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "max_features": 0.6,
        },
        "conservative": {
            "n_estimators": 600,
            "max_depth": 4,
            "min_samples_leaf": 12,
            "max_features": 0.5,
        },
        "medium": {
            "n_estimators": 600,
            "max_depth": 6,
            "min_samples_leaf": 8,
            "max_features": 0.6,
        },
        "flexible": {
            "n_estimators": 600,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "max_features": 0.7,
        },
    }
    if model_profile not in profile_params:
        raise ValueError("model_profile 只能是 quick、conservative、medium 或 flexible")
    return profile_params[model_profile]


def build_regression_model(
    model_type: str,
    model_profile: str,
    random_state: int,
):
    params = get_regression_model_profile_params(model_profile)

    if model_type == "random_forest":
        return RandomForestRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            learning_rate=0.05 if model_profile == "conservative" else 0.06,
            l2_regularization=0.1 if model_profile in {"quick", "conservative"} else 0.03,
            random_state=random_state,
        )

    raise ValueError(
        f"model_type 只能是 {', '.join(MODEL_TYPE_CHOICES)}"
    )


def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    model_profile: str = "conservative",
    model_type: str = "random_forest",
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = build_regression_model(
        model_type=model_type,
        model_profile=model_profile,
        random_state=random_state,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_classifier_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    model_profile: str = "conservative",
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    profile_params = {
        "quick": {
            "n_estimators": 150,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "max_features": 0.6,
        },
        "conservative": {
            "n_estimators": 600,
            "max_depth": 4,
            "min_samples_leaf": 12,
            "max_features": 0.5,
        },
        "medium": {
            "n_estimators": 600,
            "max_depth": 6,
            "min_samples_leaf": 8,
            "max_features": 0.6,
        },
        "flexible": {
            "n_estimators": 600,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "max_features": 0.7,
        },
    }
    if model_profile not in profile_params:
        raise ValueError("model_profile 只能是 quick、conservative、medium 或 flexible")

    model = RandomForestClassifier(
        **profile_params[model_profile],
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

def safe_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def build_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_size: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if min_train_size is None:
        min_train_size = min(max(30, n_samples // 2), n_samples - 2)

    validation_size = n_samples - min_train_size
    if validation_size < 2:
        raise ValueError("walk-forward validation 的驗證資料太少，請增加歷史資料年數。")

    n_splits = min(n_splits, validation_size)
    test_size = max(1, validation_size // n_splits)
    splits = []

    for fold_no in range(n_splits):
        train_end = min_train_size + fold_no * test_size
        test_start = train_end
        test_end = n_samples if fold_no == n_splits - 1 else min(test_start + test_size, n_samples)

        if test_start >= n_samples or test_start >= test_end:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    if len(splits) < 2:
        raise ValueError("walk-forward validation 至少需要 2 折，請增加歷史資料年數。")

    return splits


# ============================================================
# Walk-forward 清洗期（purge，2026-10-26 新增）
#   標籤 target_*_10d 會用到之後 10 個交易日的價格；若訓練集一路用到測試開始前一天，
#   訓練集最後 10 天的標籤就已經包含測試期的價格（資料洩漏）。
#   清洗期 = 在訓練集結尾刪掉 purge_dates 個交易日，讓訓練標籤在測試開始前就已實現。
#   預設 10（= horizon）；設 0 可重現 2026-09-21 論文版的舊做法。
# ============================================================
WALK_FORWARD_SETTINGS: dict = {"purge_dates": 10}


def set_walk_forward_purge(purge_dates: int) -> None:
    WALK_FORWARD_SETTINGS["purge_dates"] = max(0, int(purge_dates))
    label = "（舊做法：不清洗，可重現論文版）" if WALK_FORWARD_SETTINGS["purge_dates"] == 0 else ""
    print(f"Walk-forward 清洗期：{WALK_FORWARD_SETTINGS['purge_dates']} 個交易日{label}")


def get_walk_forward_purge() -> int:
    return int(WALK_FORWARD_SETTINGS.get("purge_dates", 0))


def build_walk_forward_splits_by_date(
    model_df: pd.DataFrame,
    n_splits: int = 8,
    min_train_dates: Optional[int] = None,
    purge_dates: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    purge = get_walk_forward_purge() if purge_dates is None else max(0, int(purge_dates))
    if "date" not in model_df.columns:
        return build_walk_forward_splits(len(model_df), n_splits=n_splits)

    dates = pd.to_datetime(model_df["date"])
    unique_dates = pd.Series(dates.dropna().unique()).sort_values().reset_index(drop=True)

    if len(unique_dates) < n_splits + 2:
        return build_walk_forward_splits(len(model_df), n_splits=n_splits)

    if min_train_dates is not None:
        min_train_dates = min(min_train_dates, len(unique_dates) - 2)
    else:
        min_train_dates = min(max(30, len(unique_dates) // 2), len(unique_dates) - 2)
    validation_dates = len(unique_dates) - min_train_dates
    n_splits = min(n_splits, validation_dates)
    test_date_size = max(1, validation_dates // n_splits)
    splits = []

    for fold_no in range(n_splits):
        train_end_pos = min_train_dates + fold_no * test_date_size
        test_start_pos = train_end_pos
        test_end_pos = len(unique_dates) if fold_no == n_splits - 1 else min(test_start_pos + test_date_size, len(unique_dates))

        if test_start_pos >= len(unique_dates) or test_start_pos >= test_end_pos:
            continue

        train_last_pos = train_end_pos - 1 - purge
        if train_last_pos < 0:
            continue
        train_end_date = unique_dates.iloc[train_last_pos]
        test_start_date = unique_dates.iloc[test_start_pos]
        test_end_date = unique_dates.iloc[test_end_pos - 1]

        train_mask = dates <= train_end_date
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    if len(splits) < 2:
        return build_walk_forward_splits(len(model_df), n_splits=n_splits)

    return splits


def calculate_ranking_metrics(
    validation_dates: pd.Series,
    y_true,
    y_pred,
) -> dict:
    ranking_df = pd.DataFrame({
        "date": pd.to_datetime(validation_dates).to_numpy(),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    }).dropna(subset=["date", "y_true", "y_pred"])

    if ranking_df.empty:
        return {
            "spearman_corr": np.nan,
            "top5_realized_mean": np.nan,
            "all_realized_mean": np.nan,
            "top5_lift": np.nan,
            "top10_realized_mean": np.nan,
            "top10_lift": np.nan,
            "long_short_spread": np.nan,
        }

    daily_rows = []

    for _, day_df in ranking_df.groupby("date"):
        day_df = day_df.sort_values("y_pred", ascending=False).reset_index(drop=True)
        n = len(day_df)
        all_mean = day_df["y_true"].mean()

        spearman_corr = np.nan
        if n >= 5:
            spearman_corr = day_df["y_true"].corr(day_df["y_pred"], method="spearman")

        top5_mean = np.nan
        if n >= 5:
            top5_mean = day_df.head(5)["y_true"].mean()

        top10_mean = np.nan
        if n >= 10:
            top10_mean = day_df.head(10)["y_true"].mean()

        long_short_spread = np.nan
        if n >= 5:
            side_n = max(1, int(np.floor(n * 0.2)))
            long_mean = day_df.head(side_n)["y_true"].mean()
            short_mean = day_df.tail(side_n)["y_true"].mean()
            long_short_spread = long_mean - short_mean

        daily_rows.append({
            "spearman_corr": spearman_corr,
            "top5_realized_mean": top5_mean,
            "all_realized_mean": all_mean,
            "top5_lift": top5_mean - all_mean if pd.notna(top5_mean) else np.nan,
            "top10_realized_mean": top10_mean,
            "top10_lift": top10_mean - all_mean if pd.notna(top10_mean) else np.nan,
            "long_short_spread": long_short_spread,
        })

    daily_metrics = pd.DataFrame(daily_rows)
    return {
        "spearman_corr": daily_metrics["spearman_corr"].mean(),
        "top5_realized_mean": daily_metrics["top5_realized_mean"].mean(),
        "all_realized_mean": daily_metrics["all_realized_mean"].mean(),
        "top5_lift": daily_metrics["top5_lift"].mean(),
        "top10_realized_mean": daily_metrics["top10_realized_mean"].mean(),
        "top10_lift": daily_metrics["top10_lift"].mean(),
        "long_short_spread": daily_metrics["long_short_spread"].mean(),
    }


def summarize_distribution(series: pd.Series, prefix: str) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_count": int(values.count()),
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(),
        f"{prefix}_min": values.min(),
        f"{prefix}_p05": values.quantile(0.05),
        f"{prefix}_p25": values.quantile(0.25),
        f"{prefix}_median": values.median(),
        f"{prefix}_p75": values.quantile(0.75),
        f"{prefix}_p95": values.quantile(0.95),
        f"{prefix}_max": values.max(),
    }


def safe_corr(df: pd.DataFrame, left: str, right: str, method: str = "pearson") -> float:
    df = clean_duplicate_columns(df.copy())
    if left not in df.columns or right not in df.columns:
        return np.nan
    left_series = df[left]
    right_series = df[right]
    if isinstance(left_series, pd.DataFrame):
        left_series = left_series.iloc[:, 0]
    if isinstance(right_series, pd.DataFrame):
        right_series = right_series.iloc[:, 0]

    work = pd.DataFrame({
        left: pd.to_numeric(left_series, errors="coerce"),
        right: pd.to_numeric(right_series, errors="coerce"),
    }).dropna()
    if len(work) < 3 or work[left].nunique() <= 1 or work[right].nunique() <= 1:
        return np.nan
    return work[left].corr(work[right], method=method)


def build_prediction_distribution_metrics(fold_context: pd.DataFrame) -> dict:
    metrics = {}
    metrics.update(summarize_distribution(fold_context["target"], "y_true"))
    metrics.update(summarize_distribution(fold_context["prediction"], "y_pred"))

    target_std = pd.to_numeric(fold_context["target"], errors="coerce").std()
    prediction_std = pd.to_numeric(fold_context["prediction"], errors="coerce").std()
    metrics["prediction_std_target_std_ratio"] = (
        prediction_std / target_std
        if pd.notna(target_std) and target_std > 0
        else np.nan
    )

    prediction = pd.to_numeric(fold_context["prediction"], errors="coerce")
    metrics["prediction_0_4_0_6_share"] = prediction.between(0.4, 0.6, inclusive="both").mean()

    ranking_df = fold_context.copy()
    ranking_df["target"] = pd.to_numeric(ranking_df["target"], errors="coerce")
    ranking_df["prediction"] = pd.to_numeric(ranking_df["prediction"], errors="coerce")
    if "future_return" in ranking_df.columns:
        ranking_df["future_return"] = pd.to_numeric(ranking_df["future_return"], errors="coerce")
    ranking_df = ranking_df.dropna(subset=["target", "prediction"])

    if ranking_df.empty:
        metrics.update({
            "y_pred_top10_y_true_mean": np.nan,
            "y_pred_bottom10_y_true_mean": np.nan,
            "y_pred_top10_future_return_mean": np.nan,
            "y_pred_top10_future_return_win_rate": np.nan,
            "y_pred_bottom10_future_return_mean": np.nan,
            "y_pred_bottom10_future_return_win_rate": np.nan,
        })
        return metrics

    side_n = max(1, int(np.ceil(len(ranking_df) * 0.10)))
    ranked = ranking_df.sort_values("prediction", ascending=False)
    top_df = ranked.head(side_n)
    bottom_df = ranked.tail(side_n)

    metrics["y_pred_top10_y_true_mean"] = top_df["target"].mean()
    metrics["y_pred_bottom10_y_true_mean"] = bottom_df["target"].mean()

    if "future_return" in ranking_df.columns:
        metrics["y_pred_top10_future_return_mean"] = top_df["future_return"].mean()
        metrics["y_pred_top10_future_return_win_rate"] = (top_df["future_return"] > 0).mean()
        metrics["y_pred_bottom10_future_return_mean"] = bottom_df["future_return"].mean()
        metrics["y_pred_bottom10_future_return_win_rate"] = (bottom_df["future_return"] > 0).mean()
    else:
        metrics["y_pred_top10_future_return_mean"] = np.nan
        metrics["y_pred_top10_future_return_win_rate"] = np.nan
        metrics["y_pred_bottom10_future_return_mean"] = np.nan
        metrics["y_pred_bottom10_future_return_win_rate"] = np.nan

    return metrics


def build_target_group_report(
    df: pd.DataFrame,
    target_col: str,
    future_return_col: str,
    future_drawdown_col: str,
) -> pd.DataFrame:
    required_cols = [target_col, future_return_col]
    work = df[[col for col in [target_col, future_return_col, future_drawdown_col] if col in df.columns]].copy()
    if not all(col in work.columns for col in required_cols):
        return pd.DataFrame()

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work[future_return_col] = pd.to_numeric(work[future_return_col], errors="coerce")
    if future_drawdown_col in work.columns:
        work[future_drawdown_col] = pd.to_numeric(work[future_drawdown_col], errors="coerce")
    work = work.dropna(subset=[target_col, future_return_col])

    if work.empty:
        return pd.DataFrame()

    try:
        work["target_score_group"] = pd.qcut(
            work[target_col],
            q=5,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        work["target_score_group"] = "all"

    rows = []
    for group_name, group_df in work.groupby("target_score_group", sort=True):
        row = {
            "target_score_group": group_name,
            "rows": len(group_df),
            "target_score_mean": group_df[target_col].mean(),
            "future_return_mean": group_df[future_return_col].mean(),
            "future_return_median": group_df[future_return_col].median(),
            "win_rate": (group_df[future_return_col] > 0).mean(),
        }
        if future_drawdown_col in group_df.columns:
            row["future_drawdown_mean"] = group_df[future_drawdown_col].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def build_top_bottom_target_report(
    df: pd.DataFrame,
    target_col: str,
    future_return_col: str,
    future_drawdown_col: str,
) -> pd.DataFrame:
    required_cols = [target_col, future_return_col]
    work = df[[col for col in [target_col, future_return_col, future_drawdown_col] if col in df.columns]].copy()
    if not all(col in work.columns for col in required_cols):
        return pd.DataFrame()

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work[future_return_col] = pd.to_numeric(work[future_return_col], errors="coerce")
    if future_drawdown_col in work.columns:
        work[future_drawdown_col] = pd.to_numeric(work[future_drawdown_col], errors="coerce")
    work = work.dropna(subset=[target_col, future_return_col])

    if work.empty:
        return pd.DataFrame()

    top_threshold = work[target_col].quantile(0.90)
    bottom_threshold = work[target_col].quantile(0.10)
    segments = {
        "top_10pct_target_score": work[work[target_col] >= top_threshold],
        "bottom_10pct_target_score": work[work[target_col] <= bottom_threshold],
    }

    rows = []
    for segment_name, segment_df in segments.items():
        row = {
            "segment": segment_name,
            "rows": len(segment_df),
            "target_score_mean": segment_df[target_col].mean(),
            "future_return_mean": segment_df[future_return_col].mean(),
            "win_rate": (segment_df[future_return_col] > 0).mean(),
        }
        if future_drawdown_col in segment_df.columns:
            row["future_drawdown_mean"] = segment_df[future_drawdown_col].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def explain_fold_diagnostics(fold_report: pd.DataFrame) -> pd.DataFrame:
    if fold_report is None or fold_report.empty:
        return pd.DataFrame()

    out = fold_report.copy()
    median_mae = out["mae"].median()
    median_r2 = out["r2"].median()
    median_future_return_std = (
        out["future_return_std"].median()
        if "future_return_std" in out.columns
        else np.nan
    )

    notes = []
    for _, row in out.iterrows():
        reasons = []
        if pd.notna(row.get("mae")) and pd.notna(row.get("baseline_mae")) and row["mae"] >= row["baseline_mae"]:
            reasons.append("模型 MAE 未優於 baseline")
        if pd.notna(row.get("r2")) and row["r2"] < 0:
            reasons.append("R2 為負，預測誤差高於均值解釋")
        if pd.notna(row.get("mae")) and row["mae"] > median_mae:
            reasons.append("MAE 高於各 fold 中位數")
        if pd.notna(row.get("r2")) and row["r2"] < median_r2:
            reasons.append("R2 低於各 fold 中位數")
        if pd.notna(row.get("spearman_corr")) and row["spearman_corr"] <= 0:
            reasons.append("排序相關為非正，分數排序訊號偏弱")
        target_std = row.get("target_std", np.nan)
        pred_std = row.get("prediction_std", np.nan)
        if pd.notna(target_std) and pd.notna(pred_std) and target_std > 0 and pred_std < target_std * 0.35:
            reasons.append("預測分布明顯壓縮，模型難以拉開高低分")
        if pd.notna(row.get("target_future_return_corr")) and abs(row["target_future_return_corr"]) < 0.1:
            reasons.append("該 fold target 與未來報酬相關性偏低")
        if pd.notna(row.get("prediction_future_return_corr")) and row["prediction_future_return_corr"] <= 0:
            reasons.append("預測分數與未來報酬方向不一致")
        if pd.notna(row.get("future_return_std")) and pd.notna(median_future_return_std) and row["future_return_std"] > median_future_return_std:
            reasons.append("future_return 波動高於多數 fold")
        if pd.notna(row.get("future_return_mean")) and row["future_return_mean"] < 0:
            reasons.append("驗證期平均 future_return 為負")
        if not reasons:
            reasons.append("未見單一明顯異常，可能是整體訊號接近 baseline")
        if int(row.get("fold", 0)) in {2, 5}:
            notes.append("重點 Fold 診斷：" + "；".join(reasons))
        else:
            notes.append("；".join(reasons))

    out["diagnostic_note"] = notes
    out["is_focus_fold"] = out["fold"].isin([2, 5])
    return out


def build_target_score_diagnostic_report(
    model_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    fold_report: pd.DataFrame,
) -> dict:
    future_return_col = f"future_return_{horizon}d"
    future_drawdown_col = f"future_drawdown_{horizon}d"
    risk_reward_candidates = [
        f"future_risk_reward_{horizon}d",
        "future_risk_reward",
        f"risk_adjusted_return_{horizon}d",
        f"future_upside_{horizon}d",
        f"future_downside_{horizon}d",
        "target_score",
        "target_score_v2",
    ]

    correlation_cols = [future_return_col, future_drawdown_col]
    correlation_cols += [col for col in risk_reward_candidates if col in model_df.columns]
    correlation_cols = list(dict.fromkeys(correlation_cols))

    corr_rows = []
    for col in correlation_cols:
        if col not in model_df.columns:
            continue
        corr_rows.append({
            "target_col": target_col,
            "comparison_col": col,
            "pearson_corr": safe_corr(model_df, target_col, col, method="pearson"),
            "spearman_corr": safe_corr(model_df, target_col, col, method="spearman"),
            "rows": int(model_df[[target_col, col]].apply(pd.to_numeric, errors="coerce").dropna().shape[0]),
        })
    correlation_report = pd.DataFrame(corr_rows)

    group_report = build_target_group_report(
        model_df,
        target_col=target_col,
        future_return_col=future_return_col,
        future_drawdown_col=future_drawdown_col,
    )
    top_bottom_report = build_top_bottom_target_report(
        model_df,
        target_col=target_col,
        future_return_col=future_return_col,
        future_drawdown_col=future_drawdown_col,
    )
    fold_diagnostic_report = explain_fold_diagnostics(fold_report)

    return {
        "correlation_report": correlation_report,
        "group_report": group_report,
        "top_bottom_report": top_bottom_report,
        "fold_diagnostic_report": fold_diagnostic_report,
    }


def print_target_score_diagnostics(diagnostics: dict, title: str = "target score 診斷報告") -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    correlation_report = diagnostics.get("correlation_report", pd.DataFrame())
    group_report = diagnostics.get("group_report", pd.DataFrame())
    top_bottom_report = diagnostics.get("top_bottom_report", pd.DataFrame())
    fold_diagnostic_report = diagnostics.get("fold_diagnostic_report", pd.DataFrame())

    print("\nTarget correlation")
    if correlation_report.empty:
        print("沒有足夠欄位計算 correlation。")
    else:
        print(correlation_report)

    print("\nTarget score groups")
    if group_report.empty:
        print("沒有足夠欄位建立 target score 分組報告。")
    else:
        print(group_report)

    print("\nTop / Bottom 10% target score")
    if top_bottom_report.empty:
        print("沒有足夠欄位建立 top/bottom 10% 報告。")
    else:
        print(top_bottom_report)

    print("\nFold-level diagnostic")
    if fold_diagnostic_report.empty:
        print("沒有 fold-level diagnostic。")
    else:
        display_cols = [
            "fold", "mae", "baseline_mae", "r2", "spearman_corr",
            "y_true_mean", "y_true_std", "y_pred_mean", "y_pred_std",
            "prediction_std_target_std_ratio", "prediction_0_4_0_6_share",
            "y_pred_top10_y_true_mean", "y_pred_bottom10_y_true_mean",
            "y_pred_top10_future_return_mean", "y_pred_top10_future_return_win_rate",
            "future_return_mean", "future_return_std", "diagnostic_note",
        ]
        print(fold_diagnostic_report[[col for col in display_cols if col in fold_diagnostic_report.columns]])


def walk_forward_validate(
    model_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    model_profile: str = "conservative",
    model_type: str = "random_forest",
    n_splits: int = 8,
) -> tuple[dict, pd.DataFrame]:
    splits = build_walk_forward_splits_by_date(
        model_df=model_df,
        n_splits=n_splits,
    )

    all_y_true = []
    all_y_pred = []
    all_baseline_pred = []
    fold_rows = []
    future_return_col = next(
        (col for col in model_df.columns if col.startswith("future_return_") and col.endswith("d")),
        None,
    )
    future_drawdown_col = next(
        (col for col in model_df.columns if col.startswith("future_drawdown_") and col.endswith("d")),
        None,
    )

    print("\n" + "=" * 70)
    print("Walk-forward Fold Results")
    print("=" * 70)

    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state + fold_no,
            model_profile=model_profile,
            model_type=model_type,
        )

        if "date" in model_df.columns:
            train_dates = pd.to_datetime(model_df.iloc[train_idx]["date"], errors="coerce")
            latest_train_date = train_dates.max()
            days_from_latest = (latest_train_date - train_dates).dt.days.clip(lower=0)
            fold_sample_weight = np.exp(-days_from_latest / 365).to_numpy(dtype=float)
        else:
            fold_sample_weight = None

        if fold_sample_weight is not None:
            pipeline.fit(X_train, y_train, model__sample_weight=fold_sample_weight)
        else:
            pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        baseline_pred = np.full(len(y_test), y_train.mean(), dtype=float)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = safe_r2(y_test, y_pred)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
        baseline_r2 = safe_r2(y_test, baseline_pred)
        ranking_metrics = calculate_ranking_metrics(
            validation_dates=model_df.iloc[test_idx]["date"],
            y_true=y_test,
            y_pred=y_pred,
        )
        fold_context = pd.DataFrame({
            "target": y_test.to_numpy(dtype=float),
            "prediction": np.asarray(y_pred, dtype=float),
        })
        if future_return_col:
            fold_context["future_return"] = pd.to_numeric(
                model_df.iloc[test_idx][future_return_col],
                errors="coerce",
            ).to_numpy(dtype=float)
        if future_drawdown_col:
            fold_context["future_drawdown"] = pd.to_numeric(
                model_df.iloc[test_idx][future_drawdown_col],
                errors="coerce",
            ).to_numpy(dtype=float)

        fold_distribution = {}
        fold_distribution.update(summarize_distribution(fold_context["target"], "target"))
        fold_distribution.update(summarize_distribution(fold_context["prediction"], "prediction"))
        fold_distribution.update(build_prediction_distribution_metrics(fold_context))
        if "future_return" in fold_context.columns:
            fold_distribution.update(summarize_distribution(fold_context["future_return"], "future_return"))
            fold_distribution["target_future_return_corr"] = safe_corr(
                fold_context, "target", "future_return", method="spearman"
            )
            fold_distribution["prediction_future_return_corr"] = safe_corr(
                fold_context, "prediction", "future_return", method="spearman"
            )
        else:
            fold_distribution["target_future_return_corr"] = np.nan
            fold_distribution["prediction_future_return_corr"] = np.nan
        if "future_drawdown" in fold_context.columns:
            fold_distribution.update(summarize_distribution(fold_context["future_drawdown"], "future_drawdown"))
        fold_distribution["prediction_target_corr"] = safe_corr(
            fold_context, "prediction", "target", method="spearman"
        )

        all_y_true.append(y_test.to_numpy(dtype=float))
        all_y_pred.append(y_pred)
        all_baseline_pred.append(baseline_pred)

        print(
            f"Fold {fold_no}: "
            f"train={len(train_idx)}, "
            f"valid={len(test_idx)}, "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}, "
            f"Baseline MAE={baseline_mae:.4f}, "
            f"Baseline RMSE={baseline_rmse:.4f}, "
            f"Baseline R2={baseline_r2:.4f}, "
            f"Spearman={ranking_metrics['spearman_corr']:.4f}, "
            f"Top5 Lift={ranking_metrics['top5_lift']:.4f}, "
            f"Top10 Lift={ranking_metrics['top10_lift']:.4f}, "
            f"Long-Short Spread={ranking_metrics['long_short_spread']:.4f}"
        )

        fold_rows.append({
            "fold": fold_no,
            "train_start_date": model_df.iloc[train_idx[0]]["date"],
            "train_end_date": model_df.iloc[train_idx[-1]]["date"],
            "test_start_date": model_df.iloc[test_idx[0]]["date"],
            "test_end_date": model_df.iloc[test_idx[-1]]["date"],
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2,
            "spearman_corr": ranking_metrics["spearman_corr"],
            "top5_realized_mean": ranking_metrics["top5_realized_mean"],
            "all_realized_mean": ranking_metrics["all_realized_mean"],
            "top5_lift": ranking_metrics["top5_lift"],
            "top10_realized_mean": ranking_metrics["top10_realized_mean"],
            "top10_lift": ranking_metrics["top10_lift"],
            "long_short_spread": ranking_metrics["long_short_spread"],
            **fold_distribution,
        })

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    baseline_all = np.concatenate(all_baseline_pred)

    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = mean_squared_error(y_true_all, y_pred_all) ** 0.5
    r2 = safe_r2(pd.Series(y_true_all), y_pred_all)
    baseline_mae = mean_absolute_error(y_true_all, baseline_all)
    baseline_rmse = mean_squared_error(y_true_all, baseline_all) ** 0.5
    baseline_r2 = safe_r2(pd.Series(y_true_all), baseline_all)

    print("\n" + "=" * 70)
    print("Walk-forward Summary")
    print("=" * 70)
    print(f"Model MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    print(f"Baseline MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, R2={baseline_r2:.4f}")

    fold_metrics_df = explain_fold_diagnostics(pd.DataFrame(fold_rows))
    summary_spearman = fold_metrics_df["spearman_corr"].mean()
    summary_top5_realized_mean = fold_metrics_df["top5_realized_mean"].mean()
    summary_all_realized_mean = fold_metrics_df["all_realized_mean"].mean()
    summary_top5_lift = fold_metrics_df["top5_lift"].mean()
    summary_top10_realized_mean = fold_metrics_df["top10_realized_mean"].mean()
    summary_top10_lift = fold_metrics_df["top10_lift"].mean()
    summary_long_short_spread = fold_metrics_df["long_short_spread"].mean()

    print(
        f"Ranking Spearman={summary_spearman:.4f}, "
        f"Top5 Lift={summary_top5_lift:.4f}, "
        f"Top10 Lift={summary_top10_lift:.4f}, "
        f"Long-Short Spread={summary_long_short_spread:.4f}"
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "validation_method": "walk_forward_time_series_split",
        "walk_forward_folds": len(splits),
        "train_rows": int(fold_rows[-1]["train_rows"]),
        "test_rows": int(sum(row["test_rows"] for row in fold_rows)),
        "spearman_corr": summary_spearman,
        "top5_realized_mean": summary_top5_realized_mean,
        "all_realized_mean": summary_all_realized_mean,
        "top5_lift": summary_top5_lift,
        "top10_realized_mean": summary_top10_realized_mean,
        "top10_lift": summary_top10_lift,
        "long_short_spread": summary_long_short_spread,
    }

    return metrics, pd.DataFrame(fold_rows)


def train_model(
    data: pd.DataFrame,
    target_col: str = "target_score_v2",
    config: Optional[PipelineConfig] = None,
) -> tuple[Pipeline, list[str], dict]:
    if config is None:
        config = CONFIG

    work = clean_duplicate_columns(data.copy())
    work = work.sort_values(["date", "symbol"] if "symbol" in work.columns else ["date"]).reset_index(drop=True)
    if config.feature_set == "structural":
        work = add_structural_features(work)
    warn_if_deprecated_feature_set(config.feature_set)
    if feature_set_uses_realized_history(config.feature_set):
        work = add_realized_history_features(work, horizon=config.horizon)

    numeric_features, categorical_features = get_feature_columns(work, feature_set=config.feature_set)
    initial_numeric_features = list(numeric_features)
    initial_categorical_features = list(categorical_features)
    structural_missing_features = []
    retained_raw_excluded_features = []
    if config.feature_set == "structural":
        structural_missing_features = [col for col in STRUCTURAL_MODEL_FEATURES if col not in work.columns]
        retained_raw_excluded_features = [col for col in EXCLUDED_RAW_FEATURES if col in work.columns]
        if structural_missing_features:
            print("missing structural features warning：")
            print(structural_missing_features)
        else:
            print("missing structural features warning：[]")
        print("被保留但未進模型的原始指標欄位：")
        print(retained_raw_excluded_features)

    if target_col not in work.columns:
        raise ValueError(f"target_col 不存在：{target_col}")

    # 修正重點：
    # pandas 的 pd.NA 可能讓部分欄位變成 object，
    # scikit-learn 轉 float 時會出現：
    # float() argument must be a string or a real number, not 'NAType'
    # 因此在進模型前，數值欄位統一轉 numeric，類別欄位統一轉 string。
    for col in numeric_features:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in categorical_features:
        if col in work.columns:
            work[col] = work[col].astype("object").where(work[col].notna(), "none").astype(str)

    target_like_cols = [
        "target_score",
        "target_score_v2",
        "target_score_v3",
        "investment_score",
        f"target_score_raw_{config.horizon}d",
        target_col,
    ]
    for col in dict.fromkeys(target_like_cols):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    diagnostic_cols = [
        f"future_max_high_{config.horizon}d",
        f"future_min_low_{config.horizon}d",
        f"future_upside_{config.horizon}d",
        f"future_downside_{config.horizon}d",
        f"future_close_{config.horizon}d",
        f"future_return_{config.horizon}d",
        f"future_drawdown_{config.horizon}d",
        f"future_risk_reward_{config.horizon}d",
        "future_risk_reward",
        f"risk_adjusted_return_{config.horizon}d",
        f"target_score_raw_{config.horizon}d",
    ]
    base_cols = [
        col for col in [
            "date", "symbol", "market", "close",
            "target_score", "target_score_v2", "target_score_v3", "investment_score",
            *diagnostic_cols,
        ]
        if col in work.columns
    ]
    if target_col in work.columns and target_col not in base_cols:
        base_cols.append(target_col)
    model_cols = list(dict.fromkeys(base_cols + numeric_features + categorical_features))
    model_df = work[model_cols].copy()
    print_target_distribution(model_df, target_col)

    numeric_features, categorical_features, leakage_removed_features = remove_leakage_features(
        numeric_features,
        categorical_features,
        horizon=config.horizon,
    )

    numeric_features, categorical_features, high_missing_removed_features = remove_high_missing_features(
        model_df,
        numeric_features,
        categorical_features,
        threshold=0.3,
    )

    numeric_features, categorical_features, low_variance_removed_features = remove_low_variance_features(
        model_df,
        numeric_features,
        categorical_features,
    )

    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError(f"feature_set={config.feature_set} 沒有可用模型特徵，請確認輸入資料欄位。")

    if config.feature_set == "structural":
        raw_overlap = [col for col in feature_cols if col in EXCLUDED_RAW_FEATURES]
        if raw_overlap:
            raise ValueError(f"structural feature_set 不應包含原始指標欄位：{raw_overlap}")

    model_df = model_df.dropna(subset=feature_cols + [target_col])

    if len(model_df) < 50:
        raise ValueError("可訓練資料太少，請確認 final_stock_table 是否有足夠歷史資料。")

    X = model_df[feature_cols]
    y = model_df[target_col]

    removed_features = (
        leakage_removed_features
        + high_missing_removed_features
        + low_variance_removed_features
    )

    if ENABLE_IMPORTANCE_FEATURE_SELECTION and config.run_feature_importance:
        numeric_features, categorical_features, removed_by_importance, feature_importance_report = select_important_features(
            X=X,
            y=y,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=config.random_state,
            model_profile=config.model_profile,
            cumulative_importance=0.95,
            min_features=12,
            max_features=24,
        )
        removed_features = removed_features + removed_by_importance
        feature_cols = numeric_features + categorical_features
        X = model_df[feature_cols]
    elif config.run_feature_importance:
        removed_by_importance = []
        feature_importance_report = build_feature_importance_report(
            X=X,
            y=y,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=config.random_state,
            model_profile=config.model_profile,
            model_type=config.model_type,
        )
    else:
        removed_by_importance = []
        feature_importance_report = pd.DataFrame()
        print("已略過 feature importance report，以提升訓練效率。")

    print("\n" + "=" * 70)
    print("Feature Importance Report")
    print("=" * 70)
    if config.run_feature_importance:
        print("只輸出，不自動刪除")
    if not feature_importance_report.empty:
        print(feature_importance_report.head(20))

    print("\n" + "=" * 70)
    print("模型訓練特徵設定")
    print("=" * 70)
    print(f"model_type={config.model_type}")
    print(f"model_profile={config.model_profile}")
    print(f"target_col={target_col}")
    print(f"目前 feature_set 模式：{config.feature_set}")
    print(f"模型實際使用特徵數量：{len(feature_cols)}")
    print("模型實際使用特徵清單：")
    print(feature_cols)
    print(f"feature_set={config.feature_set}")
    print(f"model_feature_count={len(feature_cols)}")
    print(f"model_numeric_features={numeric_features}")
    print(f"model_categorical_features={categorical_features}")
    if config.feature_set == "structural":
        print("missing structural features：")
        print(structural_missing_features)
        print("被保留但未進模型的原始指標欄位：")
        print(retained_raw_excluded_features)

    validation_metrics, validation_report = walk_forward_validate(
        model_df=model_df,
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=config.random_state,
        model_profile=config.model_profile,
        model_type=config.model_type,
        n_splits=8,
    )
    target_score_diagnostics = build_target_score_diagnostic_report(
        model_df=model_df,
        target_col=target_col,
        horizon=config.horizon,
        fold_report=validation_report,
    )
    print_target_score_diagnostics(
        target_score_diagnostics,
        title=f"{target_col} 診斷報告 ({config.feature_set})",
    )

    pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=config.random_state,
        model_profile=config.model_profile,
        model_type=config.model_type,
    )

    if "date" in model_df.columns:
        dates = pd.to_datetime(model_df["date"], errors="coerce")
        latest_date = dates.max()
        days_from_latest = (latest_date - dates).dt.days.clip(lower=0)
        sample_weight = np.exp(-days_from_latest / 365).to_numpy(dtype=float)
    else:
        sample_weight = None

    if sample_weight is not None:
        pipeline.fit(X, y, model__sample_weight=sample_weight)
    else:
        pipeline.fit(X, y)

    metrics = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "initial_numeric_features": initial_numeric_features,
        "initial_categorical_features": initial_categorical_features,
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "missing_structural_features": structural_missing_features,
        "retained_raw_excluded_features": retained_raw_excluded_features,
        "removed_features": removed_features,
        "leakage_removed_features": leakage_removed_features,
        "high_missing_removed_features": high_missing_removed_features,
        "low_variance_removed_features": low_variance_removed_features,
        "mae": validation_metrics["mae"],
        "rmse": validation_metrics["rmse"],
        "r2": validation_metrics["r2"],
        "baseline_mae": validation_metrics["baseline_mae"],
        "baseline_rmse": validation_metrics["baseline_rmse"],
        "baseline_r2": validation_metrics["baseline_r2"],
        "train_rows": validation_metrics["train_rows"],
        "test_rows": validation_metrics["test_rows"],
        "validation_method": validation_metrics["validation_method"],
        "walk_forward_folds": validation_metrics["walk_forward_folds"],
        "spearman_corr": validation_metrics["spearman_corr"],
        "top5_realized_mean": validation_metrics["top5_realized_mean"],
        "all_realized_mean": validation_metrics["all_realized_mean"],
        "top5_lift": validation_metrics["top5_lift"],
        "top10_realized_mean": validation_metrics["top10_realized_mean"],
        "top10_lift": validation_metrics["top10_lift"],
        "long_short_spread": validation_metrics["long_short_spread"],
        "walk_forward_report": validation_report,
        "target_score_diagnostics": target_score_diagnostics,
        "feature_importance_report": feature_importance_report,
        "removed_by_importance": removed_by_importance,
        "target_col": target_col,
        "feature_set": config.feature_set,
        "model_type": config.model_type,
        "model_profile": config.model_profile,
        "run_feature_importance": config.run_feature_importance,
        "importance_feature_selection_enabled": ENABLE_IMPORTANCE_FEATURE_SELECTION and config.run_feature_importance,
        "training_rows": len(model_df),
        "training_symbols": int(model_df["symbol"].nunique()) if "symbol" in model_df.columns else 1,
    }

    return pipeline, feature_cols, metrics


def prepare_classification_model_data(
    data: pd.DataFrame,
    config: PipelineConfig,
    target_col: str = "target_score_v2",
) -> dict:
    work = clean_duplicate_columns(data.copy())
    work = work.sort_values(["date", "symbol"] if "symbol" in work.columns else ["date"]).reset_index(drop=True)
    if config.feature_set == "structural":
        work = add_structural_features(work)
    warn_if_deprecated_feature_set(config.feature_set)
    if feature_set_uses_realized_history(config.feature_set):
        work = add_realized_history_features(work, horizon=config.horizon)

    numeric_features, categorical_features = get_feature_columns(work, feature_set=config.feature_set)
    if target_col not in work.columns:
        raise ValueError(f"classification target_col 不存在：{target_col}")

    for col in numeric_features:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in categorical_features:
        if col in work.columns:
            work[col] = work[col].astype("object").where(work[col].notna(), "none").astype(str)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    base_cols = [col for col in ["date", "symbol", "market", target_col] if col in work.columns]
    model_cols = list(dict.fromkeys(base_cols + numeric_features + categorical_features))
    model_df = work[model_cols].copy()

    numeric_features, categorical_features, leakage_removed_features = remove_leakage_features(
        numeric_features,
        categorical_features,
        horizon=config.horizon,
    )
    numeric_features, categorical_features, high_missing_removed_features = remove_high_missing_features(
        model_df,
        numeric_features,
        categorical_features,
        threshold=0.3,
    )
    numeric_features, categorical_features, low_variance_removed_features = remove_low_variance_features(
        model_df,
        numeric_features,
        categorical_features,
    )

    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError(f"classification feature_set={config.feature_set} 沒有可用模型特徵。")
    if config.feature_set == "structural":
        raw_overlap = [col for col in feature_cols if col in EXCLUDED_RAW_FEATURES]
        if raw_overlap:
            raise ValueError(f"structural classification 不應包含原始指標欄位：{raw_overlap}")

    model_df = model_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X = model_df[feature_cols]

    return {
        "model_df": model_df,
        "X": X,
        "target_score": model_df[target_col],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_cols": feature_cols,
        "removed_features": (
            leakage_removed_features
            + high_missing_removed_features
            + low_variance_removed_features
        ),
    }


def safe_binary_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = pd.Series(np.asarray(y_true)).astype(int)
    y_prob = pd.Series(np.asarray(y_prob), dtype=float)
    valid = y_true.notna() & y_prob.notna()
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if y_true.empty:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "confusion_tn": 0,
            "confusion_fp": 0,
            "confusion_fn": 0,
            "confusion_tp": 0,
            "top10_probability_hit_rate": np.nan,
        }

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if y_true.nunique() > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    ranked = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob", ascending=False)
    top_n = max(1, int(np.ceil(len(ranked) * 0.10)))
    top10_hit_rate = ranked.head(top_n)["y_true"].mean()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_tn": int(tn),
        "confusion_fp": int(fp),
        "confusion_fn": int(fn),
        "confusion_tp": int(tp),
        "top10_probability_hit_rate": top10_hit_rate,
    }


def walk_forward_classification_validate(
    model_df: pd.DataFrame,
    X: pd.DataFrame,
    y_class: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    model_profile: str,
    target_name: str,
    n_splits: int = 5,
) -> tuple[dict, pd.DataFrame]:
    splits = build_walk_forward_splits_by_date(
        model_df=model_df,
        n_splits=n_splits,
    )

    all_y_true = []
    all_y_prob = []
    fold_rows = []

    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_class.iloc[train_idx].astype(int)
        y_test = y_class.iloc[test_idx].astype(int)

        note = ""
        if y_train.nunique() < 2:
            y_prob = np.full(len(y_test), float(y_train.mean()), dtype=float)
            note = "training fold has one class; used constant probability fallback"
        else:
            pipeline = build_classifier_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_state=random_state + fold_no,
                model_profile=model_profile,
            )
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

        fold_metrics = safe_binary_classification_metrics(y_test, y_prob)
        prob_dist = summarize_distribution(pd.Series(y_prob), "predicted_probability")

        fold_rows.append({
            "target_name": target_name,
            "fold": fold_no,
            "train_start_date": model_df.iloc[train_idx[0]]["date"] if "date" in model_df.columns else pd.NaT,
            "train_end_date": model_df.iloc[train_idx[-1]]["date"] if "date" in model_df.columns else pd.NaT,
            "test_start_date": model_df.iloc[test_idx[0]]["date"] if "date" in model_df.columns else pd.NaT,
            "test_end_date": model_df.iloc[test_idx[-1]]["date"] if "date" in model_df.columns else pd.NaT,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_positive_rate": y_train.mean(),
            "test_positive_rate": y_test.mean(),
            **fold_metrics,
            **prob_dist,
            "diagnostic_note": note,
        })

        all_y_true.append(y_test.to_numpy(dtype=int))
        all_y_prob.append(np.asarray(y_prob, dtype=float))

    y_true_all = np.concatenate(all_y_true)
    y_prob_all = np.concatenate(all_y_prob)
    summary_metrics = safe_binary_classification_metrics(y_true_all, y_prob_all)
    summary_metrics.update({
        "target_name": target_name,
        "rows": int(len(y_true_all)),
        "positive_rate": float(np.mean(y_true_all)) if len(y_true_all) else np.nan,
        "folds": len(fold_rows),
    })

    return summary_metrics, pd.DataFrame(fold_rows)


def build_classification_targets_for_score(
    target_score: pd.Series,
    target_col: str,
    horizon: int = 10,
) -> dict[str, tuple[pd.Series, str]]:
    raw_target_col = f"target_score_raw_{horizon}d"
    if target_col == raw_target_col:
        return {
            "high_value": ((target_score >= 0.03).astype(int), f"{target_col} >= 0.03"),
            "low_value": ((target_score <= -0.05).astype(int), f"{target_col} <= -0.05"),
        }

    return {
        "high_value": ((target_score >= 0.7).astype(int), f"{target_col} >= 0.7"),
        "low_value": ((target_score <= 0.3).astype(int), f"{target_col} <= 0.3"),
    }


def run_classification_experiment_for_config(
    data: pd.DataFrame,
    config: PipelineConfig,
) -> dict:
    prepared = prepare_classification_model_data(
        data=data,
        config=config,
        target_col=config.target_col,
    )
    model_df = prepared["model_df"]
    target_score = pd.to_numeric(prepared["target_score"], errors="coerce")

    class_targets = build_classification_targets_for_score(
        target_score=target_score,
        target_col=config.target_col,
        horizon=config.horizon,
    )

    results = {}
    for target_name, (y_class, threshold_rule) in class_targets.items():
        summary, fold_report = walk_forward_classification_validate(
            model_df=model_df,
            X=prepared["X"],
            y_class=y_class,
            numeric_features=prepared["numeric_features"],
            categorical_features=prepared["categorical_features"],
            random_state=config.random_state,
            model_profile=config.model_profile,
            target_name=target_name,
            n_splits=5,
        )
        summary["threshold_rule"] = threshold_rule
        results[target_name] = {
            "summary": summary,
            "fold_report": fold_report,
            "feature_cols": prepared["feature_cols"],
            "removed_features": prepared["removed_features"],
        }

    return results


def print_classification_experiment_report(feature_set: str, results: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Classification Experiment ({feature_set})")
    print("=" * 70)
    for target_name, result in results.items():
        summary = result["summary"]
        print(f"\n{target_name}")
        print(f"- Feature count: {len(result.get('feature_cols', []))}")
        print(f"- Threshold rule: {summary.get('threshold_rule', '')}")
        print(f"- Positive rate: {summary.get('positive_rate', np.nan):.4f}")
        print(f"- Accuracy: {summary.get('accuracy', np.nan):.4f}")
        print(f"- Precision: {summary.get('precision', np.nan):.4f}")
        print(f"- Recall: {summary.get('recall', np.nan):.4f}")
        print(f"- F1: {summary.get('f1', np.nan):.4f}")
        print(f"- ROC-AUC: {summary.get('roc_auc', np.nan):.4f}")
        print(f"- PR-AUC: {summary.get('pr_auc', np.nan):.4f}")
        print(
            "- Confusion matrix: "
            f"TN={summary.get('confusion_tn', 0)}, FP={summary.get('confusion_fp', 0)}, "
            f"FN={summary.get('confusion_fn', 0)}, TP={summary.get('confusion_tp', 0)}"
        )
        print(f"- Top 10% predicted probability hit rate: {summary.get('top10_probability_hit_rate', np.nan):.4f}")


def save_classification_experiment_report(results: dict, config: PipelineConfig) -> None:
    if not results:
        return

    file_prefix = f"{config.market}_{config.stock_no}_{config.feature_set}_{config.target_col}"
    summary_csv = OUTPUT_DIR / f"{file_prefix}_classification_summary.csv"
    fold_csv = OUTPUT_DIR / f"{file_prefix}_classification_fold_report.csv"
    report_xlsx = OUTPUT_DIR / f"{file_prefix}_classification_experiment.xlsx"

    summary_rows = []
    fold_frames = []
    for target_name, result in results.items():
        row = dict(result["summary"])
        row["feature_set"] = config.feature_set
        row["feature_count"] = len(result.get("feature_cols", []))
        summary_rows.append(row)

        fold_report = result.get("fold_report", pd.DataFrame()).copy()
        if not fold_report.empty:
            fold_report["feature_set"] = config.feature_set
            fold_frames.append(fold_report)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if not fold_df.empty:
        fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not fold_df.empty:
            fold_df.to_excel(writer, sheet_name="fold_report", index=False)

    print("\n已輸出 classification experiment 報告：")
    print(f"1. {report_xlsx}")
    print(f"2. {summary_csv}")
    if not fold_df.empty:
        print(f"3. {fold_csv}")


def train_and_predict(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    out = add_risk_reward_target_score(
        df,
        horizon=config.horizon,
        risk_penalty=config.risk_penalty,
        opportunity_threshold=config.opportunity_threshold,
    )
    out = add_realized_history_features(out, horizon=config.horizon, report=False)

    pipeline, feature_cols, metrics = train_model(
        data=out,
        target_col=config.target_col,
        config=config,
    )

    # 用全部資料做最新分數預測，包括最後 horizon 天沒有 target 的資料。
    for col in metrics["numeric_features"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in metrics["categorical_features"]:
        if col in out.columns:
            out[col] = out[col].astype("object").where(out[col].notna(), "none").astype(str)

    X_all = out[feature_cols]
    out["predicted_target_score"] = pipeline.predict(X_all).clip(0, 1)
    out["predicted_score"] = out["predicted_target_score"]
    out["model_score"] = out["predicted_target_score"]
    out["predicted_investment_score"] = out["predicted_target_score"]
    out["signal"] = out["predicted_target_score"].apply(get_signal)

    return out, metrics




__all__ = [name for name in globals() if not name.startswith('__')]
