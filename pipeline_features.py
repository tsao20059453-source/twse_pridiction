from __future__ import annotations

from pipeline_data import *

# ============================================================
# 2. 基礎特徵與技術指標
# ============================================================

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["return_1d"] = out["close"].pct_change()
    out["return_3d"] = out["close"].pct_change(3)
    out["return_5d"] = out["close"].pct_change(5)
    out["return_10d"] = out["close"].pct_change(10)

    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["ma60"] = out["close"].rolling(60, min_periods=60).mean()
    out["ma20_slope"] = out["ma20"] / out["ma20"].shift(5) - 1
    out["ma60_slope"] = out["ma60"] / out["ma60"].shift(10) - 1

    out["ma5_distance"] = out["close"] / out["ma5"] - 1
    out["ma20_distance"] = out["close"] / out["ma20"] - 1
    out["ma60_distance"] = out["close"] / out["ma60"] - 1

    out["volatility_5d"] = out["return_1d"].rolling(5, min_periods=5).std()
    out["volatility_10d"] = out["return_1d"].rolling(10, min_periods=10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20, min_periods=20).std()

    out["high_low_range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["close_to_5d_high"] = out["close"] / out["high"].rolling(5, min_periods=5).max() - 1
    out["close_to_5d_low"] = out["close"] / out["low"].rolling(5, min_periods=5).min() - 1
    out["close_to_10d_high"] = out["close"] / out["high"].rolling(10, min_periods=10).max() - 1
    out["close_to_10d_low"] = out["close"] / out["low"].rolling(10, min_periods=10).min() - 1
    rolling_20d_high = out["high"].rolling(20, min_periods=20).max()
    rolling_20d_low = out["low"].rolling(20, min_periods=20).min()
    rolling_5d_high = out["high"].rolling(5, min_periods=5).max()
    rolling_10d_high = out["high"].rolling(10, min_periods=10).max()
    out["close_to_20d_high"] = out["close"] / rolling_20d_high - 1
    out["close_to_20d_low"] = out["close"] / rolling_20d_low - 1
    out["recent_5d_return_from_high"] = out["close"] / rolling_5d_high - 1
    out["recent_10d_return_from_high"] = out["close"] / rolling_10d_high - 1

    out["vol_ma5"] = out["volume"].rolling(5, min_periods=5).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["volume_ratio_5"] = out["volume"] / out["vol_ma5"]
    out["volume_ratio_20"] = out["volume"] / out["vol_ma20"]
    out["volume_ratio_20_change_3d"] = out["volume_ratio_20"] - out["volume_ratio_20"].shift(3)

    return out.replace([np.inf, -np.inf], np.nan)


def safe_divide(numerator, denominator):
    denom = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    result = pd.to_numeric(numerator, errors="coerce") / denom
    return result.replace([np.inf, -np.inf], np.nan)


def add_amount_heat_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = (
        pd.to_numeric(out["close"], errors="coerce")
        if "close" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    volume = (
        pd.to_numeric(out["volume"], errors="coerce")
        if "volume" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    # TWSE STOCK_DAY raw volume is share count, so do not multiply by 1000 here.
    out["trading_amount"] = close * volume
    out["amount_ma5"] = out["trading_amount"].rolling(5, min_periods=5).mean()
    out["amount_ma20"] = out["trading_amount"].rolling(20, min_periods=20).mean()
    out["amount_ratio_5"] = safe_divide(out["trading_amount"], out["amount_ma5"])
    out["amount_ratio_20"] = safe_divide(out["trading_amount"], out["amount_ma20"])
    out["amount_change_5d"] = safe_divide(out["amount_ma5"], out["amount_ma5"].shift(5)) - 1
    out["amount_change_10d"] = safe_divide(out["amount_ma5"], out["amount_ma5"].shift(10)) - 1

    amount_mean20 = out["trading_amount"].rolling(20, min_periods=20).mean()
    amount_std20 = out["trading_amount"].rolling(20, min_periods=20).std().replace(0, np.nan)
    out["amount_zscore_20"] = safe_divide(out["trading_amount"] - amount_mean20, amount_std20)

    return out.replace([np.inf, -np.inf], np.nan)


def build_market_environment_frame(market_df: pd.DataFrame) -> pd.DataFrame:
    market = standardize_market_index_data(market_df)
    feature_cols = ["date"] + MARKET_RAW_FEATURES + MARKET_NUMERIC_FEATURES + MARKET_CATEGORICAL_FEATURES
    if market.empty:
        return pd.DataFrame(columns=feature_cols)

    market = market.sort_values("date").reset_index(drop=True)
    market["market_return_1d"] = market["market_close"].pct_change(1)
    market["market_return_3d"] = market["market_close"] / market["market_close"].shift(3) - 1
    market["market_return_5d"] = market["market_close"] / market["market_close"].shift(5) - 1
    market["market_return_10d"] = market["market_close"] / market["market_close"].shift(10) - 1
    market["market_return_20d"] = market["market_close"] / market["market_close"].shift(20) - 1

    market["market_ma5"] = market["market_close"].rolling(5, min_periods=5).mean()
    market["market_ma20"] = market["market_close"].rolling(20, min_periods=20).mean()
    market["market_ma60"] = market["market_close"].rolling(60, min_periods=60).mean()
    market["market_ma5_distance"] = market["market_close"] / market["market_ma5"] - 1
    market["market_ma20_distance"] = market["market_close"] / market["market_ma20"] - 1
    market["market_ma60_distance"] = market["market_close"] / market["market_ma60"] - 1

    market["market_volatility_10d"] = market["market_return_1d"].rolling(10, min_periods=10).std()
    market["market_volatility_20d"] = market["market_return_1d"].rolling(20, min_periods=20).std()
    market["market_above_ma20"] = np.where(
        market["market_close"].notna() & market["market_ma20"].notna(),
        (market["market_close"] > market["market_ma20"]).astype(int),
        np.nan,
    )
    market["market_above_ma60"] = np.where(
        market["market_close"].notna() & market["market_ma60"].notna(),
        (market["market_close"] > market["market_ma60"]).astype(int),
        np.nan,
    )

    valid_regime = (
        market["market_close"].notna()
        & market["market_ma20"].notna()
        & market["market_ma60"].notna()
    )
    regime = pd.Series(pd.NA, index=market.index, dtype="object")
    regime.loc[valid_regime] = "neutral"
    regime.loc[
        valid_regime
        & (market["market_close"] > market["market_ma20"])
        & (market["market_ma20"] > market["market_ma60"])
    ] = "bullish"
    regime.loc[
        valid_regime
        & (market["market_close"] < market["market_ma20"])
        & (market["market_close"] > market["market_ma60"])
    ] = "pullback"
    regime.loc[
        valid_regime
        & (market["market_close"] < market["market_ma60"])
    ] = "bearish"

    market["market_overheat_threshold"] = (
        market["market_ma20_distance"]
        .rolling(252, min_periods=252)
        .quantile(0.90)
    )
    overheat = (
        valid_regime
        & market["market_overheat_threshold"].notna()
        & (market["market_ma20_distance"] > market["market_overheat_threshold"])
    )
    regime.loc[overheat] = "overheat"
    market["market_trend_regime"] = regime

    for col in feature_cols:
        if col not in market.columns:
            market[col] = pd.NA

    return market[feature_cols].replace([np.inf, -np.inf], np.nan)


def add_market_environment_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    market_features = build_market_environment_frame(market_df)
    derived_cols = [col for col in market_features.columns if col != "date"]
    for col in derived_cols:
        if col in out.columns:
            out = out.drop(columns=[col])

    if market_features.empty:
        for col in derived_cols:
            out[col] = pd.NA
        return out.replace([np.inf, -np.inf], np.nan)

    out = out.merge(market_features, on="date", how="left")
    return out.replace([np.inf, -np.inf], np.nan)


def add_relative_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["symbol", "date"] if "symbol" in df.columns else ["date"]
    out = df.copy().sort_values(sort_cols).reset_index(drop=True)

    def num_col(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index)
        return pd.to_numeric(out[col], errors="coerce")

    if "return_20d" not in out.columns:
        if "symbol" in out.columns and "close" in out.columns:
            out["return_20d"] = out.groupby("symbol", sort=False)["close"].transform(
                lambda s: pd.to_numeric(s, errors="coerce") / pd.to_numeric(s, errors="coerce").shift(20) - 1
            )
        elif "close" in out.columns:
            close = pd.to_numeric(out["close"], errors="coerce")
            out["return_20d"] = close / close.shift(20) - 1
        else:
            out["return_20d"] = np.nan

    out["relative_return_5d"] = num_col("return_5d") - num_col("market_return_5d")
    out["relative_return_10d"] = num_col("return_10d") - num_col("market_return_10d")
    out["relative_return_20d"] = num_col("return_20d") - num_col("market_return_20d")
    out["stock_vs_market_ma20_distance"] = num_col("ma20_distance") - num_col("market_ma20_distance")
    out["stock_vs_market_ma60_distance"] = num_col("ma60_distance") - num_col("market_ma60_distance")

    return out.replace([np.inf, -np.inf], np.nan)


def add_stock_level_context_features(
    df: pd.DataFrame,
    config: PipelineConfig,
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    market_data = market_df if market_df is not None else get_market_index_data(config)
    if "symbol" in df.columns:
        frames = []
        for _, stock_df in df.groupby("symbol", sort=False):
            stock_out = stock_df.copy().sort_values("date").reset_index(drop=True)
            stock_out = add_amount_heat_features(stock_out)
            stock_out = add_market_environment_features(stock_out, market_data)
            stock_out = add_relative_strength_features(stock_out)
            frames.append(stock_out)
        out = pd.concat(frames, ignore_index=True) if frames else df.copy()
    else:
        out = df.copy().sort_values("date").reset_index(drop=True)
        out = add_amount_heat_features(out)
        out = add_market_environment_features(out, market_data)
        out = add_relative_strength_features(out)

    sort_cols = ["date", "symbol"] if "symbol" in out.columns else ["date"]
    return clean_duplicate_columns(out.sort_values(sort_cols).reset_index(drop=True))



def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["DIF"] = out["EMA12"] - out["EMA26"]
    out["DEA"] = out["DIF"].ewm(span=9, adjust=False).mean()
    out["MACD"] = (out["DIF"] - out["DEA"]) * 2
    return out


def add_kdj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["low_9"] = out["low"].rolling(window=9, min_periods=9).min()
    out["high_9"] = out["high"].rolling(window=9, min_periods=9).max()
    out["RSV"] = (out["close"] - out["low_9"]) / (out["high_9"] - out["low_9"]) * 100
    out["RSV"] = out["RSV"].replace([np.inf, -np.inf], np.nan)

    rsv = out["RSV"].to_numpy(dtype=float)
    n = len(rsv)
    alpha = 1.0 / 3.0

    # Positions where EWM update applies: index >= 1 and RSV is valid
    update_mask = ~np.isnan(rsv)
    if n > 0:
        update_mask[0] = False  # K[0]=D[0]=50 is fixed; the original loop never touches index 0
    update_idx = np.where(update_mask)[0]

    K = np.full(n, np.nan)
    D = np.full(n, np.nan)
    if n > 0:
        K[0] = 50.0
        D[0] = 50.0

    if len(update_idx) > 0:
        rsv_valid = rsv[update_idx]

        # Prepend the seed 50 so pandas EWM treats it as the initial state.
        # adjust=False: y[0]=x[0], y[i]=(1-α)*y[i-1]+α*x[i]  →  same as original 2/3 & 1/3 recurrence
        k_vals = (
            pd.Series(np.concatenate([[50.0], rsv_valid]))
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy()[1:]
        )
        d_vals = (
            pd.Series(np.concatenate([[50.0], k_vals]))
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy()[1:]
        )

        K[update_idx] = k_vals
        D[update_idx] = d_vals

    # NaN positions hold the previous value (forward-fill replicates the original else-branch)
    K = pd.Series(K).ffill().to_numpy()
    D = pd.Series(D).ffill().to_numpy()

    out["K"] = K
    out["D"] = D
    out["J"] = 3 * K - 2 * D
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["delta"] = out["close"].diff()
    out["gain"] = out["delta"].clip(lower=0)
    out["loss"] = -out["delta"].clip(upper=0)
    out["avg_gain"] = out["gain"].ewm(alpha=1 / period, adjust=False).mean()
    out["avg_loss"] = out["loss"].ewm(alpha=1 / period, adjust=False).mean()
    out["RS"] = out["avg_gain"] / out["avg_loss"]
    out["RSI14"] = 100 - (100 / (1 + out["RS"]))
    return out.replace([np.inf, -np.inf], np.nan)


def add_bias(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA6"] = out["close"].rolling(window=6, min_periods=6).mean()
    out["MA12"] = out["close"].rolling(window=12, min_periods=12).mean()
    out["MA24"] = out["close"].rolling(window=24, min_periods=24).mean()
    out["BIAS6"] = (out["close"] - out["MA6"]) / out["MA6"] * 100
    out["BIAS12"] = (out["close"] - out["MA12"]) / out["MA12"] * 100
    out["BIAS24"] = (out["close"] - out["MA24"]) / out["MA24"] * 100
    return out.replace([np.inf, -np.inf], np.nan)


def add_atr_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["prev_close"] = out["close"].shift(1)
    out["tr_hl"] = out["high"] - out["low"]
    out["tr_hpc"] = (out["high"] - out["prev_close"]).abs()
    out["tr_lpc"] = (out["low"] - out["prev_close"]).abs()
    out["true_range"] = out[["tr_hl", "tr_hpc", "tr_lpc"]].max(axis=1)
    out["atr_14"] = out["true_range"].rolling(period, min_periods=period).mean()
    out["atr_pct"] = out["atr_14"] / out["close"]
    out["atr_pct_change_5d"] = out["atr_pct"] / out["atr_pct"].shift(5) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def add_vr_features(df: pd.DataFrame, periods: tuple[int, ...] = (14, 26)) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    price_change = out["close"].diff()
    volume = pd.to_numeric(out["volume"], errors="coerce")

    up_volume = volume.where(price_change > 0, 0)
    down_volume = volume.where(price_change < 0, 0)
    flat_volume = volume.where(price_change == 0, 0)

    for period in periods:
        up_sum = up_volume.rolling(period, min_periods=period).sum()
        down_sum = down_volume.rolling(period, min_periods=period).sum()
        flat_sum = flat_volume.rolling(period, min_periods=period).sum()
        denominator = down_sum + 0.5 * flat_sum
        vr_col = f"VR_{period}"
        out[vr_col] = (up_sum + 0.5 * flat_sum) / denominator * 100
        out.loc[denominator <= 0, vr_col] = np.nan
        out[f"{vr_col}_change"] = out[vr_col] - out[vr_col].shift(1)

    return out.replace([np.inf, -np.inf], np.nan)


def add_technical_indicators(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    out = add_basic_features(out)
    out = add_macd(out)
    out = add_kdj(out)
    out = add_rsi(out, period=14)
    out = add_bias(out)
    out = add_atr_features(out, period=config.atr_period)
    out = add_vr_features(out)
    return out


# ============================================================
# 3. 市場結構特徵：FVG / Pivot / CHoCH / SR / OB / Fib
# ============================================================

def add_fvg_features(df: pd.DataFrame, min_gap_pct: float = 0.002) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    bull_mask = out["low"] > out["high"].shift(2)
    bear_mask = out["high"] < out["low"].shift(2)

    out["fvg_type"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[bull_mask, "fvg_type"] = "bullish"
    out.loc[bear_mask, "fvg_type"] = "bearish"

    out["fvg_low"] = np.nan
    out["fvg_high"] = np.nan

    out.loc[bull_mask, "fvg_low"] = out["high"].shift(2)[bull_mask]
    out.loc[bull_mask, "fvg_high"] = out["low"][bull_mask]
    out.loc[bear_mask, "fvg_low"] = out["high"][bear_mask]
    out.loc[bear_mask, "fvg_high"] = out["low"].shift(2)[bear_mask]

    out["fvg_mid"] = (out["fvg_low"] + out["fvg_high"]) / 2
    out["fvg_size"] = out["fvg_high"] - out["fvg_low"]
    out["fvg_size_pct"] = out["fvg_size"] / out["close"]

    small_gap_mask = out["fvg_size_pct"] < min_gap_pct
    out.loc[small_gap_mask, "fvg_type"] = pd.NA
    out.loc[small_gap_mask, ["fvg_low", "fvg_high", "fvg_mid", "fvg_size", "fvg_size_pct"]] = np.nan

    return out.replace([np.inf, -np.inf], np.nan)


def add_latest_active_gap_asof_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    逐日計算截至當天仍未回補的最近 FVG。
    這裡只使用當天與過去資料，不看未來，避免資料洩漏。
    """
    out = df.copy().sort_values("date").reset_index(drop=True)

    gap_cols = [
        "recent_active_gap_type", "recent_active_gap_date", "recent_active_gap_lower",
        "recent_active_gap_upper", "recent_active_gap_mid", "recent_active_gap_size",
        "recent_active_gap_size_pct", "recent_active_gap_distance_pct",
    ]
    for col in gap_cols:
        out[col] = pd.NA

    active_gaps: list[dict] = []

    for i, row in out.iterrows():
        current_low = row["low"]
        current_high = row["high"]
        current_close = row["close"]

        still_active = []
        for gap in active_gaps:
            if gap["type"] == "bullish":
                filled = pd.notna(current_low) and current_low <= gap["lower"]
            else:
                filled = pd.notna(current_high) and current_high >= gap["upper"]
            if not filled:
                still_active.append(gap)
        active_gaps = still_active

        if pd.notna(row["fvg_type"]):
            active_gaps.append({
                "type": row["fvg_type"],
                "date": row["date"],
                "lower": float(row["fvg_low"]),
                "upper": float(row["fvg_high"]),
                "mid": float(row["fvg_mid"]),
                "size": float(row["fvg_size"]),
                "size_pct": float(row["fvg_size_pct"]),
            })

        if active_gaps:
            latest = active_gaps[-1]
            lower = latest["lower"]
            upper = latest["upper"]

            if lower <= current_close <= upper:
                distance_pct = 0.0
            elif current_close < lower:
                distance_pct = (lower - current_close) / current_close
            else:
                distance_pct = (current_close - upper) / current_close

            out.loc[i, "recent_active_gap_type"] = latest["type"]
            out.loc[i, "recent_active_gap_date"] = latest["date"]
            out.loc[i, "recent_active_gap_lower"] = lower
            out.loc[i, "recent_active_gap_upper"] = upper
            out.loc[i, "recent_active_gap_mid"] = latest["mid"]
            out.loc[i, "recent_active_gap_size"] = latest["size"]
            out.loc[i, "recent_active_gap_size_pct"] = latest["size_pct"]
            out.loc[i, "recent_active_gap_distance_pct"] = distance_pct
        else:
            out.loc[i, "recent_active_gap_type"] = "none"

    return out.replace([np.inf, -np.inf], np.nan)


def add_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["pivot_high"] = np.nan
    out["pivot_low"] = np.nan

    for i in range(left, len(out) - right):
        high_window = out.loc[i - left:i + right, "high"]
        low_window = out.loc[i - left:i + right, "low"]
        current_high = out.at[i, "high"]
        current_low = out.at[i, "low"]

        if pd.notna(current_high) and current_high == high_window.max() and (high_window == current_high).sum() == 1:
            out.at[i, "pivot_high"] = current_high
        if pd.notna(current_low) and current_low == low_window.min() and (low_window == current_low).sum() == 1:
            out.at[i, "pivot_low"] = current_low

    return out


def add_recent_fib_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["fib_lookback_high"] = out["high"].rolling(lookback, min_periods=2).max()
    out["fib_lookback_low"] = out["low"].rolling(lookback, min_periods=2).min()

    fib_range = out["fib_lookback_high"] - out["fib_lookback_low"]
    out["fib_position"] = (out["close"] - out["fib_lookback_low"]) / fib_range
    out.loc[fib_range <= 0, "fib_position"] = np.nan
    out["fib_position"] = out["fib_position"].clip(lower=0, upper=1)

    out["fib_0500_price"] = out["fib_lookback_low"] + fib_range * 0.5
    out["fib_0618_price"] = out["fib_lookback_low"] + fib_range * 0.618
    out["fib_0786_price"] = out["fib_lookback_low"] + fib_range * 0.786

    out["fib_zone"] = pd.NA
    out.loc[out["fib_position"] < 0.5, "fib_zone"] = "cheap"
    out.loc[out["fib_position"].between(0.5, 0.618, inclusive="left"), "fib_zone"] = "neutral"
    out.loc[out["fib_position"].between(0.618, 0.786, inclusive="both"), "fib_zone"] = "discount"
    out.loc[out["fib_position"] > 0.786, "fib_zone"] = "premium"

    out["fib_score"] = 0.0
    out.loc[out["fib_position"].between(0.618, 0.786, inclusive="both"), "fib_score"] = 1.0

    cheap_mask = out["fib_position"] < 0.5
    out.loc[cheap_mask, "fib_score"] = 1 - (out.loc[cheap_mask, "fib_position"] / 0.5)
    out.loc[out["fib_position"].isna(), "fib_score"] = np.nan
    out["fib_score"] = out["fib_score"].clip(lower=0, upper=1)

    return out.replace([np.inf, -np.inf], np.nan)


def detect_recent_2week_order_block_for_window(window: pd.DataFrame) -> dict:
    if len(window) < 12:
        return {
            "ob_type": "none",
            "ob_date": pd.NaT,
            "ob_lower": pd.NA,
            "ob_upper": pd.NA,
            "ob_impulse_date": pd.NaT,
            "ob_volume_ratio": pd.NA,
            "ob_days_since": pd.NA,
            "ob_strength_score": pd.NA,
        }

    work = window.copy().sort_values("date").reset_index(drop=True)
    last_10 = work.tail(10).copy()
    candidate = None

    for i in range(len(last_10) - 1, 0, -1):
        impulse = last_10.iloc[i]
        prev_candle = last_10.iloc[i - 1]

        bullish_impulse = (
            pd.notna(impulse["return_1d"])
            and impulse["return_1d"] > 0.02
            and pd.notna(impulse["vol_ma5"])
            and impulse["volume"] > impulse["vol_ma5"]
            and prev_candle["close"] < prev_candle["open"]
        )

        bearish_impulse = (
            pd.notna(impulse["return_1d"])
            and impulse["return_1d"] < -0.02
            and pd.notna(impulse["vol_ma5"])
            and impulse["volume"] > impulse["vol_ma5"]
            and prev_candle["close"] > prev_candle["open"]
        )

        if bullish_impulse or bearish_impulse:
            volume_ratio = (
                float(impulse["volume"] / impulse["vol_ma20"])
                if pd.notna(impulse["vol_ma20"]) and impulse["vol_ma20"] > 0
                else np.nan
            )
            candidate = {
                "ob_type": "bullish" if bullish_impulse else "bearish",
                "ob_date": prev_candle["date"],
                "ob_lower": float(prev_candle["low"]),
                "ob_upper": float(prev_candle["high"]),
                "ob_impulse_date": impulse["date"],
                "ob_volume_ratio": volume_ratio,
                "ob_days_since": int((work["date"].iloc[-1] - prev_candle["date"]).days),
            }
            break

    if candidate is None:
        latest_row = last_10.iloc[-1]

        latest_volume_ratio = (
            float(latest_row["volume"] / latest_row["vol_ma20"])
            if pd.notna(latest_row["vol_ma20"]) and latest_row["vol_ma20"] > 0
            else 0.0
        )

        candidate = {
            "ob_type": "fallback_range",
            "ob_date": latest_row["date"],
            "ob_lower": float(last_10["low"].min()),
            "ob_upper": float(last_10["high"].max()),

            # 注意：
            # fallback_range 不是由真正的衝動 K 棒觸發。
            # 這裡用最後一日作為參考日，避免輸出報表中 OB 後續欄位整欄空白。
            "ob_impulse_date": latest_row["date"],
            "ob_volume_ratio": latest_volume_ratio,
            "ob_days_since": 0,
        }

    volume_score = min(candidate["ob_volume_ratio"] / 2, 1) if pd.notna(candidate["ob_volume_ratio"]) else 0
    time_score = max(0, 1 - candidate["ob_days_since"] / 10)
    candidate["ob_strength_score"] = round(0.7 * volume_score + 0.3 * time_score, 4)
    return candidate


def add_order_block_asof_features(df: pd.DataFrame, lookback_days: int = 10) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    ob_cols = [
        "recent_2w_ob_type",
        "recent_2w_ob_date",
        "recent_2w_ob_lower",
        "recent_2w_ob_upper",
        "recent_2w_ob_impulse_date",
        "recent_2w_ob_volume_ratio",
        "recent_2w_ob_days_since",
        "recent_2w_ob_strength_score",
        "recent_2w_ob_is_fallback",
    ]
    for col in ob_cols:
        out[col] = pd.NA

    min_required = max(lookback_days + 2, 12)
    RECALC_INTERVAL = 5  # trading days (rows) between forced recalculations

    fvg_col = out["fvg_type"].to_numpy() if "fvg_type" in out.columns else None
    dates = out["date"]

    last_ob = None   # cached result from the last recalculation
    last_calc_i = -1  # row index of that recalculation

    for i in range(len(out)):
        if i + 1 < min_required:
            out.loc[i, "recent_2w_ob_type"] = "none"
            continue

        has_fvg = fvg_col is not None and pd.notna(fvg_col[i])
        need_recalc = (
            last_ob is None
            or has_fvg
            or (i - last_calc_i) >= RECALC_INTERVAL
        )

        if need_recalc:
            window = out.iloc[:i + 1].copy()
            ob = detect_recent_2week_order_block_for_window(window)
            last_ob = ob.copy()
            last_calc_i = i
        else:
            # Reuse cached result; advance ob_days_since by calendar days elapsed
            ob = last_ob.copy()
            elapsed = (dates.iloc[i] - dates.iloc[last_calc_i]).days
            base = ob["ob_days_since"]
            if pd.notna(base):
                ob["ob_days_since"] = int(base) + elapsed
                vr = ob["ob_volume_ratio"]
                volume_score = min(float(vr) / 2, 1) if pd.notna(vr) else 0
                time_score = max(0, 1 - ob["ob_days_since"] / 10)
                ob["ob_strength_score"] = round(0.7 * volume_score + 0.3 * time_score, 4)

        out.loc[i, "recent_2w_ob_type"] = ob["ob_type"]
        out.loc[i, "recent_2w_ob_date"] = ob["ob_date"]
        out.loc[i, "recent_2w_ob_lower"] = ob["ob_lower"]
        out.loc[i, "recent_2w_ob_upper"] = ob["ob_upper"]
        out.loc[i, "recent_2w_ob_impulse_date"] = ob["ob_impulse_date"]
        out.loc[i, "recent_2w_ob_volume_ratio"] = ob["ob_volume_ratio"]
        out.loc[i, "recent_2w_ob_days_since"] = ob["ob_days_since"]
        out.loc[i, "recent_2w_ob_strength_score"] = ob["ob_strength_score"]
        out.loc[i, "recent_2w_ob_is_fallback"] = 1 if ob["ob_type"] == "fallback_range" else 0

    return out.replace([np.inf, -np.inf], np.nan)


def cluster_levels(values: list[float], tol: float) -> list[dict]:
    vals = sorted([float(v) for v in values if pd.notna(v)])
    if not vals:
        return []

    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])

    zones = []
    for g in groups:
        zones.append({
            "level": float(np.mean(g)),
            "zone_low": float(min(g)),
            "zone_high": float(max(g)),
            "touches": len(g),
        })
    return zones


def detect_nearest_sr_for_window(window: pd.DataFrame, tol_pct: float = 0.003) -> dict:
    work = window.copy().sort_values("date").reset_index(drop=True)
    if work.empty:
        return {"nearest_support": None, "nearest_resistance": None}

    last_close = float(work["close"].iloc[-1])
    tol = last_close * tol_pct

    support_candidates = work["pivot_low"].dropna().tolist() if "pivot_low" in work.columns else []
    resistance_candidates = work["pivot_high"].dropna().tolist() if "pivot_high" in work.columns else []

    if {"recent_active_gap_type", "recent_active_gap_lower", "recent_active_gap_upper", "recent_active_gap_mid"}.issubset(work.columns):
        latest_gap = work.iloc[-1]
        if latest_gap["recent_active_gap_type"] == "bullish":
            support_candidates += [
                latest_gap["recent_active_gap_lower"],
                latest_gap["recent_active_gap_mid"],
                latest_gap["recent_active_gap_upper"],
            ]
        elif latest_gap["recent_active_gap_type"] == "bearish":
            resistance_candidates += [
                latest_gap["recent_active_gap_lower"],
                latest_gap["recent_active_gap_mid"],
                latest_gap["recent_active_gap_upper"],
            ]

    supports = cluster_levels(support_candidates, tol)
    resistances = cluster_levels(resistance_candidates, tol)

    supports = [z for z in supports if z["level"] <= last_close]
    resistances = [z for z in resistances if z["level"] >= last_close]

    supports = sorted(supports, key=lambda z: (-z["touches"], abs(last_close - z["level"])))
    resistances = sorted(resistances, key=lambda z: (-z["touches"], abs(last_close - z["level"])))

    return {
        "nearest_support": supports[0] if supports else None,
        "nearest_resistance": resistances[0] if resistances else None,
    }


def add_support_resistance_asof_features(df: pd.DataFrame, lookback: int = 150, tol_pct: float = 0.003) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    sr_cols = [
        "nearest_support_level", "nearest_support_zone_low", "nearest_support_zone_high",
        "nearest_resistance_level", "nearest_resistance_zone_low", "nearest_resistance_zone_high",
        "support_distance", "resistance_distance",
    ]
    for col in sr_cols:
        out[col] = np.nan

    RECALC_INTERVAL = 10  # trading days (rows) between forced recalculations

    pivot_high_col = out["pivot_high"].to_numpy() if "pivot_high" in out.columns else None
    pivot_low_col  = out["pivot_low"].to_numpy()  if "pivot_low"  in out.columns else None

    last_sr = None    # cached result from last recalculation
    last_calc_i = -1  # row index of that recalculation

    for i in range(len(out)):
        has_new_pivot = (
            (pivot_high_col is not None and pd.notna(pivot_high_col[i])) or
            (pivot_low_col  is not None and pd.notna(pivot_low_col[i]))
        )
        need_recalc = (
            last_sr is None
            or has_new_pivot
            or (i - last_calc_i) >= RECALC_INTERVAL
        )

        if need_recalc:
            start = max(0, i - lookback + 1)
            window = out.iloc[start: i + 1].copy()
            last_sr = detect_nearest_sr_for_window(window, tol_pct=tol_pct)
            last_calc_i = i

        # Always recompute distances from today's close, even when reusing cached levels
        last_close = out.loc[i, "close"]
        ns = last_sr["nearest_support"]
        nr = last_sr["nearest_resistance"]

        if ns is not None:
            out.loc[i, "nearest_support_level"]     = ns["level"]
            out.loc[i, "nearest_support_zone_low"]  = ns["zone_low"]
            out.loc[i, "nearest_support_zone_high"] = ns["zone_high"]
            out.loc[i, "support_distance"]          = last_close / ns["level"] - 1

        if nr is not None:
            out.loc[i, "nearest_resistance_level"]     = nr["level"]
            out.loc[i, "nearest_resistance_zone_low"]  = nr["zone_low"]
            out.loc[i, "nearest_resistance_zone_high"] = nr["zone_high"]
            out.loc[i, "resistance_distance"]          = nr["level"] / last_close - 1

    return out.replace([np.inf, -np.inf], np.nan)


def _compute_vol_high_lows(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vol_high1_low / vol_high2_low per-stock using 20-day rolling window."""
    if "volume" not in df.columns or "low" not in df.columns:
        df["vol_high1_low"] = np.nan
        df["vol_high2_low"] = np.nan
        return df

    out = df.copy()

    def _rolling_vol_lows(group: pd.DataFrame) -> pd.DataFrame:
        vol_arr = pd.to_numeric(group["volume"], errors="coerce").to_numpy()
        low_arr = pd.to_numeric(group["low"], errors="coerce").to_numpy()
        n = len(group)
        h1 = np.full(n, np.nan)
        h2 = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - 19)
            w_vol = vol_arr[start : i + 1]
            w_low = low_arr[start : i + 1]
            valid = ~np.isnan(w_vol) & ~np.isnan(w_low)
            if valid.sum() >= 1:
                sorted_lows = w_low[valid][np.argsort(w_vol[valid])[::-1]]
                h1[i] = sorted_lows[0]
                if len(sorted_lows) >= 2:
                    h2[i] = sorted_lows[1]
        result = pd.DataFrame({"vol_high1_low": h1, "vol_high2_low": h2}, index=group.index)
        return result

    if "symbol" in out.columns:
        vhl = out.groupby("symbol", group_keys=False).apply(_rolling_vol_lows)
    else:
        vhl = _rolling_vol_lows(out)

    out["vol_high1_low"] = vhl["vol_high1_low"]
    out["vol_high2_low"] = vhl["vol_high2_low"]
    return out


def calculate_support_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score support quality [0, 1] based on proximity, pivot confirmation,
    volume confirmation, and order block credibility.
    """
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce")
    support = (
        pd.to_numeric(out["nearest_support_level"], errors="coerce")
        if "nearest_support_level" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    # Component 1: proximity — closer to support → higher score (0..1)
    if "support_distance_pct" in out.columns:
        dist_pct = pd.to_numeric(out["support_distance_pct"], errors="coerce") / 100
    else:
        safe_close = close.where(close > 0)
        dist_pct = (close - support) / safe_close
    proximity_score = (1 - (dist_pct / 0.15).clip(0, 1)).fillna(0.0)

    # Component 2: recent pivot low exists near the support level
    pivot_conf = pd.Series(0.0, index=out.index)
    if "pivot_low" in out.columns:
        pl = pd.to_numeric(out["pivot_low"], errors="coerce")
        safe_support = support.where(support > 0)
        near = (pl - support).abs() / safe_support < 0.02
        pivot_conf = (pd.notna(pl) & pd.notna(support) & near).astype(float)

    # Component 3: volume ratio at support tests
    vol_score = pd.Series(0.5, index=out.index)
    if "volume_ratio_20" in out.columns:
        vr = pd.to_numeric(out["volume_ratio_20"], errors="coerce")
        vol_score = ((vr.clip(0.5, 2.0) - 0.5) / 1.5).fillna(0.5)

    # Component 4: real (non-fallback) order block adds credibility
    ob_score = pd.Series(0.0, index=out.index)
    if "recent_2w_ob_type" in out.columns and "recent_2w_ob_is_fallback" in out.columns:
        has_ob = out["recent_2w_ob_type"].notna() & ~out["recent_2w_ob_type"].isin(["none", ""])
        is_fb = pd.to_numeric(out["recent_2w_ob_is_fallback"], errors="coerce").fillna(1) == 1
        ob_score = (has_ob & ~is_fb).astype(float)

    quality = (
        0.35 * proximity_score
        + 0.25 * pivot_conf
        + 0.20 * vol_score
        + 0.20 * ob_score
    ).clip(0, 1)

    out["support_quality_score"] = quality.where(pd.notna(support), other=np.nan)
    return out


def add_market_structure_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    out = add_fvg_features(out, min_gap_pct=config.fvg_min_gap_pct)
    out = add_latest_active_gap_asof_features(out)
    out = add_pivots(out, left=3, right=3)
    out = add_recent_fib_features(out, lookback=config.fib_lookback_days)
    out = add_order_block_asof_features(out, lookback_days=config.ob_lookback_days)
    out = add_support_resistance_asof_features(
        out,
        lookback=config.sr_lookback_days,
        tol_pct=config.sr_tolerance_pct,
    )
    out = _compute_vol_high_lows(out)
    out = calculate_support_quality_score(out)
    return out


# ============================================================
# 4. 風險控管特徵
# ============================================================

def add_stoploss_features(df: pd.DataFrame, atr_multiplier: float = 2.0) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    out["risk_entry_price"] = pd.to_numeric(out["close"], errors="coerce")
    out["risk_atr_14"] = pd.to_numeric(out["atr_14"], errors="coerce")
    out["risk_ob_bottom"] = pd.to_numeric(out["recent_2w_ob_lower"], errors="coerce")

    out["risk_atr_stop"] = out["risk_entry_price"] - atr_multiplier * out["risk_atr_14"]

    # 若 OB bottom 不存在，則使用 ATR stop；若兩者都存在，取較低者作為保守停損。
    out["risk_stoploss_price"] = out[["risk_ob_bottom", "risk_atr_stop"]].min(axis=1, skipna=True)
    out["risk_per_share"] = out["risk_entry_price"] - out["risk_stoploss_price"]
    out["risk_stoploss_pct"] = out["risk_per_share"] / out["risk_entry_price"]
    out["risk_atr_stop_distance_pct"] = (out["close"] - out["risk_atr_stop"]) / out["close"]
    out["risk_ob_bottom_distance_pct"] = (out["close"] - out["risk_ob_bottom"]) / out["close"]
    out["risk_stoploss_distance_pct"] = (out["close"] - out["risk_stoploss_price"]) / out["close"]
    out["risk_valid"] = np.where(out["risk_per_share"] > 0, 1, 0)

    return out.replace([np.inf, -np.inf], np.nan)


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sort_cols = ["date", "symbol"] if "symbol" in out.columns else ["date"]
    existing_sort_cols = [col for col in sort_cols if col in out.columns]
    if existing_sort_cols:
        out = out.sort_values(existing_sort_cols).reset_index(drop=True)

    missing_sources = set()
    group_key = "symbol" if "symbol" in out.columns else None

    def numeric_col(col: str) -> Optional[pd.Series]:
        if col not in out.columns:
            missing_sources.add(col)
            return None
        return pd.to_numeric(out[col], errors="coerce")

    def grouped_pct_change(col: str) -> Optional[pd.Series]:
        values = numeric_col(col)
        if values is None:
            return None
        if group_key:
            return values.groupby(out[group_key]).pct_change() * 100
        return values.pct_change() * 100

    def grouped_diff(col: str) -> Optional[pd.Series]:
        values = numeric_col(col)
        if values is None:
            return None
        if group_key:
            return values.groupby(out[group_key]).diff()
        return values.diff()

    ma_map = {
        "week": ("ma5", "ma5_distance"),
        "month": ("ma20", "ma20_distance"),
        "season": ("ma60", "ma60_distance"),
    }
    close = numeric_col("close")
    for label, (ma_col, distance_col) in ma_map.items():
        distance_feature = f"ma_{label}_distance_pct"
        if distance_feature not in out.columns:
            if distance_col in out.columns:
                out[distance_feature] = pd.to_numeric(out[distance_col], errors="coerce") * 100
            elif close is not None and ma_col in out.columns:
                ma_value = pd.to_numeric(out[ma_col], errors="coerce")
                out[distance_feature] = (close / ma_value - 1) * 100
            else:
                missing_sources.add(distance_col)
                missing_sources.add(ma_col)

        slope_feature = f"ma_{label}_slope"
        if slope_feature not in out.columns:
            slope = grouped_pct_change(ma_col)
            if slope is not None:
                out[slope_feature] = slope

    if "support_distance_pct" not in out.columns:
        support = numeric_col("nearest_support_level")
        if close is not None and support is not None:
            out["support_distance_pct"] = (close - support) / close * 100

    if "resistance_distance_pct" not in out.columns:
        resistance = numeric_col("nearest_resistance_level")
        if close is not None and resistance is not None:
            out["resistance_distance_pct"] = (resistance - close) / close * 100

    if "volume_ratio" not in out.columns:
        if "volume_ratio_20" in out.columns:
            out["volume_ratio"] = pd.to_numeric(out["volume_ratio_20"], errors="coerce")
        else:
            volume = numeric_col("volume")
            if volume is not None:
                if group_key:
                    vol_ma20 = volume.groupby(out[group_key]).transform(
                        lambda s: s.rolling(20, min_periods=20).mean()
                    )
                else:
                    vol_ma20 = volume.rolling(20, min_periods=20).mean()
                out["volume_ratio"] = volume / vol_ma20

    if "macd_cross_signal" not in out.columns:
        dif = numeric_col("DIF")
        dea = numeric_col("DEA")
        if dif is not None and dea is not None:
            out["macd_cross_signal"] = np.nan
            valid = dif.notna() & dea.notna()
            out.loc[valid, "macd_cross_signal"] = (dif[valid] > dea[valid]).astype(int)

    if "macd_hist_trend" not in out.columns:
        macd_trend = grouped_diff("MACD")
        if macd_trend is not None:
            out["macd_hist_trend"] = macd_trend

    if "kdj_cross_signal" not in out.columns:
        k_value = numeric_col("K")
        d_value = numeric_col("D")
        if k_value is not None and d_value is not None:
            out["kdj_cross_signal"] = np.nan
            valid = k_value.notna() & d_value.notna()
            out.loc[valid, "kdj_cross_signal"] = (k_value[valid] > d_value[valid]).astype(int)

    if "rsi_zone" not in out.columns:
        rsi = numeric_col("RSI14")
        if rsi is not None:
            out["rsi_zone"] = pd.cut(
                rsi,
                bins=[-np.inf, 30, 45, 55, 70, np.inf],
                labels=["oversold", "weak", "neutral", "strong", "overheated"],
                right=False,
            ).astype("object")

    if "bias_zone" not in out.columns:
        bias = numeric_col("BIAS24")
        if bias is not None:
            out["bias_zone"] = pd.cut(
                bias,
                bins=[-np.inf, -5, 5, np.inf],
                labels=["low", "neutral", "high"],
                right=False,
            ).astype("object")

    if "volume_trend_signal" not in out.columns and "volume_ratio" in out.columns:
        ratio = pd.to_numeric(out["volume_ratio"], errors="coerce")
        ratio_change = ratio.groupby(out[group_key]).diff() if group_key else ratio.diff()
        out["volume_trend_signal"] = np.select(
            [(ratio >= 1.2) & (ratio_change >= 0), (ratio <= 0.8) & (ratio_change <= 0)],
            [1, -1],
            default=0,
        )
        out.loc[ratio.isna(), "volume_trend_signal"] = np.nan

    if "VR_14_zone" not in out.columns and "VR_14" in out.columns:
        vr_14 = pd.to_numeric(out["VR_14"], errors="coerce")
        out["VR_14_zone"] = pd.cut(
            vr_14,
            bins=[-np.inf, 70, 100, 160, np.inf],
            labels=[
                "volume_selling_pressure",
                "weak_volume",
                "balanced_to_bullish_volume",
                "hot_volume",
            ],
            right=False,
        ).astype("object")

    if missing_sources:
        print("Missing structural source warning：")
        print(sorted(missing_sources))

    out = out.replace([np.inf, -np.inf], np.nan)
    # 籌碼與營收特徵（--chip-features 開啟時才加入；2026-10-26）
    from pipeline_chip import add_chip_features
    return add_chip_features(out)


__all__ = [name for name in globals() if not name.startswith('__')]
