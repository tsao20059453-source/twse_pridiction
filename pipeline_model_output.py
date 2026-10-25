from __future__ import annotations

from pipeline_model_core import *

def prepare_single_stock_feature_data(config: PipelineConfig) -> pd.DataFrame:
    print("\nSingle diagnostic Step 1/5：抓取與清理股價資料")
    price_df = get_last_n_years_trading_data(config)
    market_index_df = get_market_index_data(config)

    print("\nSingle diagnostic Step 2/5：計算技術指標")
    feature_df = add_technical_indicators(price_df, config)
    feature_df = add_stock_level_context_features(feature_df, config, market_df=market_index_df)

    print("\nSingle diagnostic Step 3/5：建立市場結構與風險特徵")
    feature_df = add_market_structure_features(feature_df, config)
    feature_df = add_stoploss_features(feature_df, atr_multiplier=config.atr_multiplier)
    feature_df = add_structural_features(feature_df)
    feature_df = add_cross_sectional_pool_features(feature_df)
    feature_df = clean_duplicate_columns(feature_df)
    print_new_feature_availability(feature_df, context="single_diagnostic single scope")

    print("\nSingle diagnostic Step 4/5：建立 target_score")
    diagnostic_df = add_risk_reward_target_score(
        feature_df,
        horizon=config.horizon,
        risk_penalty=config.risk_penalty,
        opportunity_threshold=config.opportunity_threshold,
    )
    diagnostic_df = add_realized_history_features(diagnostic_df, horizon=config.horizon, report=False)
    diagnostic_df = clean_duplicate_columns(diagnostic_df)

    if "symbol" not in diagnostic_df.columns:
        diagnostic_df["symbol"] = config.stock_no
    if "market" not in diagnostic_df.columns:
        diagnostic_df["market"] = config.market.upper()
    if "date" in diagnostic_df.columns:
        diagnostic_df["date"] = pd.to_datetime(diagnostic_df["date"], errors="coerce")
        diagnostic_df = diagnostic_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return diagnostic_df


def prepare_common_single_diagnostic_feature_data(config: PipelineConfig, top_n: int = 30) -> pd.DataFrame:
    if config.market.upper() != "TWSE":
        raise ValueError("single_diagnostic common scope 目前只支援 TWSE")

    print("\nSingle diagnostic common scope：建立多股票共同訓練資料")
    stock_list = fetch_twse_top_turnover_stocks(limit=top_n)
    symbols = stock_list["symbol"].astype(str).tolist()
    target_symbol = str(config.stock_no).strip()

    if target_symbol not in symbols:
        extra_stock = pd.DataFrame([{
            "symbol": target_symbol,
            "stock_name": "",
            "today_turnover": np.nan,
        }])
        stock_list = pd.concat([stock_list, extra_stock], ignore_index=True)
        symbols.append(target_symbol)
        print("指定股票不在 Top-N 清單中，已強制加入共同診斷股票池")

    common_config = replace(config, stock_no="ALL")
    combined_data = fetch_and_prepare_multi_stock_data_with_config(
        symbols=symbols,
        base_config=common_config,
    )
    combined_data = clean_duplicate_columns(combined_data)

    if "symbol" not in combined_data.columns:
        raise ValueError("common diagnostic 合併資料缺少 symbol 欄位。")
    if target_symbol not in set(combined_data["symbol"].astype(str)):
        raise ValueError(f"common diagnostic 合併資料中找不到指定股票：{target_symbol}")

    print("\nSingle diagnostic common scope 資料完成")
    print(f"共同訓練股票數量：{combined_data['symbol'].nunique()}")
    print(f"共同訓練資料筆數：{len(combined_data)}")
    print(f"指定股票 {target_symbol} 資料筆數：{int((combined_data['symbol'].astype(str) == target_symbol).sum())}")

    return combined_data


def get_series_or_default(df: pd.DataFrame, col: str, default=np.nan):
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def prepare_single_diagnostic_model_data(
    data: pd.DataFrame,
    config: PipelineConfig,
) -> dict:
    work = clean_duplicate_columns(data.copy())
    work = work.sort_values(["date", "symbol"] if "symbol" in work.columns else ["date"]).reset_index(drop=True)
    warn_if_deprecated_feature_set(config.feature_set)
    if feature_set_uses_realized_history(config.feature_set):
        work = add_realized_history_features(work, horizon=config.horizon)

    numeric_features, categorical_features = get_feature_columns(work, feature_set=config.feature_set)
    if config.target_col not in work.columns:
        raise ValueError(f"target_col 不存在：{config.target_col}")

    for col in numeric_features:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in categorical_features:
        if col in work.columns:
            work[col] = work[col].astype("object").where(work[col].notna(), "none").astype(str)

    diagnostic_cols = [
        "target_score_v2",
        config.target_col,
        "target_return_10d_cs_rank",
        "target_safety_10d_cs_rank",
        f"target_return_{config.horizon}d_cs_rank",
        f"target_safety_{config.horizon}d_cs_rank",
        f"target_abs_drawdown_safety_{config.horizon}d",
        f"future_return_{config.horizon}d",
        f"future_drawdown_{config.horizon}d",
        f"risk_adjusted_return_{config.horizon}d",
    ]
    for col in dict.fromkeys(diagnostic_cols):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    base_cols = [
        "date", "symbol", "market", "close",
        "target_score_v2", config.target_col,
        "target_return_10d_cs_rank",
        "target_safety_10d_cs_rank",
        f"target_return_{config.horizon}d_cs_rank",
        f"target_safety_{config.horizon}d_cs_rank",
        f"target_abs_drawdown_safety_{config.horizon}d",
        f"future_return_{config.horizon}d",
        f"future_drawdown_{config.horizon}d",
        f"risk_adjusted_return_{config.horizon}d",
        "RSI14", "MACD", "K", "D", "J", "BIAS24",
        "VR_14", "VR_26",
        "return_5d", "return_10d",
        "nearest_support_level", "nearest_resistance_level",
        "atr_14",
        "risk_entry_price", "risk_atr_14", "risk_ob_bottom", "risk_atr_stop", "risk_stoploss_price",
        "recent_2w_ob_type", "recent_2w_ob_is_fallback",
        "recent_active_gap_type",
        "atr_pct", "volatility_10d", "risk_atr_stop_distance_pct", "risk_stoploss_pct",
        "fib_position", "fib_zone", "support_distance", "resistance_distance",
    ]
    base_cols = [col for col in dict.fromkeys(base_cols) if col in work.columns]
    model_cols = list(dict.fromkeys(base_cols + numeric_features + categorical_features))
    model_df = work[model_cols].copy()
    model_df = clean_duplicate_columns(model_df)

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

    prediction_feature_df = model_df.dropna(subset=["date"]).copy()
    training_df = model_df.dropna(subset=[config.target_col]).copy()
    if len(training_df) < 50:
        raise ValueError("可診斷資料太少，請增加歷史資料或確認 target/feature 欄位。")

    return {
        "model_df": training_df,
        "training_df": training_df,
        "prediction_feature_df": prediction_feature_df,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "removed_features": (
            leakage_removed_features
            + high_missing_removed_features
            + low_variance_removed_features
        ),
    }

def get_summary_value(summary, keys, default=np.nan):
    if summary is None:
        return default
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(summary, pd.DataFrame):
        if summary.empty:
            return default
        return get_summary_value(summary.iloc[0].to_dict(), keys, default=default)
    if isinstance(summary, pd.Series):
        return get_summary_value(summary.to_dict(), keys, default=default)
    if isinstance(summary, dict):
        for key in keys:
            if key in summary:
                return summary.get(key, default)
        lower_key_map = {str(existing_key).lower(): existing_key for existing_key in summary.keys()}
        for key in keys:
            matched_key = lower_key_map.get(str(key).lower())
            if matched_key is not None:
                return summary.get(matched_key, default)
    return default


def get_summary_metric(summary, key: str):
    return get_summary_value(summary, key, default=np.nan)


def format_validation_metric(value) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "N/A"
    return f"{numeric_value:.4f}"


def format_validation_method(value) -> str:
    if pd.isna(value):
        return "N/A"
    text = str(value).strip()
    if not text:
        return "N/A"
    if text == "walk_forward_time_series_split":
        return "walk-forward time-series validation"
    return text


def build_model_performance_text(r2, spearman_corr) -> str:
    r2_value = pd.to_numeric(pd.Series([r2]), errors="coerce").iloc[0]
    spearman_value = pd.to_numeric(pd.Series([spearman_corr]), errors="coerce").iloc[0]
    if pd.notna(r2_value) and pd.notna(spearman_value) and r2_value >= 0.05 and spearman_value >= 0.20:
        return "模型在此股票上具有一定解釋力與排序能力，可作為投資決策的重要輔助依據。"
    if pd.notna(r2_value) and pd.notna(spearman_value) and r2_value < 0 and spearman_value >= 0.10:
        return "模型在此股票上的單點解釋力有限，但排序方向仍有弱正相關，因此適合作為輔助判斷，不宜單獨作為進出場依據。"
    if pd.notna(spearman_value) and spearman_value < 0:
        return "模型在此股票上的排序方向不穩，建議主要參考風險警示與指標依據，不宜單獨依賴分數。"
    return "模型表現接近中性，結果應搭配技術指標、風險警示與人工判斷共同使用。"


def build_baseline_comparison_text(mae, rmse, baseline_mae, baseline_rmse) -> str:
    mae_value = pd.to_numeric(pd.Series([mae]), errors="coerce").iloc[0]
    rmse_value = pd.to_numeric(pd.Series([rmse]), errors="coerce").iloc[0]
    baseline_mae_value = pd.to_numeric(pd.Series([baseline_mae]), errors="coerce").iloc[0]
    baseline_rmse_value = pd.to_numeric(pd.Series([baseline_rmse]), errors="coerce").iloc[0]
    if pd.isna(mae_value) or pd.isna(rmse_value) or pd.isna(baseline_mae_value) or pd.isna(baseline_rmse_value):
        return "目前無 baseline 資訊可比較。"
    if mae_value < baseline_mae_value and rmse_value < baseline_rmse_value:
        return "模型誤差低於 baseline，代表模型相較於單純使用歷史平均值預測具有較佳表現。"
    if mae_value < baseline_mae_value and rmse_value >= baseline_rmse_value:
        return "模型 MAE 優於 baseline，但 RMSE 未明顯改善，代表模型在一般誤差上有改善，但仍可能受到部分極端誤差影響。"
    if mae_value >= baseline_mae_value and rmse_value >= baseline_rmse_value:
        return (
            "模型誤差未優於 baseline，代表此股票的單股預測難度較高，因此結果應主要作為輔助參考。"
            "由於模型未穩定優於 baseline，本系統不將模型分數作為直接買賣依據，而是將其定位為第一層輔助訊號。"
            "最終決策將更重視第二層規則式過濾、風險報酬比與停損可行性。"
        )
    return "模型與 baseline 各有優劣，建議搭配技術指標與風險警示共同判斷。"


def build_ranking_performance_text(spearman_corr) -> str:
    spearman_value = pd.to_numeric(pd.Series([spearman_corr]), errors="coerce").iloc[0]
    if pd.isna(spearman_value):
        return "目前無排序能力資訊可解讀。"
    if spearman_value > 0.20:
        return "模型排序方向具有一定參考價值。"
    if spearman_value >= 0:
        return (
            "模型排序方向偏弱，僅適合作為輔助。"
            "由於模型未穩定優於 baseline，本系統不將模型分數作為直接買賣依據，而是將其定位為第一層輔助訊號。"
            "最終決策將更重視第二層規則式過濾、風險報酬比與停損可行性。"
        )
    return (
        "模型排序方向不穩，不宜單獨依賴分數。"
        "由於模型未穩定優於 baseline，本系統不將模型分數作為直接買賣依據，而是將其定位為第一層輔助訊號。"
        "最終決策將更重視第二層規則式過濾、風險報酬比與停損可行性。"
    )


def format_report_count(value) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "N/A"
    return str(int(numeric_value))


def print_presentation_process_header() -> None:
    print("=" * 70)
    print("資料處理與模型建置流程")
    print("=" * 70)
    print()
    print("Step 1/6：取得市場股票清單與成交金額排名")
    print("Step 2/6：抓取近三年個股歷史資料")
    print("Step 3/6：計算技術指標 RSI / MACD / KDJ / BIAS / ATR / VR")
    print("Step 4/6：計算價格位置、支撐壓力、Fib、Order Block、Gap 與風險特徵")
    print("Step 5/6：建立共同模型並產生 safety / return / combined 決策分數")
    print("Step 6/6：輸出投資決策報告與模型驗證摘要")
    print()


def print_presentation_data_summary(
    config: PipelineConfig,
    data: pd.DataFrame,
    top_n: Optional[int] = None,
    target_symbol: Optional[str] = None,
) -> None:
    print("-" * 70)
    print("資料合併摘要")
    print("-" * 70)
    print(f"市場：{config.market.upper()}")
    if top_n is not None:
        print(f"多股票模式使用的 top_n：{top_n}")
    if "symbol" in data.columns:
        print(f"股票數量：{data['symbol'].nunique()}")
        if target_symbol:
            target_rows = int((data["symbol"].astype(str) == str(target_symbol)).sum())
            print(f"指定股票資料筆數：{target_rows}")
    print(f"總資料筆數：{len(data)}")
    if "date" in data.columns and not data.empty:
        date_values = pd.to_datetime(data["date"], errors="coerce").dropna()
        if not date_values.empty:
            print(f"日期範圍：{date_values.min().strftime('%Y-%m-%d')} ~ {date_values.max().strftime('%Y-%m-%d')}")
    print()


def print_presentation_model_settings(config: PipelineConfig, top_n: Optional[int] = None, investor_mode: bool = True) -> None:
    importance_enabled = bool(ENABLE_IMPORTANCE_FEATURE_SELECTION and config.run_feature_importance)
    print("=" * 70)
    print("本次模型設定")
    print("=" * 70)
    print()
    if not investor_mode:
        print(f"target_col：{config.target_col}")
        print(f"feature_set：{config.feature_set}")
        print(f"model_type：{config.model_type}")
        print(f"model_profile：{config.model_profile}")
        print(f"diagnostic_scope：{config.diagnostic_scope}")
        print(f"top_n：{format_report_count(top_n)}")
        print(f"display_mode：{config.display_mode}")
        print(f"importance_based_feature_removal：{importance_enabled}")
        print(f"run_feature_importance：{config.run_feature_importance}")
        print()
    print("目前主模型 target：")
    if config.target_col == "target_safety_10d_cs_rank":
        print("- target_safety_10d_cs_rank：安全 / 下行風險優先的 10 日橫斷面相對分數")
    elif config.target_col == "target_return_10d_cs_rank":
        print("- target_return_10d_cs_rank：報酬機會優先的 10 日橫斷面相對分數")
    else:
        print(f"- {config.target_col}：使用者指定的主模型 target")
    print()
    print("雙模型決策分數定義：")
    print("- target_safety_10d_cs_rank：安全 / 下行風險優先的 10 日橫斷面相對分數")
    print("- target_return_10d_cs_rank：報酬機會輔助分數")
    print("- combined_decision_score = 0.65 * safety_score + 0.35 * return_score")
    print()
    print("最終決策分數公式：")
    print(f"- {FINAL_SCORE_FORMULA_TEXT}")
    print(f"- final_score_weight_profile：{FINAL_SCORE_WEIGHT_PROFILE}")
    print(f"- 公式說明：{FINAL_SCORE_FORMULA_EXPLANATION}")
    print()


def print_model_validation_body(summary, is_multi_stock: bool = False) -> None:
    validation_method = get_summary_value(summary, "validation_method", default="walk-forward time-series validation")
    n_folds = get_summary_value(summary, ["walk_forward_folds", "n_folds"], default=np.nan)
    total_valid_rows = get_summary_value(summary, ["total_valid_rows", "test_rows", "oos_rows"], default=np.nan)
    mae = get_summary_value(summary, ["MAE", "mae"], default=np.nan)
    rmse = get_summary_value(summary, ["RMSE", "rmse"], default=np.nan)
    r2 = get_summary_value(summary, ["R2", "r2"], default=np.nan)
    spearman_corr = get_summary_value(summary, ["ranking_spearman", "spearman_corr"], default=np.nan)
    baseline_mae = get_summary_value(summary, ["baseline_mae", "baseline_MAE", "Baseline MAE"], default=np.nan)
    baseline_rmse = get_summary_value(summary, ["baseline_rmse", "baseline_RMSE", "Baseline RMSE"], default=np.nan)
    baseline_r2 = get_summary_value(summary, ["baseline_r2", "baseline_R2", "Baseline R2"], default=np.nan)
    top5_lift = get_summary_value(summary, "top5_lift", default=np.nan)
    top10_lift = get_summary_value(summary, "top10_lift", default=np.nan)
    long_short_spread = get_summary_value(summary, "long_short_spread", default=np.nan)

    print(f"驗證方式：{format_validation_method(validation_method)}")
    print(f"Walk-forward 折數：{format_report_count(n_folds)}")
    print(f"驗證總筆數：{format_report_count(total_valid_rows)}")
    print()
    print(f"模型 MAE：{format_validation_metric(mae)}")
    print(f"Baseline MAE：{format_validation_metric(baseline_mae)}")
    print(f"模型 RMSE：{format_validation_metric(rmse)}")
    print(f"Baseline RMSE：{format_validation_metric(baseline_rmse)}")
    print(f"模型 R2：{format_validation_metric(r2)}")
    print(f"Baseline R2：{format_validation_metric(baseline_r2)}")
    print(f"Spearman：{format_validation_metric(spearman_corr)}")
    if is_multi_stock:
        print()
        print(f"Ranking Spearman：{format_validation_metric(spearman_corr)}")
        print(f"Top5 Lift：{format_validation_metric(top5_lift)}")
        print(f"Top10 Lift：{format_validation_metric(top10_lift)}")
        print(f"Long-Short Spread：{format_validation_metric(long_short_spread)}")
    print()
    print("【模型表現解讀】")
    print(build_baseline_comparison_text(mae, rmse, baseline_mae, baseline_rmse))
    print()
    print("【排序能力解讀】")
    print(build_ranking_performance_text(spearman_corr))


def print_presentation_model_validation_summary(summary, is_multi_stock: bool = False) -> None:
    print("=" * 70)
    print("模型驗證與表現摘要")
    print("=" * 70)
    print()
    print_model_validation_body(summary, is_multi_stock=is_multi_stock)
    print()


def report_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, pd.Series):
        return value.dropna().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if pd.isna(value):
        return []
    return [value]


def print_report_list(values) -> None:
    items = report_list(values)
    if not items:
        print("[]")
        return
    for item in items:
        print(f"- {item}")


def build_fold_info_from_metrics(metrics: Optional[dict]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    report = metrics.get("walk_forward_report")
    if isinstance(report, pd.DataFrame):
        return report.copy()
    return pd.DataFrame()


def print_presentation_feature_fold_appendix(
    config: PipelineConfig,
    feature_cols=None,
    numeric_features=None,
    categorical_features=None,
    removed_features=None,
    fold_info=None,
    metrics: Optional[dict] = None,
) -> None:
    metrics = metrics or {}
    feature_cols = feature_cols if feature_cols is not None else metrics.get("feature_cols", [])
    numeric_features = numeric_features if numeric_features is not None else metrics.get("numeric_features", [])
    categorical_features = categorical_features if categorical_features is not None else metrics.get("categorical_features", [])
    removed_features = removed_features if removed_features is not None else metrics.get("removed_features", [])
    fold_df = pd.DataFrame(fold_info) if fold_info is not None else build_fold_info_from_metrics(metrics)
    importance_enabled = bool(metrics.get(
        "importance_feature_selection_enabled",
        ENABLE_IMPORTANCE_FEATURE_SELECTION and config.run_feature_importance,
    ))

    print("=" * 70)
    print("模型特徵與驗證切分摘要")
    print("=" * 70)
    print()
    print(f"模型實際使用特徵數量：{len(report_list(feature_cols))}")
    print()
    print("模型實際使用特徵清單：")
    print_report_list(feature_cols)
    print()
    print("數值特徵：")
    print_report_list(numeric_features)
    print()
    print("類別特徵：")
    print_report_list(categorical_features)
    print()
    print("被移除的特徵：")
    print_report_list(removed_features)
    print()
    print(f"Importance-based feature removal enabled：{importance_enabled}")
    print("Feature importance 目前只輸出報告，不自動刪除特徵。")
    print()
    print("Walk-forward 驗證切分：")
    if fold_df.empty:
        print("N/A")
    else:
        for _, row in fold_df.iterrows():
            fold = row.get("fold", "N/A")
            train_rows = row.get("train_rows", "N/A")
            valid_rows = row.get("valid_rows", row.get("test_rows", "N/A"))
            start_date = row.get("valid_start_date", row.get("test_start_date", "N/A"))
            end_date = row.get("valid_end_date", row.get("test_end_date", "N/A"))
            target_rows = row.get("target_symbol_valid_rows", "N/A")
            print(
                f"Fold {format_report_count(fold)}: "
                f"train={format_report_count(train_rows)}, "
                f"valid={format_report_count(valid_rows)}, "
                f"target_symbol_valid={format_report_count(target_rows)}, "
                f"date={format_report_date(start_date)} ~ {format_report_date(end_date)}"
            )
    print()


def print_final_investment_decision(output_df: pd.DataFrame, config: PipelineConfig, summary=None) -> None:
    if output_df.empty:
        print("沒有可輸出的投資決策資料。")
        return

    latest_row = output_df.sort_values("date").tail(1).iloc[0]
    symbol = latest_row.get("symbol", config.stock_no)
    market = latest_row.get("market", config.market.upper())
    date_text = format_report_date(latest_row.get("date"))
    close_text = format_report_price(latest_row.get("close"))
    final_label = translate_decision_label(latest_row.get("final_decision_label", "中性觀察"))
    decision_summary = normalize_text_for_report(
        latest_row.get("enhanced_decision_summary_text", latest_row.get("decision_summary_text", ""))
    )
    suggested_action = normalize_text_for_report(latest_row.get("suggested_action_text", ""))
    risk_warning = normalize_text_for_report(latest_row.get("risk_warning", "無明顯風險警示"))

    print("=" * 70)
    print("未來第 10 個交易日投資決策建議")
    print("=" * 70)
    print()
    print(f"股票代碼：{symbol}")
    print(f"市場：{market}")
    print(f"資料蒐集截止日：{date_text}")
    print(f"最新收盤價：{close_text}")
    print("預測目標：未來第 10 個交易日")
    print()
    print("-" * 70)
    print("一、決策結論")
    print("-" * 70)
    print(f"最終決策標籤：{final_label}")
    print(f"第一層模型綜合分數：{format_decision_score(latest_row.get('combined_decision_score'))}")
    print(f"最終決策分數：{format_decision_score(latest_row.get('final_decision_score'))}")
    print(f"安全分數：{format_decision_score(latest_row.get('safety_score'))}")
    print(f"報酬機會分數：{format_decision_score(latest_row.get('return_score'))}")
    print(f"decision_action：{translate_decision_action(latest_row.get('decision_action', 'NA'))}")
    entry_plan_override = bool(latest_row.get("entry_plan_override", False))
    print(f"進場計畫狀態：{'回測進場 R/R 達標，建議等待回測' if entry_plan_override else '維持原決策'}")
    # ── 大盤過濾狀態（同步回測引擎）────────────────────────────────────
    _market_above_raw = latest_row.get("market_above_ma20")
    try:
        _market_above = bool(int(float(_market_above_raw))) if pd.notna(_market_above_raw) else None
    except (TypeError, ValueError):
        _market_above = None
    _mkt_dist = pd.to_numeric(pd.Series([latest_row.get("market_ma20_distance")]), errors="coerce").iloc[0]
    if _market_above is False:
        _d = f"（距 MA20：{_mkt_dist:.2%}）" if pd.notna(_mkt_dist) else ""
        print(f"大盤過濾：TAIEX 低於 MA20{_d}，回測引擎將封鎖所有新進場")
    elif _market_above is True:
        _d = f"（超越 MA20：+{_mkt_dist:.2%}）" if pd.notna(_mkt_dist) else ""
        print(f"大盤狀態：TAIEX 高於 MA20{_d}")
    else:
        print("大盤狀態：無法取得 TAIEX 資料（回測引擎視為不過濾）")
    print()
    print(decision_summary)
    print()
    print("-" * 70)
    print("二、目前技術面依據")
    print("-" * 70)
    print(f"RSI14：{format_report_value(latest_row.get('RSI14'))}")
    print(f"MACD：{format_report_value(latest_row.get('MACD'))}")
    print(f"K：{format_report_value(latest_row.get('K'))}")
    print(f"D：{format_report_value(latest_row.get('D'))}")
    print(f"J：{format_report_value(latest_row.get('J'))}")
    print(f"BIAS24：{format_bias_value(latest_row.get('BIAS24'))}")
    print(f"VR_14：{format_report_value(latest_row.get('VR_14'))}")
    print(f"VR_26：{format_report_value(latest_row.get('VR_26'))}")
    print(f"return_5d：{format_report_value(latest_row.get('return_5d'), pct=True)}")
    print(f"return_10d：{format_report_value(latest_row.get('return_10d'), pct=True)}")
    print()
    print("-" * 70)
    print("三、價格位置與風險依據")
    print("-" * 70)
    print(f"fib_position：{format_report_value(latest_row.get('fib_position'))}")
    print(f"fib_zone：{normalize_text_for_report(latest_row.get('fib_zone', 'NA'))}")
    print(f"nearest_support_level：{format_report_price(latest_row.get('nearest_support_level'))}")
    print(f"nearest_resistance_level：{format_report_price(latest_row.get('nearest_resistance_level'))}")
    print(f"support_distance：{format_report_value(latest_row.get('support_distance'), pct=True)}")
    print(f"resistance_distance：{format_report_value(latest_row.get('resistance_distance'), pct=True)}")
    print(f"atr_14：{format_report_value(latest_row.get('atr_14'))}")
    print(f"atr_pct：{format_report_value(latest_row.get('atr_pct'), pct=True)}")
    print(f"主要停損價 stop_loss_price_adjusted：{format_report_price(latest_row.get('stop_loss_price_adjusted'))}")
    print(f"主要停損距離 stop_loss_pct：{format_report_value(latest_row.get('stop_loss_pct'), pct=True)}")
    print(f"原始風險欄位 risk_entry_price：{format_report_price(latest_row.get('risk_entry_price'))}")
    print(f"原始風險欄位 risk_stoploss_price：{format_report_price(latest_row.get('risk_stoploss_price'))}")
    print(f"原始風險欄位 risk_stoploss_pct：{format_report_value(latest_row.get('risk_stoploss_pct'), pct=True)}")
    print(f"risk_atr_stop_distance_pct：{format_report_value(latest_row.get('risk_atr_stop_distance_pct'), pct=True)}")
    print(f"recent_2w_ob_type：{normalize_text_for_report(latest_row.get('recent_2w_ob_type', 'NA'))}")
    print(f"recent_2w_ob_is_fallback：{format_report_value(latest_row.get('recent_2w_ob_is_fallback'), decimals=0)}")
    print(f"recent_active_gap_type：{normalize_text_for_report(latest_row.get('recent_active_gap_type', 'NA'))}")
    print(f"support_quality_score：{format_report_value(latest_row.get('support_quality_score'))}")
    support_quality_score_val = pd.to_numeric(pd.Series([latest_row.get('support_quality_score')]), errors='coerce').iloc[0]
    if pd.notna(support_quality_score_val):
        if support_quality_score_val >= 0.7:
            quality_text = '強支撐，歷史測試有效且量能配合'
        elif support_quality_score_val >= 0.4:
            quality_text = '中性支撐，需觀察量能確認'
        else:
            quality_text = '弱支撐，歷史測試有效率偏低'
        print(f"支撐品質說明：{quality_text}")
    print()
    print("-" * 70)
    print("四、第二層規則式過濾")
    print("-" * 70)
    print(f"rule_filter_score：{format_decision_score(latest_row.get('rule_filter_score'))}")
    print(f"technical_filter_result：{translate_technical_filter_result(latest_row.get('technical_filter_result', 'NA'))}")
    print(f"rule_penalty_points：{format_report_value(latest_row.get('rule_penalty_points'))}")
    print(f"rule_filter_notes：{translate_rule_filter_notes(latest_row.get('rule_filter_notes', ''))}")
    print()
    print("-" * 70)
    print("五、交易可行性與風險報酬比")
    print("-" * 70)
    print(f"entry_price：{format_report_price(latest_row.get('entry_price'))}")
    print(f"stop_loss_price_adjusted：{format_report_price(latest_row.get('stop_loss_price_adjusted'))}")
    print(f"stop_loss_pct：{format_report_value(latest_row.get('stop_loss_pct'), pct=True)}")
    print(f"take_profit_price：{format_report_price(latest_row.get('take_profit_price'))}")
    print(f"risk_per_share：{format_report_price(latest_row.get('risk_per_share'))}")
    print(f"stop_loss_too_close_flag：{bool(latest_row.get('stop_loss_too_close_flag', False))}")
    print(f"practical_risk_per_share：{format_report_price(latest_row.get('practical_risk_per_share'))}")
    print(f"reward_per_share：{format_report_price(latest_row.get('reward_per_share'))}")
    print(f"risk_reward_ratio：{format_report_value(latest_row.get('risk_reward_ratio'))}")
    print(f"practical_risk_reward_ratio：{format_report_value(latest_row.get('practical_risk_reward_ratio'))}")
    print(f"risk_reward_score：{format_decision_score(latest_row.get('risk_reward_score'))}")
    print(f"practical_risk_reward_score：{format_decision_score(latest_row.get('practical_risk_reward_score'))}")
    print("practical_risk_reward_note：原始 R/R 可能因停損過近而被放大，因此系統採用 practical R/R 作為風險報酬判斷依據。")
    print(f"trade_veto_flag：{bool(latest_row.get('trade_veto_flag', False))}")
    print(f"trade_veto_reason：{translate_veto_reason(latest_row.get('trade_veto_reason', '')) or '無'}")
    print(f"trade_plan_notes：{translate_trade_plan_notes(latest_row.get('trade_plan_notes', ''))}")
    print()
    print("-" * 70)
    print("五之一、最佳進場計畫")
    print("-" * 70)
    print(f"【回測支撐進場】")
    print(f"建議進場區間：{format_report_price(latest_row.get('support_entry_zone_low'))} - {format_report_price(latest_row.get('support_entry_zone_high'))}")
    print(f"進場參考價：{format_report_price(latest_row.get('support_entry_price'))}")
    print(f"【突破確認進場】")
    print(f"建議進場區間：{format_report_price(latest_row.get('breakout_entry_zone_low'))} - {format_report_price(latest_row.get('breakout_entry_zone_high'))}")
    print(f"進場參考價：{format_report_price(latest_row.get('breakout_entry_price'))}")
    print(f"止損點：{format_report_price(latest_row.get('entry_stop_loss_price'))}（距離 {format_report_value(latest_row.get('entry_stop_loss_pct'), pct=True)}）")
    print(f"止損來源：{latest_row.get('entry_support_source', 'NA')}")
    print(f"支撐可信度：{latest_row.get('support_quality_warning', 'NA')}")
    print(f"第一止盈：{format_report_price(latest_row.get('entry_take_profit_1'))}（R/R = {format_report_value(latest_row.get('entry_rr_1'))}）")
    print(f"第二止盈：{format_report_price(latest_row.get('entry_take_profit_2'))}（R/R = {format_report_value(latest_row.get('entry_rr_2'))}）")
    print(f"進場條件說明：{latest_row.get('entry_condition_text', 'NA')}")
    print("【進場計畫說明】")
    if entry_plan_override:
        print("目前收盤價進場的 R/R 不足，但回測支撐進場具備合理 R/R 結構。")
        print("本系統建議：不追收盤價，等待價格回測至支撐區間再進場。")
        print(f"回測進場 R/R：{format_report_value(latest_row.get('entry_rr_1'))}（第一止盈）/ {format_report_value(latest_row.get('entry_rr_2'))}（第二止盈）")
        print("此交易的成立條件為價格結構而非時間限制，持有邏輯以停損條件為準。")
    else:
        print("目前收盤價與回測進場均未達到足夠 R/R 結構，維持原決策。")
    print()
    print("-" * 70)
    print("五之二、回測引擎進場模擬（5 項硬條件）")
    print("-" * 70)

    def _lrf(col, dfl=np.nan):
        v = latest_row.get(col, dfl)
        try:
            return float(v) if pd.notna(v) else dfl
        except (TypeError, ValueError):
            return dfl

    def _ok(ok: bool) -> str:
        return "✓ 通過" if ok else "✗ 未達"

    # 條件 1：大盤過濾
    _sim_market_ok = (_market_above is not False)
    if _market_above is True:
        _sim_mkt_str = "TAIEX 高於 MA20"
    elif _market_above is False:
        _sim_mkt_str = "TAIEX 低於 MA20 — 封鎖進場"
    else:
        _sim_mkt_str = "資料未知，視為通過"
    print(f"條件1 大盤過濾          [{_ok(_sim_market_ok)}]  {_sim_mkt_str}")

    # 條件 2：支撐進場路徑
    _sim_close     = _lrf("close")
    _sim_zone_low  = _lrf("support_entry_zone_low")
    _sim_zone_high = _lrf("support_entry_zone_high")
    _sim_n_supp    = _lrf("nearest_support_level")
    _sim_n_res     = _lrf("nearest_resistance_level")
    _sim_vol       = _lrf("volume_ratio_20")
    _sim_rsi       = _lrf("RSI14")
    _sim_score     = _lrf("final_decision_score", 0.0)
    _sim_r20_rank  = _lrf("return_20d_score_daily_rank")
    _sim_r20_sc    = _lrf("return_20d_score")
    _sim_ma20d     = _lrf("ma20_distance")
    _sim_c10h      = _lrf("close_to_10d_high")
    _sim_entry_mode = getattr(config, "backtest_entry_mode", "balanced_support_v2")

    _sim_path = None
    if (pd.notna(_sim_zone_low) and pd.notna(_sim_zone_high) and pd.notna(_sim_close)
            and _sim_zone_low <= _sim_close <= _sim_zone_high):
        _sim_path = "strict_support"
    if _sim_path is None:
        _relax_score_thr = 0.70 if _sim_entry_mode == "balanced_support" else 0.65
        _relax_high_thr  = 1.03 if _sim_entry_mode == "balanced_support" else 1.05
        if (pd.notna(_sim_close) and pd.notna(_sim_zone_high)
                and pd.notna(_sim_score) and _sim_score >= _relax_score_thr
                and _sim_close <= _sim_zone_high * _relax_high_thr):
            _sim_path = "near_support_relaxed"
    if (_sim_path is None and pd.notna(_sim_n_supp) and _sim_n_supp > 0
            and pd.notna(_sim_close) and _sim_close > 0
            and _sim_n_supp * 0.97 <= _sim_close <= _sim_n_supp * 1.03):
        _sim_path = "support_proximity"
    if (_sim_path is None and pd.notna(_sim_n_res) and _sim_n_res > 0
            and pd.notna(_sim_close) and _sim_close > 0
            and _sim_close > _sim_n_res * 1.005
            and pd.notna(_sim_vol) and _sim_vol >= 1.3
            and pd.notna(_sim_rsi) and 50 <= _sim_rsi <= 75):
        _sim_path = "breakout_entry"
    if _sim_path is None:
        _rank_top30 = (
            (pd.notna(_sim_r20_rank) and _sim_r20_rank >= 0.70)
            or (pd.isna(_sim_r20_rank) and pd.notna(_sim_r20_sc) and _sim_r20_sc >= 0.60)
        )
        if (_rank_top30 and pd.notna(_sim_ma20d) and _sim_ma20d > 0
                and pd.notna(_sim_c10h) and _sim_c10h >= -0.03
                and pd.notna(_sim_vol) and _sim_vol >= 1.20):
            _sim_path = "strong_continuation"

    _sim_supp_ok = _sim_path is not None
    _sim_path_str = _sim_path if _sim_path else "無有效進場路徑"
    print(f"條件2 支撐進場路徑      [{_ok(_sim_supp_ok)}]  {_sim_path_str}")

    # 條件 3：停損距離（1.2%～8%）
    _sim_stop_pct = _lrf("entry_stop_loss_pct")
    _sim_stop_ok  = pd.notna(_sim_stop_pct) and 0.012 <= _sim_stop_pct <= 0.08
    _sim_stop_str = f"{_sim_stop_pct:.2%}" if pd.notna(_sim_stop_pct) else "N/A"
    print(f"條件3 停損距離（1.2%～8%） [{_ok(_sim_stop_ok)}]  entry_stop_loss_pct = {_sim_stop_str}")

    # 條件 4：R/R 門檻（entry_rr_1 ≥ 1.5，不使用 practical_risk_reward_ratio）
    _sim_rr1    = _lrf("entry_rr_1")
    _sim_rr_ok  = pd.notna(_sim_rr1) and _sim_rr1 >= 1.5
    _sim_rr_str = f"{_sim_rr1:.2f}" if pd.notna(_sim_rr1) else "N/A"
    print(f"條件4 R/R 門檻（≥ 1.5）   [{_ok(_sim_rr_ok)}]  entry_rr_1 = {_sim_rr_str}")

    # 條件 5：分數門檻（final_decision_score ≥ 0.55 或 entry_plan_override）
    _sim_override  = bool(latest_row.get("entry_plan_override", False))
    _sim_score_ok  = bool(_sim_override or (pd.notna(_sim_score) and _sim_score >= 0.55))
    _sim_score_str = f"{_sim_score:.2f}" if pd.notna(_sim_score) else "N/A"
    _sim_ov_note   = " + entry_plan_override=True" if _sim_override else ""
    print(f"條件5 分數門檻（≥ 0.55）  [{_ok(_sim_score_ok)}]  final_decision_score = {_sim_score_str}{_sim_ov_note}")

    # 綜合結果
    _sim_all    = [_sim_market_ok, _sim_supp_ok, _sim_stop_ok, _sim_rr_ok, _sim_score_ok]
    _sim_n_pass = sum(_sim_all)
    print()
    if all(_sim_all):
        print(f">>> 模擬結果：{_sim_n_pass}/5 條件通過 — 此訊號在回測引擎中將觸發進場")
    else:
        print(f">>> 模擬結果：{_sim_n_pass}/5 條件通過 — 此訊號在回測引擎中不會觸發進場")
    print()
    print("-" * 70)
    print("六、倉位建議")
    print("-" * 70)
    print(f"risk_per_trade_pct：{format_report_value(latest_row.get('risk_per_trade_pct'), pct=True)}")
    print(f"max_position_pct：{format_report_value(latest_row.get('max_position_pct'), pct=True)}")
    print(f"suggested_position_pct：{format_report_value(latest_row.get('suggested_position_pct'), pct=True)}")
    print(f"suggested_shares：{format_report_value(latest_row.get('suggested_shares'), decimals=0)}")
    print(f"position_sizing_label：{normalize_text_for_report(latest_row.get('position_sizing_label', ''))}")
    print(f"position_sizing_reason：{normalize_text_for_report(latest_row.get('position_sizing_reason', ''))}")
    print()
    print("-" * 70)
    print("七、決策輔助指南")
    print("-" * 70)
    print(build_decision_guide_text(latest_row))
    print()
    print("-" * 70)
    print("八、風險警示")
    print("-" * 70)
    print(risk_warning if risk_warning else "無明顯風險警示")
    print()
    print("-" * 70)
    print("九、操作建議")
    print("-" * 70)
    print(suggested_action)
    print()
    print("-" * 70)
    print("十、模型驗證與表現摘要")
    print("-" * 70)
    print_model_validation_body(summary, is_multi_stock=False)
    print()
    print("-" * 70)
    print("十一、使用提醒")
    print("-" * 70)
    print("本系統為投資決策支援工具，並非保證獲利模型。")
    print("模型結果應搭配基本面、籌碼面、市場環境與個人風險承受度共同判斷。")


def build_single_diagnostic_summary(diagnostic_df: pd.DataFrame, config: PipelineConfig) -> tuple[dict, pd.DataFrame]:
    group_df = diagnostic_df.copy()
    for text_col in ["final_decision_label", "risk_warning", "model_confidence_note", "signal"]:
        if text_col in group_df.columns:
            group_df[text_col] = group_df[text_col].apply(normalize_text_for_report)
    target_value_col = "actual_target" if "actual_target" in group_df.columns else "actual_target_score_v2"
    if target_value_col not in group_df.columns:
        raise ValueError(f"single_diagnostic output missing actual target column for {config.target_col}")

    if "actual_target" not in group_df.columns:
        group_df["actual_target"] = pd.to_numeric(group_df[target_value_col], errors="coerce")
        target_value_col = "actual_target"
    else:
        group_df["actual_target"] = pd.to_numeric(group_df["actual_target"], errors="coerce")

    if "actual_target_score_v2" in group_df.columns:
        group_df["actual_target_score_v2"] = pd.to_numeric(group_df["actual_target_score_v2"], errors="coerce")
    if "future_drawdown_10d" in group_df.columns:
        group_df["future_drawdown_10d"] = pd.to_numeric(group_df["future_drawdown_10d"], errors="coerce")
        group_df["abs_future_drawdown_10d"] = group_df["future_drawdown_10d"].abs()
    if "future_return_10d" in group_df.columns:
        group_df["future_return_10d"] = pd.to_numeric(group_df["future_return_10d"], errors="coerce")
    if "future_return_10d" in group_df.columns and "future_drawdown_10d" in group_df.columns:
        group_df["risk_adjusted_return_10d"] = (
            group_df["future_return_10d"] / (group_df["future_drawdown_10d"].abs() + 1e-6)
        ).replace([np.inf, -np.inf], np.nan)
    if "risk_adjusted_return_10d_capped" not in group_df.columns:
        if "future_return_10d" in group_df.columns and "abs_future_drawdown_10d" in group_df.columns:
            group_df["risk_adjusted_return_10d_capped"] = calculate_risk_adjusted_return_10d_capped(
                group_df["future_return_10d"],
                group_df["abs_future_drawdown_10d"],
            )
    for col in ["safety_score", "return_score", "combined_decision_score", "risk_adjusted_return_10d_capped"]:
        if col in group_df.columns:
            group_df[col] = pd.to_numeric(group_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    eval_df = group_df.dropna(subset=["predicted_score", target_value_col]).copy()
    if eval_df.empty:
        raise ValueError("沒有可計算診斷指標的 out-of-sample 預測資料。")

    y_true = pd.to_numeric(eval_df[target_value_col], errors="coerce")
    y_pred = pd.to_numeric(eval_df["predicted_score"], errors="coerce")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = safe_r2(y_true, y_pred.to_numpy(dtype=float))
    spearman_corr = eval_df[["predicted_score", target_value_col]].corr(method="spearman").iloc[0, 1]
    baseline_pred = np.full(len(y_true), y_true.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = mean_squared_error(y_true, baseline_pred) ** 0.5
    baseline_r2 = safe_r2(y_true, baseline_pred)

    score_values = pd.to_numeric(group_df["predicted_score"], errors="coerce")
    group_df["score_group_fixed"] = pd.cut(
        score_values,
        bins=[-np.inf, 0.45, 0.55, 0.65, np.inf],
        labels=["<=0.45", "0.45-0.55", "0.55-0.65", ">=0.65"],
    )
    group_agg = {
        "days": ("date", "count"),
        "predicted_score_mean": ("predicted_score", "mean"),
        "actual_target_mean": ("actual_target", "mean"),
    }
    if "actual_target_score_v2" in group_df.columns:
        group_agg["actual_target_score_v2_mean"] = ("actual_target_score_v2", "mean")
    if "future_return_10d" in group_df.columns:
        group_agg["future_return_mean"] = ("future_return_10d", "mean")
        group_agg["win_rate"] = ("future_return_10d", positive_return_rate)
    if "future_drawdown_10d" in group_df.columns:
        group_agg["future_drawdown_10d_mean"] = ("future_drawdown_10d", "mean")
    if "abs_future_drawdown_10d" in group_df.columns:
        group_agg["abs_future_drawdown_10d_mean"] = ("abs_future_drawdown_10d", "mean")
    if "risk_adjusted_return_10d" in group_df.columns:
        group_agg["risk_adjusted_return_10d_mean"] = ("risk_adjusted_return_10d", "mean")
    if "risk_adjusted_return_10d_capped" in group_df.columns:
        group_agg["risk_adjusted_return_10d_capped_mean"] = ("risk_adjusted_return_10d_capped", "mean")
    if "safety_score" in group_df.columns:
        group_agg["safety_score_mean"] = ("safety_score", "mean")
    if "return_score" in group_df.columns:
        group_agg["return_score_mean"] = ("return_score", "mean")
    if "combined_decision_score" in group_df.columns:
        group_agg["combined_decision_score_mean"] = ("combined_decision_score", "mean")

    fixed_group_summary = (
        group_df
        .dropna(subset=["score_group_fixed"])
        .groupby("score_group_fixed", observed=False)
        .agg(**group_agg)
        .reset_index()
        .rename(columns={"score_group_fixed": "score_group"})
    )
    fixed_group_summary.insert(0, "group_type", "fixed_threshold")

    rank_pct = score_values.rank(method="first", pct=True)
    group_df["score_group_quantile"] = pd.Series(pd.NA, index=group_df.index, dtype="object")
    group_df.loc[rank_pct <= 0.20, "score_group_quantile"] = "Q1 lowest 20%"
    group_df.loc[(rank_pct > 0.20) & (rank_pct <= 0.40), "score_group_quantile"] = "Q2"
    group_df.loc[(rank_pct > 0.40) & (rank_pct <= 0.60), "score_group_quantile"] = "Q3"
    group_df.loc[(rank_pct > 0.60) & (rank_pct <= 0.80), "score_group_quantile"] = "Q4"
    group_df.loc[rank_pct > 0.80, "score_group_quantile"] = "Q5 highest 20%"
    group_df["score_group_quantile"] = pd.Categorical(
        group_df["score_group_quantile"],
        categories=["Q1 lowest 20%", "Q2", "Q3", "Q4", "Q5 highest 20%"],
        ordered=True,
    )
    quantile_group_summary = (
        group_df
        .dropna(subset=["score_group_quantile"])
        .groupby("score_group_quantile", observed=False)
        .agg(**group_agg)
        .reset_index()
        .rename(columns={"score_group_quantile": "score_group"})
    )
    quantile_group_summary.insert(0, "group_type", "quantile_20pct")
    group_summary = pd.concat(
        [fixed_group_summary, quantile_group_summary],
        ignore_index=True,
    )

    score_series = pd.to_numeric(group_df["predicted_score"], errors="coerce")
    high_score_fixed = group_df[score_series >= 0.65]
    low_score_fixed = group_df[score_series <= 0.45]

    q80_threshold = score_series.quantile(0.80)
    q20_threshold = score_series.quantile(0.20)
    high_score_q80 = (
        group_df[score_series >= q80_threshold]
        if pd.notna(q80_threshold)
        else group_df.iloc[0:0]
    )
    low_score_q20 = (
        group_df[score_series <= q20_threshold]
        if pd.notna(q20_threshold)
        else group_df.iloc[0:0]
    )

    def group_mean(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns:
            return np.nan
        return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()

    fixed_high_actual_target_mean = group_mean(high_score_fixed, "actual_target")
    fixed_low_actual_target_mean = group_mean(low_score_fixed, "actual_target")
    q80_high_actual_target_mean = group_mean(high_score_q80, "actual_target")
    q20_low_actual_target_mean = group_mean(low_score_q20, "actual_target")
    if pd.notna(q80_high_actual_target_mean) and pd.notna(q20_low_actual_target_mean):
        q80_minus_q20_actual_target_spread = q80_high_actual_target_mean - q20_low_actual_target_mean
    else:
        q80_minus_q20_actual_target_spread = np.nan

    fixed_high_future_return_mean = group_mean(high_score_fixed, "future_return_10d")
    fixed_low_future_return_mean = group_mean(low_score_fixed, "future_return_10d")
    q80_high_future_return_mean = group_mean(high_score_q80, "future_return_10d")
    q20_low_future_return_mean = group_mean(low_score_q20, "future_return_10d")
    if pd.notna(q80_high_future_return_mean) and pd.notna(q20_low_future_return_mean):
        q80_minus_q20_future_return_spread = (
            q80_high_future_return_mean - q20_low_future_return_mean
        )
    else:
        q80_minus_q20_future_return_spread = np.nan

    fixed_high_future_drawdown_mean = group_mean(high_score_fixed, "future_drawdown_10d")
    fixed_low_future_drawdown_mean = group_mean(low_score_fixed, "future_drawdown_10d")
    q80_high_future_drawdown_mean = group_mean(high_score_q80, "future_drawdown_10d")
    q20_low_future_drawdown_mean = group_mean(low_score_q20, "future_drawdown_10d")

    fixed_high_abs_drawdown_mean = group_mean(high_score_fixed, "abs_future_drawdown_10d")
    fixed_low_abs_drawdown_mean = group_mean(low_score_fixed, "abs_future_drawdown_10d")
    q80_high_abs_drawdown_mean = group_mean(high_score_q80, "abs_future_drawdown_10d")
    q20_low_abs_drawdown_mean = group_mean(low_score_q20, "abs_future_drawdown_10d")

    fixed_high_rar_mean = group_mean(high_score_fixed, "risk_adjusted_return_10d")
    fixed_low_rar_mean = group_mean(low_score_fixed, "risk_adjusted_return_10d")
    q80_high_rar_mean = group_mean(high_score_q80, "risk_adjusted_return_10d")
    q20_low_rar_mean = group_mean(low_score_q20, "risk_adjusted_return_10d")

    total_days = len(group_df)
    if "risk_warning" in group_df.columns:
        risk_warning_text = group_df["risk_warning"].fillna("").astype(str).map(clean_diagnostic_text)
        high_risk_warning_days = int(
            ((risk_warning_text != "") & (risk_warning_text != "無明顯風險警示")).sum()
        )
    else:
        high_risk_warning_days = 0
    high_risk_warning_ratio = high_risk_warning_days / total_days if total_days else np.nan

    if "final_decision_label" in group_df.columns:
        label_text = group_df["final_decision_label"].fillna("").astype(str).map(clean_diagnostic_text)
        weak_label_days = int((label_text == "分數偏弱，暫以觀察為主").sum())
        neutral_label_days = int((label_text == "中性觀察").sum())
    else:
        weak_label_days = 0
        neutral_label_days = 0

    summary = {
        "market": config.market.upper(),
        "symbol": config.stock_no,
        "target_col": config.target_col,
        "actual_target_col": config.target_col,
        "diagnostic_scope": config.diagnostic_scope,
        "feature_set": config.feature_set,
        "model_type": config.model_type,
        "model_profile": config.model_profile,
        "validation_method": "walk-forward time-series validation",
        "safety_score_mean": group_mean(group_df, "safety_score"),
        "return_score_mean": group_mean(group_df, "return_score"),
        "combined_decision_score_mean": group_mean(group_df, "combined_decision_score"),
        "high_risk_warning_days": high_risk_warning_days,
        "high_risk_warning_ratio": high_risk_warning_ratio,
        "weak_label_days": weak_label_days,
        "neutral_label_days": neutral_label_days,
        "oos_rows": len(eval_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "spearman_corr": spearman_corr,
        "high_score_threshold": 0.65,
        "low_score_threshold": 0.45,
        "high_score_days": len(high_score_fixed),
        "low_score_days": len(low_score_fixed),
        "high_score_actual_target_mean": fixed_high_actual_target_mean,
        "low_score_actual_target_mean": fixed_low_actual_target_mean,
        "high_score_future_return_10d_mean": fixed_high_future_return_mean,
        "low_score_future_return_10d_mean": fixed_low_future_return_mean,
        "high_score_future_drawdown_10d_mean": fixed_high_future_drawdown_mean,
        "low_score_future_drawdown_10d_mean": fixed_low_future_drawdown_mean,
        "high_score_abs_future_drawdown_10d_mean": fixed_high_abs_drawdown_mean,
        "low_score_abs_future_drawdown_10d_mean": fixed_low_abs_drawdown_mean,
        "high_score_risk_adjusted_return_10d_mean": fixed_high_rar_mean,
        "low_score_risk_adjusted_return_10d_mean": fixed_low_rar_mean,
        "high_score_days_fixed": len(high_score_fixed),
        "low_score_days_fixed": len(low_score_fixed),
        "high_score_days_q80": len(high_score_q80),
        "low_score_days_q20": len(low_score_q20),
        "fixed_high_count": len(high_score_fixed),
        "fixed_high_actual_target_mean": fixed_high_actual_target_mean,
        "fixed_high_future_return_mean": fixed_high_future_return_mean,
        "fixed_high_future_drawdown_10d_mean": fixed_high_future_drawdown_mean,
        "fixed_high_abs_future_drawdown_10d_mean": fixed_high_abs_drawdown_mean,
        "fixed_high_risk_adjusted_return_10d_mean": fixed_high_rar_mean,
        "fixed_low_count": len(low_score_fixed),
        "fixed_low_actual_target_mean": fixed_low_actual_target_mean,
        "fixed_low_future_return_mean": fixed_low_future_return_mean,
        "fixed_low_future_drawdown_10d_mean": fixed_low_future_drawdown_mean,
        "fixed_low_abs_future_drawdown_10d_mean": fixed_low_abs_drawdown_mean,
        "fixed_low_risk_adjusted_return_10d_mean": fixed_low_rar_mean,
        "q80_threshold": q80_threshold,
        "q20_threshold": q20_threshold,
        "q80_high_count": len(high_score_q80),
        "q80_high_actual_target_mean": q80_high_actual_target_mean,
        "q80_high_future_return_mean": q80_high_future_return_mean,
        "q80_high_future_drawdown_10d_mean": q80_high_future_drawdown_mean,
        "q80_high_abs_future_drawdown_10d_mean": q80_high_abs_drawdown_mean,
        "q80_high_risk_adjusted_return_10d_mean": q80_high_rar_mean,
        "q20_low_count": len(low_score_q20),
        "q20_low_actual_target_mean": q20_low_actual_target_mean,
        "q20_low_future_return_mean": q20_low_future_return_mean,
        "q20_low_future_drawdown_10d_mean": q20_low_future_drawdown_mean,
        "q20_low_abs_future_drawdown_10d_mean": q20_low_abs_drawdown_mean,
        "q20_low_risk_adjusted_return_10d_mean": q20_low_rar_mean,
        "q80_minus_q20_actual_target_spread": q80_minus_q20_actual_target_spread,
        "q80_minus_q20_future_return_spread": q80_minus_q20_future_return_spread,
    }

    return summary, group_summary


def predict_single_diagnostic_auxiliary_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: PipelineConfig,
    fold_no: int,
    random_state_offset: int,
) -> np.ndarray:
    y_train_clean = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = y_train_clean.notna()
    if int(valid_mask.sum()) < 10:
        return np.full(len(X_test), np.nan)

    pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=config.random_state + random_state_offset + fold_no,
        model_profile=config.model_profile,
        model_type=config.model_type,
    )
    pipeline.fit(X_train.loc[valid_mask], y_train_clean.loc[valid_mask])
    return np.clip(pipeline.predict(X_test), 0, 1)


def fit_predict_single_diagnostic_auxiliary_latest(
    X_train: pd.DataFrame,
    X_latest: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: PipelineConfig,
    random_state_offset: int,
) -> np.ndarray:
    y_train_clean = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = y_train_clean.notna()
    if int(valid_mask.sum()) < 10:
        return np.full(len(X_latest), np.nan)

    pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=config.random_state + random_state_offset,
        model_profile=config.model_profile,
        model_type=config.model_type,
    )
    pipeline.fit(X_train.loc[valid_mask], y_train_clean.loc[valid_mask])
    return np.clip(pipeline.predict(X_latest), 0, 1)


def build_single_diagnostic_output_frame(
    scored_df: pd.DataFrame,
    config: PipelineConfig,
    safety_target_col: str,
    return_target_col: str,
) -> pd.DataFrame:
    source_df = scored_df.copy()
    actual_target = pd.to_numeric(get_series_or_default(source_df, config.target_col), errors="coerce")
    actual_target_score_v2 = pd.to_numeric(get_series_or_default(source_df, "target_score_v2"), errors="coerce")
    if actual_target_score_v2.isna().all():
        actual_target_score_v2 = actual_target
    future_return_10d = pd.to_numeric(
        get_series_or_default(source_df, f"future_return_{config.horizon}d"),
        errors="coerce",
    )
    future_drawdown_10d = pd.to_numeric(
        get_series_or_default(source_df, f"future_drawdown_{config.horizon}d"),
        errors="coerce",
    )
    abs_future_drawdown_10d = future_drawdown_10d.abs()
    risk_adjusted_return_10d = (
        future_return_10d / (abs_future_drawdown_10d + 1e-6)
    ).replace([np.inf, -np.inf], np.nan)
    risk_adjusted_return_10d_capped = calculate_risk_adjusted_return_10d_capped(
        future_return_10d,
        abs_future_drawdown_10d,
    )

    output_df = pd.DataFrame({
        "date": pd.to_datetime(source_df["date"], errors="coerce"),
        "market": get_series_or_default(source_df, "market", config.market.upper()),
        "symbol": get_series_or_default(source_df, "symbol", config.stock_no),
        "close": pd.to_numeric(get_series_or_default(source_df, "close"), errors="coerce"),
        "prediction_source": get_series_or_default(source_df, "prediction_source", "walk_forward_oos"),
        "predicted_score": pd.to_numeric(source_df["predicted_score"], errors="coerce"),
        "safety_score": pd.to_numeric(source_df["safety_score"], errors="coerce"),
        "return_score": pd.to_numeric(source_df["return_score"], errors="coerce"),
        "actual_target": actual_target,
        "actual_target_score_v2": actual_target_score_v2,
        safety_target_col: pd.to_numeric(get_series_or_default(source_df, safety_target_col), errors="coerce"),
        return_target_col: pd.to_numeric(get_series_or_default(source_df, return_target_col), errors="coerce"),
        "future_return_10d": future_return_10d,
        "future_drawdown_10d": future_drawdown_10d,
        "abs_future_drawdown_10d": abs_future_drawdown_10d,
        "risk_adjusted_return_10d": risk_adjusted_return_10d,
        "risk_adjusted_return_10d_capped": risk_adjusted_return_10d_capped,
        "RSI14": pd.to_numeric(get_series_or_default(source_df, "RSI14"), errors="coerce"),
        "MACD": pd.to_numeric(get_series_or_default(source_df, "MACD"), errors="coerce"),
        "K": pd.to_numeric(get_series_or_default(source_df, "K"), errors="coerce"),
        "D": pd.to_numeric(get_series_or_default(source_df, "D"), errors="coerce"),
        "J": pd.to_numeric(get_series_or_default(source_df, "J"), errors="coerce"),
        "BIAS24": pd.to_numeric(get_series_or_default(source_df, "BIAS24"), errors="coerce"),
        "VR_14": pd.to_numeric(get_series_or_default(source_df, "VR_14"), errors="coerce"),
        "VR_26": pd.to_numeric(get_series_or_default(source_df, "VR_26"), errors="coerce"),
        "return_5d": pd.to_numeric(get_series_or_default(source_df, "return_5d"), errors="coerce"),
        "return_10d": pd.to_numeric(get_series_or_default(source_df, "return_10d"), errors="coerce"),
        "nearest_support_level": pd.to_numeric(get_series_or_default(source_df, "nearest_support_level"), errors="coerce"),
        "nearest_resistance_level": pd.to_numeric(get_series_or_default(source_df, "nearest_resistance_level"), errors="coerce"),
        "atr_14": pd.to_numeric(get_series_or_default(source_df, "atr_14"), errors="coerce"),
        "risk_atr_14": pd.to_numeric(get_series_or_default(source_df, "risk_atr_14"), errors="coerce"),
        "risk_ob_bottom": pd.to_numeric(get_series_or_default(source_df, "risk_ob_bottom"), errors="coerce"),
        "risk_atr_stop": pd.to_numeric(get_series_or_default(source_df, "risk_atr_stop"), errors="coerce"),
        "atr_pct": pd.to_numeric(get_series_or_default(source_df, "atr_pct"), errors="coerce"),
        "volatility_10d": pd.to_numeric(get_series_or_default(source_df, "volatility_10d"), errors="coerce"),
        "risk_entry_price": pd.to_numeric(get_series_or_default(source_df, "risk_entry_price"), errors="coerce"),
        "risk_stoploss_price": pd.to_numeric(get_series_or_default(source_df, "risk_stoploss_price"), errors="coerce"),
        "risk_atr_stop_distance_pct": pd.to_numeric(get_series_or_default(source_df, "risk_atr_stop_distance_pct"), errors="coerce"),
        "risk_stoploss_pct": pd.to_numeric(get_series_or_default(source_df, "risk_stoploss_pct"), errors="coerce"),
        "fib_position": pd.to_numeric(get_series_or_default(source_df, "fib_position"), errors="coerce"),
        "fib_zone": get_series_or_default(source_df, "fib_zone", "none"),
        "recent_2w_ob_type": get_series_or_default(source_df, "recent_2w_ob_type", "none"),
        "recent_2w_ob_is_fallback": pd.to_numeric(get_series_or_default(source_df, "recent_2w_ob_is_fallback"), errors="coerce"),
        "recent_active_gap_type": get_series_or_default(source_df, "recent_active_gap_type", "none"),
        "support_distance": pd.to_numeric(get_series_or_default(source_df, "support_distance"), errors="coerce"),
        "resistance_distance": pd.to_numeric(get_series_or_default(source_df, "resistance_distance"), errors="coerce"),
        "ma5": pd.to_numeric(get_series_or_default(source_df, "ma5"), errors="coerce"),
        "ma20": pd.to_numeric(get_series_or_default(source_df, "ma20"), errors="coerce"),
        "ma60": pd.to_numeric(get_series_or_default(source_df, "ma60"), errors="coerce"),
        "fib_0618_price": pd.to_numeric(get_series_or_default(source_df, "fib_0618_price"), errors="coerce"),
        "close_to_20d_low": pd.to_numeric(get_series_or_default(source_df, "close_to_20d_low"), errors="coerce"),
        "nearest_resistance_zone_high": pd.to_numeric(get_series_or_default(source_df, "nearest_resistance_zone_high"), errors="coerce"),
        "volume_ratio_20": pd.to_numeric(get_series_or_default(source_df, "volume_ratio_20"), errors="coerce"),
        "vol_high1_low": pd.to_numeric(get_series_or_default(source_df, "vol_high1_low"), errors="coerce"),
        "vol_high2_low": pd.to_numeric(get_series_or_default(source_df, "vol_high2_low"), errors="coerce"),
        "pivot_low": pd.to_numeric(get_series_or_default(source_df, "pivot_low"), errors="coerce"),
        "recent_active_gap_lower": pd.to_numeric(get_series_or_default(source_df, "recent_active_gap_lower"), errors="coerce"),
        "support_quality_score": pd.to_numeric(get_series_or_default(source_df, "support_quality_score"), errors="coerce"),
    })
    extra_diagnostic_cols = [
        col for col in NEW_DIAGNOSTIC_FEATURES
        if col in source_df.columns and col not in output_df.columns
    ]
    for col in extra_diagnostic_cols:
        output_df[col] = source_df[col].to_numpy()

    output_df["combined_decision_score"] = (
        0.65 * output_df["safety_score"] + 0.35 * output_df["return_score"]
    )
    return output_df


def apply_single_diagnostic_decision_outputs(
    output_df: pd.DataFrame,
    risk_thresholds: dict[str, float],
) -> pd.DataFrame:
    out = output_df.copy()
    out["final_decision_label"] = out.apply(
        get_single_diagnostic_final_decision_label,
        axis=1,
    )
    out["model_confidence_note"] = out.apply(
        get_single_diagnostic_model_confidence_note,
        axis=1,
    )
    out["risk_warning"] = out.apply(build_risk_warning, axis=1)
    out["prediction_error"] = out["predicted_score"] - out["actual_target"]
    out["signal"] = out["predicted_score"].apply(get_signal)
    decision_text_df = out.apply(build_decision_text, axis=1, result_type="expand")
    out = _concat_overwrite(out, decision_text_df)

    trade_plan_df = out.apply(build_trade_plan, axis=1, result_type="expand")
    out = _concat_overwrite(out, trade_plan_df)

    rule_filter_df = out.apply(apply_second_layer_rule_filter, axis=1, result_type="expand")
    out = _concat_overwrite(out, rule_filter_df)

    out["risk_warning"] = out.apply(build_risk_warning, axis=1)

    entry_plan_df = out.apply(build_entry_plan, axis=1, result_type="expand")
    out = _concat_overwrite(out, entry_plan_df)

    override_df = out.apply(_apply_entry_plan_override_row, axis=1, result_type="expand")
    out = _concat_overwrite(out, override_df)

    final_decision_df = out.apply(build_final_decision, axis=1, result_type="expand")
    out = _concat_overwrite(out, final_decision_df)
    if "technical_filter_result" in out.columns:
        out["technical_filter_result"] = out["technical_filter_result"].apply(translate_technical_filter_result)

    position_df = out.apply(lambda row: build_position_sizing(row), axis=1, result_type="expand")
    out = _concat_overwrite(out, position_df)

    enhanced_text_df = out.apply(build_enhanced_decision_text, axis=1, result_type="expand")
    out = _concat_overwrite(out, enhanced_text_df)
    return out


def get_single_diagnostic_final_decision_label(row: pd.Series) -> str:
    safety_score = pd.to_numeric(pd.Series([row.get("safety_score", np.nan)]), errors="coerce").iloc[0]
    return_score = pd.to_numeric(pd.Series([row.get("return_score", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(safety_score) or pd.isna(return_score):
        return "中性觀察"
    if safety_score >= 0.65 and return_score >= 0.55:
        return "安全與報酬分數偏高，仍需確認"
    if safety_score >= 0.65 and return_score < 0.55:
        return "穩健偏多，優先觀察"
    if safety_score < 0.45 and return_score >= 0.55:
        return "短線機會型，但需控管風險"
    if safety_score < 0.45 and return_score < 0.55:
        return "分數偏弱，暫以觀察為主"
    return "中性觀察"


def get_single_diagnostic_risk_warning(row: pd.Series, thresholds: dict[str, float]) -> str:
    return build_risk_warning(row)


def get_single_diagnostic_model_confidence_note(row: pd.Series) -> str:
    safety_score = pd.to_numeric(pd.Series([row.get("safety_score", np.nan)]), errors="coerce").iloc[0]
    return_score = pd.to_numeric(pd.Series([row.get("return_score", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(safety_score) or pd.isna(return_score):
        return "模型分數可作為輔助參考"
    if 0.45 <= safety_score <= 0.55 and 0.45 <= return_score <= 0.55:
        return "模型分數集中，決策信心較低"
    if abs(safety_score - return_score) >= 0.25:
        return "安全與報酬模型分歧，需人工檢查"
    return "模型分數可作為輔助參考"


def run_single_stock_diagnostic(config: PipelineConfig, top_n: int = 30, investor_mode: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.stock_no.strip().upper() in {"ALL", "*", "全部"}:
        raise ValueError("single_diagnostic 需要指定單一股票代碼，例如 --symbol 2330")
    verbose = config.display_mode == "research"

    if verbose:
        print("\n" + "=" * 70)
        print("Single Stock Diagnostic")
        print("=" * 70)
        print(f"market={config.market.upper()}")
        print(f"symbol={config.stock_no}")
        print(f"top_n={top_n}")
        print(f"feature_set={config.feature_set}")
        print(f"model_type={config.model_type}")
        print(f"model_profile={config.model_profile}")
        print(f"target_col={config.target_col}")

    return run_shared_single_stock_diagnostic(config, top_n=top_n, verbose=verbose, investor_mode=investor_mode)

    if verbose:
        data = prepare_common_single_diagnostic_feature_data(config, top_n=top_n)
        prepared = prepare_single_diagnostic_model_data(data, config)
    else:
        print_presentation_process_header()
        with contextlib.redirect_stdout(io.StringIO()):
            data = prepare_common_single_diagnostic_feature_data(config, top_n=top_n)
            prepared = prepare_single_diagnostic_model_data(data, config)
        print_presentation_data_summary(
            config,
            data,
            top_n=top_n,
            target_symbol=config.stock_no,
        )
    model_df = prepared["model_df"]
    training_df = prepared.get("training_df", model_df)
    prediction_feature_df = prepared["prediction_feature_df"]
    model_df = training_df
    feature_cols = prepared["feature_cols"]
    numeric_features = prepared["numeric_features"]
    categorical_features = prepared["categorical_features"]
    safety_target_col = "target_safety_10d_cs_rank"
    return_target_col = "target_return_10d_cs_rank"
    missing_aux_target_cols = [
        col for col in [safety_target_col, return_target_col]
        if col not in model_df.columns
    ]
    if missing_aux_target_cols:
        raise ValueError(f"single_diagnostic 輔助模型 target 欄位不存在：{missing_aux_target_cols}")

    if verbose:
        print("\nSingle diagnostic Step 5/5：walk-forward out-of-sample 預測")
        print(f"模型實際使用特徵數量：{len(feature_cols)}")
        print("模型實際使用特徵清單：")
        print(feature_cols)
        print("被移除的特徵：")
        print(prepared["removed_features"])
    else:
        print("Step 5/6：建立共同模型並產生 safety / return / combined 決策分數")

    X = model_df[feature_cols]
    y = pd.to_numeric(model_df[config.target_col], errors="coerce")
    y_safety = pd.to_numeric(model_df[safety_target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_return = pd.to_numeric(model_df[return_target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    splits = build_walk_forward_splits_by_date(model_df=model_df, n_splits=8)
    target_symbol = str(config.stock_no).strip()

    diagnostic_frames = []
    fold_info_rows = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        valid_dates = pd.to_datetime(model_df.iloc[test_idx]["date"], errors="coerce").dropna()
        fold_info = {
            "fold": fold_no,
            "train_rows": len(train_idx),
            "valid_rows": len(test_idx),
            "valid_start_date": valid_dates.min() if not valid_dates.empty else pd.NaT,
            "valid_end_date": valid_dates.max() if not valid_dates.empty else pd.NaT,
            "target_symbol_valid_rows": np.nan,
        }

        pipeline = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=config.random_state + fold_no,
            model_profile=config.model_profile,
            model_type=config.model_type,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if not config.target_col.startswith("target_score_raw_"):
            y_pred = np.clip(y_pred, 0, 1)
        safety_score = predict_single_diagnostic_auxiliary_score(
            X_train=X_train,
            X_test=X_test,
            y_train=y_safety.iloc[train_idx],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=config,
            fold_no=fold_no,
            random_state_offset=1000,
        )
        return_score = predict_single_diagnostic_auxiliary_score(
            X_train=X_train,
            X_test=X_test,
            y_train=y_return.iloc[train_idx],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=config,
            fold_no=fold_no,
            random_state_offset=2000,
        )

        fold_df = model_df.iloc[test_idx].copy()
        fold_df["predicted_score"] = y_pred
        fold_df["safety_score"] = safety_score
        fold_df["return_score"] = return_score
        symbol_fold_rows = len(fold_df)
        if config.diagnostic_scope == "common":
            fold_df = fold_df[fold_df["symbol"].astype(str) == target_symbol].copy()
            symbol_fold_rows = len(fold_df)
            fold_info["target_symbol_valid_rows"] = symbol_fold_rows
            fold_info_rows.append(fold_info)
            if fold_df.empty:
                if verbose:
                    print(
                        f"Fold {fold_no}: train={len(train_idx)}, "
                        f"valid={len(test_idx)}, target_symbol_valid=0，略過輸出"
                    )
                continue
        else:
            fold_info["target_symbol_valid_rows"] = len(fold_df)
            fold_info_rows.append(fold_info)
        diagnostic_frames.append(fold_df)

        if verbose:
            print(
                f"Fold {fold_no}: train={len(train_idx)}, "
                f"valid={len(test_idx)}, "
                f"target_symbol_valid={symbol_fold_rows}, "
                f"date={pd.to_datetime(fold_df['date']).min().date()} ~ {pd.to_datetime(fold_df['date']).max().date()}"
            )

    if not diagnostic_frames:
        raise ValueError("無法產生 walk-forward 診斷預測，請確認資料長度。")

    if not verbose:
        print("Step 6/6：輸出投資決策報告")

    oos_df = pd.concat(diagnostic_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos_df["prediction_source"] = "walk_forward_oos"

    latest_candidates = prediction_feature_df.copy()
    latest_candidates["date"] = pd.to_datetime(latest_candidates["date"], errors="coerce")
    latest_candidates = latest_candidates.dropna(subset=["date"])
    if config.diagnostic_scope == "common":
        latest_candidates = latest_candidates[
            latest_candidates["symbol"].astype(str) == target_symbol
        ].copy()
    if latest_candidates.empty:
        raise ValueError(f"prediction_feature_df 中找不到指定股票 {target_symbol} 的最新 feature row。")

    latest_feature_row = latest_candidates.sort_values("date").tail(1).copy()
    latest_X = latest_feature_row[feature_cols]

    final_pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=config.random_state + 9000,
        model_profile=config.model_profile,
        model_type=config.model_type,
    )
    final_pipeline.fit(X, y)
    latest_pred = final_pipeline.predict(latest_X)
    if not config.target_col.startswith("target_score_raw_"):
        latest_pred = np.clip(latest_pred, 0, 1)
    latest_feature_row["predicted_score"] = latest_pred
    latest_feature_row["safety_score"] = fit_predict_single_diagnostic_auxiliary_latest(
        X_train=X,
        X_latest=latest_X,
        y_train=y_safety,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
        random_state_offset=9100,
    )
    latest_feature_row["return_score"] = fit_predict_single_diagnostic_auxiliary_latest(
        X_train=X,
        X_latest=latest_X,
        y_train=y_return,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
        random_state_offset=9200,
    )
    latest_feature_row["prediction_source"] = "latest_feature_full_model"

    scored_output_source_df = pd.concat([oos_df, latest_feature_row], ignore_index=True, sort=False)
    scored_output_source_df["date"] = pd.to_datetime(scored_output_source_df["date"], errors="coerce")
    scored_output_source_df["_latest_priority"] = np.where(
        scored_output_source_df["prediction_source"].eq("latest_feature_full_model"),
        1,
        0,
    )
    dedupe_cols = ["date", "symbol"] if "symbol" in scored_output_source_df.columns else ["date"]
    scored_output_source_df = (
        scored_output_source_df
        .sort_values(["date", "_latest_priority"])
        .drop_duplicates(subset=dedupe_cols, keep="last")
        .drop(columns=["_latest_priority"])
        .reset_index(drop=True)
    )

    validation_base_df = build_single_diagnostic_output_frame(
        oos_df,
        config,
        safety_target_col,
        return_target_col,
    )
    risk_thresholds = {
        "volatility_10d_q80": validation_base_df["volatility_10d"].quantile(0.80),
        "support_distance_q80": validation_base_df["support_distance"].quantile(0.80),
        "resistance_distance_q20": validation_base_df["resistance_distance"].quantile(0.20),
    }
    validation_output_df = apply_single_diagnostic_decision_outputs(
        validation_base_df,
        risk_thresholds,
    )
    output_df = build_single_diagnostic_output_frame(
        scored_output_source_df,
        config,
        safety_target_col,
        return_target_col,
    )
    output_df = apply_single_diagnostic_decision_outputs(output_df, risk_thresholds)

    extra_diagnostic_cols = [col for col in NEW_DIAGNOSTIC_FEATURES if col in output_df.columns]
    output_cols = [
        "date",
        "market",
        "symbol",
        "close",
        "prediction_source",
        "predicted_score",
        "safety_score",
        "return_score",
        "combined_decision_score",
        "final_decision_label",
        "risk_warning",
        "model_confidence_note",
        "actual_target",
        "actual_target_score_v2",
        safety_target_col,
        return_target_col,
        "prediction_error",
        "future_return_10d",
        "future_drawdown_10d",
        "abs_future_drawdown_10d",
        "risk_adjusted_return_10d",
        "risk_adjusted_return_10d_capped",
        "RSI14",
        "MACD",
        "K",
        "D",
        "J",
        "BIAS24",
        "VR_14",
        "VR_26",
        "return_5d",
        "return_10d",
        "atr_pct",
        "volatility_10d",
        "nearest_support_level",
        "nearest_resistance_level",
        "atr_14",
        "risk_atr_14",
        "risk_ob_bottom",
        "risk_atr_stop",
        "risk_entry_price",
        "risk_stoploss_price",
        "risk_atr_stop_distance_pct",
        "risk_stoploss_pct",
        "fib_position",
        "fib_zone",
        "recent_2w_ob_type",
        "recent_2w_ob_is_fallback",
        "recent_active_gap_type",
        "support_distance",
        "resistance_distance",
        "support_quality_score",
        "signal",
        "decision_summary_text",
        "decision_reason_text",
        "suggested_action_text",
        "decision_report_text",
    ] + ENHANCED_DECISION_COLUMNS + ENHANCED_DECISION_TEXT_COLUMNS + extra_diagnostic_cols
    output_cols = [col for col in dict.fromkeys(output_cols) if col in output_df.columns]
    output_df = output_df[output_cols]
    text_cols = [
        "risk_warning",
        "model_confidence_note",
        "final_decision_label",
        "signal",
        "decision_summary_text",
        "decision_reason_text",
        "suggested_action_text",
        "technical_filter_result",
        "rule_filter_notes",
        "trade_veto_reason",
        "trade_plan_notes",
        "score_band",
        "decision_action",
        "position_sizing_label",
        "position_sizing_reason",
        *[col for col in ENHANCED_DECISION_TEXT_COLUMNS if col != "enhanced_decision_report_text"],
    ]
    for col in text_cols:
        if col in output_df.columns:
            output_df[col] = output_df[col].apply(normalize_text_for_report)
        if col in validation_output_df.columns:
            validation_output_df[col] = validation_output_df[col].apply(normalize_text_for_report)
    validation_cols = [col for col in output_cols if col in validation_output_df.columns]
    validation_output_df = validation_output_df[validation_cols]

    summary, group_summary = build_single_diagnostic_summary(validation_output_df, config)
    fold_info_df = pd.DataFrame(fold_info_rows)
    summary["walk_forward_folds"] = len(fold_info_df)
    summary["test_rows"] = int(fold_info_df["valid_rows"].sum()) if "valid_rows" in fold_info_df.columns else summary.get("oos_rows")
    single_summary_keys = [
        "market", "symbol", "target_col", "actual_target_col",
        "diagnostic_scope", "feature_set", "model_type", "model_profile",
        "validation_method",
        "safety_score_mean", "return_score_mean", "combined_decision_score_mean",
        "high_risk_warning_days", "high_risk_warning_ratio",
        "weak_label_days", "neutral_label_days",
        "walk_forward_folds", "test_rows", "oos_rows", "MAE", "RMSE", "R2",
        "baseline_mae", "baseline_rmse", "baseline_r2",
        "spearman_corr",
    ]
    summary_df = pd.DataFrame([{key: summary.get(key) for key in single_summary_keys}])
    mode_1_metrics = {
        **summary,
        "top_n": top_n,
        "walk_forward_folds": summary.get("walk_forward_folds", len(fold_info_df)),
        "test_rows": summary.get("test_rows"),
    }
    register_model_flow_audit(
        collect_model_flow_audit(
            context_name="MODE_1_SINGLE_DIAGNOSTIC",
            config=config,
            data=model_df,
            feature_cols=feature_cols,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            removed_features=prepared.get("removed_features", []),
            metrics=mode_1_metrics,
            validation_scope=(
                "target_symbol_only_after_common_pool_split"
                if config.diagnostic_scope == "common"
                else "single_symbol_only"
            ),
            model_flow="manual_single_diagnostic_walk_forward_plus_latest_full_model",
        )
    )
    decision_label_backtest_summary = build_decision_label_backtest_summary(validation_output_df)
    latest_decision_df = output_df.sort_values("date").tail(1).copy()
    enhanced_decision_backtest = build_enhanced_decision_backtest_summary(validation_output_df, config)
    first_layer_score_band_summary = build_score_band_summary(
        validation_output_df,
        "combined_decision_score",
        "first_layer_score_band",
    )
    final_decision_score_band_summary = build_score_band_summary(
        validation_output_df,
        "final_decision_score",
        "final_decision_score_band",
    )
    veto_summary = build_veto_summary(validation_output_df)
    feature_and_rule_config = build_feature_and_rule_config(config)

    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    file_prefix = OUTPUT_DIR / (
        f"{config.market.upper()}_{config.stock_no}_{config.diagnostic_scope}_single_diagnostic_{timestamp}"
    )
    csv_path = Path(f"{file_prefix}.csv")
    xlsx_path = Path(f"{file_prefix}.xlsx")
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        validation_output_df.to_excel(writer, sheet_name="daily_oos_predictions", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        decision_label_backtest_summary.to_excel(writer, sheet_name="decision_label_backtest", index=False)
        latest_decision_df.to_excel(writer, sheet_name="latest_decision", index=False)
        output_df.to_excel(writer, sheet_name="all_diagnostics", index=False)
        summary_df.to_excel(writer, sheet_name="model_validation", index=False)
        enhanced_decision_backtest.to_excel(writer, sheet_name="enhanced_decision_backtest", index=False)
        first_layer_score_band_summary.to_excel(writer, sheet_name="first_layer_score_band_summary", index=False)
        final_decision_score_band_summary.to_excel(writer, sheet_name="final_decision_score_band_summary", index=False)
        veto_summary.to_excel(writer, sheet_name="veto_summary", index=False)
        feature_and_rule_config.to_excel(writer, sheet_name="feature_and_rule_config", index=False)
        if not fold_info_df.empty:
            fold_info_df.to_excel(writer, sheet_name="walk_forward_fold_info", index=False)

    if verbose:
        print("\n" + "=" * 70)
        print("單一個股診斷摘要")
        print("=" * 70)
        for key in [
            "target_col", "actual_target_col",
            "safety_score_mean", "return_score_mean", "combined_decision_score_mean",
            "high_risk_warning_days", "high_risk_warning_ratio",
            "weak_label_days", "neutral_label_days",
            "MAE", "RMSE", "R2", "spearman_corr",
            "oos_rows",
        ]:
            print(f"{key}: {summary.get(key)}")
        print("\ndecision_label_backtest_summary:")
        print(decision_label_backtest_summary.to_string(index=False))

        print("\n最近 20 筆 out-of-sample 預測：")
        preview_cols = [
            "date", "close", "predicted_score", "safety_score", "return_score",
            "combined_decision_score", "final_decision_label", "risk_warning",
            "model_confidence_note", "actual_target",
            "decision_summary_text", "suggested_action_text",
            "prediction_error", "future_return_10d", "future_drawdown_10d",
            "abs_future_drawdown_10d", "risk_adjusted_return_10d",
            "risk_adjusted_return_10d_capped", "signal",
        ]
        print(output_df.tail(20)[preview_cols].to_string(index=False))
        if "enhanced_decision_report_text" in output_df.columns and not output_df.empty:
            print("\n最新一筆教授展示版決策報告：")
            print(output_df.sort_values("date").tail(1)["enhanced_decision_report_text"].iloc[0])

        print("\n已輸出單一個股診斷報告：")
        print(f"1. {csv_path}")
        print(f"2. {xlsx_path}")
    else:
        print_final_investment_decision(output_df, config, summary=summary)
        print_presentation_model_settings(config, top_n=top_n if config.diagnostic_scope == "common" else None)
        print_presentation_feature_fold_appendix(
            config,
            feature_cols=feature_cols,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            removed_features=prepared.get("removed_features", []),
            fold_info=fold_info_df,
        )

    return output_df, summary_df


def get_dual_decision_label(row: pd.Series, safety_q40: float, safety_q70: float, return_q60: float) -> str:
    safety_score = row.get("safety_score", np.nan)
    return_score = row.get("return_score", np.nan)
    if pd.isna(safety_score) or pd.isna(return_score):
        return "中性觀察"
    if pd.notna(safety_q70) and pd.notna(return_q60) and safety_score >= safety_q70 and return_score < return_q60:
        return "穩健偏多，優先觀察"
    if pd.notna(safety_q70) and pd.notna(return_q60) and safety_score >= safety_q70 and return_score >= return_q60:
        return "安全與報酬分數偏高，仍需確認"
    if pd.notna(safety_q40) and pd.notna(return_q60) and safety_score < safety_q40 and return_score >= return_q60:
        return "短線機會型，但需控管風險"
    if pd.notna(safety_q40) and pd.notna(return_q60) and safety_score < safety_q40 and return_score < return_q60:
        return "分數偏弱，暫以觀察為主"
    return "中性觀察"


def get_dual_risk_warning(row: pd.Series, thresholds: dict[str, float]) -> str:
    warnings = []
    risk_checks = [
        ("atr_pct", "high", "ATR 波動偏高"),
        ("risk_atr_stop_distance_pct", "high", "停損距離偏大"),
        ("volatility_10d", "high", "短期波動偏高"),
        ("support_distance", "high", "距離支撐較遠"),
        ("resistance_distance", "low", "上方壓力接近"),
    ]
    for col, direction, message in risk_checks:
        value = row.get(col, np.nan)
        threshold = thresholds.get(col)
        if pd.isna(value) or pd.isna(threshold):
            continue
        if direction == "high" and value >= threshold:
            warnings.append(message)
        if direction == "low" and value <= threshold:
            warnings.append(message)
    warning_text = "；".join(warnings) if warnings else "無明顯風險警示"
    return clean_diagnostic_text(warning_text)


def build_decision_label_backtest_summary(output_df: pd.DataFrame) -> pd.DataFrame:
    group_df = output_df.copy()
    label_col = "final_decision_label" if "final_decision_label" in group_df.columns else "decision_label"
    if label_col not in group_df.columns:
        return pd.DataFrame()
    group_df[label_col] = group_df[label_col].apply(clean_diagnostic_text)
    if "risk_adjusted_return_10d_capped" not in group_df.columns:
        if "future_return_10d" in group_df.columns and "abs_future_drawdown_10d" in group_df.columns:
            group_df["risk_adjusted_return_10d_capped"] = calculate_risk_adjusted_return_10d_capped(
                group_df["future_return_10d"],
                group_df["abs_future_drawdown_10d"],
            )
    numeric_cols = [
        "combined_decision_score",
        "safety_score",
        "return_score",
        "future_return_10d",
        "future_drawdown_10d",
        "abs_future_drawdown_10d",
        "risk_adjusted_return_10d",
        "risk_adjusted_return_10d_capped",
        "target_safety_10d_cs_rank",
        "target_return_10d_cs_rank",
    ]
    for col in numeric_cols:
        if col in group_df.columns:
            group_df[col] = pd.to_numeric(group_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    group_agg = {
        "days": ("date", "count"),
        "safety_score_mean": ("safety_score", "mean"),
        "return_score_mean": ("return_score", "mean"),
        "combined_decision_score_mean": ("combined_decision_score", "mean"),
    }
    if "future_return_10d" in group_df.columns:
        group_agg["future_return_10d_mean"] = ("future_return_10d", "mean")
        group_agg["future_return_10d_median"] = ("future_return_10d", "median")
        group_agg["win_rate"] = ("future_return_10d", positive_return_rate)
    if "future_drawdown_10d" in group_df.columns:
        group_agg["future_drawdown_10d_mean"] = ("future_drawdown_10d", "mean")
    if "abs_future_drawdown_10d" in group_df.columns:
        group_agg["abs_future_drawdown_10d_mean"] = ("abs_future_drawdown_10d", "mean")
    if "risk_adjusted_return_10d_capped" in group_df.columns:
        group_agg["risk_adjusted_return_10d_capped_mean"] = ("risk_adjusted_return_10d_capped", "mean")
    if "risk_adjusted_return_10d" in group_df.columns:
        group_agg["risk_adjusted_return_10d_mean"] = ("risk_adjusted_return_10d", "mean")
    if "target_safety_10d_cs_rank" in group_df.columns:
        group_agg["target_safety_10d_cs_rank_mean"] = ("target_safety_10d_cs_rank", "mean")
    if "target_return_10d_cs_rank" in group_df.columns:
        group_agg["target_return_10d_cs_rank_mean"] = ("target_return_10d_cs_rank", "mean")

    summary = (
        group_df
        .dropna(subset=[label_col])
        .groupby(label_col, observed=False)
        .agg(**group_agg)
        .reset_index()
    )
    if "combined_decision_score_mean" in summary.columns:
        summary = summary.sort_values("combined_decision_score_mean", ascending=False).reset_index(drop=True)
    return summary


def build_dual_model_summary(
    output_df: pd.DataFrame,
    config: PipelineConfig,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    decision_counts = output_df["decision_label"].value_counts(dropna=False).to_dict()
    summary = {
        "market": config.market.upper(),
        "symbol": config.stock_no,
        "diagnostic_scope": config.diagnostic_scope,
        "feature_set": "selected_v2",
        "model_type": "extra_trees",
        "model_profile": "medium",
        "safety_target_col": f"target_safety_{config.horizon}d_cs_rank",
        "return_target_col": f"target_return_{config.horizon}d_cs_rank",
        "oos_rows": len(output_df),
        "safety_score_mean": pd.to_numeric(output_df["safety_score"], errors="coerce").mean(),
        "return_score_mean": pd.to_numeric(output_df["return_score"], errors="coerce").mean(),
        "combined_decision_score_mean": pd.to_numeric(output_df["combined_decision_score"], errors="coerce").mean(),
        "safety_q40": thresholds.get("safety_q40"),
        "safety_q60": thresholds.get("safety_q60"),
        "safety_q70": thresholds.get("safety_q70"),
        "return_q40": thresholds.get("return_q40"),
        "return_q60": thresholds.get("return_q60"),
        "return_q70": thresholds.get("return_q70"),
        "safety_score_interpretation": "單一個股目前風險結構參考分數",
        "return_score_interpretation": "單一個股目前上行潛力參考分數",
        "combined_decision_score_interpretation": "單一個股綜合診斷分數",
    }
    for label, count in decision_counts.items():
        summary[f"decision_count_{label}"] = count
    return pd.DataFrame([summary])


def format_optional_pct(value, decimals: int = 2) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "NA"
    return f"{numeric_value * 100:.{decimals}f}%"


def build_single_stock_decision_explanation(row: pd.Series) -> str:
    label = str(row.get("decision_label", "中性觀察"))
    risk_warning = str(row.get("risk_warning", "無明顯風險警示"))
    support_distance = pd.to_numeric(pd.Series([row.get("support_distance", np.nan)]), errors="coerce").iloc[0]
    resistance_distance = pd.to_numeric(pd.Series([row.get("resistance_distance", np.nan)]), errors="coerce").iloc[0]
    risk_stop_distance = pd.to_numeric(pd.Series([row.get("risk_atr_stop_distance_pct", np.nan)]), errors="coerce").iloc[0]
    volatility_10d = pd.to_numeric(pd.Series([row.get("volatility_10d", np.nan)]), errors="coerce").iloc[0]
    rsi14 = pd.to_numeric(pd.Series([row.get("RSI14", np.nan)]), errors="coerce").iloc[0]

    if label == "穩健偏多，優先觀察":
        parts = ["目前風險結構較穩，適合列入觀察清單，但仍應等待價格回到合理支撐或出現突破確認。"]
    elif label == "安全與報酬分數偏高，仍需確認":
        parts = ["目前風險與上行潛力分數都偏高，但不代表可以直接追價，仍需確認量價、支撐與停損距離。"]
    elif label == "短線機會型，但需控管風險":
        parts = ["短線上行潛力存在，但風險結構偏高，若要觀察應降低部位並先規劃停損。"]
    elif label == "分數偏弱，暫以觀察為主":
        parts = ["目前綜合分數偏弱，不宜急著進場，較適合等待支撐確認、波動下降或動能轉強。"]
    else:
        parts = ["目前訊號偏中性，建議以觀察為主，等待風險下降或方向更明確後再評估。"]

    if risk_warning and risk_warning != "無明顯風險警示":
        parts.append(f"主要風險提醒：{risk_warning}。")
    if pd.notna(rsi14) and rsi14 >= 70:
        parts.append("RSI 偏高，短線不宜追高。")
    if pd.notna(support_distance) and support_distance > 0.08:
        parts.append(f"距離支撐約 {format_optional_pct(support_distance)}，若進場需要更保守的風險控管。")
    if pd.notna(resistance_distance) and resistance_distance < 0.03:
        parts.append(f"距離上方壓力約 {format_optional_pct(resistance_distance)}，上檔空間需再確認。")
    if pd.notna(risk_stop_distance) and risk_stop_distance > 0.08:
        parts.append(f"ATR 停損距離約 {format_optional_pct(risk_stop_distance)}，停損成本偏高。")
    if pd.notna(volatility_10d) and volatility_10d > 0.035:
        parts.append("短期波動偏高，較適合等待波動收斂後再評估。")
    return " ".join(parts)


def build_latest_single_stock_decision_report(output_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if output_df.empty:
        return pd.DataFrame()

    latest = output_df.sort_values("date").tail(1).copy()
    latest["market"] = config.market.upper()
    latest["latest_date"] = latest["date"]
    latest["decision_explanation"] = latest.apply(build_single_stock_decision_explanation, axis=1)
    report_cols = [
        "symbol", "market", "latest_date", "close",
        "safety_score", "return_score", "combined_decision_score", "decision_label", "risk_warning",
        "RSI14", "MACD", "K", "D", "J", "BIAS24", "fib_position", "fib_zone",
        "atr_pct", "volatility_10d", "risk_atr_stop_distance_pct", "risk_stoploss_pct",
        "support_distance", "resistance_distance",
        "decision_explanation",
    ]
    for col in report_cols:
        if col not in latest.columns:
            latest[col] = np.nan
    return latest[report_cols].reset_index(drop=True)


def run_dual_model_diagnostic(config: PipelineConfig, top_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.stock_no.strip().upper() in {"ALL", "*", "全部"}:
        raise ValueError("dual_diagnostic 需要指定單一股票代碼，例如 --symbol 2330")
    if config.diagnostic_scope not in DIAGNOSTIC_SCOPE_CHOICES:
        raise ValueError(f"diagnostic_scope 只能是 {DIAGNOSTIC_SCOPE_CHOICES}")

    safety_target_col = f"target_safety_{config.horizon}d_cs_rank"
    return_target_col = f"target_return_{config.horizon}d_cs_rank"
    abs_drawdown_safety_col = f"target_abs_drawdown_safety_{config.horizon}d"
    dual_config = replace(
        config,
        feature_set="selected_v2",
        model_type="extra_trees",
        model_profile="medium",
        target_col=safety_target_col,
    )

    print("\n" + "=" * 70)
    print("Dual Model Diagnostic")
    print("=" * 70)
    print(f"market={dual_config.market.upper()}")
    print(f"symbol={dual_config.stock_no}")
    print(f"diagnostic_scope={dual_config.diagnostic_scope}")
    if dual_config.diagnostic_scope == "common":
        print(f"top_n={top_n}")
    print("feature_set=selected_v2")
    print("model_type=extra_trees")
    print("model_profile=medium")
    print(f"safety_target_col={safety_target_col}")
    print(f"return_target_col={return_target_col}")
    print(f"{abs_drawdown_safety_col} is kept as a diagnostic field, not a main model target.")

    if dual_config.diagnostic_scope == "common":
        data = prepare_common_single_diagnostic_feature_data(dual_config, top_n=top_n)
    else:
        data = prepare_single_stock_feature_data(dual_config)

    prepared = prepare_single_diagnostic_model_data(data, dual_config)
    model_df = prepared["model_df"]
    feature_cols = prepared["feature_cols"]
    numeric_features = prepared["numeric_features"]
    categorical_features = prepared["categorical_features"]

    if return_target_col not in model_df.columns:
        raise ValueError(f"return target column missing: {return_target_col}")
    model_df = model_df.dropna(subset=[return_target_col]).copy()
    if len(model_df) < 50:
        raise ValueError("dual_diagnostic 可用資料不足，請確認 target/feature 欄位。")

    print("\nDual diagnostic walk-forward out-of-sample")
    print(f"model_feature_count={len(feature_cols)}")
    print(f"model_numeric_features={numeric_features}")
    print(f"model_categorical_features={categorical_features}")
    print(f"removed_features={prepared['removed_features']}")

    X = model_df[feature_cols]
    y_safety = pd.to_numeric(model_df[safety_target_col], errors="coerce")
    y_return = pd.to_numeric(model_df[return_target_col], errors="coerce")
    splits = build_walk_forward_splits_by_date(model_df=model_df, n_splits=5)
    target_symbol = str(dual_config.stock_no).strip()

    diagnostic_frames = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        safety_model = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=dual_config.random_state + fold_no,
            model_profile="medium",
            model_type="extra_trees",
        )
        return_model = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=dual_config.random_state + 1000 + fold_no,
            model_profile="medium",
            model_type="extra_trees",
        )
        safety_model.fit(X_train, y_safety.iloc[train_idx])
        return_model.fit(X_train, y_return.iloc[train_idx])

        fold_df = model_df.iloc[test_idx].copy()
        fold_df["safety_score"] = np.clip(safety_model.predict(X_test), 0, 1)
        fold_df["return_score"] = np.clip(return_model.predict(X_test), 0, 1)

        symbol_fold_rows = len(fold_df)
        if dual_config.diagnostic_scope == "common":
            fold_df = fold_df[fold_df["symbol"].astype(str) == target_symbol].copy()
            symbol_fold_rows = len(fold_df)
            if fold_df.empty:
                print(
                    f"Fold {fold_no}: train={len(train_idx)}, "
                    f"valid={len(test_idx)}, target_symbol_valid=0，略過輸出"
                )
                continue
        diagnostic_frames.append(fold_df)

        print(
            f"Fold {fold_no}: train={len(train_idx)}, "
            f"valid={len(test_idx)}, "
            f"target_symbol_valid={symbol_fold_rows}, "
            f"date={pd.to_datetime(fold_df['date']).min().date()} ~ {pd.to_datetime(fold_df['date']).max().date()}"
        )

    if not diagnostic_frames:
        raise ValueError("無法產生 dual_diagnostic walk-forward 診斷預測，請確認資料長度。")

    oos_df = pd.concat(diagnostic_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    future_return_10d = pd.to_numeric(
        get_series_or_default(oos_df, f"future_return_{dual_config.horizon}d"),
        errors="coerce",
    )
    future_drawdown_10d = pd.to_numeric(
        get_series_or_default(oos_df, f"future_drawdown_{dual_config.horizon}d"),
        errors="coerce",
    )
    abs_future_drawdown_10d = future_drawdown_10d.abs()
    risk_adjusted_return_10d = (
        future_return_10d / (abs_future_drawdown_10d + 1e-6)
    ).replace([np.inf, -np.inf], np.nan)

    output_df = pd.DataFrame({
        "date": pd.to_datetime(oos_df["date"], errors="coerce"),
        "symbol": get_series_or_default(oos_df, "symbol", dual_config.stock_no),
        "close": pd.to_numeric(get_series_or_default(oos_df, "close"), errors="coerce"),
        "safety_score": pd.to_numeric(oos_df["safety_score"], errors="coerce"),
        "return_score": pd.to_numeric(oos_df["return_score"], errors="coerce"),
        "actual_safety_target": pd.to_numeric(get_series_or_default(oos_df, safety_target_col), errors="coerce"),
        "actual_return_target": pd.to_numeric(get_series_or_default(oos_df, return_target_col), errors="coerce"),
        safety_target_col: pd.to_numeric(get_series_or_default(oos_df, safety_target_col), errors="coerce"),
        return_target_col: pd.to_numeric(get_series_or_default(oos_df, return_target_col), errors="coerce"),
        "target_abs_drawdown_safety_10d": pd.to_numeric(get_series_or_default(oos_df, abs_drawdown_safety_col), errors="coerce"),
        "future_return_10d": future_return_10d,
        "future_drawdown_10d": future_drawdown_10d,
        "abs_future_drawdown_10d": abs_future_drawdown_10d,
        "risk_adjusted_return_10d": risk_adjusted_return_10d,
        "return_5d": pd.to_numeric(get_series_or_default(oos_df, "return_5d"), errors="coerce"),
        "return_10d": pd.to_numeric(get_series_or_default(oos_df, "return_10d"), errors="coerce"),
        "atr_pct": pd.to_numeric(get_series_or_default(oos_df, "atr_pct"), errors="coerce"),
        "atr_14": pd.to_numeric(get_series_or_default(oos_df, "atr_14"), errors="coerce"),
        "risk_entry_price": pd.to_numeric(get_series_or_default(oos_df, "risk_entry_price"), errors="coerce"),
        "risk_stoploss_price": pd.to_numeric(get_series_or_default(oos_df, "risk_stoploss_price"), errors="coerce"),
        "risk_atr_stop_distance_pct": pd.to_numeric(get_series_or_default(oos_df, "risk_atr_stop_distance_pct"), errors="coerce"),
        "risk_stoploss_pct": pd.to_numeric(get_series_or_default(oos_df, "risk_stoploss_pct"), errors="coerce"),
        "volatility_10d": pd.to_numeric(get_series_or_default(oos_df, "volatility_10d"), errors="coerce"),
        "nearest_support_level": pd.to_numeric(get_series_or_default(oos_df, "nearest_support_level"), errors="coerce"),
        "nearest_resistance_level": pd.to_numeric(get_series_or_default(oos_df, "nearest_resistance_level"), errors="coerce"),
        "support_distance": pd.to_numeric(get_series_or_default(oos_df, "support_distance"), errors="coerce"),
        "resistance_distance": pd.to_numeric(get_series_or_default(oos_df, "resistance_distance"), errors="coerce"),
        "RSI14": pd.to_numeric(get_series_or_default(oos_df, "RSI14"), errors="coerce"),
        "MACD": pd.to_numeric(get_series_or_default(oos_df, "MACD"), errors="coerce"),
        "K": pd.to_numeric(get_series_or_default(oos_df, "K"), errors="coerce"),
        "D": pd.to_numeric(get_series_or_default(oos_df, "D"), errors="coerce"),
        "J": pd.to_numeric(get_series_or_default(oos_df, "J"), errors="coerce"),
        "BIAS24": pd.to_numeric(get_series_or_default(oos_df, "BIAS24"), errors="coerce"),
        "VR_14": pd.to_numeric(get_series_or_default(oos_df, "VR_14"), errors="coerce"),
        "VR_26": pd.to_numeric(get_series_or_default(oos_df, "VR_26"), errors="coerce"),
        "fib_position": pd.to_numeric(get_series_or_default(oos_df, "fib_position"), errors="coerce"),
        "fib_zone": get_series_or_default(oos_df, "fib_zone", "none"),
        "ma5": pd.to_numeric(get_series_or_default(oos_df, "ma5"), errors="coerce"),
        "ma20": pd.to_numeric(get_series_or_default(oos_df, "ma20"), errors="coerce"),
        "ma60": pd.to_numeric(get_series_or_default(oos_df, "ma60"), errors="coerce"),
        "fib_0618_price": pd.to_numeric(get_series_or_default(oos_df, "fib_0618_price"), errors="coerce"),
        "risk_atr_stop": pd.to_numeric(get_series_or_default(oos_df, "risk_atr_stop"), errors="coerce"),
        "close_to_20d_low": pd.to_numeric(get_series_or_default(oos_df, "close_to_20d_low"), errors="coerce"),
        "nearest_resistance_zone_high": pd.to_numeric(get_series_or_default(oos_df, "nearest_resistance_zone_high"), errors="coerce"),
        "volume_ratio_20": pd.to_numeric(get_series_or_default(oos_df, "volume_ratio_20"), errors="coerce"),
    })

    output_df["combined_decision_score"] = (
        0.65 * output_df["safety_score"] + 0.35 * output_df["return_score"]
    )
    safety_q40 = output_df["safety_score"].quantile(0.40)
    safety_q60 = output_df["safety_score"].quantile(0.60)
    safety_q70 = output_df["safety_score"].quantile(0.70)
    return_q40 = output_df["return_score"].quantile(0.40)
    return_q60 = output_df["return_score"].quantile(0.60)
    return_q70 = output_df["return_score"].quantile(0.70)
    output_df["decision_label"] = output_df.apply(
        lambda row: get_dual_decision_label(row, safety_q40, safety_q70, return_q60),
        axis=1,
    )

    risk_thresholds = {
        "atr_pct": output_df["atr_pct"].quantile(0.80),
        "risk_atr_stop_distance_pct": output_df["risk_atr_stop_distance_pct"].quantile(0.80),
        "volatility_10d": output_df["volatility_10d"].quantile(0.80),
        "support_distance": output_df["support_distance"].quantile(0.80),
        "resistance_distance": output_df["resistance_distance"].quantile(0.20),
        "safety_q40": safety_q40,
        "safety_q60": safety_q60,
        "safety_q70": safety_q70,
        "return_q40": return_q40,
        "return_q60": return_q60,
        "return_q70": return_q70,
    }
    output_df["risk_warning"] = output_df.apply(
        lambda row: get_dual_risk_warning(row, risk_thresholds),
        axis=1,
    )
    output_df["final_decision_label"] = output_df["decision_label"]
    output_df["model_confidence_note"] = ""
    output_df["signal"] = output_df["combined_decision_score"].apply(get_signal)
    decision_text_df = output_df.apply(build_decision_text, axis=1, result_type="expand")
    output_df = pd.concat([output_df, decision_text_df], axis=1)
    output_df = apply_enhanced_decision_layers(output_df)

    output_cols = [
        "date", "symbol", "close",
        "safety_score", "return_score",
        "combined_decision_score", "decision_label", "final_decision_label", "risk_warning",
        "model_confidence_note",
        "actual_safety_target", "actual_return_target",
        safety_target_col, return_target_col,
        "target_abs_drawdown_safety_10d",
        "future_return_10d", "future_drawdown_10d", "abs_future_drawdown_10d", "risk_adjusted_return_10d",
        "return_5d", "return_10d",
        "atr_pct", "atr_14", "volatility_10d",
        "risk_entry_price", "risk_stoploss_price", "risk_atr_stop_distance_pct", "risk_stoploss_pct",
        "nearest_support_level", "nearest_resistance_level",
        "support_distance", "resistance_distance",
        "RSI14", "MACD", "K", "D", "J", "BIAS24", "VR_14", "VR_26", "fib_position", "fib_zone",
        "signal",
        "decision_summary_text",
        "decision_reason_text",
        "suggested_action_text",
        "decision_report_text",
    ] + ENHANCED_DECISION_COLUMNS + ENHANCED_DECISION_TEXT_COLUMNS
    output_cols = [col for col in dict.fromkeys(output_cols) if col in output_df.columns]
    output_df = output_df[output_cols]
    text_cols = [
        "decision_label",
        "final_decision_label",
        "risk_warning",
        "model_confidence_note",
        "signal",
        "decision_summary_text",
        "decision_reason_text",
        "suggested_action_text",
        "technical_filter_result",
        "rule_filter_notes",
        "trade_veto_reason",
        "trade_plan_notes",
        "score_band",
        "decision_action",
        "position_sizing_label",
        "position_sizing_reason",
        *[col for col in ENHANCED_DECISION_TEXT_COLUMNS if col != "enhanced_decision_report_text"],
    ]
    for col in text_cols:
        if col in output_df.columns:
            output_df[col] = output_df[col].apply(normalize_text_for_report)
    summary_df = build_dual_model_summary(
        output_df,
        dual_config,
        risk_thresholds,
    )
    decision_label_backtest_summary = build_decision_label_backtest_summary(output_df)
    latest_single_stock_decision_report = build_latest_single_stock_decision_report(output_df, dual_config)
    latest_decision_df = output_df.sort_values("date").tail(1).copy()
    enhanced_decision_backtest = build_enhanced_decision_backtest_summary(output_df, dual_config)
    first_layer_score_band_summary = build_score_band_summary(
        output_df,
        "combined_decision_score",
        "first_layer_score_band",
    )
    final_decision_score_band_summary = build_score_band_summary(
        output_df,
        "final_decision_score",
        "final_decision_score_band",
    )
    veto_summary = build_veto_summary(output_df)
    feature_and_rule_config = build_feature_and_rule_config(dual_config)

    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    file_prefix = OUTPUT_DIR / (
        f"{dual_config.market.upper()}_{dual_config.stock_no}_{dual_config.diagnostic_scope}_dual_diagnostic_{timestamp}"
    )
    csv_path = Path(f"{file_prefix}.csv")
    xlsx_path = Path(f"{file_prefix}.xlsx")
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="daily_oos_predictions", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        decision_label_backtest_summary.to_excel(writer, sheet_name="decision_label_backtest", index=False)
        latest_single_stock_decision_report.to_excel(writer, sheet_name="latest_decision_report", index=False)
        latest_decision_df.to_excel(writer, sheet_name="latest_decision", index=False)
        output_df.to_excel(writer, sheet_name="all_diagnostics", index=False)
        summary_df.to_excel(writer, sheet_name="model_validation", index=False)
        enhanced_decision_backtest.to_excel(writer, sheet_name="enhanced_decision_backtest", index=False)
        first_layer_score_band_summary.to_excel(writer, sheet_name="first_layer_score_band_summary", index=False)
        final_decision_score_band_summary.to_excel(writer, sheet_name="final_decision_score_band_summary", index=False)
        veto_summary.to_excel(writer, sheet_name="veto_summary", index=False)
        feature_and_rule_config.to_excel(writer, sheet_name="feature_and_rule_config", index=False)

    print("\n" + "=" * 70)
    print("latest_single_stock_decision_report")
    print("=" * 70)
    if latest_single_stock_decision_report.empty:
        print("No latest decision report available.")
    else:
        print(latest_single_stock_decision_report.to_string(index=False))

    print("\nscore interpretation:")
    print("safety_score = 單一個股目前風險結構參考分數")
    print("return_score = 單一個股目前上行潛力參考分數")
    print("combined_decision_score = 單一個股綜合診斷分數")

    print("\nDual model diagnostic summary")
    print("=" * 70)
    for key in [
        "feature_set", "model_type", "model_profile",
        "safety_target_col", "return_target_col",
        "oos_rows", "safety_score_mean", "return_score_mean",
        "combined_decision_score_mean",
        "safety_q40", "safety_q60", "safety_q70",
        "return_q40", "return_q60", "return_q70",
    ]:
        print(f"{key}: {summary_df.iloc[0].get(key)}")
    print("\ndecision_label_backtest_summary:")
    print(decision_label_backtest_summary.to_string(index=False))
    print("\n最近 20 筆 dual model 預測：")
    preview_cols = [
        "date", "close", "safety_score", "return_score",
        "combined_decision_score", "decision_label", "risk_warning",
    ]
    print(output_df.tail(20)[preview_cols].to_string(index=False))
    print("\n已輸出 dual model 診斷報告：")
    print(f"1. {csv_path}")
    print(f"2. {xlsx_path}")

    return output_df, summary_df


# ============================================================
# 6. 輸出與主流程
# ============================================================

def clean_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """保留第一個同名欄位，避免 merge 或 concat 後欄位重複。"""
    return df.loc[:, ~df.columns.duplicated()].copy()

def build_column_audit_report(
    df: pd.DataFrame,
    config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    建立欄位檢查報告：

    1. column_report：
       每個欄位的型態、缺失數、缺失率、唯一值數量、範例值

    2. categorical_report：
       重要類別欄位的分布

    3. target_report：
       target_score 與未來上行 / 下行風險欄位的摘要統計
    """

    total_rows = len(df)

    column_rows = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_rate = missing_count / total_rows if total_rows > 0 else np.nan
        non_missing = df[col].dropna()

        sample_values = non_missing.astype(str).head(5).tolist()

        column_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 4),
            "non_missing_count": int(total_rows - missing_count),
            "unique_count": int(non_missing.nunique()) if not non_missing.empty else 0,
            "sample_values": " | ".join(sample_values),
        })

    column_report = pd.DataFrame(column_rows).sort_values(
        by=["missing_rate", "column"],
        ascending=[False, True],
    )

    categorical_candidates = [
        "fib_zone",
        "recent_2w_ob_type",
        "recent_2w_ob_is_fallback",
        "recent_active_gap_type",
        "signal",
    ]

    categorical_rows = []

    for col in categorical_candidates:
        if col not in df.columns:
            continue

        value_counts = (
            df[col]
            .astype("object")
            .where(df[col].notna(), "missing")
            .value_counts(dropna=False)
        )

        for value, count in value_counts.items():
            categorical_rows.append({
                "column": col,
                "value": value,
                "count": int(count),
                "rate": round(count / total_rows, 4) if total_rows > 0 else np.nan,
            })

    categorical_report = pd.DataFrame(categorical_rows)

    target_cols = [
        "target_score",
        "target_score_v2",
        "target_score_v3",
        "investment_score",
        f"future_max_high_{config.horizon}d",
        f"future_min_low_{config.horizon}d",
        f"future_upside_{config.horizon}d",
        f"future_downside_{config.horizon}d",
        f"future_close_{config.horizon}d",
        f"future_return_{config.horizon}d",
        f"future_drawdown_{config.horizon}d",
        f"risk_adjusted_return_{config.horizon}d",
        f"target_return_{config.horizon}d",
        f"target_return_{config.horizon}d_cs_rank",
        f"target_downside_risk_{config.horizon}d",
        f"target_downside_risk_{config.horizon}d_cs_rank",
        f"target_safety_{config.horizon}d_cs_rank",
        f"target_abs_drawdown_safety_{config.horizon}d",
        f"target_rar_{config.horizon}d",
        f"target_rar_{config.horizon}d_cs_rank",
        "predicted_target_score",
        "predicted_investment_score",
    ]

    target_cols = [col for col in target_cols if col in df.columns]

    target_rows = []

    for col in target_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        target_rows.append({
            "column": col,
            "count": int(s.count()),
            "missing_count": int(s.isna().sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max(),
        })

    target_report = pd.DataFrame(target_rows)

    return column_report, categorical_report, target_report

def save_audit_report(df: pd.DataFrame, config: PipelineConfig) -> None:
    column_report, categorical_report, target_report = build_column_audit_report(
        df,
        config
    )

    file_prefix = f"{config.market}_{config.stock_no}_{config.feature_set}"

    audit_csv_path = OUTPUT_DIR / f"{file_prefix}_column_audit_report.csv"
    audit_xlsx_path = OUTPUT_DIR / f"{file_prefix}_column_audit_report.xlsx"

    column_report.to_csv(
        audit_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(audit_xlsx_path, engine="openpyxl") as writer:
        column_report.to_excel(
            writer,
            sheet_name="column_missing_report",
            index=False
        )

        categorical_report.to_excel(
            writer,
            sheet_name="categorical_distribution",
            index=False
        )

        target_report.to_excel(
            writer,
            sheet_name="target_summary",
            index=False
        )

    print("\n已輸出欄位檢查報告：")
    print(f"1. {audit_csv_path}")
    print(f"2. {audit_xlsx_path}")




# 新增在這裡
def append_latest_prediction_summary(df: pd.DataFrame, config: PipelineConfig) -> None:
    """
    將每一檔股票的最新預測結果追加到總表。
    若同一檔股票同一天已存在，則更新該列，避免重複。
    """

    latest = df.iloc[-1].copy()

    summary_path = OUTPUT_DIR / "all_latest_predictions.csv"

    row = {
        "date": latest.get("date"),
        "symbol": latest.get("symbol", config.stock_no),
        "market": latest.get("market", config.market),
        "close": latest.get("close"),
        "predicted_target_score": latest.get("predicted_target_score"),
        "predicted_investment_score": latest.get("predicted_investment_score"),
        "signal": latest.get("signal"),
        "model_target_col": config.target_col,
        "target_score": latest.get("target_score"),
        "target_score_v2": latest.get("target_score_v2"),
        "target_score_v3": latest.get("target_score_v3"),
        f"future_max_high_{config.horizon}d": latest.get(f"future_max_high_{config.horizon}d"),
        f"future_min_low_{config.horizon}d": latest.get(f"future_min_low_{config.horizon}d"),
        f"future_upside_{config.horizon}d": latest.get(f"future_upside_{config.horizon}d"),
        f"future_downside_{config.horizon}d": latest.get(f"future_downside_{config.horizon}d"),
        f"future_close_{config.horizon}d": latest.get(f"future_close_{config.horizon}d"),
        f"future_return_{config.horizon}d": latest.get(f"future_return_{config.horizon}d"),
        f"future_drawdown_{config.horizon}d": latest.get(f"future_drawdown_{config.horizon}d"),
        f"risk_adjusted_return_{config.horizon}d": latest.get(f"risk_adjusted_return_{config.horizon}d"),
        "RSI14": latest.get("RSI14"),
        "MACD": latest.get("MACD"),
        "K": latest.get("K"),
        "D": latest.get("D"),
        "J": latest.get("J"),
        "BIAS24": latest.get("BIAS24"),
        "VR_14": latest.get("VR_14"),
        "VR_26": latest.get("VR_26"),
        "VR_14_change": latest.get("VR_14_change"),
        "VR_14_zone": latest.get("VR_14_zone"),
        "fib_position": latest.get("fib_position"),
        "fib_zone": latest.get("fib_zone"),
        "nearest_support_level": latest.get("nearest_support_level"),
        "nearest_resistance_level": latest.get("nearest_resistance_level"),
        "atr_14": latest.get("atr_14"),
        "atr_pct": latest.get("atr_pct"),
        "risk_stoploss_price": latest.get("risk_stoploss_price"),
        "risk_per_share": latest.get("risk_per_share"),
        "risk_stoploss_pct": latest.get("risk_stoploss_pct"),
        "recent_2w_ob_type": latest.get("recent_2w_ob_type"),
        "recent_2w_ob_is_fallback": latest.get("recent_2w_ob_is_fallback"),
        "recent_active_gap_type": latest.get("recent_active_gap_type"),
    }

    new_row_df = pd.DataFrame([row])

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df["date"] = summary_df["date"].astype(str)
        new_row_df["date"] = new_row_df["date"].astype(str)

        mask = ~(
            (summary_df["market"].astype(str) == str(row["market"]))
            & (summary_df["symbol"].astype(str) == str(row["symbol"]))
            & (summary_df["date"].astype(str) == str(row["date"]))
        )

        summary_df = pd.concat(
            [summary_df.loc[mask], new_row_df],
            ignore_index=True
        )
    else:
        summary_df = new_row_df

    summary_df = summary_df.sort_values(
        by=["date", "market", "symbol"],
        ascending=[False, True, True]
    )

    try:
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        saved_path = summary_path
    except PermissionError:
        fallback_path = OUTPUT_DIR / f"all_latest_predictions_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        saved_path = fallback_path
        print("\n提醒：all_latest_predictions.csv 可能正被 Excel 或其他程式開啟，已改存成備用檔。")

    print("\n已更新所有股票最新預測總表：")
    print(saved_path)


def save_walk_forward_validation_report(metrics: dict, config: PipelineConfig) -> None:
    report = metrics.get("walk_forward_report")
    if report is None or report.empty:
        return

    file_prefix = f"{config.market}_{config.stock_no}_{config.feature_set}_{config.target_col}"
    report_csv = OUTPUT_DIR / f"{file_prefix}_walk_forward_validation.csv"
    report_xlsx = OUTPUT_DIR / f"{file_prefix}_walk_forward_validation.xlsx"

    report.to_csv(report_csv, index=False, encoding="utf-8-sig")
    report.to_excel(report_xlsx, index=False)

    print("\n已輸出 walk-forward validation 報告：")
    print(f"1. {report_csv}")
    print(f"2. {report_xlsx}")


def save_feature_importance_report(metrics: dict, config: PipelineConfig) -> None:
    report = metrics.get("feature_importance_report")
    if report is None or report.empty:
        return

    file_prefix = f"{config.market}_{config.stock_no}_{config.feature_set}_{config.target_col}"
    report_csv = OUTPUT_DIR / f"{file_prefix}_feature_importance.csv"
    report_xlsx = OUTPUT_DIR / f"{file_prefix}_feature_importance.xlsx"

    report.to_csv(report_csv, index=False, encoding="utf-8-sig")
    report.to_excel(report_xlsx, index=False)

    print("\n已輸出特徵重要度報告（只輸出，不自動刪除）：")
    print(f"1. {report_csv}")
    print(f"2. {report_xlsx}")


def save_target_diagnostic_report(metrics: dict, config: PipelineConfig) -> None:
    diagnostics = metrics.get("target_score_diagnostics")
    if not diagnostics:
        return

    file_prefix = f"{config.market}_{config.stock_no}_{config.feature_set}_{config.target_col}"
    report_xlsx = OUTPUT_DIR / f"{file_prefix}_target_score_diagnostic.xlsx"
    corr_csv = OUTPUT_DIR / f"{file_prefix}_target_score_correlation.csv"
    group_csv = OUTPUT_DIR / f"{file_prefix}_target_score_groups.csv"
    top_bottom_csv = OUTPUT_DIR / f"{file_prefix}_target_score_top_bottom.csv"
    fold_csv = OUTPUT_DIR / f"{file_prefix}_fold_level_diagnostic.csv"
    prediction_distribution_csv = OUTPUT_DIR / f"{file_prefix}_prediction_distribution_diagnostic.csv"

    correlation_report = diagnostics.get("correlation_report", pd.DataFrame())
    group_report = diagnostics.get("group_report", pd.DataFrame())
    top_bottom_report = diagnostics.get("top_bottom_report", pd.DataFrame())
    fold_diagnostic_report = diagnostics.get("fold_diagnostic_report", pd.DataFrame())

    if not correlation_report.empty:
        correlation_report.to_csv(corr_csv, index=False, encoding="utf-8-sig")
    if not group_report.empty:
        group_report.to_csv(group_csv, index=False, encoding="utf-8-sig")
    if not top_bottom_report.empty:
        top_bottom_report.to_csv(top_bottom_csv, index=False, encoding="utf-8-sig")
    if not fold_diagnostic_report.empty:
        fold_diagnostic_report.to_csv(fold_csv, index=False, encoding="utf-8-sig")
        prediction_distribution_cols = [
            col for col in fold_diagnostic_report.columns
            if (
                col in {"fold", "test_start_date", "test_end_date", "test_rows"}
                or col.startswith("y_true_")
                or col.startswith("y_pred_")
                or col in {
                    "prediction_std_target_std_ratio",
                    "prediction_0_4_0_6_share",
                    "future_return_mean",
                    "future_return_std",
                    "diagnostic_note",
                }
            )
        ]
        fold_diagnostic_report[prediction_distribution_cols].to_csv(
            prediction_distribution_csv,
            index=False,
            encoding="utf-8-sig",
        )

    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as writer:
        wrote_sheet = False
        if not correlation_report.empty:
            correlation_report.to_excel(writer, sheet_name="correlation", index=False)
            wrote_sheet = True
        if not group_report.empty:
            group_report.to_excel(writer, sheet_name="target_groups", index=False)
            wrote_sheet = True
        if not top_bottom_report.empty:
            top_bottom_report.to_excel(writer, sheet_name="top_bottom_10pct", index=False)
            wrote_sheet = True
        if not fold_diagnostic_report.empty:
            fold_diagnostic_report.to_excel(writer, sheet_name="fold_diagnostic", index=False)
            wrote_sheet = True
            fold_diagnostic_report[prediction_distribution_cols].to_excel(
                writer,
                sheet_name="prediction_distribution",
                index=False,
            )
        if not wrote_sheet:
            pd.DataFrame({"message": ["No target diagnostics available."]}).to_excel(
                writer,
                sheet_name="diagnostic",
                index=False,
            )

    print(f"\n已輸出 {config.target_col} 診斷報告：")
    print(f"1. {report_xlsx}")
    if not correlation_report.empty:
        print(f"2. {corr_csv}")
    if not group_report.empty:
        print(f"3. {group_csv}")
    if not top_bottom_report.empty:
        print(f"4. {top_bottom_csv}")
    if not fold_diagnostic_report.empty:
        print(f"5. {fold_csv}")
        print(f"6. {prediction_distribution_csv}")


def save_outputs(df: pd.DataFrame, config: PipelineConfig, metrics: Optional[dict] = None) -> None:
    file_prefix = f"{config.market}_{config.stock_no}_{config.feature_set}"

    final_csv = OUTPUT_DIR / f"{file_prefix}_final_stock_table.csv"
    final_xlsx = OUTPUT_DIR / f"{file_prefix}_final_stock_table.xlsx"

    ml_csv = OUTPUT_DIR / f"{file_prefix}_ml_prediction_with_risk.csv"
    ml_xlsx = OUTPUT_DIR / f"{file_prefix}_ml_prediction_with_risk.xlsx"

    df = clean_duplicate_columns(df)

    df.to_csv(final_csv, index=False, encoding="utf-8-sig")
    df.to_excel(final_xlsx, index=False)

    ml_output_cols = [
        "date", "symbol", "market", "close",
        "predicted_target_score", "predicted_score", "model_score",
        "predicted_investment_score", "signal",
        "target_score", "target_score_v2", "target_score_v3",
        "investment_score",
        f"future_max_high_{config.horizon}d",
        f"future_min_low_{config.horizon}d",
        f"future_upside_{config.horizon}d",
        f"future_downside_{config.horizon}d",
        f"future_close_{config.horizon}d",
        f"future_return_{config.horizon}d",
        f"future_drawdown_{config.horizon}d",
        f"risk_adjusted_return_{config.horizon}d",
        "risk_entry_price", "risk_atr_14", "risk_ob_bottom", "risk_atr_stop",
        "risk_stoploss_price", "risk_per_share", "risk_stoploss_pct",
        "risk_atr_stop_distance_pct", "risk_ob_bottom_distance_pct",
        "risk_stoploss_distance_pct", "risk_valid",
        "RSI14", "MACD", "K", "D", "J", "BIAS24",
        "VR_14", "VR_26", "VR_14_change", "VR_14_zone",
        "fib_position", "fib_zone", "fib_score",
        "nearest_support_level", "nearest_resistance_level",
        "atr_pct", "recent_2w_ob_type", "recent_active_gap_type",
    ]
    ml_output_cols = [col for col in ml_output_cols if col in df.columns]
    ml_result = df[ml_output_cols].copy()

    ml_result.to_csv(ml_csv, index=False, encoding="utf-8-sig")
    ml_result.to_excel(ml_xlsx, index=False)

    print("\n已輸出檔案：")
    print(f"1. {final_csv}")
    print(f"2. {final_xlsx}")
    print(f"3. {ml_csv}")
    print(f"4. {ml_xlsx}")
    save_audit_report(df, config)
    if metrics is not None:
        save_walk_forward_validation_report(metrics, config)
        save_feature_importance_report(metrics, config)
    append_latest_prediction_summary(df, config)

def build_latest_decision_guide(latest: pd.Series) -> pd.DataFrame:
    """
    建立最新一日的決策輔助指南。
    這不是模型訓練用 target，而是把模型輸出與技術 / 風險指標
    轉成可解釋的決策說明。
    """

    rows = []

    def add_row(indicator: str, value, status: str, interpretation: str, decision_effect: str) -> None:
        rows.append({
            "indicator": indicator,
            "value": value,
            "status": status,
            "interpretation": interpretation,
            "decision_effect": decision_effect,
        })

    # =========================
    # RSI
    # =========================
    if "RSI14" in latest.index and pd.notna(latest["RSI14"]):
        rsi = float(latest["RSI14"])

        if rsi >= 70:
            status = "過熱"
            interpretation = "短線可能過度樂觀，追高風險增加。"
            effect = "降低積極進場意願，等待回檔或確認突破。"
        elif rsi >= 55:
            status = "中性偏強"
            interpretation = "動能偏多，但尚未明顯過熱。"
            effect = "若其他條件配合，可列入觀察。"
        elif rsi >= 45:
            status = "中性"
            interpretation = "多空力量接近平衡，方向尚未明確。"
            effect = "以觀察為主，避免單靠 RSI 進場。"
        elif rsi >= 30:
            status = "中性偏弱"
            interpretation = "短線動能偏弱，買盤尚未明顯恢復。"
            effect = "降低進場信心，等待轉強訊號。"
        else:
            status = "超賣"
            interpretation = "價格可能短線過度下跌，但不代表立即反彈。"
            effect = "可觀察反彈機會，但需搭配止損與結構確認。"

        add_row("RSI14", round(rsi, 2), status, interpretation, effect)

    # =========================
    # MACD
    # =========================
    if "MACD" in latest.index and pd.notna(latest["MACD"]):
        macd = float(latest["MACD"])

        if macd > 0:
            status = "偏強"
            interpretation = "中短期動能偏多，趨勢可能正在改善。"
            effect = "提高模型分數可信度，支援偏多判斷。"
        elif macd < 0:
            status = "偏弱"
            interpretation = "動能仍未完全翻多，趨勢仍有壓力。"
            effect = "降低追高意願，需等待動能改善。"
        else:
            status = "中性"
            interpretation = "MACD 接近零軸，方向尚不明確。"
            effect = "不作為主要決策依據。"

        add_row("MACD", round(macd, 2), status, interpretation, effect)

    # =========================
    # KDJ
    # =========================
    if all(col in latest.index and pd.notna(latest[col]) for col in ["K", "D", "J"]):
        k = float(latest["K"])
        d = float(latest["D"])
        j = float(latest["J"])

        if k > d and k < 80:
            status = "短線偏多"
            interpretation = "K 值高於 D 值，短線動能有轉強跡象。"
            effect = "若價格未接近壓力，可提高觀察或分批進場意願。"
        elif k < d:
            status = "短線偏弱"
            interpretation = "K 值低於 D 值，短線動能偏弱。"
            effect = "降低短線進場意願，等待 K 值重新轉強。"
        elif k >= 80:
            status = "高檔區"
            interpretation = "短線可能進入高檔區，容易震盪。"
            effect = "避免追高，需搭配壓力位與停利策略。"
        else:
            status = "中性"
            interpretation = "KDJ 尚未出現明確方向。"
            effect = "不作為主要決策依據。"

        add_row("KDJ", f"K={k:.2f}, D={d:.2f}, J={j:.2f}", status, interpretation, effect)

    # =========================
    # Fib Position
    # =========================
    if "fib_position" in latest.index and pd.notna(latest["fib_position"]):
        fib = float(latest["fib_position"])
        fib_zone = latest.get("fib_zone", "unknown")

        if fib < 0.5:
            status = "偏低位階"
            interpretation = "價格位於近 10 日區間偏低位置，具有相對便宜特徵。"
            effect = "可提高觀察價值，但仍需確認動能與支撐品質。"
        elif fib <= 0.786:
            status = "中間至折價區"
            interpretation = "價格位於常見回檔觀察區，仍具一定風險報酬空間。"
            effect = "若模型分數轉強，可列入候選。"
        else:
            status = "偏高位階"
            interpretation = "價格接近短期區間高位，追價風險增加。"
            effect = "降低追高意願，等待回檔或突破確認。"

        add_row("Fib Position", f"{fib:.2f} / {fib_zone}", status, interpretation, effect)

    # =========================
    # Resistance
    # =========================
    if (
        "nearest_resistance_level" in latest.index
        and pd.notna(latest["nearest_resistance_level"])
        and "close" in latest.index
        and pd.notna(latest["close"])
    ):
        close = float(latest["close"])
        resistance = float(latest["nearest_resistance_level"])
        distance = resistance / close - 1

        if distance < 0.03:
            status = "上方壓力近"
            interpretation = "現價距離最近壓力位很近，上檔空間有限。"
            effect = "不適合追高，需等待突破或回檔。"
        elif distance < 0.08:
            status = "上方仍有空間"
            interpretation = "距離壓力位仍有一定空間。"
            effect = "若其他條件配合，可維持觀察。"
        else:
            status = "壓力較遠"
            interpretation = "短線上檔壓力相對不近。"
            effect = "若風險可控，可提高候選排序。"

        add_row("Resistance", f"{resistance:.2f}，距離 {distance * 100:.2f}%", status, interpretation, effect)

    # =========================
    # Stoploss Risk
    # =========================
    if "stop_loss_pct" in latest.index and pd.notna(latest["stop_loss_pct"]):
        risk_pct = float(latest["stop_loss_pct"])

        if risk_pct <= 0:
            status = "停損無效"
            interpretation = "停損距離異常，需檢查 OB 或 ATR 欄位。"
            effect = "不建議使用該筆結果做決策。"
        elif risk_pct < 0.03:
            status = "風險成本低"
            interpretation = "停損距離較小，單筆風險較容易控制。"
            effect = "提高可操作性。"
        elif risk_pct < 0.06:
            status = "風險成本中等"
            interpretation = "停損距離可接受，但需控制部位。"
            effect = "適合小部位或等待更好進場價。"
        else:
            status = "停損距離偏大"
            interpretation = "若進場，需承擔較大價格波動。"
            effect = "降低進場意願，建議等待更靠近支撐的位置。"

        add_row("Stoploss Risk", f"{risk_pct * 100:.2f}%", status, interpretation, effect)

    # =========================
    # Order Block
    # =========================
    if "recent_2w_ob_type" in latest.index and pd.notna(latest["recent_2w_ob_type"]):
        ob_type = latest["recent_2w_ob_type"]
        is_fallback = latest.get("recent_2w_ob_is_fallback", np.nan)

        if ob_type == "bullish":
            status = "多方 OB"
            interpretation = "近兩週出現由上漲衝動確認的多方訂單塊。"
            effect = "提高支撐可信度，可作為進場與停損參考。"
        elif ob_type == "bearish":
            status = "空方 OB"
            interpretation = "近兩週出現由下跌衝動確認的空方訂單塊。"
            effect = "降低偏多決策信心，需留意賣壓。"
        elif ob_type == "fallback_range":
            status = "沒有明確 OB"
            interpretation = "未偵測到明確衝動 K 棒形成的 OB，目前用近 10 日區間替代。"
            effect = "結構支撐可信度較低，停損與進場需更保守。"
        else:
            status = "無 OB"
            interpretation = "目前沒有足夠資料或沒有偵測到 OB。"
            effect = "不以 OB 作為主要決策依據。"

        add_row("Order Block", f"{ob_type}, fallback={is_fallback}", status, interpretation, effect)

    return pd.DataFrame(rows)

def build_final_decision_summary(latest: pd.Series) -> str:
    """
    把最新一日的模型分數與決策指南整合成一段自然語言說明。
    """

    score = float(latest["predicted_investment_score"])
    signal = latest.get("signal", "無判斷")

    reasons = []

    # 價格位置
    if "fib_position" in latest.index and pd.notna(latest["fib_position"]):
        fib = float(latest["fib_position"])

        if fib < 0.5:
            reasons.append("價格位置偏低，具備相對便宜特徵")
        elif fib > 0.786:
            reasons.append("價格接近短期高位，追高風險較高")

    # MACD
    if "MACD" in latest.index and pd.notna(latest["MACD"]):
        if float(latest["MACD"]) < 0:
            reasons.append("MACD 仍偏弱，動能尚未完全轉強")
        else:
            reasons.append("MACD 偏強，動能有所支撐")

    # KDJ
    if all(col in latest.index and pd.notna(latest[col]) for col in ["K", "D"]):
        if float(latest["K"]) < float(latest["D"]):
            reasons.append("K 值低於 D 值，短線動能偏弱")
        else:
            reasons.append("K 值高於 D 值，短線動能偏多")

    # 壓力位距離
    if (
        "nearest_resistance_level" in latest.index
        and pd.notna(latest["nearest_resistance_level"])
        and "close" in latest.index
        and pd.notna(latest["close"])
    ):
        resistance_distance = float(latest["nearest_resistance_level"]) / float(latest["close"]) - 1

        if resistance_distance < 0.03:
            reasons.append("上方壓力位接近，短線上檔空間受限")

    # 停損風險
    if "stop_loss_pct" in latest.index and pd.notna(latest["stop_loss_pct"]):
        risk_pct = float(latest["stop_loss_pct"])

        if risk_pct >= 0.06:
            reasons.append("停損距離偏大，風險成本較高")
        elif risk_pct >= 0.03:
            reasons.append("停損距離中等，需控制部位")

    # OB 品質
    if (
        "recent_2w_ob_type" in latest.index
        and pd.notna(latest["recent_2w_ob_type"])
        and latest["recent_2w_ob_type"] == "fallback_range"
    ):
        reasons.append("目前沒有明確 OB，結構支撐可信度較低")

    if reasons:
        reason_text = "；".join(reasons)
    else:
        reason_text = "目前指標未形成明確單邊訊號"

    if score >= 0.70:
        action_text = "因此本系統傾向列為投資吸引力較高的觀察標的，但仍需設定停損。"
    elif score >= 0.55:
        action_text = "因此本系統傾向中性偏多，可等待回檔或突破確認後再評估進場。"
    elif score >= 0.45:
        action_text = "因此本系統傾向中性觀察，不建議直接追高。"
    elif score >= 0.30:
        action_text = "因此本系統傾向偏弱觀察，建議等待動能或結構轉強。"
    else:
        action_text = "因此本系統判斷風險偏高，不建議積極進場。"

    return (
        f"{latest.get('target_col', 'target_score_v3')} 預測值為 {score:.4f}，判斷為「{signal}」。"
        f"綜合解讀：{reason_text}。"
        f"{action_text}"
    )

def print_latest_report(df: pd.DataFrame, metrics: dict, config: PipelineConfig) -> None:
    latest = df.iloc[-1]
    latest_date = pd.to_datetime(latest["date"])
    future_start_date = latest_date + BDay(1)
    future_end_date = latest_date + BDay(config.horizon)

    print("\n" + "=" * 70)
    print("股票機器學習投資價值預測結果")
    print("=" * 70)
    print(f"股票代碼：{latest.get('symbol', config.stock_no)}")
    print(f"市場：{latest.get('market', config.market)}")
    print(f"資料蒐集截止日：{latest_date.strftime('%Y-%m-%d')}")
    print(f"最新收盤價：{latest['close']:.2f}")

    print("\n" + "-" * 70)
    print(f"未來 {config.horizon} 個交易日風險報酬結構預測")
    print("-" * 70)
    print(f"預測起始日：約 {future_start_date.strftime('%Y-%m-%d')}")
    print(f"預測結束日：約 {future_end_date.strftime('%Y-%m-%d')}")
    print(f"預測期間：未來 {config.horizon} 個交易日")
    print(f"模型 target 欄位：{metrics.get('target_col', config.target_col)}")
    print(f"預測 target score：{latest['predicted_target_score']:.4f}")
    print(f"模型判斷：{latest['signal']}")
    print("目前主模型 target 是 target_score_v2。")
    print("target_score 並非未來報酬率，也不是上漲機率。")
    print("本研究將 target_score 定義為未來 10 個交易日內的風險報酬結構分數。")
    print("future_upside_10d 代表從當前收盤價到未來 10 日最高價的潛在上行空間。")
    print("future_downside_10d 代表從當前收盤價到未來 10 日最低價的潛在下行風險。")
    print("target_score 越接近 1，代表上行空間相對大於下行風險；越接近 0，代表下行風險相對較高。")
    print("target_score_v2 是正式主模型 target：在 target_score 的風險報酬比例基礎上加入 opportunity_score，避免小幅上行/下行被高估。")
    print("target_score_v3 保留為舊版相對排名對照 target。")
    print(f"target_score_raw_{config.horizon}d 保留為原始風險調整後報酬對照 target。")

    print("\n" + "-" * 70)
    print("目前技術面依據")
    print("-" * 70)
    for col in ["RSI14", "MACD", "K", "D", "J", "BIAS24"]:
        if col in latest.index and pd.notna(latest[col]):
            if col == "BIAS24":
                print(f"{col}：{latest[col]:.2f}%")
            else:
                print(f"{col}：{latest[col]:.2f}")

    print("\n" + "-" * 70)
    print("價格位置與風險依據")
    print("-" * 70)
    display_cols = [
        f"future_max_high_{config.horizon}d", f"future_min_low_{config.horizon}d",
        f"future_upside_{config.horizon}d", f"future_downside_{config.horizon}d",
        "target_score", "target_score_v2", "target_score_v3", f"target_score_raw_{config.horizon}d",
        "fib_position", "fib_zone", "nearest_support_level", "nearest_resistance_level",
        "atr_14", "atr_pct", "risk_entry_price", "risk_ob_bottom", "risk_atr_stop",
        "risk_stoploss_price", "risk_per_share", "risk_stoploss_pct",
        "recent_2w_ob_type", "recent_active_gap_type",
    ]
    for col in display_cols:
        if col in latest.index and pd.notna(latest[col]):
            value = latest[col]
            if col.endswith("pct") or col in {"atr_pct", "risk_stoploss_pct"}:
                print(f"{col}：{float(value) * 100:.2f}%")
            elif col.startswith("future_upside_") or col.startswith("future_downside_"):
                print(f"{col}：{float(value) * 100:.2f}%")
            elif isinstance(value, (float, np.floating)):
                print(f"{col}：{float(value):.4f}" if col in {"target_score", "target_score_v2", "target_score_v3", f"target_score_raw_{config.horizon}d"} else f"{col}：{float(value):.2f}")
            else:
                print(f"{col}：{value}")

    print("\n" + "-" * 70)
    print("決策輔助指南")
    print("-" * 70)

    decision_guide = build_latest_decision_guide(latest)

    if not decision_guide.empty:
        for _, row in decision_guide.iterrows():
            print(f"{row['indicator']}｜{row['value']}｜{row['status']}")
            print(f"  解讀：{row['interpretation']}")
            print(f"  決策影響：{row['decision_effect']}")

    print("\n綜合決策說明：")
    print(build_final_decision_summary(latest))
    print("\n" + "-" * 70)
    print("模型評估")
    print("-" * 70)
    print(f"驗證方式：{metrics.get('validation_method', 'holdout')}")
    if "walk_forward_folds" in metrics:
        print(f"Walk-forward 折數：{metrics['walk_forward_folds']}")
        print(f"驗證總筆數：{metrics['test_rows']}")
    print(f"MAE：{metrics['mae']:.4f}")
    print(f"RMSE：{metrics['rmse']:.4f}")
    print(f"R2：{metrics['r2']:.4f}")
    print(f"Baseline MAE：{metrics['baseline_mae']:.4f}")
    print(f"Baseline RMSE：{metrics['baseline_rmse']:.4f}")
    print(f"Baseline R2：{metrics['baseline_r2']:.4f}")

    print("\n" + "-" * 70)
    print("模型實際使用特徵")
    print("-" * 70)
    print(f"feature_set：{metrics.get('feature_set', config.feature_set)}")
    print(f"model_type：{metrics.get('model_type', config.model_type)}")
    print(f"model_profile：{metrics.get('model_profile', config.model_profile)}")
    print(f"模型實際使用特徵數量：{metrics.get('feature_count', len(metrics['feature_cols']))}")
    print("數值特徵：")
    print(metrics["numeric_features"])
    print("類別特徵：")
    print(metrics["categorical_features"])
    if metrics.get("feature_set") == "structural":
        print("missing structural features：")
        print(metrics.get("missing_structural_features", []))
        print("被保留但未進模型的原始指標欄位：")
        print(metrics.get("retained_raw_excluded_features", []))
    print("被移除的特徵：")
    print(metrics["removed_features"])
    print(f"Importance-based feature removal enabled：{metrics.get('importance_feature_selection_enabled', False)}")
    if metrics.get("removed_by_importance"):
        print("因模型重要度較低而移除的特徵：")
        print(metrics["removed_by_importance"])
    elif not metrics.get("run_feature_importance", True):
        print("本次已略過 feature importance report，以提升訓練效率。")
    else:
        print("Feature importance 目前只輸出報告，不自動刪除特徵。")

    print("\n" + "=" * 70)
    print("說明：target_score 是未來 10 個交易日內的風險報酬結構分數。")
    print("target_score_v2 是目前主模型 target，進一步加入 opportunity_score，避免只看比例而忽略上行機會大小。")
    print("target_score_v3 是舊版相對排名對照 target。")
    print(f"target_score_raw_{config.horizon}d 是原始風險調整後報酬對照 target。")
    print("它不是未來報酬率，也不是上漲機率，更不是直接買賣建議。")
    print("=" * 70)


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    print("\nStep 1/6：抓取與清理股價資料")
    price_df = get_last_n_years_trading_data(config)
    market_index_df = get_market_index_data(config)

    print("\nStep 2/6：計算技術指標")
    feature_df = add_technical_indicators(price_df, config)
    feature_df = add_stock_level_context_features(feature_df, config, market_df=market_index_df)

    print("\nStep 3/6：建立市場結構特徵")
    feature_df = add_market_structure_features(feature_df, config)

    print("\nStep 4/6：建立風險控管特徵")
    feature_df = add_stoploss_features(feature_df, atr_multiplier=config.atr_multiplier)
    feature_df = add_structural_features(feature_df)
    feature_df = add_cross_sectional_pool_features(feature_df)
    feature_df = clean_duplicate_columns(feature_df)
    print_new_feature_availability(feature_df, context="single-stock pipeline")

    print("\nStep 5/6：建立 target_score、訓練模型、預測風險報酬結構")
    result_df, metrics = train_and_predict(feature_df, config)
    result_df = clean_duplicate_columns(result_df)

    print("\nStep 6/6：輸出結果")

    save_outputs(result_df, config, metrics)
    print_latest_report(result_df, metrics, config)

    return result_df


def build_stock_config(base_config: PipelineConfig, stock_no: str, market: str = "TWSE") -> PipelineConfig:
    return PipelineConfig(
        stock_no=stock_no,
        market=market,
        years=base_config.years,
        horizon=base_config.horizon,
        output_dir=base_config.output_dir,
        sleep_sec=base_config.sleep_sec,
        atr_period=base_config.atr_period,
        atr_multiplier=base_config.atr_multiplier,
        ob_lookback_days=base_config.ob_lookback_days,
        fib_lookback_days=base_config.fib_lookback_days,
        sr_lookback_days=base_config.sr_lookback_days,
        sr_tolerance_pct=base_config.sr_tolerance_pct,
        fvg_min_gap_pct=base_config.fvg_min_gap_pct,
        risk_penalty=base_config.risk_penalty,
        opportunity_threshold=base_config.opportunity_threshold,
        target_col=base_config.target_col,
        feature_set=base_config.feature_set,
        model_type=base_config.model_type,
        model_profile=base_config.model_profile,
        random_state=base_config.random_state,
        run_feature_importance=base_config.run_feature_importance,
        target_compare_cols=base_config.target_compare_cols,
        compare_feature_sets=base_config.compare_feature_sets,
        compare_model_types=base_config.compare_model_types,
        diagnostic_scope=base_config.diagnostic_scope,
        display_mode=base_config.display_mode,
        skip_classification=base_config.skip_classification,
    )


def add_cross_sectional_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or "symbol" not in df.columns:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    rank_cols = [
        "return_5d",
        "return_10d",
        "ma20_distance",
        "ma60_distance",
        "volume_ratio_20",
        "VR_14",
        "VR_26",
        "VR_14_change",
        "RSI14",
        "MACD",
        "K",
        "BIAS24",
        "atr_pct",
        "volatility_10d",
        "fib_position",
        "support_distance",
        "resistance_distance",
    ]

    for col in rank_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_cs_rank"] = out.groupby("date")[col].rank(pct=True)

    return out


def add_cross_sectional_amount_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    out = combined_df.copy()
    if "date" not in out.columns or "trading_amount" not in out.columns:
        for col in CROSS_SECTIONAL_AMOUNT_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
        return out

    sort_cols = ["symbol", "date"] if "symbol" in out.columns else ["date"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out["trading_amount"] = pd.to_numeric(out["trading_amount"], errors="coerce")

    out["amount_rank"] = out.groupby("date")["trading_amount"].rank(
        ascending=False,
        method="min",
    )
    out["amount_rank_pct"] = out.groupby("date")["trading_amount"].rank(
        pct=True,
        ascending=True,
    )
    pool_amount = out.groupby("date")["trading_amount"].transform("sum").replace(0, np.nan)
    out["amount_share_in_pool"] = out["trading_amount"] / pool_amount

    if "symbol" in out.columns:
        out["amount_rank_change_5d"] = (
            out.groupby("symbol", sort=False)["amount_rank"].shift(5) - out["amount_rank"]
        )
        out["amount_rank_change_10d"] = (
            out.groupby("symbol", sort=False)["amount_rank"].shift(10) - out["amount_rank"]
        )
        out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    else:
        out["amount_rank_change_5d"] = out["amount_rank"].shift(5) - out["amount_rank"]
        out["amount_rank_change_10d"] = out["amount_rank"].shift(10) - out["amount_rank"]
        out = out.sort_values("date").reset_index(drop=True)

    return out.replace([np.inf, -np.inf], np.nan)


def add_cross_sectional_relative_strength_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    out = combined_df.copy()
    if "date" not in out.columns or "relative_return_10d" not in out.columns:
        for col in CROSS_SECTIONAL_RELATIVE_STRENGTH_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["relative_return_10d"] = pd.to_numeric(out["relative_return_10d"], errors="coerce")
    out["relative_strength_rank_10d"] = out.groupby("date")["relative_return_10d"].rank(
        pct=True,
        ascending=True,
    )
    sort_cols = ["date", "symbol"] if "symbol" in out.columns else ["date"]
    return out.sort_values(sort_cols).reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def add_cross_sectional_pool_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    out = add_cross_sectional_amount_features(combined_df)
    out = add_cross_sectional_relative_strength_features(out)
    out = add_cross_sectional_rank_features(out)
    return clean_duplicate_columns(out)


def print_new_feature_availability(df: pd.DataFrame, context: str = "") -> None:
    title = f"New feature availability check{f' - {context}' if context else ''}"
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    def report(label: str, expected_cols: list[str]) -> None:
        present = [col for col in expected_cols if col in df.columns]
        missing = [col for col in expected_cols if col not in df.columns]
        print(f"{label} feature count: {len(present)} / {len(expected_cols)}")
        print(f"missing {label} feature list: {missing}")
        high_missing = {}
        for col in present:
            missing_rate = df[col].isna().mean()
            if missing_rate >= 0.80:
                high_missing[col] = round(float(missing_rate), 4)
        if high_missing:
            print(f"WARNING: high missing ratio in {label} features: {high_missing}")

    report("market", MARKET_RAW_FEATURES + MARKET_NUMERIC_FEATURES + MARKET_CATEGORICAL_FEATURES)
    report("amount", AMOUNT_FEATURES + CROSS_SECTIONAL_AMOUNT_FEATURES)
    report("relative strength", RELATIVE_STRENGTH_FEATURES + CROSS_SECTIONAL_RELATIVE_STRENGTH_FEATURES)
    print("Note: amount_rank/amount_rank_pct/relative_strength_rank_10d are ranking features inside the current analysis pool, not full-market ranks.")


def get_local_range_sensitivity_profiles() -> list[dict]:
    return [
        {
            "profile_name": "short_term",
            "purpose": "偏短線交易",
            "fib_lookback_days": 10,
            "ob_lookback_days": 10,
            "sr_lookback_days": 60,
            "atr_period": 14,
        },
        {
            "profile_name": "mid_term",
            "purpose": "平衡版",
            "fib_lookback_days": 20,
            "ob_lookback_days": 20,
            "sr_lookback_days": 120,
            "atr_period": 14,
        },
        {
            "profile_name": "swing",
            "purpose": "偏波段",
            "fib_lookback_days": 30,
            "ob_lookback_days": 20,
            "sr_lookback_days": 150,
            "atr_period": 20,
        },
        {
            "profile_name": "long_term",
            "purpose": "偏中長線",
            "fib_lookback_days": 60,
            "ob_lookback_days": 30,
            "sr_lookback_days": 250,
            "atr_period": 20,
        },
    ]


def fetch_and_prepare_multi_stock_data_with_config(
    symbols: list[str],
    base_config: PipelineConfig,
) -> pd.DataFrame:
    """
    使用傳入的 base_config 逐支股票重新抓資料、計算技術指標、市場結構特徵、風險特徵與 target。
    這個函式會完整使用 base_config 裡面的：
    - fib_lookback_days
    - ob_lookback_days
    - sr_lookback_days
    - atr_period
    - atr_multiplier
    - sr_tolerance_pct
    - fvg_min_gap_pct
    - horizon
    - risk_penalty
    - opportunity_threshold
    - feature_set
    - model_profile

    sensitivity 模式必須走這條路徑，避免讀舊 combined_training_data 或忽略 profile 參數。
    """
    all_frames = []
    market_index_df = get_market_index_data(base_config)
    is_sensitivity_mode = (
        getattr(base_config, "analysis_mode", None) == "sensitivity"
        or getattr(base_config, "mode", None) == "sensitivity"
        or getattr(base_config, "use_sensitivity", False) is True
        or getattr(base_config, "require_sensitivity_data", False) is True
    )
    process_label = "sensitivity" if is_sensitivity_mode else "multi-stock"

    for symbol in symbols:
        try:
            print("\n" + "-" * 70)
            print(f"開始處理 {process_label} 股票：{base_config.market} {symbol}")
            print(
                "profile config："
                f"fib={base_config.fib_lookback_days}, "
                f"ob={base_config.ob_lookback_days}, "
                f"sr={base_config.sr_lookback_days}, "
                f"atr_period={base_config.atr_period}, "
                f"atr_multiplier={base_config.atr_multiplier}"
            )
            print("-" * 70)

            config = replace(base_config, stock_no=symbol)
            df = get_last_n_years_trading_data(config)

            if df is None or df.empty:
                print(f"股票 {symbol} 無資料，略過。")
                continue

            df["symbol"] = symbol
            df["market"] = config.market

            _t0 = time.time()
            df = add_technical_indicators(df, config)
            print(f'[TIMER] add_technical_indicators: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_stock_level_context_features(df, config, market_df=market_index_df)
            print(f'[TIMER] add_stock_level_context_features: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_market_structure_features(df, config)
            print(f'[TIMER] add_market_structure_features: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_stoploss_features(df, atr_multiplier=config.atr_multiplier)
            print(f'[TIMER] add_stoploss_features: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_structural_features(df)
            print(f'[TIMER] add_structural_features: {time.time() - _t0:.1f}s')

            df = clean_duplicate_columns(df)
            df = add_risk_reward_target_score(
                df,
                horizon=config.horizon,
                risk_penalty=config.risk_penalty,
                opportunity_threshold=config.opportunity_threshold,
            )
            df = add_realized_history_features(df, horizon=config.horizon, report=False)
            df["symbol"] = symbol
            df["market"] = config.market

            all_frames.append(df)
            print(f"完成 {process_label} 股票 {symbol}，資料筆數：{len(df)}")

        except Exception as e:
            print(f"{process_label} 股票 {symbol} 處理失敗：{e}")
            continue

    if not all_frames:
        if is_sensitivity_mode:
            raise ValueError("所有股票皆無法取得有效 sensitivity 資料，無法訓練模型。")
        raise ValueError("所有股票皆無法取得有效歷史資料或特徵資料，無法訓練模型。")

    combined_data = pd.concat(all_frames, ignore_index=True)

    if "date" in combined_data.columns:
        combined_data["date"] = pd.to_datetime(combined_data["date"])
        combined_data = combined_data.sort_values(["date", "symbol"]).reset_index(drop=True)

    combined_data = add_low_subjective_targets(combined_data, horizon=base_config.horizon)
    combined_data = add_cross_sectional_pool_features(combined_data)
    combined_data = add_cross_sectional_rank_features(combined_data)
    print_new_feature_availability(combined_data, context="multi-stock prepared data")

    print("\n" + "=" * 70)
    print(("Sensitivity" if is_sensitivity_mode else "Multi-stock") + " 多股票資料合併完成")
    print("=" * 70)
    print(f"股票數量：{combined_data['symbol'].nunique()}")
    print(f"總資料筆數：{len(combined_data)}")
    print(f"日期範圍：{combined_data['date'].min()} ~ {combined_data['date'].max()}")

    return combined_data


def fetch_and_prepare_multi_stock_data_historical(
    symbols: list[str],
    base_config: PipelineConfig,
    start_date: str,
    end_date: str,
    burn_in_months: int = 12,
) -> pd.DataFrame:
    """
    歷史區間版本的多股票資料準備函式。
    與 fetch_and_prepare_multi_stock_data_with_config() 的差別：
    改呼叫 get_historical_trading_data() / get_historical_market_index_data()，
    抓取 [start_date - burn_in_months, end_date] 區間資料（不依賴 datetime.today()）。
    特徵計算、target 建構邏輯與原函式完全相同。
    """
    all_frames = []
    market_index_df = get_historical_market_index_data(
        config=base_config,
        start_date=start_date,
        end_date=end_date,
        burn_in_months=burn_in_months,
    )

    for symbol in symbols:
        try:
            print("\n" + "-" * 70)
            print(f"[HIST] 開始處理歷史回測股票：{base_config.market} {symbol}")
            print(
                "profile config："
                f"fib={base_config.fib_lookback_days}, "
                f"ob={base_config.ob_lookback_days}, "
                f"sr={base_config.sr_lookback_days}, "
                f"atr_period={base_config.atr_period}, "
                f"atr_multiplier={base_config.atr_multiplier}"
            )
            print("-" * 70)

            config = replace(base_config, stock_no=symbol)
            df = get_historical_trading_data(
                config=config,
                start_date=start_date,
                end_date=end_date,
                burn_in_months=burn_in_months,
            )

            if df is None or df.empty:
                print(f"股票 {symbol} 無資料，略過。")
                continue

            df["symbol"] = symbol
            df["market"] = config.market

            _t0 = time.time()
            df = add_technical_indicators(df, config)
            print(f'[TIMER] add_technical_indicators: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_stock_level_context_features(df, config, market_df=market_index_df)
            print(f'[TIMER] add_stock_level_context_features: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_market_structure_features(df, config)
            print(f'[TIMER] add_market_structure_features: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_stoploss_features(df, atr_multiplier=config.atr_multiplier)
            print(f'[TIMER] add_stoploss_features: {time.time() - _t0:.1f}s')

            _t0 = time.time()
            df = add_structural_features(df)
            print(f'[TIMER] add_structural_features: {time.time() - _t0:.1f}s')

            df = clean_duplicate_columns(df)
            df = add_risk_reward_target_score(
                df,
                horizon=config.horizon,
                risk_penalty=config.risk_penalty,
                opportunity_threshold=config.opportunity_threshold,
            )
            df = add_realized_history_features(df, horizon=config.horizon, report=False)
            df["symbol"] = symbol
            df["market"] = config.market

            all_frames.append(df)
            print(f"[HIST] 完成歷史回測股票 {symbol}，資料筆數：{len(df)}")

        except Exception as e:
            print(f"[HIST] 歷史回測股票 {symbol} 處理失敗：{e}")
            continue

    if not all_frames:
        raise ValueError("所有股票皆無法取得有效歷史資料或特徵資料，無法訓練模型。")

    combined_data = pd.concat(all_frames, ignore_index=True)

    if "date" in combined_data.columns:
        combined_data["date"] = pd.to_datetime(combined_data["date"])
        combined_data = combined_data.sort_values(["date", "symbol"]).reset_index(drop=True)

    combined_data = add_low_subjective_targets(combined_data, horizon=base_config.horizon)
    combined_data = add_cross_sectional_pool_features(combined_data)
    combined_data = add_cross_sectional_rank_features(combined_data)
    print_new_feature_availability(combined_data, context="historical multi-stock prepared data")

    print("\n" + "=" * 70)
    print("歷史回測 多股票資料合併完成")
    print("=" * 70)
    print(f"股票數量：{combined_data['symbol'].nunique()}")
    print(f"總資料筆數：{len(combined_data)}")
    print(f"日期範圍：{combined_data['date'].min()} ~ {combined_data['date'].max()}")

    return combined_data


def fetch_and_prepare_multi_stock_data(
    symbols: list[str],
    market: str,
    years: int = 3,
    horizon: int = 10,
    risk_penalty: float = 1.0,
    opportunity_threshold: float = 0.05,
    feature_set: str = "full",
) -> pd.DataFrame:
    """
    逐支股票抓取資料、清理、計算特徵與 target，
    最後合併成一張多股票訓練資料表。
    """
    all_frames = []
    market_index_config = replace(CONFIG, market=market, years=years, horizon=horizon)
    market_index_df = get_market_index_data(market_index_config)

    for symbol in symbols:
        try:
            print("\n" + "-" * 70)
            print(f"開始處理股票：{market} {symbol}")
            print("-" * 70)

            config = PipelineConfig(
                stock_no=symbol,
                market=market,
                years=years,
                horizon=horizon,
                output_dir=CONFIG.output_dir,
                sleep_sec=CONFIG.sleep_sec,
                atr_period=CONFIG.atr_period,
                atr_multiplier=CONFIG.atr_multiplier,
                ob_lookback_days=CONFIG.ob_lookback_days,
                fib_lookback_days=CONFIG.fib_lookback_days,
                sr_lookback_days=CONFIG.sr_lookback_days,
                sr_tolerance_pct=CONFIG.sr_tolerance_pct,
                fvg_min_gap_pct=CONFIG.fvg_min_gap_pct,
                risk_penalty=risk_penalty,
                opportunity_threshold=opportunity_threshold,
                target_col=CONFIG.target_col,
                feature_set=feature_set,
                model_type=CONFIG.model_type,
                model_profile=CONFIG.model_profile,
                random_state=CONFIG.random_state,
                run_feature_importance=CONFIG.run_feature_importance,
                target_compare_cols=CONFIG.target_compare_cols,
                compare_feature_sets=CONFIG.compare_feature_sets,
                compare_model_types=CONFIG.compare_model_types,
                skip_classification=CONFIG.skip_classification,
            )

            df = get_last_n_years_trading_data(config)

            if df is None or df.empty:
                print(f"股票 {symbol} 無資料，略過。")
                continue

            df["symbol"] = symbol
            df["market"] = market

            df = add_technical_indicators(df, config)
            df = add_stock_level_context_features(df, config, market_df=market_index_df)
            df = add_market_structure_features(df, config)
            df = add_stoploss_features(df, atr_multiplier=config.atr_multiplier)
            df = add_structural_features(df)
            df = clean_duplicate_columns(df)
            df = add_risk_reward_target_score(
                df,
                horizon=horizon,
                risk_penalty=risk_penalty,
                opportunity_threshold=opportunity_threshold,
            )
            df = add_realized_history_features(df, horizon=horizon, report=False)
            df["symbol"] = symbol
            df["market"] = market

            all_frames.append(df)

            print(f"完成股票 {symbol}，資料筆數：{len(df)}")

        except Exception as e:
            print(f"股票 {symbol} 處理失敗：{e}")
            continue

    if not all_frames:
        raise ValueError("所有股票皆無法取得有效資料，無法訓練模型。")

    combined_data = pd.concat(all_frames, ignore_index=True)

    if "date" in combined_data.columns:
        combined_data["date"] = pd.to_datetime(combined_data["date"])
        combined_data = combined_data.sort_values(["date", "symbol"]).reset_index(drop=True)

    combined_data = add_low_subjective_targets(combined_data, horizon=horizon)
    combined_data = add_cross_sectional_pool_features(combined_data)
    combined_data = add_cross_sectional_rank_features(combined_data)
    print_new_feature_availability(combined_data, context="multi-stock prepared data")

    print("\n" + "=" * 70)
    print("多股票資料合併完成")
    print("=" * 70)
    print(f"股票數量：{combined_data['symbol'].nunique()}")
    print(f"總資料筆數：{len(combined_data)}")
    print(f"日期範圍：{combined_data['date'].min()} ~ {combined_data['date'].max()}")

    return combined_data


def prepare_latest_features_for_model(model, latest_row: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    latest_features = latest_row[feature_cols].replace({pd.NA: np.nan})

    try:
        transformers = model.named_steps["preprocessor"].transformers
        numeric_cols = list(transformers[0][2])
        categorical_cols = list(transformers[1][2])
    except Exception:
        numeric_cols = []
        categorical_cols = []

    for col in numeric_cols:
        if col in latest_features.columns:
            latest_features[col] = pd.to_numeric(latest_features[col], errors="coerce")
    for col in categorical_cols:
        if col in latest_features.columns:
            latest_features[col] = (
                latest_features[col]
                .astype("object")
                .where(latest_features[col].notna(), "none")
                .astype(str)
            )
    return latest_features


def predict_latest_score_for_model(model, latest_row: pd.DataFrame, feature_cols: list[str]) -> float:
    latest_features = prepare_latest_features_for_model(model, latest_row, feature_cols)
    return float(np.clip(model.predict(latest_features)[0], 0, 1))


def sort_and_rank_multi_stock_results(prediction_results: pd.DataFrame) -> pd.DataFrame:
    if prediction_results.empty:
        return prediction_results
    out = prediction_results.copy()

    if (
        "final_decision_score" in out.columns
        and pd.to_numeric(out["final_decision_score"], errors="coerce").notna().any()
    ):
        if "trade_veto_flag" in out.columns:
            trade_veto_sort = out["trade_veto_flag"].apply(
                lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
            )
        else:
            trade_veto_sort = pd.Series(False, index=out.index)
        out["_trade_veto_sort"] = trade_veto_sort
        out["_final_decision_score_sort"] = pd.to_numeric(out["final_decision_score"], errors="coerce")
        out = (
            out.sort_values(
                ["_trade_veto_sort", "_final_decision_score_sort"],
                ascending=[True, False],
                na_position="last",
            )
            .drop(columns=["_trade_veto_sort", "_final_decision_score_sort"])
            .reset_index(drop=True)
        )
    else:
        if "predicted_score" in out.columns:
            sort_col = "predicted_score"
        else:
            sort_col = "predicted_target_score"
        out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def enrich_multi_stock_decision_fields(prediction_results: pd.DataFrame) -> pd.DataFrame:
    if prediction_results.empty:
        return prediction_results

    out = prediction_results.copy()
    for col in ["predicted_score", "predicted_target_score", "safety_score", "return_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "safety_score" not in out.columns:
        out["safety_score"] = out.get("predicted_score", out.get("predicted_target_score", np.nan))
    if "return_score" not in out.columns:
        out["return_score"] = out.get("predicted_score", out.get("predicted_target_score", np.nan))

    out["combined_decision_score"] = (
        0.65 * pd.to_numeric(out["safety_score"], errors="coerce")
        + 0.35 * pd.to_numeric(out["return_score"], errors="coerce")
    )

    risk_thresholds = {
        "volatility_10d_q80": pd.to_numeric(out.get("volatility_10d", pd.Series(dtype=float)), errors="coerce").quantile(0.80),
        "support_distance_q80": pd.to_numeric(out.get("support_distance", pd.Series(dtype=float)), errors="coerce").quantile(0.80),
        "resistance_distance_q20": pd.to_numeric(out.get("resistance_distance", pd.Series(dtype=float)), errors="coerce").quantile(0.20),
    }
    out["final_decision_label"] = out.apply(get_single_diagnostic_final_decision_label, axis=1)
    out["model_confidence_note"] = out.apply(get_single_diagnostic_model_confidence_note, axis=1)
    out["signal"] = out["combined_decision_score"].apply(get_signal)

    decision_text_df = out.apply(build_decision_text, axis=1, result_type="expand")
    out = _concat_overwrite(out, decision_text_df)

    trade_plan_df = out.apply(build_trade_plan, axis=1, result_type="expand")
    out = _concat_overwrite(out, trade_plan_df)

    rule_filter_df = out.apply(apply_second_layer_rule_filter, axis=1, result_type="expand")
    out = _concat_overwrite(out, rule_filter_df)

    out["risk_warning"] = out.apply(lambda row: get_single_diagnostic_risk_warning(row, risk_thresholds), axis=1)

    final_decision_df = out.apply(build_final_decision, axis=1, result_type="expand")
    out = _concat_overwrite(out, final_decision_df)
    if "technical_filter_result" in out.columns:
        out["technical_filter_result"] = out["technical_filter_result"].apply(translate_technical_filter_result)

    position_df = out.apply(lambda row: build_position_sizing(row), axis=1, result_type="expand")
    out = _concat_overwrite(out, position_df)

    enhanced_text_df = out.apply(build_enhanced_decision_text, axis=1, result_type="expand")
    out = _concat_overwrite(out, enhanced_text_df)

    text_cols = [
        "risk_warning",
        "model_confidence_note",
        "final_decision_label",
        "signal",
        "decision_summary_text",
        "decision_reason_text",
        "suggested_action_text",
        "technical_filter_result",
        "rule_filter_notes",
        "trade_veto_reason",
        "trade_plan_notes",
        "score_band",
        "decision_action",
        "position_sizing_label",
        "position_sizing_reason",
        *[col for col in ENHANCED_DECISION_TEXT_COLUMNS if col != "enhanced_decision_report_text"],
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(normalize_text_for_report)

    return sort_and_rank_multi_stock_results(out)


def predict_latest_for_each_stock(
    model,
    data: pd.DataFrame,
    symbols: list[str],
    feature_cols: list[str],
    extra_models: Optional[dict[str, tuple[object, list[str]]]] = None,
) -> pd.DataFrame:
    """
    使用共同訓練好的模型，對每一支股票最新一天資料進行預測。
    """
    results = []
    extra_models = extra_models or {}

    for symbol in symbols:
        stock_data = data[data["symbol"].astype(str) == str(symbol)].copy()

        if stock_data.empty:
            continue

        stock_data = stock_data.sort_values("date")
        latest_row = stock_data.tail(1).copy()
        pred_score = predict_latest_score_for_model(model, latest_row, feature_cols)

        result_row = latest_row.iloc[0].to_dict()
        result_row.update({
            "symbol": symbol,
            "date": latest_row["date"].iloc[0],
            "close": latest_row["close"].iloc[0] if "close" in latest_row.columns else None,
            "predicted_target_score": pred_score,
            "predicted_score": pred_score,
            "model_score": pred_score,
        })
        for score_col, (extra_model, extra_feature_cols) in extra_models.items():
            result_row[score_col] = predict_latest_score_for_model(extra_model, latest_row, extra_feature_cols)
        result_row["signal"] = score_to_signal(pred_score)
        result_row.update(generate_decision_explanation(result_row))
        results.append(result_row)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    return sort_and_rank_multi_stock_results(result_df)


def build_shared_common_stock_pool(
    base_config: PipelineConfig,
    top_n: int = 30,
    symbols_override: Optional[list[str]] = None,
    target_symbol: Optional[str] = None,
    quiet: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    if symbols_override is not None:
        symbols = [str(symbol).strip() for symbol in symbols_override if str(symbol).strip()]
        stock_list = pd.DataFrame({
            "symbol": symbols,
            "stock_name": [""] * len(symbols),
            "today_turnover": [np.nan] * len(symbols),
        })
    else:
        if base_config.market.upper() != "TWSE":
            raise ValueError("shared common model stock pool currently supports TWSE top turnover ranking.")
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                stock_list = fetch_twse_top_turnover_stocks(limit=top_n)
        else:
            stock_list = fetch_twse_top_turnover_stocks(limit=top_n)
        stock_list = stock_list.copy()

    required_cols = ["symbol", "stock_name", "today_turnover"]
    for col in required_cols:
        if col not in stock_list.columns:
            stock_list[col] = "" if col == "stock_name" else np.nan
    stock_list["symbol"] = stock_list["symbol"].astype(str).str.strip()
    stock_list = stock_list[stock_list["symbol"] != ""].copy()

    if target_symbol:
        target_symbol = str(target_symbol).strip()
        if target_symbol and target_symbol not in set(stock_list["symbol"].astype(str)):
            stock_list = pd.concat(
                [
                    stock_list,
                    pd.DataFrame([{
                        "symbol": target_symbol,
                        "stock_name": "",
                        "today_turnover": np.nan,
                    }]),
                ],
                ignore_index=True,
            )

    stock_list = stock_list.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    symbols = stock_list["symbol"].astype(str).tolist()
    return stock_list, symbols


def build_shared_common_model_config(base_config: PipelineConfig) -> PipelineConfig:
    safety_target_col = f"target_safety_{base_config.horizon}d_cs_rank"
    return replace(
        base_config,
        stock_no="ALL",
        feature_set="selected_v2",
        model_type="extra_trees",
        model_profile="medium",
        target_col=safety_target_col,
    )


def metric_value(metrics: Optional[dict], *keys, default=np.nan):
    metrics = metrics or {}
    for key in keys:
        if key in metrics:
            return metrics.get(key)
    return default


def build_shared_common_data_fingerprint(
    combined_data: pd.DataFrame,
    symbols: list[str],
    safety_target_col: str,
    return_target_col: str,
) -> str:
    date_min, date_max = audit_date_range(combined_data)
    payload = {
        "rows": int(len(combined_data)),
        "unique_symbols": int(combined_data["symbol"].nunique()) if "symbol" in combined_data.columns else 0,
        "symbols_fingerprint": audit_md5_items(sorted(str(symbol) for symbol in symbols)),
        "row_count_by_symbol_fingerprint": audit_row_count_by_symbol_hash(combined_data),
        "date_min": str(date_min),
        "date_max": str(date_max),
        "safety_target_non_null": (
            int(pd.to_numeric(combined_data[safety_target_col], errors="coerce").notna().sum())
            if safety_target_col in combined_data.columns
            else 0
        ),
        "return_target_non_null": (
            int(pd.to_numeric(combined_data[return_target_col], errors="coerce").notna().sum())
            if return_target_col in combined_data.columns
            else 0
        ),
    }
    return audit_md5_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def build_shared_common_feature_fingerprint(
    combined_data: pd.DataFrame,
    safety_feature_cols: list[str],
    return_feature_cols: list[str],
) -> str:
    payload = {
        "safety_feature_cols_hash": audit_md5_items(safety_feature_cols),
        "return_feature_cols_hash": audit_md5_items(return_feature_cols),
        "safety_feature_non_null_hash": audit_feature_non_null_summary_hash(combined_data, safety_feature_cols),
        "return_feature_non_null_hash": audit_feature_non_null_summary_hash(combined_data, return_feature_cols),
    }
    return audit_md5_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def print_shared_common_model_audit(shared_result: dict) -> None:
    combined_data = shared_result["combined_data"]
    shared_config = shared_result["shared_config"]
    safety_metrics = shared_result["safety_metrics"]
    return_metrics = shared_result["return_metrics"]
    date_min, date_max = audit_date_range(combined_data)

    print("\n" + "=" * 70)
    print("SHARED COMMON MODEL AUDIT")
    print("=" * 70)
    audit_lines = {
        "context_name": shared_result.get("context_name"),
        "target_col": f"{shared_result.get('safety_target_col')} / {shared_result.get('return_target_col')}",
        "safety_target_col": shared_result.get("safety_target_col"),
        "return_target_col": shared_result.get("return_target_col"),
        "feature_set": shared_config.feature_set,
        "model_type": shared_config.model_type,
        "model_profile": shared_config.model_profile,
        "walk_forward_folds": metric_value(safety_metrics, "walk_forward_folds", default="-"),
        "combined_data rows": int(len(combined_data)),
        "unique symbols": int(combined_data["symbol"].nunique()) if "symbol" in combined_data.columns else 0,
        "date range": f"{date_min} ~ {date_max}",
        "feature_count": len(shared_result.get("safety_feature_cols", [])),
        "safety_feature_count": len(shared_result.get("safety_feature_cols", [])),
        "return_feature_count": len(shared_result.get("return_feature_cols", [])),
        "safety_feature_cols hash": shared_result.get("safety_feature_cols_hash"),
        "return_feature_cols hash": shared_result.get("return_feature_cols_hash"),
        "data_fingerprint": shared_result.get("data_fingerprint"),
        "feature_fingerprint": shared_result.get("feature_fingerprint"),
        "symbols_fingerprint": shared_result.get("symbols_fingerprint"),
        "row_count_by_symbol_fingerprint": shared_result.get("row_count_by_symbol_fingerprint"),
        "safety_mae": metric_value(safety_metrics, "mae"),
        "safety_rmse": metric_value(safety_metrics, "rmse"),
        "safety_r2": metric_value(safety_metrics, "r2"),
        "safety_baseline_mae": metric_value(safety_metrics, "baseline_mae"),
        "safety_spearman": metric_value(safety_metrics, "spearman_corr"),
        "return_mae": metric_value(return_metrics, "mae"),
        "return_rmse": metric_value(return_metrics, "rmse"),
        "return_r2": metric_value(return_metrics, "r2"),
        "return_baseline_mae": metric_value(return_metrics, "baseline_mae"),
        "return_spearman": metric_value(return_metrics, "spearman_corr"),
    }
    for key, value in audit_lines.items():
        print(f"{key}={value}")


def run_shared_common_model_core(
    base_config: PipelineConfig,
    stock_list: pd.DataFrame,
    symbols: list[str],
    context_name: str = "shared_common_model",
) -> dict:
    symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not symbols:
        raise ValueError("shared common model requires at least one symbol.")

    shared_config = build_shared_common_model_config(base_config)
    safety_target_col = f"target_safety_{shared_config.horizon}d_cs_rank"
    return_target_col = f"target_return_{shared_config.horizon}d_cs_rank"
    presentation = shared_config.display_mode == "presentation"

    if presentation:
        with contextlib.redirect_stdout(io.StringIO()):
            combined_data = fetch_and_prepare_multi_stock_data_with_config(
                symbols=symbols,
                base_config=shared_config,
            )
        print_presentation_data_summary(
            shared_config,
            combined_data,
            top_n=len(symbols),
            target_symbol=None if shared_config.stock_no == "ALL" else shared_config.stock_no,
        )
    else:
        combined_data = fetch_and_prepare_multi_stock_data_with_config(
            symbols=symbols,
            base_config=shared_config,
        )

    def train_shared_target(target_col: str):
        target_config = replace(shared_config, target_col=target_col)
        if presentation:
            with contextlib.redirect_stdout(io.StringIO()):
                return train_model(
                    data=combined_data,
                    target_col=target_col,
                    config=target_config,
                )
        return train_model(
            data=combined_data,
            target_col=target_col,
            config=target_config,
        )

    if presentation:
        print("Step 5/6: training shared safety and return models")
    safety_model, safety_feature_cols, safety_metrics = train_shared_target(safety_target_col)
    return_model, return_feature_cols, return_metrics = train_shared_target(return_target_col)

    if presentation:
        print("Step 6/6: generating latest shared common predictions")
    prediction_results = predict_latest_for_each_stock(
        model=safety_model,
        data=combined_data,
        symbols=symbols,
        feature_cols=safety_feature_cols,
        extra_models={
            "safety_score": (safety_model, safety_feature_cols),
            "return_score": (return_model, return_feature_cols),
        },
    )
    prediction_results = enrich_multi_stock_decision_fields(prediction_results)

    symbols_fingerprint = audit_md5_items(sorted(str(symbol) for symbol in symbols))
    row_count_by_symbol_fingerprint = audit_row_count_by_symbol_hash(combined_data)
    data_fingerprint = build_shared_common_data_fingerprint(
        combined_data=combined_data,
        symbols=symbols,
        safety_target_col=safety_target_col,
        return_target_col=return_target_col,
    )
    feature_fingerprint = build_shared_common_feature_fingerprint(
        combined_data=combined_data,
        safety_feature_cols=safety_feature_cols,
        return_feature_cols=return_feature_cols,
    )

    shared_result = {
        "context_name": context_name,
        "shared_config": shared_config,
        "combined_data": combined_data,
        "prediction_results": prediction_results,
        "safety_model": safety_model,
        "return_model": return_model,
        "safety_metrics": safety_metrics,
        "return_metrics": return_metrics,
        "safety_feature_cols": safety_feature_cols,
        "return_feature_cols": return_feature_cols,
        "stock_list": stock_list,
        "symbols": symbols,
        "safety_target_col": safety_target_col,
        "return_target_col": return_target_col,
        "safety_feature_cols_hash": audit_md5_items(safety_feature_cols),
        "return_feature_cols_hash": audit_md5_items(return_feature_cols),
        "data_fingerprint": data_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "symbols_fingerprint": symbols_fingerprint,
        "row_count_by_symbol_fingerprint": row_count_by_symbol_fingerprint,
    }
    print_shared_common_model_audit(shared_result)
    return shared_result


def build_shared_common_metrics_summary(
    shared_result: dict,
    config: PipelineConfig,
    output_df: Optional[pd.DataFrame] = None,
) -> dict:
    shared_config = shared_result["shared_config"]
    safety_metrics = shared_result["safety_metrics"]
    return_metrics = shared_result["return_metrics"]
    output_df = output_df if output_df is not None else pd.DataFrame()

    summary = {
        "market": config.market.upper(),
        "symbol": config.stock_no,
        "target_col": shared_result["safety_target_col"],
        "actual_target_col": shared_result["safety_target_col"],
        "return_target_col": shared_result["return_target_col"],
        "diagnostic_scope": "common",
        "feature_set": shared_config.feature_set,
        "model_type": shared_config.model_type,
        "model_profile": shared_config.model_profile,
        "validation_method": "shared common model walk-forward validation",
        "walk_forward_folds": metric_value(safety_metrics, "walk_forward_folds", default=8),
        "test_rows": metric_value(safety_metrics, "test_rows"),
        "oos_rows": metric_value(safety_metrics, "test_rows"),
        "MAE": metric_value(safety_metrics, "mae"),
        "RMSE": metric_value(safety_metrics, "rmse"),
        "R2": metric_value(safety_metrics, "r2"),
        "baseline_mae": metric_value(safety_metrics, "baseline_mae"),
        "baseline_rmse": metric_value(safety_metrics, "baseline_rmse"),
        "baseline_r2": metric_value(safety_metrics, "baseline_r2"),
        "spearman_corr": metric_value(safety_metrics, "spearman_corr"),
        "safety_mae": metric_value(safety_metrics, "mae"),
        "safety_rmse": metric_value(safety_metrics, "rmse"),
        "safety_r2": metric_value(safety_metrics, "r2"),
        "safety_baseline_mae": metric_value(safety_metrics, "baseline_mae"),
        "safety_baseline_rmse": metric_value(safety_metrics, "baseline_rmse"),
        "safety_baseline_r2": metric_value(safety_metrics, "baseline_r2"),
        "safety_spearman": metric_value(safety_metrics, "spearman_corr"),
        "return_mae": metric_value(return_metrics, "mae"),
        "return_rmse": metric_value(return_metrics, "rmse"),
        "return_r2": metric_value(return_metrics, "r2"),
        "return_baseline_mae": metric_value(return_metrics, "baseline_mae"),
        "return_baseline_rmse": metric_value(return_metrics, "baseline_rmse"),
        "return_baseline_r2": metric_value(return_metrics, "baseline_r2"),
        "return_spearman": metric_value(return_metrics, "spearman_corr"),
        "data_fingerprint": shared_result.get("data_fingerprint"),
        "feature_fingerprint": shared_result.get("feature_fingerprint"),
        "symbols_fingerprint": shared_result.get("symbols_fingerprint"),
        "row_count_by_symbol_fingerprint": shared_result.get("row_count_by_symbol_fingerprint"),
        "common_model_note": (
            "本報告使用與多股票排名模式相同之共同模型。"
            "共同模型先於 Top 30 股票池中進行訓練與驗證，再針對指定股票輸出單股診斷、"
            "交易可行性、風險報酬比與進場計畫。"
        ),
    }
    for col in ["safety_score", "return_score", "combined_decision_score", "final_decision_score"]:
        if col in output_df.columns:
            summary[f"{col}_mean"] = pd.to_numeric(output_df[col], errors="coerce").mean()
    return summary


def print_shared_common_validation_metrics(shared_result: dict) -> None:
    safety_metrics = shared_result["safety_metrics"]
    return_metrics = shared_result["return_metrics"]
    print("\n" + "=" * 70)
    print("Shared Common Model Validation Metrics")
    print("=" * 70)
    print("Safety model common validation:")
    print_model_validation_body(safety_metrics, is_multi_stock=True)
    print("Return model common validation:")
    print_model_validation_body(return_metrics, is_multi_stock=True)


def metrics_row(prefix: str, metrics: dict) -> dict:
    return {
        "model": prefix,
        "target_col": metrics.get("target_col"),
        "feature_set": metrics.get("feature_set"),
        "model_type": metrics.get("model_type"),
        "model_profile": metrics.get("model_profile"),
        "walk_forward_folds": metrics.get("walk_forward_folds"),
        "test_rows": metrics.get("test_rows"),
        "MAE": metrics.get("mae"),
        "RMSE": metrics.get("rmse"),
        "R2": metrics.get("r2"),
        "baseline_mae": metrics.get("baseline_mae"),
        "baseline_rmse": metrics.get("baseline_rmse"),
        "baseline_r2": metrics.get("baseline_r2"),
        "spearman_corr": metrics.get("spearman_corr"),
        "top5_lift": metrics.get("top5_lift"),
        "top10_lift": metrics.get("top10_lift"),
        "long_short_spread": metrics.get("long_short_spread"),
    }


def run_shared_single_stock_diagnostic(
    config: PipelineConfig,
    top_n: int = 30,
    verbose: bool = False,
    investor_mode: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_symbol = str(config.stock_no).strip()
    if not verbose:
        if investor_mode:
            print("正在分析個股資料...")
        else:
            print_presentation_process_header()

    stock_list, symbols = build_shared_common_stock_pool(
        base_config=config,
        top_n=top_n,
        target_symbol=target_symbol,
        quiet=not verbose,
    )
    shared_config = build_shared_common_model_config(config)
    shared_config = replace(shared_config, stock_no=target_symbol)
    shared_result = run_shared_common_model_core(
        base_config=shared_config,
        stock_list=stock_list,
        symbols=symbols,
        context_name="MODE_1_SINGLE_DIAGNOSTIC",
    )

    prediction_results = shared_result["prediction_results"].copy()
    prediction_results["symbol"] = prediction_results["symbol"].astype(str)
    target_prediction = prediction_results[prediction_results["symbol"] == target_symbol].copy()
    if target_prediction.empty:
        raise ValueError(f"shared common model prediction_results missing target symbol: {target_symbol}")

    safety_target_col = shared_result["safety_target_col"]
    return_target_col = shared_result["return_target_col"]
    diagnostic_base_all = build_single_diagnostic_output_frame(
        prediction_results,
        shared_config,
        safety_target_col,
        return_target_col,
    )
    risk_thresholds = {
        "volatility_10d_q80": pd.to_numeric(diagnostic_base_all["volatility_10d"], errors="coerce").quantile(0.80),
        "support_distance_q80": pd.to_numeric(diagnostic_base_all["support_distance"], errors="coerce").quantile(0.80),
        "resistance_distance_q20": pd.to_numeric(diagnostic_base_all["resistance_distance"], errors="coerce").quantile(0.20),
    }
    diagnostic_output_all = apply_single_diagnostic_decision_outputs(diagnostic_base_all, risk_thresholds)
    output_df = diagnostic_output_all[diagnostic_output_all["symbol"].astype(str) == target_symbol].copy()
    if output_df.empty:
        raise ValueError(f"single diagnostic output missing target symbol after decision enrichment: {target_symbol}")

    output_df["prediction_source"] = "shared_common_model_latest"
    output_df = output_df.sort_values("date").tail(1).reset_index(drop=True)
    summary = build_shared_common_metrics_summary(shared_result, config, output_df=output_df)
    summary_df = pd.DataFrame([summary])

    decision_label_backtest_summary = build_decision_label_backtest_summary(output_df)
    latest_decision_df = output_df.copy()
    enhanced_decision_backtest = build_enhanced_decision_backtest_summary(output_df, shared_config)
    first_layer_score_band_summary = build_score_band_summary(
        output_df,
        "combined_decision_score",
        "first_layer_score_band",
    )
    final_decision_score_band_summary = build_score_band_summary(
        output_df,
        "final_decision_score",
        "final_decision_score_band",
    )
    veto_summary = build_veto_summary(output_df)
    feature_and_rule_config = build_feature_and_rule_config(shared_config)
    common_validation_df = pd.DataFrame([
        metrics_row("safety", shared_result["safety_metrics"]),
        metrics_row("return", shared_result["return_metrics"]),
    ])
    safety_fold_info_df = build_fold_info_from_metrics(shared_result["safety_metrics"])
    return_fold_info_df = build_fold_info_from_metrics(shared_result["return_metrics"])

    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    file_prefix = OUTPUT_DIR / (
        f"{config.market.upper()}_{target_symbol}_common_single_diagnostic_{timestamp}"
    )
    csv_path = Path(f"{file_prefix}.csv")
    xlsx_path = Path(f"{file_prefix}.xlsx")
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="all_diagnostics", index=False)
        latest_decision_df.to_excel(writer, sheet_name="latest_decision", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        summary_df.to_excel(writer, sheet_name="model_validation", index=False)
        common_validation_df.to_excel(writer, sheet_name="common_model_validation", index=False)
        decision_label_backtest_summary.to_excel(writer, sheet_name="decision_label_backtest", index=False)
        enhanced_decision_backtest.to_excel(writer, sheet_name="enhanced_decision_backtest", index=False)
        first_layer_score_band_summary.to_excel(writer, sheet_name="first_layer_score_band_summary", index=False)
        final_decision_score_band_summary.to_excel(writer, sheet_name="final_decision_score_band_summary", index=False)
        veto_summary.to_excel(writer, sheet_name="veto_summary", index=False)
        feature_and_rule_config.to_excel(writer, sheet_name="feature_and_rule_config", index=False)
        if not safety_fold_info_df.empty:
            safety_fold_info_df.to_excel(writer, sheet_name="safety_walk_forward_folds", index=False)
        if not return_fold_info_df.empty:
            return_fold_info_df.to_excel(writer, sheet_name="return_walk_forward_folds", index=False)

    if verbose:
        print("\nShared common model single diagnostic summary:")
        print(summary_df.to_string(index=False))
        print("\nLatest shared common diagnostic row:")
        print(output_df.to_string(index=False))
        print(f"\nSaved files:\n1. {csv_path}\n2. {xlsx_path}")
    else:
        print(
            "本報告使用與多股票排名模式相同之共同模型。"
            "共同模型先於 Top 30 股票池中進行訓練與驗證，再針對指定股票輸出單股診斷、"
            "交易可行性、風險報酬比與進場計畫。"
        )
        print_final_investment_decision(output_df, shared_config, summary=summary)
        print_shared_common_validation_metrics(shared_result)
        print_presentation_model_settings(shared_config, top_n=len(symbols), investor_mode=investor_mode)
        if not investor_mode:
            print_presentation_feature_fold_appendix(shared_config, metrics=shared_result["safety_metrics"])
        print("Output files:")
        print(f"1. {csv_path}")
        print(f"2. {xlsx_path}")

    return output_df, summary_df


def get_common_model_paths(market: str) -> tuple[Path, Path]:
    market = market.upper()
    return (
        MODEL_DIR / f"{market}_common_model.joblib",
        MODEL_DIR / f"{market}_common_model_features.json",
    )


def save_common_model(model, feature_cols: list[str], market: str) -> tuple[Path, Path]:
    model_path, features_path = get_common_model_paths(market)
    joblib.dump(model, model_path)
    features_path.write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_path, features_path


def load_common_model(market: str):
    model_path, features_path = get_common_model_paths(market)
    if not model_path.exists() or not features_path.exists():
        print("找不到既有共同模型，請先執行 train 模式。")
        print(f"缺少模型檔：{model_path}")
        print(f"缺少特徵檔：{features_path}")
        return None, None
    model = joblib.load(model_path)
    feature_cols = json.loads(features_path.read_text(encoding="utf-8"))
    return model, feature_cols


def save_multi_stock_ranking(
    prediction_results: pd.DataFrame,
    stock_list: pd.DataFrame,
    market: str,
    mode: str,
    combined_data: Optional[pd.DataFrame] = None,
    feature_set: Optional[str] = None,
) -> tuple[Path, Path, Optional[Path]]:
    if not prediction_results.empty:
        stock_meta_cols = [col for col in ["symbol", "stock_name", "today_turnover"] if col in stock_list.columns]
        stock_meta = stock_list[stock_meta_cols].copy()
        stock_meta["symbol"] = stock_meta["symbol"].astype(str)
        prediction_results = prediction_results.merge(stock_meta, on="symbol", how="left")
        prediction_results = sort_and_rank_multi_stock_results(prediction_results)

        front_cols = [
            "rank", "symbol", "stock_name", "date", "close", "today_turnover",
            "predicted_target_score", "predicted_score", "model_score",
            "safety_score", "return_score", "combined_decision_score",
            "rule_filter_score", "technical_filter_result",
            "entry_price", "stop_loss_price_adjusted", "stop_loss_pct",
            "take_profit_price", "risk_reward_ratio",
            "stop_loss_too_close_flag", "practical_risk_per_share",
            "practical_risk_reward_ratio", "risk_reward_score", "practical_risk_reward_score",
            "trade_veto_flag", "trade_veto_reason",
            "final_decision_score", "final_score_weight_profile",
            "final_decision_label", "decision_action",
            "position_sizing_label", "position_sizing_reason",
            "risk_per_trade_pct", "max_position_pct", "suggested_position_pct",
            "risk_warning", "suggested_action_text",
            "decision_summary_text", "signal",
            "RSI14", "MACD", "K", "D", "J", "atr_pct", "fib_position", "fib_zone",
            "nearest_support_level", "nearest_resistance_level",
            "support_distance", "resistance_distance",
            "rule_penalty_points", "rule_filter_notes",
            "trade_plan_notes", "risk_per_share", "reward_per_share",
            "suggested_shares", "enhanced_decision_report_text",
            "decision_summary", "technical_reason", "risk_reason", "action_suggestion",
        ]
        ordered_cols = [col for col in front_cols if col in prediction_results.columns]
        remaining_cols = [col for col in prediction_results.columns if col not in ordered_cols]
        prediction_results = prediction_results[ordered_cols + remaining_cols]

    feature_tag = f"_{feature_set}" if feature_set else ""
    summary_prefix = f"{market.upper()}_COMMON_MODEL_{mode.upper()}_{len(prediction_results)}_{datetime.today().strftime('%Y%m%d_%H%M%S')}{feature_tag}"
    ranking_csv = OUTPUT_DIR / f"{summary_prefix}_ranking.csv"
    ranking_xlsx = OUTPUT_DIR / f"{summary_prefix}_ranking.xlsx"
    combined_csv = None
    per_stock_decision_summary = build_per_stock_decision_summary_df(prediction_results)

    prediction_results.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(ranking_xlsx, engine="openpyxl") as writer:
        prediction_results.to_excel(writer, sheet_name="multi_stock_ranking", index=False)
        per_stock_decision_summary.to_excel(writer, sheet_name="per_stock_decision_summary", index=False)

    if combined_data is not None:
        combined_csv = OUTPUT_DIR / f"{summary_prefix}_training_data.csv"
        combined_data.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    return ranking_csv, ranking_xlsx, combined_csv


def get_terminal_label_short(value) -> str:
    text = translate_decision_label(value)
    text = normalize_text_for_report(text).strip()
    mapping = {
        "高品質機會，可條件式進場": "高品質",
        "穩健偏多，優先觀察": "偏多",
        "中性偏多，等待確認": "等待",
        "中性觀察": "中性",
        "保守觀察": "保守",
        "避免進場": "避免",
        "否決交易": "否決",
    }
    return mapping.get(text, text[:4] if text else "NA")


def get_terminal_position_short(value) -> str:
    text = normalize_text_for_report(value).strip()
    mapping = {
        "高品質機會倉位": "高倉",
        "標準小倉位": "小倉",
        "小倉試探": "試探",
        "保守觀察，不配置實際倉位": "觀察",
        "條件不足，暫不配置倉位": "不配",
        "不配置倉位": "不配",
    }
    return mapping.get(text, text[:4] if text else "NA")


def get_terminal_veto_short(value) -> str:
    veto_value = str(value).strip().lower() in {"true", "1", "yes", "y"}
    return "是" if veto_value else "否"


def format_terminal_number(value, decimals: int = 2) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "NA"
    return f"{float(numeric_value):.{decimals}f}"


def terminal_display_width(value) -> int:
    import unicodedata

    text = str(value)
    return sum(2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1 for char in text)


def format_terminal_cell(value, width: int, align: str = "<") -> str:
    text = str(value)
    padding = max(width - terminal_display_width(text), 0)
    if align == ">":
        return (" " * padding) + text
    return text + (" " * padding)


def get_effective_terminal_rr(row: pd.Series):
    practical_rr = _row_number(row, "practical_risk_reward_ratio")
    if pd.notna(practical_rr):
        return practical_rr
    return _row_number(row, "risk_reward_ratio")


def get_terminal_risk_note(row: pd.Series) -> str:
    stop_loss_too_close = str(row.get("stop_loss_too_close_flag", False)).strip().lower() in {"true", "1", "yes", "y"}
    return "停損近" if stop_loss_too_close else ""


def print_ranking_preview(prediction_results: pd.DataFrame, top_n: Optional[int] = None) -> None:
    if prediction_results.empty:
        print("沒有可輸出的 ranking 結果。")
        return

    preview = prediction_results.head(top_n) if top_n is not None else prediction_results
    print(
        " ".join([
            format_terminal_cell("Rank", 4),
            format_terminal_cell("Symbol", 6),
            format_terminal_cell("Date", 10),
            format_terminal_cell("Close", 10, ">"),
            format_terminal_cell("Final", 6, ">"),
            format_terminal_cell("Label", 6),
            format_terminal_cell("Eff R/R", 7, ">"),
            format_terminal_cell("Veto", 4),
            format_terminal_cell("Position", 6),
            format_terminal_cell("RiskNote", 6),
        ])
    )
    for _, row in preview.iterrows():
        date_text = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        score_value = row.get("final_decision_score", row.get("predicted_score", row.get("predicted_target_score", np.nan)))
        print(
            " ".join([
                format_terminal_cell(int(row["rank"]), 4),
                format_terminal_cell(row.get("symbol", "NA"), 6),
                format_terminal_cell(date_text, 10),
                format_terminal_cell(format_terminal_number(row.get("close")), 10, ">"),
                format_terminal_cell(format_terminal_number(score_value), 6, ">"),
                format_terminal_cell(get_terminal_label_short(row.get("final_decision_label", row.get("signal", "NA"))), 6),
                format_terminal_cell(format_terminal_number(get_effective_terminal_rr(row)), 7, ">"),
                format_terminal_cell(get_terminal_veto_short(row.get("trade_veto_flag", False)), 4),
                format_terminal_cell(get_terminal_position_short(row.get("position_sizing_label", "NA")), 6),
                format_terminal_cell(get_terminal_risk_note(row), 6),
            ])
        )


MULTI_STOCK_DECISION_SUMMARY_COLS = [
    "rank",
    "symbol",
    "date",
    "close",
    "safety_score",
    "return_score",
    "combined_decision_score",
    "rule_filter_score",
    "technical_filter_result",
    "entry_price",
    "stop_loss_price_adjusted",
    "stop_loss_pct",
    "take_profit_price",
    "risk_reward_ratio",
    "stop_loss_too_close_flag",
    "practical_risk_per_share",
    "practical_risk_reward_ratio",
    "risk_reward_score",
    "practical_risk_reward_score",
    "trade_veto_flag",
    "trade_veto_reason",
    "final_decision_score",
    "final_decision_label",
    "decision_action",
    "position_sizing_label",
    "position_sizing_reason",
    "risk_per_trade_pct",
    "max_position_pct",
    "suggested_position_pct",
    "risk_warning",
    "suggested_action_text",
    "RSI14",
    "MACD",
    "K",
    "D",
    "J",
    "BIAS24",
    "VR_14",
    "VR_26",
    "return_5d",
    "return_10d",
    "atr_14",
    "atr_pct",
    "risk_entry_price",
    "risk_stoploss_price",
    "risk_stoploss_pct",
    "risk_atr_stop_distance_pct",
    "fib_position",
    "fib_zone",
    "nearest_support_level",
    "nearest_resistance_level",
    "support_distance",
    "resistance_distance",
    "recent_2w_ob_type",
    "recent_2w_ob_is_fallback",
    "recent_active_gap_type",
]


def build_per_stock_decision_summary_df(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return pd.DataFrame(columns=MULTI_STOCK_DECISION_SUMMARY_COLS)
    out = ranking_df[[col for col in MULTI_STOCK_DECISION_SUMMARY_COLS if col in ranking_df.columns]].copy()
    for col in [
        "final_decision_label",
        "decision_action",
        "technical_filter_result",
        "trade_veto_reason",
        "position_sizing_label",
        "position_sizing_reason",
        "risk_warning",
        "suggested_action_text",
        "fib_zone",
    ]:
        if col in out.columns:
            out[col] = out[col].apply(normalize_text_for_report)
    return out


def get_rsi_status(value) -> str:
    rsi = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(rsi):
        return "無資料"
    if rsi >= 70:
        return "過熱"
    if rsi >= 55:
        return "中性偏強"
    if rsi >= 45:
        return "中性"
    if rsi >= 30:
        return "中性偏弱"
    return "偏弱或接近超賣"


def get_macd_status(value) -> str:
    macd = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(macd):
        return "無資料"
    return "偏多" if macd > 0 else "偏弱"


def get_kdj_status(row: pd.Series) -> str:
    k_value = pd.to_numeric(pd.Series([row.get("K", np.nan)]), errors="coerce").iloc[0]
    d_value = pd.to_numeric(pd.Series([row.get("D", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(k_value) or pd.isna(d_value):
        return "無資料"
    return "短線偏多" if k_value > d_value else "短線偏弱"


def print_multi_stock_presentation_report(ranking_df: pd.DataFrame) -> None:
    if ranking_df.empty:
        print("沒有可輸出的多股票投資決策資料。")
        return

    ranking_df = sort_and_rank_multi_stock_results(ranking_df)

    print("=" * 70)
    print("多股票最新投資決策排名")
    print("=" * 70)
    print()
    print("Rank  Symbol  Date        Close     Safety  Return  Combined  Decision                         Risk Warning")
    for _, row in ranking_df.iterrows():
        print(
            f"{int(row.get('rank', 0)):<5} "
            f"{str(row.get('symbol', 'NA')):<7} "
            f"{format_report_date(row.get('date')):<10} "
            f"{format_report_price(row.get('close')):<9} "
            f"{format_decision_score(row.get('safety_score')):<7} "
            f"{format_decision_score(row.get('return_score')):<7} "
            f"{format_decision_score(row.get('combined_decision_score')):<9} "
            f"{normalize_text_for_report(row.get('final_decision_label', '中性觀察')):<30} "
            f"{normalize_text_for_report(row.get('risk_warning', '無明顯風險警示'))}"
        )

    print()
    print("=" * 70)
    print("逐檔投資決策摘要")
    print("=" * 70)

    for _, row in ranking_df.iterrows():
        print()
        print("-" * 70)
        print(f"Rank {int(row.get('rank', 0))}｜{row.get('symbol', 'NA')}")
        print("-" * 70)
        print(f"資料日期：{format_report_date(row.get('date'))}")
        print(f"最新收盤價：{format_report_price(row.get('close'))}")
        print(f"最終決策標籤：{translate_decision_label(row.get('final_decision_label', '中性觀察'))}")
        print(f"第一層模型綜合分數：{format_decision_score(row.get('combined_decision_score'))}")
        print(f"最終決策分數：{format_decision_score(row.get('final_decision_score'))}")
        print(f"安全分數：{format_decision_score(row.get('safety_score'))}")
        print(f"報酬機會分數：{format_decision_score(row.get('return_score'))}")
        print()
        print("【目前技術面依據】")
        print(f"RSI14：{format_report_value(row.get('RSI14'))}")
        print(f"MACD：{format_report_value(row.get('MACD'))}")
        print(f"K：{format_report_value(row.get('K'))}")
        print(f"D：{format_report_value(row.get('D'))}")
        print(f"J：{format_report_value(row.get('J'))}")
        print(f"BIAS24：{format_bias_value(row.get('BIAS24'))}")
        print(f"VR_14：{format_report_value(row.get('VR_14'))}")
        print(f"VR_26：{format_report_value(row.get('VR_26'))}")
        print(f"return_5d：{format_report_value(row.get('return_5d'), pct=True)}")
        print(f"return_10d：{format_report_value(row.get('return_10d'), pct=True)}")
        print()
        print("【價格位置與風險依據】")
        print(f"fib_position：{format_report_value(row.get('fib_position'))}")
        print(f"fib_zone：{normalize_text_for_report(row.get('fib_zone', 'NA'))}")
        print(f"nearest_support_level：{format_report_price(row.get('nearest_support_level'))}")
        print(f"nearest_resistance_level：{format_report_price(row.get('nearest_resistance_level'))}")
        print(f"support_distance：{format_report_value(row.get('support_distance'), pct=True)}")
        print(f"resistance_distance：{format_report_value(row.get('resistance_distance'), pct=True)}")
        print(f"atr_14：{format_report_value(row.get('atr_14'))}")
        print(f"atr_pct：{format_report_value(row.get('atr_pct'), pct=True)}")
        print(f"主要停損價 stop_loss_price_adjusted：{format_report_price(row.get('stop_loss_price_adjusted'))}")
        print(f"主要停損距離 stop_loss_pct：{format_report_value(row.get('stop_loss_pct'), pct=True)}")
        print(f"原始風險欄位 risk_entry_price：{format_report_price(row.get('risk_entry_price'))}")
        print(f"原始風險欄位 risk_stoploss_price：{format_report_price(row.get('risk_stoploss_price'))}")
        print(f"原始風險欄位 risk_stoploss_pct：{format_report_value(row.get('risk_stoploss_pct'), pct=True)}")
        print(f"risk_atr_stop_distance_pct：{format_report_value(row.get('risk_atr_stop_distance_pct'), pct=True)}")
        print(f"recent_2w_ob_type：{normalize_text_for_report(row.get('recent_2w_ob_type', 'NA'))}")
        print(f"recent_2w_ob_is_fallback：{format_report_value(row.get('recent_2w_ob_is_fallback'), decimals=0)}")
        print(f"recent_active_gap_type：{normalize_text_for_report(row.get('recent_active_gap_type', 'NA'))}")
        print()
        print("【決策輔助指南】")
        print(build_decision_guide_text(row, include_combined=False))
        print()
        print("【風險警示】")
        print(normalize_text_for_report(row.get("risk_warning", "無明顯風險警示")))
        print()
        print("【操作建議】")
        print(normalize_text_for_report(row.get("suggested_action_text", "")))
        print()
        print("【綜合決策說明】")
        print(build_combined_decision_explanation_text(row))


def prepare_top_turnover_combined_data(base_config: PipelineConfig, top_n: int, quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if quiet:
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            stock_list = fetch_twse_top_turnover_stocks(limit=top_n)
            symbols = stock_list["symbol"].astype(str).tolist()
            combined_data = fetch_and_prepare_multi_stock_data(
                symbols=symbols,
                market=base_config.market,
                years=base_config.years,
                horizon=base_config.horizon,
                risk_penalty=base_config.risk_penalty,
                opportunity_threshold=base_config.opportunity_threshold,
                feature_set=base_config.feature_set,
            )
        for line in captured_output.getvalue().splitlines():
            if (
                "TWSE OpenAPI" in line
                or "成交金額排行快取" in line
                or "ranking CSV" in line
                or "無法取得成交金額排行" in line
            ):
                print(line)
    else:
        stock_list = fetch_twse_top_turnover_stocks(limit=top_n)
        symbols = stock_list["symbol"].astype(str).tolist()
        combined_data = fetch_and_prepare_multi_stock_data(
            symbols=symbols,
            market=base_config.market,
            years=base_config.years,
            horizon=base_config.horizon,
            risk_penalty=base_config.risk_penalty,
            opportunity_threshold=base_config.opportunity_threshold,
            feature_set=base_config.feature_set,
        )

    return stock_list, combined_data, symbols


def run_common_model_train(
    base_config: PipelineConfig,
    top_n: int = 30,
    symbols_override: list[str] | None = None,
    investor_mode: bool = True,
) -> pd.DataFrame:
    presentation = base_config.display_mode == "presentation"
    if presentation:
        if investor_mode:
            print("正在分析市場資料...")
        else:
            print_presentation_process_header()
    stock_list, symbols = build_shared_common_stock_pool(
        base_config=base_config,
        top_n=top_n,
        symbols_override=symbols_override,
        quiet=presentation,
    )
    total = len(symbols)
    pool_label = (
        f"自定義股票池共 {total} 檔股票"
        if symbols_override is not None
        else f"{base_config.market.upper()} 成交金額 Top {top_n} 股票"
    )
    if not presentation:
        print(f"本次將以 {pool_label}，訓練 1 個共同模型。")
    else:
        print(f"本次將以 {pool_label}建立共同決策模型。")

    shared_result = run_shared_common_model_core(
        base_config=base_config,
        stock_list=stock_list,
        symbols=symbols,
        context_name="MODE_2_MULTI_STOCK_RANKING",
    )
    shared_config = shared_result["shared_config"]
    combined_data = shared_result["combined_data"]
    prediction_results = shared_result["prediction_results"]
    metrics = shared_result["safety_metrics"]
    model_path, features_path = save_common_model(
        shared_result["safety_model"],
        shared_result["safety_feature_cols"],
        shared_config.market,
    )

    if not presentation:
        print(f"Saved shared safety common model: {model_path}")
        print(f"Saved shared safety feature list: {features_path}")

    ranking_csv, ranking_xlsx, combined_csv = save_multi_stock_ranking(
        prediction_results=prediction_results,
        stock_list=stock_list,
        market=shared_config.market,
        mode="train",
        combined_data=combined_data,
        feature_set=shared_config.feature_set,
    )

    if presentation:
        with contextlib.redirect_stdout(io.StringIO()):
            save_walk_forward_validation_report(metrics, shared_config)
            save_feature_importance_report(metrics, shared_config)
            save_target_diagnostic_report(metrics, shared_config)
        print()
        print_multi_stock_presentation_report(prediction_results)
        print()
        print_presentation_model_validation_summary(metrics, is_multi_stock=True)
        print_shared_common_validation_metrics(shared_result)
        print_presentation_model_settings(shared_config, top_n=None if symbols_override is not None else top_n, investor_mode=investor_mode)
        if not investor_mode:
            print_presentation_feature_fold_appendix(shared_config, metrics=metrics)
        print("Output files:")
        print(f"ranking CSV: {ranking_csv}")
        print(f"ranking Excel: {ranking_xlsx}")
        print(f"training data CSV: {combined_csv}")
    else:
        save_walk_forward_validation_report(metrics, shared_config)
        save_feature_importance_report(metrics, shared_config)
        save_target_diagnostic_report(metrics, shared_config)

        print("\n" + "=" * 70)
        print("Multi-stock shared common ranking preview")
        print("=" * 70)
        print_ranking_preview(prediction_results)

        print("\nTWSE shared common model output files:")
        print(f"ranking CSV: {ranking_csv}")
        print(f"ranking Excel: {ranking_xlsx}")
        print(f"training data CSV: {combined_csv}")

    return prediction_results

    def train_model_for_target(target_col: str):
        target_config = replace(base_config, target_col=target_col)
        if presentation:
            with contextlib.redirect_stdout(io.StringIO()):
                return train_model(
                    data=combined_data,
                    target_col=target_col,
                    config=target_config,
                )
        return train_model(
            data=combined_data,
            target_col=target_col,
            config=target_config,
        )

    if presentation:
        print("Step 5/6：建立共同模型並產生 safety / return / combined 決策分數")

    model, selected_features, metrics = train_model_for_target(base_config.target_col)
    mode_2_metrics = {**metrics, "top_n": top_n if symbols_override is None else len(symbols)}
    register_model_flow_audit(
        collect_model_flow_audit(
            context_name="MODE_2_MULTI_STOCK_RANKING",
            config=base_config,
            data=combined_data,
            feature_cols=selected_features,
            numeric_features=metrics.get("numeric_features", []),
            categorical_features=metrics.get("categorical_features", []),
            removed_features=metrics.get("removed_features", []),
            metrics=mode_2_metrics,
            validation_scope="all_symbols_common_pool",
            model_flow="train_model_walk_forward_validate_plus_latest_ranking",
        )
    )
    model_path, features_path = save_common_model(model, selected_features, base_config.market)
    if not presentation:
        print(f"已儲存共同模型：{model_path}")
        print(f"已儲存模型特徵欄位：{features_path}")

        print("\n" + "=" * 70)
        print("模型實際使用特徵")
        print("=" * 70)
        print(f"feature_set：{metrics.get('feature_set', base_config.feature_set)}")
        print(f"model_type：{metrics.get('model_type', base_config.model_type)}")
        print(f"model_profile：{metrics.get('model_profile', base_config.model_profile)}")
        print(f"模型實際使用特徵數量：{metrics.get('feature_count', len(metrics['feature_cols']))}")
        print("數值特徵：")
        print(metrics["numeric_features"])
        print("類別特徵：")
        print(metrics["categorical_features"])
        if metrics.get("feature_set") == "structural":
            print("missing structural features：")
            print(metrics.get("missing_structural_features", []))
            print("被保留但未進模型的原始指標欄位：")
            print(metrics.get("retained_raw_excluded_features", []))
        print("被基礎篩選移除的特徵：")
        print(metrics["removed_features"])

    trained_targets = {base_config.target_col: (model, selected_features, metrics)}

    def get_trained_target_model(target_col: str):
        if target_col not in trained_targets:
            trained_targets[target_col] = train_model_for_target(target_col)
        return trained_targets[target_col]

    safety_target_col = f"target_safety_{base_config.horizon}d_cs_rank"
    return_target_col = f"target_return_{base_config.horizon}d_cs_rank"
    safety_model, safety_features, _ = get_trained_target_model(safety_target_col)
    return_model, return_features, _ = get_trained_target_model(return_target_col)

    if presentation:
        print("Step 6/6：輸出投資決策報告與模型驗證摘要")
    prediction_results = predict_latest_for_each_stock(
        model=model,
        data=combined_data,
        symbols=symbols,
        feature_cols=selected_features,
        extra_models={
            "safety_score": (safety_model, safety_features),
            "return_score": (return_model, return_features),
        },
    )
    prediction_results = enrich_multi_stock_decision_fields(prediction_results)

    ranking_csv, ranking_xlsx, combined_csv = save_multi_stock_ranking(
        prediction_results=prediction_results,
        stock_list=stock_list,
        market=base_config.market,
        mode="train",
        combined_data=combined_data,
        feature_set=base_config.feature_set,
    )

    if presentation:
        with contextlib.redirect_stdout(io.StringIO()):
            save_walk_forward_validation_report(metrics, base_config)
            save_feature_importance_report(metrics, base_config)
            save_target_diagnostic_report(metrics, base_config)
        print()
        print_multi_stock_presentation_report(prediction_results)
        print()
        print_presentation_model_validation_summary(metrics, is_multi_stock=True)
        print_presentation_model_settings(base_config, top_n=None if symbols_override is not None else top_n)
        print_presentation_feature_fold_appendix(base_config, metrics=metrics)
        print("輸出檔案：")
        print(f"排名 CSV：{ranking_csv}")
        print(f"排名 Excel：{ranking_xlsx}")
        print(f"合併訓練資料：{combined_csv}")
    else:
        save_walk_forward_validation_report(metrics, base_config)
        save_feature_importance_report(metrics, base_config)
        save_target_diagnostic_report(metrics, base_config)

        print("\n" + "=" * 70)
        print("多股票最新預測排名")
        print("=" * 70)
        print_ranking_preview(prediction_results)

        print("\nTWSE 共同模型批次分析完成。")
        print(f"排名 CSV：{ranking_csv}")
        print(f"排名 Excel：{ranking_xlsx}")
        print(f"合併訓練資料：{combined_csv}")

    return prediction_results


def print_experiment_metrics_block(title: str, metrics: dict) -> None:
    print(title)
    print(f"- Feature count: {metrics.get('feature_count', len(metrics.get('feature_cols', [])))}")
    print(f"- MAE: {metrics['mae']:.4f}")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- R2: {metrics['r2']:.4f}")
    print(f"- Baseline MAE: {metrics['baseline_mae']:.4f}")
    print(f"- Baseline RMSE: {metrics['baseline_rmse']:.4f}")
    print(f"- Baseline R2: {metrics['baseline_r2']:.4f}")


def print_full_vs_structural_comparison(full_metrics: dict, structural_metrics: dict) -> None:
    print("\n" + "=" * 70)
    print("Experiment Comparison: Full vs Structural")
    print("=" * 70)
    print()
    print_experiment_metrics_block("Full Feature Model（實務主模型）", full_metrics)
    print()
    print_experiment_metrics_block("Structural-only Model（可解釋性與風險結構對照模型）", structural_metrics)
    print("\nStructural model actual features:")
    print(structural_metrics.get("feature_cols", []))
    print("Missing structural features:")
    print(structural_metrics.get("missing_structural_features", []))
    print("Retained raw indicator columns not used by structural model:")
    print(structural_metrics.get("retained_raw_excluded_features", []))


def run_single_experiment_from_processed_data(base_config: PipelineConfig, combined_data: pd.DataFrame) -> None:
    model, selected_features, metrics = train_model(
        data=combined_data,
        target_col=base_config.target_col,
        config=base_config,
    )

    model_path, features_path = save_common_model(model, selected_features, base_config.market)
    save_target_diagnostic_report(metrics, base_config)

    print("\nRanking metrics summary")
    print(f"Spearman：{metrics.get('spearman_corr', np.nan):.4f}")
    print(f"Top5 Lift：{metrics.get('top5_lift', np.nan):.4f}")
    print(f"Top10 Lift：{metrics.get('top10_lift', np.nan):.4f}")
    print(f"Long-Short Spread：{metrics.get('long_short_spread', np.nan):.4f}")
    print(f"\n模型儲存路徑：{model_path}")
    print(f"特徵欄位儲存路徑：{features_path}")


def run_full_vs_structural_experiment(base_config: PipelineConfig, combined_data: pd.DataFrame) -> None:
    full_config = replace(base_config, feature_set="full")
    structural_config = replace(base_config, feature_set="structural")

    print("\n訓練 Full Feature Model（實務主模型）...")
    _, _, full_metrics = train_model(
        data=combined_data,
        target_col=full_config.target_col,
        config=full_config,
    )

    save_target_diagnostic_report(full_metrics, full_config)

    print("\n訓練 Structural-only Model（可解釋性與風險結構對照模型）...")
    _, _, structural_metrics = train_model(
        data=combined_data,
        target_col=structural_config.target_col,
        config=structural_config,
    )
    save_target_diagnostic_report(structural_metrics, structural_config)

    print_full_vs_structural_comparison(full_metrics, structural_metrics)

    print("\n" + "=" * 70)
    print("Classification Experiment: Full vs Structural")
    print("=" * 70)
    raw_target_col = f"target_score_raw_{base_config.horizon}d"
    if base_config.target_col == raw_target_col:
        print(f"分類目標：high_value = {base_config.target_col} >= 0.03；low_value = {base_config.target_col} <= -0.05")
    else:
        print(f"分類目標：high_value = {base_config.target_col} >= 0.7；low_value = {base_config.target_col} <= 0.3")

    full_classification_results = run_classification_experiment_for_config(
        data=combined_data,
        config=full_config,
    )
    print_classification_experiment_report("full", full_classification_results)
    save_classification_experiment_report(full_classification_results, full_config)

    structural_classification_results = run_classification_experiment_for_config(
        data=combined_data,
        config=structural_config,
    )
    print_classification_experiment_report("structural", structural_classification_results)
    save_classification_experiment_report(structural_classification_results, structural_config)


def find_latest_training_data_path() -> Optional[Path]:
    candidates = (
        list(OUTPUT_DIR.glob("*_training_data.csv"))
        + list(OUTPUT_DIR.glob("*_combined_training_data.csv"))
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def ensure_target_comparison_columns(
    combined_data: pd.DataFrame,
    horizon: int = 10,
) -> pd.DataFrame:
    out = combined_data.copy()
    raw_col = f"target_score_raw_{horizon}d"
    future_return_col = f"future_return_{horizon}d"
    future_drawdown_col = f"future_drawdown_{horizon}d"
    risk_adjusted_col = f"risk_adjusted_return_{horizon}d"

    if raw_col not in out.columns:
        if future_return_col in out.columns and future_drawdown_col in out.columns:
            out[raw_col] = (
                pd.to_numeric(out[future_return_col], errors="coerce")
                + pd.to_numeric(out[future_drawdown_col], errors="coerce")
            )
        elif risk_adjusted_col in out.columns:
            out[raw_col] = pd.to_numeric(out[risk_adjusted_col], errors="coerce")

    out = add_low_subjective_targets(out, horizon=horizon)
    return out.replace([np.inf, -np.inf], np.nan)


def parse_target_compare_columns(target_compare_cols: str, horizon: int = 10) -> list[str]:
    raw_target_col = f"target_score_raw_{horizon}d"
    all_target_cols = [
        "target_score",
        "target_score_v2",
        "target_score_v3",
        raw_target_col,
        f"target_return_{horizon}d",
        f"target_return_{horizon}d_cs_rank",
        f"target_downside_risk_{horizon}d",
        f"target_downside_risk_{horizon}d_cs_rank",
        f"target_safety_{horizon}d_cs_rank",
        f"target_abs_drawdown_safety_{horizon}d",
        f"target_rar_{horizon}d",
        f"target_rar_{horizon}d_cs_rank",
    ]

    if str(target_compare_cols).strip().lower() == "all":
        return all_target_cols

    cols = [
        item.strip()
        for item in str(target_compare_cols).split(",")
        if item.strip()
    ]
    if not cols:
        return ["target_score_v2", raw_target_col]

    normalized_cols = []
    for col in cols:
        if col == "target_score_raw_10d":
            col = raw_target_col
        normalized_cols.append(col)

    return list(dict.fromkeys(normalized_cols))


def parse_compare_feature_sets(compare_feature_sets: Optional[str]) -> list[str]:
    if compare_feature_sets is None or not str(compare_feature_sets).strip():
        return list(DEFAULT_COMPARE_FEATURE_SETS)

    feature_sets = [
        item.strip()
        for item in str(compare_feature_sets).split(",")
        if item.strip()
    ]
    if not feature_sets:
        return list(DEFAULT_COMPARE_FEATURE_SETS)

    invalid_feature_sets = [
        feature_set
        for feature_set in feature_sets
        if feature_set not in FEATURE_SET_CHOICES
    ]
    if invalid_feature_sets:
        raise ValueError(
            f"未知 compare_feature_sets：{invalid_feature_sets}；"
            f"可用 feature_set：{FEATURE_SET_CHOICES}"
        )

    return list(dict.fromkeys(feature_sets))


def parse_compare_model_types(
    compare_model_types: Optional[str],
    default_model_type: str = "random_forest",
) -> list[str]:
    if compare_model_types is None or not str(compare_model_types).strip():
        return [default_model_type]

    model_types = [
        item.strip()
        for item in str(compare_model_types).split(",")
        if item.strip()
    ]
    if not model_types:
        return [default_model_type]

    invalid_model_types = [
        model_type
        for model_type in model_types
        if model_type not in MODEL_TYPE_CHOICES
    ]
    if invalid_model_types:
        raise ValueError(
            f"未知 compare_model_types：{invalid_model_types}；"
            f"可用 model_type：{MODEL_TYPE_CHOICES}"
        )

    return list(dict.fromkeys(model_types))


def run_target_score_comparison_experiment(
    base_config: PipelineConfig,
    input_data_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_data_path)

    print("使用模式：target_compare")
    print(f"input_data 路徑：{input_path}")

    combined_data = pd.read_csv(input_path)
    combined_data = clean_duplicate_columns(combined_data)
    if "date" not in combined_data.columns:
        raise ValueError("input_data 缺少 date 欄位")

    combined_data["date"] = pd.to_datetime(combined_data["date"], errors="coerce")
    combined_data = combined_data.dropna(subset=["date"]).sort_values(
        ["date", "symbol"] if "symbol" in combined_data.columns else ["date"]
    ).reset_index(drop=True)
    combined_data = ensure_target_comparison_columns(combined_data, horizon=base_config.horizon)
    combined_data = clean_duplicate_columns(combined_data)
    combined_data = add_structural_features(combined_data)
    combined_data = add_stock_level_context_features(combined_data, base_config)
    combined_data = add_cross_sectional_pool_features(combined_data)
    combined_data = add_cross_sectional_rank_features(combined_data)
    combined_data = clean_duplicate_columns(combined_data)
    print_new_feature_availability(combined_data, context="target_compare input data")

    raw_target_col = f"target_score_raw_{base_config.horizon}d"
    target_cols = parse_target_compare_columns(
        base_config.target_compare_cols,
        horizon=base_config.horizon,
    )
    feature_sets = parse_compare_feature_sets(base_config.compare_feature_sets)
    model_types = parse_compare_model_types(
        base_config.compare_model_types,
        default_model_type=base_config.model_type,
    )

    regression_rows = []
    classification_rows = []
    classification_columns = [
        "target_col",
        "feature_set",
        "class_target",
        "threshold_rule",
        "positive_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "top10_probability_hit_rate",
        "confusion_tn",
        "confusion_fp",
        "confusion_fn",
        "confusion_tp",
        "error",
    ]

    print("本次 target_compare targets：")
    print(target_cols)
    print("本次 target_compare feature_sets：")
    print(feature_sets)
    print("本次 target_compare model_types：")
    print(model_types)
    print(
        "\nTarget scale note: MAE / RMSE are not directly comparable across targets with different scales. "
        "Use R2, Spearman, Top5 Lift, Top10 Lift, and Long-Short Spread for cross-target comparison."
    )
    if base_config.skip_classification:
        print("已跳過 classification comparison。")

    for target_col in target_cols:
        if target_col not in combined_data.columns:
            print(f"略過 target_col={target_col}，資料中沒有此欄位。")
            for feature_set in feature_sets:
                for model_type in model_types:
                    regression_rows.append({
                        "target_col": target_col,
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "model_profile": base_config.model_profile,
                        "error": f"missing target column: {target_col}",
                    })
            continue

        for feature_set in feature_sets:
            feature_config = replace(
                base_config,
                feature_set=feature_set,
                target_col=target_col,
            )

            for model_type in model_types:
                temp_config = replace(feature_config, model_type=model_type)

                print("\n" + "=" * 70)
                print(
                    "Target comparison："
                    f"target={target_col} / feature_set={feature_set} / model_type={model_type}"
                )
                print("=" * 70)

                try:
                    _, _, metrics = train_model(
                        data=combined_data,
                        target_col=target_col,
                        config=temp_config,
                    )
                    regression_rows.append({
                        "target_col": target_col,
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "model_profile": temp_config.model_profile,
                        "feature_count": metrics.get("feature_count"),
                        "training_rows": metrics.get("training_rows"),
                        "training_symbols": metrics.get("training_symbols"),
                        "MAE": metrics.get("mae"),
                        "RMSE": metrics.get("rmse"),
                        "R2": metrics.get("r2"),
                        "baseline_MAE": metrics.get("baseline_mae"),
                        "baseline_RMSE": metrics.get("baseline_rmse"),
                        "baseline_R2": metrics.get("baseline_r2"),
                        "spearman_corr": metrics.get("spearman_corr"),
                        "top5_lift": metrics.get("top5_lift"),
                        "top10_lift": metrics.get("top10_lift"),
                        "long_short_spread": metrics.get("long_short_spread"),
                        "error": "",
                    })
                except Exception as exc:
                    print(f"Regression comparison 失敗：{exc}")
                    regression_rows.append({
                        "target_col": target_col,
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "model_profile": temp_config.model_profile,
                        "error": str(exc),
                    })

            if not base_config.skip_classification:
                try:
                    class_results = run_classification_experiment_for_config(
                        data=combined_data,
                        config=feature_config,
                    )
                    for class_target, result in class_results.items():
                        summary = result["summary"]
                        classification_rows.append({
                            "target_col": target_col,
                            "feature_set": feature_set,
                            "class_target": class_target,
                            "threshold_rule": summary.get("threshold_rule"),
                            "positive_rate": summary.get("positive_rate"),
                            "accuracy": summary.get("accuracy"),
                            "precision": summary.get("precision"),
                            "recall": summary.get("recall"),
                            "f1": summary.get("f1"),
                            "roc_auc": summary.get("roc_auc"),
                            "pr_auc": summary.get("pr_auc"),
                            "top10_probability_hit_rate": summary.get("top10_probability_hit_rate"),
                            "confusion_tn": summary.get("confusion_tn"),
                            "confusion_fp": summary.get("confusion_fp"),
                            "confusion_fn": summary.get("confusion_fn"),
                            "confusion_tp": summary.get("confusion_tp"),
                            "error": "",
                        })
                except Exception as exc:
                    print(f"Classification comparison 失敗：{exc}")
                    classification_rows.append({
                        "target_col": target_col,
                        "feature_set": feature_set,
                        "class_target": "",
                        "threshold_rule": "",
                        "error": str(exc),
                    })

    regression_df = pd.DataFrame(regression_rows)
    classification_df = pd.DataFrame(classification_rows, columns=classification_columns)

    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    output_prefix = output_dir / f"{base_config.market.upper()}_TARGET_COMPARISON_{timestamp}"
    report_xlsx = Path(f"{output_prefix}.xlsx")
    regression_csv = Path(f"{output_prefix}_regression_summary.csv")
    classification_csv = Path(f"{output_prefix}_classification_summary.csv")

    regression_df.to_csv(regression_csv, index=False, encoding="utf-8-sig")
    classification_df.to_csv(classification_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(report_xlsx) as writer:
        regression_df.to_excel(writer, sheet_name="regression_ranking_summary", index=False)
        classification_df.to_excel(writer, sheet_name="classification_summary", index=False)

    print("\n已輸出 target comparison 報告：")
    print(f"1. {report_xlsx}")
    print(f"2. {regression_csv}")
    print(f"3. {classification_csv}")

    display_cols = [
        "target_col",
        "feature_set",
        "MAE",
        "R2",
        "spearman_corr",
        "top5_lift",
        "top10_lift",
        "long_short_spread",
    ]
    available_display_cols = [col for col in display_cols if col in regression_df.columns]
    print("\nTarget comparison regression summary：")
    if available_display_cols:
        print(regression_df[available_display_cols].to_string(index=False))
    else:
        print(regression_df.to_string(index=False))

    class_display_cols = [
        "target_col",
        "feature_set",
        "class_target",
        "threshold_rule",
        "positive_rate",
        "top10_probability_hit_rate",
        "roc_auc",
        "pr_auc",
    ]
    available_class_display_cols = [col for col in class_display_cols if col in classification_df.columns]
    print("\nTarget comparison classification summary：")
    if available_class_display_cols:
        print(classification_df[available_class_display_cols].to_string(index=False))

    print(
        "\n提醒：不同 target 的 MAE / RMSE 尺度不同，不能只用 MAE / RMSE 判斷優劣；"
        "應優先比較 Spearman、Top5 Lift、Top10 Lift、Long-Short Spread "
        "與 classification Top10 hit rate。"
    )

    print("\nCurrent recommended baseline:")
    print("feature_set = selected_v2")
    print("model_type = extra_trees")
    print("model_profile = medium")
    print("\nRecommended target experiment:")
    print("1. target_score_v2")
    print("2. target_return_10d_cs_rank")
    print("3. target_safety_10d_cs_rank")
    print("4. target_rar_10d_cs_rank")
    print("\nInterpretation:")
    print("- target_return_10d_cs_rank = return model")
    print("- target_safety_10d_cs_rank = risk / safety model")
    print("- target_rar_10d_cs_rank = one-stage risk-adjusted comparison model")

    return regression_df, classification_df


def parse_sensitivity_profiles(profile_text: Optional[str]) -> Optional[list[str]]:
    if profile_text is None or not str(profile_text).strip():
        return None
    return [item.strip() for item in str(profile_text).split(",") if item.strip()]


def run_local_range_sensitivity_experiment(
    base_config: PipelineConfig,
    top_n: int = 30,
    selected_profile_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    profiles = get_local_range_sensitivity_profiles()
    if selected_profile_names:
        valid_profile_names = {profile["profile_name"] for profile in profiles}
        invalid_profile_names = [
            name for name in selected_profile_names
            if name not in valid_profile_names
        ]
        if invalid_profile_names:
            raise ValueError(
                "未知 sensitivity profile："
                f"{invalid_profile_names}；可用 profile：{sorted(valid_profile_names)}"
            )
        selected_set = set(selected_profile_names)
        profiles = [profile for profile in profiles if profile["profile_name"] in selected_set]

    print("本次 sensitivity profiles：")
    print([profile["profile_name"] for profile in profiles])

    results = []
    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")

    stock_list = fetch_twse_top_turnover_stocks(limit=top_n)
    symbols = stock_list["symbol"].astype(str).tolist()

    for profile in profiles:
        print("\n" + "=" * 70)
        print(f"開始局部範圍敏感度測試：{profile['profile_name']} / {profile['purpose']}")
        print("=" * 70)
        print(profile)

        profile_config = replace(
            base_config,
            stock_no="ALL",
            analysis_mode="sensitivity",
            use_sensitivity=True,
            require_sensitivity_data=True,
            fib_lookback_days=profile["fib_lookback_days"],
            ob_lookback_days=profile["ob_lookback_days"],
            sr_lookback_days=profile["sr_lookback_days"],
            atr_period=profile["atr_period"],
        )

        combined_data = fetch_and_prepare_multi_stock_data_with_config(
            symbols=symbols,
            base_config=profile_config,
        )

        training_data_path = OUTPUT_DIR / (
            f"{profile_config.market}_SENSITIVITY_{profile['profile_name']}_"
            f"{timestamp}_training_data.csv"
        )
        combined_data.to_csv(training_data_path, index=False, encoding="utf-8-sig")

        model, selected_features, metrics = train_model(
            data=combined_data,
            target_col=profile_config.target_col,
            config=profile_config,
        )

        row = {
            "profile_name": profile["profile_name"],
            "purpose": profile["purpose"],
            "feature_set": profile_config.feature_set,
            "model_type": profile_config.model_type,
            "model_profile": profile_config.model_profile,
            "top_n": top_n,
            "fib_lookback_days": profile_config.fib_lookback_days,
            "ob_lookback_days": profile_config.ob_lookback_days,
            "sr_lookback_days": profile_config.sr_lookback_days,
            "atr_period": profile_config.atr_period,
            "atr_multiplier": profile_config.atr_multiplier,
            "sr_tolerance_pct": profile_config.sr_tolerance_pct,
            "fvg_min_gap_pct": profile_config.fvg_min_gap_pct,
            "feature_count": metrics.get("feature_count"),
            "training_rows": metrics.get("training_rows"),
            "training_symbols": metrics.get("training_symbols"),
            "MAE": metrics.get("mae"),
            "RMSE": metrics.get("rmse"),
            "R2": metrics.get("r2"),
            "baseline_MAE": metrics.get("baseline_mae"),
            "baseline_RMSE": metrics.get("baseline_rmse"),
            "baseline_R2": metrics.get("baseline_r2"),
            "spearman_corr": metrics.get("spearman_corr"),
            "top5_lift": metrics.get("top5_lift"),
            "top10_lift": metrics.get("top10_lift"),
            "long_short_spread": metrics.get("long_short_spread"),
            "training_data_path": str(training_data_path),
        }

        results.append(row)

    comparison_df = pd.DataFrame(results)

    comparison_csv = OUTPUT_DIR / f"{base_config.market}_LOCAL_RANGE_SENSITIVITY_{timestamp}.csv"
    comparison_xlsx = OUTPUT_DIR / f"{base_config.market}_LOCAL_RANGE_SENSITIVITY_{timestamp}.xlsx"

    comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    comparison_df.to_excel(comparison_xlsx, index=False)

    print("\n" + "=" * 70)
    print("局部範圍敏感度測試完成")
    print("=" * 70)
    print(comparison_df)
    print(f"\n比較表 CSV：{comparison_csv}")
    print(f"比較表 Excel：{comparison_xlsx}")

    return comparison_df


def run_experiment_from_processed_data(base_config: PipelineConfig, input_data_path) -> None:
    input_path = Path(input_data_path)

    print("使用模式：experiment")
    print(f"input_data 路徑：{input_path}")

    combined_data = pd.read_csv(input_path)
    if "date" not in combined_data.columns:
        raise ValueError("input_data 缺少 date 欄位")

    combined_data["date"] = pd.to_datetime(combined_data["date"], errors="coerce")
    combined_data = combined_data.dropna(subset=["date"]).sort_values(
        ["date", "symbol"] if "symbol" in combined_data.columns else ["date"]
    ).reset_index(drop=True)
    combined_data = add_structural_features(combined_data)
    combined_data = add_stock_level_context_features(combined_data, base_config)
    combined_data = add_cross_sectional_pool_features(combined_data)
    combined_data = add_cross_sectional_rank_features(combined_data)
    combined_data = clean_duplicate_columns(combined_data)
    print_new_feature_availability(combined_data, context="experiment input data")

    symbol_count = combined_data["symbol"].nunique() if "symbol" in combined_data.columns else 1

    print(f"feature_set：{base_config.feature_set}")
    print(f"model_type：{base_config.model_type}")
    print(f"model_profile：{base_config.model_profile}")
    print(f"target_col：{base_config.target_col}")
    print(f"資料筆數：{len(combined_data)}")
    print(f"股票數量：{symbol_count}")
    print(f"日期範圍：{combined_data['date'].min()} ~ {combined_data['date'].max()}")

    if base_config.feature_set in {"full", "structural"}:
        run_full_vs_structural_experiment(base_config, combined_data)
    else:
        run_single_experiment_from_processed_data(base_config, combined_data)


def run_common_model_predict(base_config: PipelineConfig, top_n: int = 30) -> Optional[pd.DataFrame]:
    print("使用模式：predict")

    model, selected_features = load_common_model(base_config.market)
    if model is None or selected_features is None:
        return None

    stock_list, combined_data, symbols = prepare_top_turnover_combined_data(base_config, top_n, quiet=True)

    prediction_results = predict_latest_for_each_stock(
        model=model,
        data=combined_data,
        symbols=symbols,
        feature_cols=selected_features,
    )
    prediction_results = enrich_multi_stock_decision_fields(prediction_results)

    ranking_csv, ranking_xlsx, _ = save_multi_stock_ranking(
        prediction_results=prediction_results,
        stock_list=stock_list,
        market=base_config.market,
        mode="predict",
        combined_data=None,
        feature_set=base_config.feature_set,
    )

    last_date = pd.to_datetime(combined_data["date"]).max()
    print(f"股票數量：{len(symbols)}")
    print(f"資料最後日期：{last_date.strftime('%Y-%m-%d')}")
    print("是否載入既有模型：是")
    display_n = min(len(prediction_results), top_n if top_n <= 30 else 30)
    print(f"\nTop {display_n} ranking")
    print_ranking_preview(prediction_results, top_n=display_n)
    print(f"\nRanking CSV：{ranking_csv}")
    print(f"Ranking Excel：{ranking_xlsx}")

    return prediction_results


def run_top_volume_twse_pipeline(base_config: PipelineConfig, top_n: int = 30) -> pd.DataFrame:
    return run_common_model_train(base_config, top_n=top_n)


def run_all_twse_pipeline(base_config: PipelineConfig, max_stocks: Optional[int] = None) -> pd.DataFrame:
    top_n = max_stocks if max_stocks is not None else 30
    return run_top_volume_twse_pipeline(base_config, top_n=top_n)




__all__ = [name for name in globals() if not name.startswith('__')]
