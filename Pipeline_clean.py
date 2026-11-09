from __future__ import annotations

"""Main entry point for the modular Clean Stock ML Pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_backtest import *

def run_sensitivity_analysis(config: "PipelineConfig", top_n: int = 30) -> None:
    """
    敏感度分析：對五組參數進行單因子回測。
    資料準備（抓取 + OOS 預測）只執行一次；每次只改一個參數，其他維持 config 預設值。
    """
    import contextlib, io

    # ── 一次性資料準備 ──────────────────────────────────────────────
    print("=" * 60)
    print("敏感度分析：開始資料準備（只跑一次）")
    print("=" * 60)

    stock_list, symbols = build_shared_common_stock_pool(
        base_config=config, top_n=top_n, target_symbol=None, quiet=True
    )
    shared_config = build_shared_common_model_config(config)

    print(f"Step 1/3: 抓取 {len(symbols)} 支股票歷史資料...")
    with contextlib.redirect_stdout(io.StringIO()):
        combined_data = fetch_and_prepare_multi_stock_data_with_config(
            symbols=symbols, base_config=shared_config
        )

    print("Step 2/3: 執行 OOS 預測生成（12 折 walk-forward）...")
    oos_df = generate_oos_predictions_for_backtest(
        df=combined_data, config=shared_config, n_splits=12
    )
    if oos_df.empty:
        print("OOS 預測資料不足，無法執行敏感度分析。")
        return

    print("Step 3/3: 抓取大盤資料...")
    try:
        market_df = get_market_index_data(config)
    except Exception as exc:
        print(f"WARNING: 大盤資料取得失敗（{exc}），不啟用大盤過濾。")
        market_df = pd.DataFrame()

    print("資料準備完成，開始逐組敏感度測試。\n")

    # ── 建立基準信號字典 ────────────────────────────────────────────
    def _build_signals(df: pd.DataFrame) -> dict:
        out: dict = {}
        for sym in df["symbol"].astype(str).unique():
            out[str(sym)] = df[df["symbol"].astype(str) == sym].copy()
        return out

    base_signals = _build_signals(oos_df)

    # ── 執行一次回測（suppress 詳細輸出），回傳績效 dict ───────────
    def _run_one(signals: dict, cfg) -> dict:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                equity_curve, trade_log, _ = run_portfolio_backtest(
                    all_stock_signals=signals, config=cfg, market_df=market_df
                )
            return calculate_portfolio_metrics(
                equity_curve=equity_curve,
                trade_log=trade_log,
                initial_capital=cfg.backtest_initial_capital,
            )
        except Exception as exc:
            print(f"  WARNING: 回測異常（{exc}），跳過此版本。")
            return {}

    # ── 格式化並記錄一行摘要 ────────────────────────────────────────
    summary_rows: list[dict] = []

    def _record(group: str, version: str, perf: dict) -> None:
        cagr    = perf.get("cagr",         float("nan"))
        mdd     = perf.get("max_drawdown", float("nan"))
        sharpe  = perf.get("sharpe_ratio", float("nan"))
        calmar  = perf.get("calmar_ratio", float("nan"))
        entries = perf.get("total_entries", 0)
        line = (
            f"[{group}] {version:<18s} | "
            f"CAGR: {cagr:+7.2%} | MDD: {mdd:7.2%} | "
            f"Sharpe: {sharpe:5.2f} | Calmar: {calmar:5.2f} | 進場次數: {entries}"
        )
        print(line)
        summary_rows.append({
            "群組":   group,
            "版本":   version,
            "CAGR":   f"{cagr:+.2%}",
            "MDD":    f"{mdd:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Calmar": f"{calmar:.2f}",
            "進場次數": entries,
        })

    # ── 重算 final_decision_score（用新權重，不重新跑 OOS）─────────
    def _recompute_score(signals: dict, weights: dict) -> dict:
        out: dict = {}
        for sym, df in signals.items():
            df2 = df.copy()
            s = pd.to_numeric(df2.get("safety_score",      0), errors="coerce").fillna(0)
            r = pd.to_numeric(df2.get("return_score",      0), errors="coerce").fillna(0)
            f = pd.to_numeric(df2.get("rule_filter_score", 0), errors="coerce").fillna(0)
            k = pd.to_numeric(df2.get("risk_reward_score", 0), errors="coerce").fillna(0)
            df2["final_decision_score"] = (
                weights["safety_score"]      * s
                + weights["return_score"]      * r
                + weights["rule_filter_score"] * f
                + weights["risk_reward_score"] * k
            ).clip(0, 1)
            out[sym] = df2
        return out

    # ── 過濾 R/R 門檻（低於門檻的 row 清零 final_decision_score）──
    # 注意：引擎內部硬編碼 entry_rr_1 >= 1.5，rr=1.2 版本實際效果與 1.5 相同。
    def _filter_rr(signals: dict, rr_min: float) -> dict:
        out: dict = {}
        for sym, df in signals.items():
            df2 = df.copy()
            if "entry_rr_1" in df2.columns:
                rr = pd.to_numeric(df2["entry_rr_1"], errors="coerce")
                df2.loc[rr < rr_min, "final_decision_score"] = 0.0
            out[sym] = df2
        return out

    # ════════════════════════════════════════════════════════════════
    # 第一組：final_score 權重敏感度（5 個版本）
    # ════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("第一組：final_score 權重敏感度")
    print("=" * 60)
    weight_versions = [
        ("high_model", {"safety_score": 0.40, "return_score": 0.30, "rule_filter_score": 0.20, "risk_reward_score": 0.10}),
        ("current",    {"safety_score": 0.30, "return_score": 0.20, "rule_filter_score": 0.25, "risk_reward_score": 0.25}),
        ("low_model",  {"safety_score": 0.15, "return_score": 0.10, "rule_filter_score": 0.50, "risk_reward_score": 0.25}),
        ("high_rr",    {"safety_score": 0.25, "return_score": 0.15, "rule_filter_score": 0.25, "risk_reward_score": 0.35}),
        ("original",   {"safety_score": 0.30, "return_score": 0.15, "rule_filter_score": 0.35, "risk_reward_score": 0.20}),
    ]
    for version_name, weights in weight_versions:
        _record("weights", version_name, _run_one(_recompute_score(base_signals, weights), config))

    # ════════════════════════════════════════════════════════════════
    # 第二組：R/R 門檻敏感度（4 個版本）
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第二組：R/R 門檻敏感度")
    print("  注意：引擎硬編碼 entry_rr_1 >= 1.5，rr=1.2 版本效果與 1.5 相同。")
    print("=" * 60)
    for rr_min in [1.2, 1.5, 2.0, 2.5]:
        _record("rr_threshold", f"rr>={rr_min:.1f}", _run_one(_filter_rr(base_signals, rr_min), config))

    # ════════════════════════════════════════════════════════════════
    # 第三組：倉位比例敏感度（4 個版本）
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第三組：倉位比例敏感度（試倉%／加碼1%／加碼2%）")
    print("=" * 60)
    position_versions = [
        ("6/4/4",   0.06, 0.04, 0.04),
        ("8/6/6",   0.08, 0.06, 0.06),  # 目前預設
        ("10/5/5",  0.10, 0.05, 0.05),
        ("10/10/0", 0.10, 0.10, 0.00),
    ]
    for label, init_pct, add1_pct, add2_pct in position_versions:
        cfg = replace(
            config,
            backtest_initial_position_pct=init_pct,
            backtest_add1_position_pct=add1_pct,
            backtest_add2_position_pct=add2_pct,
        )
        _record("position_sizing", label, _run_one(base_signals, cfg))

    # ════════════════════════════════════════════════════════════════
    # 第四組：最大持股數敏感度（4 個版本）
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第四組：最大持股數敏感度")
    print("=" * 60)
    for max_pos in [5, 8, 10, 12]:
        cfg = replace(config, backtest_max_positions=max_pos)
        _record("max_positions", f"max_pos={max_pos}", _run_one(base_signals, cfg))

    # ════════════════════════════════════════════════════════════════
    # 第五組：最大曝險敏感度（4 個版本）
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第五組：最大曝險敏感度")
    print("=" * 60)
    for exp_pct in [0.70, 0.80, 0.85, 0.95]:
        cash_reserve = round(1.0 - exp_pct, 2)
        cfg = replace(
            config,
            backtest_max_total_exposure=exp_pct,
            backtest_min_cash_reserve=cash_reserve,
        )
        _record("max_exposure", f"exp={exp_pct:.0%}", _run_one(base_signals, cfg))

    # ════════════════════════════════════════════════════════════════
    # 彙整表
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("敏感度分析彙整")
    print("=" * 70)
    if summary_rows:
        col_order = ["群組", "版本", "CAGR", "MDD", "Sharpe", "Calmar", "進場次數"]
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
        print(summary_df.to_string(index=False))


def _has_anthropic_key() -> bool:
    import os
    try:
        from pipeline_ai import ROOT_DIR, load_env_file
        load_env_file(ROOT_DIR / ".env")
    except Exception:
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "experiment", "sensitivity", "sensitivity_analysis", "target_compare", "single_diagnostic", "dual_diagnostic", "backtest"],
        default="single_diagnostic",
    )
    parser.add_argument("--market", default="TWSE")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--input-data", default=None)
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="selected_v2")
    parser.add_argument("--model-type", choices=MODEL_TYPE_CHOICES, default="extra_trees")
    parser.add_argument("--model-profile", choices=MODEL_PROFILE_CHOICES, default="medium")
    parser.add_argument("--diagnostic-scope", choices=DIAGNOSTIC_SCOPE_CHOICES, default="common")
    parser.add_argument("--display-mode", choices=DISPLAY_MODE_CHOICES, default="presentation")
    parser.add_argument("--target-col", default="target_safety_10d_cs_rank")
    parser.add_argument(
        "--run-profile-experiments",
        action="store_true",
        help="額外執行 Portfolio profile comparison；預設只跑正式主策略",
    )
    parser.add_argument(
        "--run-feature-importance",
        action="store_true",
        help="輸出 feature importance report；預設關閉，research 模式可手動開啟",
    )
    parser.add_argument(
        "--no-feature-importance",
        action="store_true",
        help="跳過 feature importance report（保留相容舊參數）",
    )
    parser.add_argument(
        "--target-compare-cols",
        default=CONFIG.target_compare_cols,
        help="target_compare 要比較的 target 欄位，逗號分隔",
    )
    parser.add_argument(
        "--compare-feature-sets",
        default=CONFIG.compare_feature_sets,
        help=(
            "target_compare 要比較的 feature_set，逗號分隔；"
            f"可用值：{','.join(FEATURE_SET_CHOICES)}"
        ),
    )
    parser.add_argument(
        "--compare-model-types",
        default=CONFIG.compare_model_types,
        help=(
            "target_compare 要比較的 model_type，逗號分隔；"
            f"可用值：{','.join(MODEL_TYPE_CHOICES)}"
        ),
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="target_compare 時跳過 classification，只跑 regression / ranking",
    )
    parser.add_argument("--sensitivity-profiles", default=None)
    parser.add_argument(
        "--investor-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="只輸出投資人摘要；--no-investor-mode 強制顯示完整診斷",
    )
    parser.add_argument(
        "--pool-csv",
        default=None,
        help="可重現模式：固定股票池 CSV（需含 symbol 欄位），不再抓執行當天的成交金額排行",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="可重現模式：資料基準日 YYYY-MM-DD，只用到這天的本地快取，不上網、不寫回快取",
    )
    parser.add_argument(
        "--shared-raw-cache-dir",
        default=None,
        help="可選：可重現模式的唯讀個股與 TAIEX 原始快取目錄，不會寫入該目錄。",
    )
    parser.add_argument(
        "--daily-top30-universe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回測時依每個交易日的實際 Top N 股票池建模與選股；預設啟用。使用 --no-daily-top30-universe 才會改用固定 CSV 股票池。",
    )
    parser.add_argument(
        "--daily-top30-rank-metric",
        choices=["volume", "turnover"],
        default="volume",
        help="每日 Top N 排序依據，預設 volume（成交量）。",
    )
    parser.add_argument(
        "--shared-snapshot-cache-dir",
        default=None,
        help="每日 Top N 所需的唯讀全市場日快取目錄。",
    )
    parser.add_argument(
        "--purge-days",
        type=int,
        default=None,
        help="walk-forward 清洗期（交易日），預設 10；設 0 可重現 2026-09-21 論文版（舊做法）",
    )
    parser.add_argument(
        "--score-profile",
        default=None,
        choices=["low_model_weight_v1", "rule_rr_v2"],
        help="最終決策分數權重：low_model_weight_v1＝論文版（0.30 安全＋0.20 報酬＋0.25 規則＋0.25 風險報酬）；"
             "rule_rr_v2＝規則與風險報酬各 0.5（2026-10-26 新增）",
    )
    parser.add_argument(
        "--optimize-score-weights",
        action="store_true",
        help="以 DE 對四個 final score 權重與進場門檻做巢狀 walk-forward 最佳化；僅支援 daily Top-N 回測。",
    )
    parser.add_argument("--optimization-tuning-folds", type=int, default=8,
                        help="最佳化使用的前段 OOS folds 數量，剩餘 folds 保留作最終測試。")
    parser.add_argument("--optimization-maxiter", type=int, default=12,
                        help="Differential Evolution 世代數。")
    parser.add_argument("--optimization-popsize", type=int, default=5,
                        help="Differential Evolution 每個維度的母體倍率。")
    parser.add_argument("--optimization-regularization", type=float, default=0.05,
                        help="偏離目前 score profile 權重的 L2 正則化係數。")
    parser.add_argument("--optimization-min-trades", type=int, default=30,
                        help="調參期間最低總進場筆數限制。")
    parser.add_argument(
        "--optimization-ai-range-audit",
        action="store_true",
        help="讓 Claude 只根據調參 folds 產出下一次實驗的搜尋範圍建議；不會自動套用或改變本次結果。",
    )
    parser.add_argument("--optimization-max-weight", type=float, default=0.60,
                        help="單一 final score 權重的最大值。")
    parser.add_argument("--optimization-entry-threshold-min", type=float, default=0.50,
                        help="DE 搜尋的進場門檻下界。")
    parser.add_argument("--optimization-entry-threshold-max", type=float, default=0.80,
                        help="DE 搜尋的進場門檻上界。")
    parser.add_argument(
        "--optimization-evaluation-end-date",
        default=None,
        help="DE 最終評估 OOS 的截止日 YYYY-MM-DD；只截斷評估列，不重建快取或模型。",
    )
    parser.add_argument(
        "--include-otc",
        action="store_true",
        help="股票池納入上櫃股票：上市＋上櫃合併依當日成交金額取前 N 檔（個股分析會自動判別上市／上櫃，不需此參數）",
    )
    parser.add_argument(
        "--chip-features",
        action="store_true",
        help="模型加入三大法人、融資融券、月營收特徵（資料在 data/chip/，以 tools/fetch_chip_data.py 取得）",
    )
    parser.add_argument(
        "--no-price-adjust",
        action="store_true",
        help="不做除權／面額變更價格還原（重現 2026-10-26 以前的結果時使用）",
    )
    parser.add_argument("--agent", default=None, metavar="問題",
                        help='AI 決策代理：用一句話提問，例如 --agent "2330 現在可以買嗎？有沒有更好的選擇？"')
    parser.add_argument("--agent-dry-run", action="store_true", help="決策代理不呼叫 API，改用固定流程與確定性範本")
    parser.add_argument("--ai-report", action="store_true",
                        help="第十二段 AI 分析師解讀（Claude 撰寫＋Critic 稽核；不改變任何決策）")
    parser.add_argument("--ai-model", default=None, help="AI 分析師使用的模型，預設 claude-sonnet-5-5")
    parser.add_argument("--ai-dry-run", action="store_true",
                        help="AI 分析師報告不呼叫 API，只用確定性範本，並把 prompt 存進輸出 JSON")
    args, _ = parser.parse_known_args()
    set_reproducible_mode(pool_csv=args.pool_csv, as_of=args.as_of)
    set_shared_raw_cache_dir(args.shared_raw_cache_dir)
    if args.shared_snapshot_cache_dir:
        from pipeline_historical_universe import set_shared_snapshot_cache_dir
        set_shared_snapshot_cache_dir(args.shared_snapshot_cache_dir)
    set_price_adjust(not args.no_price_adjust)
    NAME_SETTINGS["allow_fetch"] = True          # 名稱查不到時可從 FinMind 補齊（可重現模式下不會上網）
    if args.include_otc:
        set_include_otc(True)
    if args.chip_features:
        from pipeline_chip import set_chip_features
        set_chip_features(True)
    if args.score_profile:
        set_final_score_profile(args.score_profile)
    if args.ai_report or args.ai_dry_run:
        from pipeline_ai import set_ai_mode
        set_ai_mode(enabled=args.ai_report, model=args.ai_model, dry_run=args.ai_dry_run)
    if args.purge_days is not None:
        set_walk_forward_purge(args.purge_days)
    if args.agent:
        from pipeline_agent import run_agent_cli
        run_agent_cli(args.agent, CONFIG, symbol=args.symbol, top_n=30,
                      dry_run=args.agent_dry_run, model=args.ai_model)
        return

    direct_run = len(sys.argv) == 1
    symbols_for_training: list[str] | None = None

    backtest_symbol: Optional[str] = None
    backtest_top_n: int = 30
    backtest_mode_selected: bool = False

    if direct_run:
        print("請選擇分析模式：")
        print("1. 個股深度分析（輸入任意股票代碼，輸出完整個股決策報告）")
        print("2. 多股票排名分析（依成交金額 Top N 建立共同模型並輸出排名）")
        print("3. 回測分析（對指定股票或 Top N 進行歷史回測）")
        print("4. AI 決策代理（用一句話提問，例如「2330 現在可以買嗎？有沒有更好的選擇？」）")
        mode_choice = input("請輸入 1、2、3 或 4，預設 1：").strip()
        if mode_choice == "4":
            question = input("請輸入你的問題：").strip() or "2330 現在可以買嗎？有沒有更好的選擇？"
            has_key = _has_anthropic_key()
            if not has_key:
                print("找不到 ANTHROPIC_API_KEY，改用固定流程（不呼叫 API）。")
            from pipeline_agent import run_agent_cli
            run_agent_cli(question, CONFIG, top_n=30, dry_run=not has_key, model=args.ai_model)
            return
        args.diagnostic_scope = "common"
        args.feature_set = "selected_v2"
        args.model_type = "extra_trees"
        args.model_profile = "medium"
        args.target_col = "target_safety_10d_cs_rank"
        args.display_mode = "presentation"

        if mode_choice == "3":
            backtest_mode_selected = True
            args.mode = "backtest"
        elif mode_choice == "2":
            args.mode = "train"
        else:
            args.mode = "single_diagnostic"

        print("請選擇市場：")
        print("1. TWSE 上市")
        print("2. TPEX 上櫃（個股會自動判別上市／上櫃；Top N 會改為上市＋上櫃合併排行）")
        market_choice = input("請輸入 1 或 2，預設 1：").strip()
        args.market = "TPEX" if market_choice == "2" else "TWSE"

        if backtest_mode_selected:
            print("請選擇回測對象：")
            print("1. 指定個股（輸入股票代碼）")
            print("2. Top N 股票池批次回測（輸入 N，預設 30）")
            bt_choice = input("請輸入 1 或 2，預設 1：").strip()
            if bt_choice == "2":
                top_n_text = input("請輸入 Top N 數量，預設 30：").strip()
                backtest_top_n = int(top_n_text) if top_n_text.isdigit() else 30
                backtest_symbol = None
                args.symbol = "ALL"
                args.top_n = backtest_top_n
                print(f"本次將對{'上市＋上櫃' if args.market == 'TPEX' else '上市'}成交金額 Top {backtest_top_n} 股票進行批次回測。")
            else:
                user_sym = input("請輸入要回測的股票代碼，例如 2330：").strip()
                backtest_symbol = user_sym if user_sym else "2330"
                backtest_top_n = 30
                args.symbol = backtest_symbol
                args.top_n = 30
                print(f"本次將對 {backtest_symbol} 進行個股回測（以 Top 30 共同模型為基礎）。")
        elif args.mode == "single_diagnostic":
            args.symbol = prompt_single_stock_symbol()
            args.top_n = 30
        else:
            top_n_text = input("請輸入 Top N 數量，預設 30：")
            args.top_n = parse_interactive_top_n(top_n_text)
            args.symbol = "ALL"
            symbols_for_training = None
            print(f"本次將以{'上市＋上櫃' if args.market == 'TPEX' else '上市'}成交金額 Top {args.top_n} 股票建立共同決策模型。")

        # AI 分析師（2026-10-27）：每條選單路徑都可產生
        has_key = _has_anthropic_key()
        print("是否產生 AI 分析師解讀？")
        print("1. 是，由 Claude 撰寫（需要 .env 的 ANTHROPIC_API_KEY，會產生少量 API 費用）")
        print("2. 是，使用確定性範本（不呼叫 API、不花費）")
        print("3. 否")
        default_ai = "1" if has_key else "2"
        ai_choice = input(f"請輸入 1、2 或 3，預設 {default_ai}：").strip() or default_ai
        if ai_choice == "1" and not has_key:
            print("找不到 ANTHROPIC_API_KEY，改用確定性範本。")
            ai_choice = "2"
        if ai_choice in ("1", "2"):
            from pipeline_ai import set_ai_mode
            set_ai_mode(enabled=True, model=args.ai_model, dry_run=(ai_choice == "2"))

    if False and direct_run:
        print("請選擇分析模式：")
        print("1. 個股深度分析（輸入任意股票代碼，輸出完整個股決策報告）")
        print("2. 多股票排名分析（自定義選股池，輸出排名與決策摘要）")
        mode_choice = input("請輸入 1 或 2，預設 1：").strip()
        args.mode = "single_diagnostic" if mode_choice != "2" else "train"
        args.diagnostic_scope = "common"

        print("請選擇市場：")
        print("1. TWSE 上市")
        print("2. TPEX 上櫃")
        market_choice = input("請輸入 1 或 2，預設 1：").strip()
        args.market = "TPEX" if market_choice == "2" else "TWSE"

        if args.mode == "single_diagnostic":
            user_symbol = input("請輸入要分析的股票代碼，例如 2330：").strip()
            args.symbol = user_symbol if user_symbol else "2330"
        else:
            print("請選擇選股方式：")
            print("  a. 使用今日成交金額 Top N（請輸入 N，預設 30）")
            print("  b. 手動輸入股票代碼清單（以逗號分隔，例如 2330,2317,2454）")
            print("  c. 兩者合併（Top N 加上手動輸入的股票）")
            method_choice = input("請輸入 a、b 或 c，預設 a：").strip().lower()
            if method_choice not in ("a", "b", "c"):
                method_choice = "a"

            top_n_symbols: list[str] = []
            manual_symbols: list[str] = []

            if method_choice in ("a", "c"):
                top_n_text = input("請輸入 Top N 數量，預設 30：").strip()
                args.top_n = int(top_n_text) if top_n_text.isdigit() else 30

            if method_choice in ("b", "c"):
                manual_input = input("請輸入股票代碼清單（以逗號分隔，例如 2330,2317,2454）：").strip()
                manual_symbols = [s.strip() for s in manual_input.split(",") if s.strip()]

            if method_choice == "a":
                args.symbol = "ALL"
            else:
                if method_choice == "c":
                    print(f"正在抓取 Top {args.top_n} 成交金額股票清單...")
                    _top_list = fetch_twse_top_turnover_stocks(limit=args.top_n)
                    top_n_symbols = _top_list["symbol"].astype(str).tolist()
                seen_syms: set[str] = set()
                merged_syms: list[str] = []
                for s in top_n_symbols + manual_symbols:
                    if s not in seen_syms:
                        seen_syms.add(s)
                        merged_syms.append(s)
                symbols_for_training = merged_syms
                args.symbol = "CUSTOM"

    if not direct_run and args.mode == "single_diagnostic" and (args.symbol is None or str(args.symbol).strip() == ""):
        args.symbol = prompt_single_stock_symbol()

    if args.mode == "single_diagnostic":
        raw_symbol = args.symbol if args.symbol is not None else "2330"
        if is_all_stock_symbol_input(raw_symbol):
            raise ValueError("個股深度分析需要指定單一股票代碼，例如 2330。")
        args.symbol = normalize_stock_symbol_input(raw_symbol)

    if args.mode == "backtest" and not backtest_mode_selected:
        raw_bt_symbol = str(args.symbol).strip() if args.symbol is not None else "ALL"
        backtest_symbol = None if is_all_stock_symbol_input(raw_bt_symbol) else normalize_stock_symbol_input(raw_bt_symbol)
        backtest_top_n = int(args.top_n or 30)

    if False and not direct_run and args.mode == "single_diagnostic" and (args.symbol is None or str(args.symbol).strip() == ""):
        user_symbol = input("請輸入要分析的股票代碼，例如 2330：").strip()
        args.symbol = user_symbol if user_symbol else "2330"

    if args.display_mode != "presentation":
        print("=" * 70)
        print("Clean Stock ML Pipeline")
        print("=" * 70)

    if args.mode is None:
        market = input(f"請輸入市場（TWSE / TPEX，預設 {CONFIG.market}）: ").strip().upper()
        stock_no = input(f"請輸入股票代碼；若要跑 TWSE 當日成交金額前 30 檔請輸入 ALL（預設 {CONFIG.stock_no}）: ").strip()
        top_n = None
    else:
        market = args.market.strip().upper()
        stock_no = args.symbol if args.symbol is not None else "ALL"
        top_n = args.top_n

    config = PipelineConfig(
        stock_no=stock_no if stock_no else CONFIG.stock_no,
        market=market if market else CONFIG.market,
        years=CONFIG.years,
        horizon=10,
        output_dir=CONFIG.output_dir,
        sleep_sec=CONFIG.sleep_sec,
        atr_period=CONFIG.atr_period,
        atr_multiplier=CONFIG.atr_multiplier,
        ob_lookback_days=CONFIG.ob_lookback_days,
        fib_lookback_days=CONFIG.fib_lookback_days,
        sr_lookback_days=CONFIG.sr_lookback_days,
        sr_tolerance_pct=CONFIG.sr_tolerance_pct,
        fvg_min_gap_pct=CONFIG.fvg_min_gap_pct,
        risk_penalty=CONFIG.risk_penalty,
        opportunity_threshold=CONFIG.opportunity_threshold,
        target_col=args.target_col,
        feature_set=args.feature_set,
        mode=args.mode,
        model_type=args.model_type,
        model_profile=args.model_profile,
        random_state=CONFIG.random_state,
        run_feature_importance=bool(args.run_feature_importance and not args.no_feature_importance),
        target_compare_cols=args.target_compare_cols,
        compare_feature_sets=args.compare_feature_sets,
        compare_model_types=args.compare_model_types,
        diagnostic_scope="common",
        skip_classification=args.skip_classification,
        display_mode=args.display_mode,
    )

    # 上櫃（2026-10-27）：市場選 TPEX 時，股票池與大盤指數仍以上市為基礎；
    # 個股自動判別上市／上櫃（判別不出時視為上櫃），Top N 則改為上市＋上櫃合併排行。
    if config.market.upper() == "TPEX":
        from dataclasses import replace as _replace
        sym = str(config.stock_no).strip()
        if sym and sym.upper() not in {"ALL", "*", "全部", "CUSTOM"}:
            mkt = lookup_symbol_market(sym, "TPEX")
            register_symbol_markets({sym: mkt})
            print(f"{sym} 判別為{'上櫃' if mkt == 'TPEX' else '上市'}股票；共同模型股票池沿用成交金額排行，大盤過濾使用加權指數。")
        else:
            set_include_otc(True)
            print("股票池改為上市＋上櫃合併，依當日成交金額排行。")
        config = _replace(config, market="TWSE")

    if args.mode == "target_compare":
        if args.input_data:
            input_path = Path(args.input_data)
        else:
            input_path = find_latest_training_data_path()
            if input_path is None:
                print("找不到 training_data，請先執行 sensitivity 或 train 模式建立資料。")
                return
            print(f"未提供 input_data，已自動使用最新檔案：{input_path}")

        if not input_path.exists():
            print(f"找不到 input_data：{input_path}")
            return
        run_target_score_comparison_experiment(config, input_path)
        return

    if backtest_mode_selected or args.mode == "backtest":
        bt_mode = "single" if backtest_symbol else "multi"
        run_backtest_mode(
            config,
            top_n=backtest_top_n,
            target_symbol=backtest_symbol,
            mode=bt_mode,
            run_profile_experiments=args.run_profile_experiments,
            investor_mode=args.investor_mode or False,
            daily_top30_universe=args.daily_top30_universe,
            daily_top30_rank_metric=args.daily_top30_rank_metric,
            optimize_score_weights=args.optimize_score_weights,
            optimization_tuning_folds=args.optimization_tuning_folds,
            optimization_maxiter=args.optimization_maxiter,
            optimization_popsize=args.optimization_popsize,
            optimization_regularization=args.optimization_regularization,
            optimization_min_trades=args.optimization_min_trades,
            optimization_ai_range_audit=args.optimization_ai_range_audit,
            optimization_max_weight=args.optimization_max_weight,
            optimization_entry_threshold_min=args.optimization_entry_threshold_min,
            optimization_entry_threshold_max=args.optimization_entry_threshold_max,
            optimization_evaluation_end_date=args.optimization_evaluation_end_date,
        )
        return

    if args.mode == "single_diagnostic":
        run_single_stock_diagnostic(config, top_n=top_n or 30, investor_mode=True if args.investor_mode is None else args.investor_mode)
        return

    if args.mode == "dual_diagnostic":
        run_dual_model_diagnostic(config, top_n=top_n or 30)
        return

    if args.mode == "sensitivity":
        if config.market != "TWSE":
            raise ValueError("局部範圍敏感度測試目前只支援 TWSE")
        selected_profile_names = parse_sensitivity_profiles(args.sensitivity_profiles)
        run_local_range_sensitivity_experiment(
            config,
            top_n=top_n or 30,
            selected_profile_names=selected_profile_names,
        )
        return

    if args.mode == "sensitivity_analysis":
        run_sensitivity_analysis(config, top_n=top_n or 30)
        return

    if args.mode == "experiment":
        if args.input_data:
            input_path = Path(args.input_data)
        else:
            candidates = list(OUTPUT_DIR.glob("*_training_data.csv")) + list(OUTPUT_DIR.glob("*_combined_training_data.csv"))
            if not candidates:
                print("找不到 combined_training_data，請先執行 train 模式建立資料。")
                return
            input_path = max(candidates, key=lambda path: path.stat().st_mtime)
            print(f"未提供 input_data，已自動使用最新檔案：{input_path}")

        if not input_path.exists():
            print(f"找不到 input_data：{input_path}")
            return
        run_experiment_from_processed_data(config, input_path)
        return

    if stock_no.strip().upper() in {"ALL", "*", "全部", "CUSTOM"}:
        if stock_no.strip().upper() in {"ALL", "*", "全部"} and config.market != "TWSE":
            raise ValueError("批次全市場目前只支援 TWSE，請市場輸入 TWSE 或直接留空。")

        if top_n is None and symbols_for_training is None:
            limit_text = input("要分析成交金額排行前幾檔？直接 Enter 代表前 30 檔：").strip()
            top_n = int(limit_text) if limit_text else 30

        if args.mode == "predict":
            run_common_model_predict(config, top_n=top_n or 30)
        else:
            if args.mode == "train":
                print("使用模式：train")
            run_common_model_train(config, top_n=top_n or 30, symbols_override=symbols_for_training, investor_mode=True if args.investor_mode is None else args.investor_mode)
    else:
        run_pipeline(config)

def _smoke_test() -> None:
    """Sanity-check add_kdj() and add_order_block_asof_features() with synthetic data."""
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 30

    # ── add_kdj ──────────────────────────────────────────────────
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    kdj_df = pd.DataFrame({
        "close": close,
        "high":  close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":   close * (1 - np.abs(rng.normal(0, 0.005, n))),
    })
    r_kdj    = add_kdj(kdj_df)
    tail_kdj = r_kdj[["K", "D", "J"]].tail(5)
    print("\n[add_kdj] last 5 rows — K, D, J:")
    print(tail_kdj.to_string())
    for col in ["K", "D", "J"]:
        assert not tail_kdj[col].isna().all(), f"FAIL: {col} is all NaN"
    print("-> K/D/J OK")

    # ── add_order_block_asof_features ────────────────────────────
    dates    = pd.date_range("2024-01-01", periods=n, freq="B")
    close_ob = np.cumprod(1 + rng.normal(0.001, 0.01, n)) * 100
    open_ob  = close_ob.copy() * (1 + rng.normal(0, 0.003, n))

    # Row 20: bearish OB candle  →  Row 21: +4% bullish impulse with volume spike
    open_ob[20]  = close_ob[19] * 1.008   # open > close → bearish candle
    close_ob[20] = close_ob[19] * 0.993
    close_ob[21] = close_ob[20] * 1.04    # strong up move on next bar

    high_ob = np.maximum(close_ob, open_ob) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low_ob  = np.minimum(close_ob, open_ob) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol_ob  = rng.integers(100_000, 500_000, n).astype(float)
    vol_ob[21] = vol_ob[16:21].mean() * 4.0   # volume spike on impulse bar

    ret_1d   = np.concatenate([[0.0], np.diff(close_ob) / close_ob[:-1]])
    vol_ma5  = pd.Series(vol_ob).rolling(5,  min_periods=1).mean().to_numpy()
    vol_ma20 = pd.Series(vol_ob).rolling(20, min_periods=1).mean().to_numpy()

    ob_df = pd.DataFrame({
        "date":      dates,
        "close":     close_ob,
        "open":      open_ob,
        "high":      high_ob,
        "low":       low_ob,
        "volume":    vol_ob,
        "return_1d": ret_1d,
        "vol_ma5":   vol_ma5,
        "vol_ma20":  vol_ma20,
        "fvg_type":  pd.NA,
    })

    r_ob    = add_order_block_asof_features(ob_df)
    tail_ob = r_ob["recent_2w_ob_type"].tail(5)
    print("\n[add_order_block_asof_features] last 5 rows — recent_2w_ob_type:")
    print(tail_ob.to_string())
    assert not tail_ob.isna().all(), "FAIL: recent_2w_ob_type tail is all NaN"
    valid_types = {"bullish", "bearish", "fallback_range", "none"}
    invalid = tail_ob.dropna()[~tail_ob.dropna().isin(valid_types)]
    assert invalid.empty, f"FAIL: unexpected ob_type values: {invalid.values}"
    print("-> recent_2w_ob_type OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
