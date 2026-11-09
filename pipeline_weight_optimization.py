"""Constrained, walk-forward score-weight optimization for portfolio backtests.

This module deliberately reuses ``run_portfolio_backtest``.  It only changes
the four components of ``final_decision_score`` and the score entry threshold;
all existing vetoes, D+1 execution, position sizing, costs, and exit rules
remain owned by the normal backtest engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pipeline_backtest_engine import run_portfolio_backtest
from pipeline_data import PipelineConfig
from pipeline_risk import get_final_score_profile_info


SCORE_COLUMNS = (
    "safety_score",
    "return_score",
    "rule_filter_score",
    "risk_reward_score",
)


@dataclass(frozen=True)
class DecisionParameters:
    safety_weight: float
    return_weight: float
    rule_weight: float
    risk_reward_weight: float
    entry_threshold: float

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (
            self.safety_weight,
            self.return_weight,
            self.rule_weight,
            self.risk_reward_weight,
        )

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class FoldMetrics:
    net_annual_return: float
    max_drawdown: float
    trade_count: int
    fold: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationConfig:
    tuning_folds: int = 8
    maxiter: int = 12
    popsize: int = 5
    risk_penalty: float = 1.5
    stability_penalty: float = 0.5
    regularization: float = 0.05
    min_total_trades: int = 30
    max_weight: float = 0.60
    entry_threshold_min: float = 0.50
    entry_threshold_max: float = 0.80
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.tuning_folds < 1:
            raise ValueError("tuning_folds must be positive")
        if self.maxiter < 1 or self.popsize < 1:
            raise ValueError("DE maxiter and popsize must be positive")
        if not 0.25 <= self.max_weight <= 1.0:
            raise ValueError("max_weight must be between 0.25 and 1.0")
        if not 0 <= self.entry_threshold_min < self.entry_threshold_max <= 1:
            raise ValueError("entry threshold bounds are invalid")


@dataclass(frozen=True)
class EvaluationSummary:
    objective: float
    mean_net_annual_return: float
    mean_max_drawdown: float
    utility_std: float
    total_trades: int
    active_folds: int
    feasible: bool
    infeasible_reason: str = ""

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def normalize_parameters(
    raw_weights: Iterable[float],
    entry_threshold: float,
    config: OptimizationConfig,
) -> DecisionParameters:
    """Normalize DE variables and enforce the declared search constraints."""
    raw = np.asarray(list(raw_weights), dtype=float)
    if raw.shape != (4,) or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("four positive finite raw weights are required")
    weights = raw / raw.sum()
    if np.any(weights > config.max_weight + 1e-12):
        raise ValueError(f"weight exceeds max_weight={config.max_weight}")
    if weights[0] + 1e-12 < weights[1]:
        raise ValueError("safety weight must be at least return weight")
    if not config.entry_threshold_min <= entry_threshold <= config.entry_threshold_max:
        raise ValueError("entry threshold is outside configured bounds")
    return DecisionParameters(
        safety_weight=float(weights[0]),
        return_weight=float(weights[1]),
        rule_weight=float(weights[2]),
        risk_reward_weight=float(weights[3]),
        entry_threshold=float(entry_threshold),
    )


def apply_decision_parameters(
    data: pd.DataFrame,
    parameters: DecisionParameters,
) -> pd.DataFrame:
    """Recalculate final score only for genuine OOS prediction rows.

    Unscored rows were appended for position exits after a security leaves the
    daily Top-N universe.  They must never become fresh entries merely because
    component-score fallbacks are available.
    """
    missing = [column for column in SCORE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"cannot optimize missing score columns: {missing}")

    out = data.copy()
    prediction_mask = out.get(
        "is_oos_prediction", pd.Series(True, index=out.index)
    ).fillna(False).astype(bool)
    components = [
        pd.to_numeric(out[column], errors="coerce").fillna(0.5)
        for column in SCORE_COLUMNS
    ]
    score = sum(weight * value for weight, value in zip(parameters.weights, components))
    out["final_decision_score"] = score.clip(0.0, 1.0).where(prediction_mask, 0.0)
    out["optimized_trade_eligible"] = (
        prediction_mask & out["final_decision_score"].ge(parameters.entry_threshold)
    )
    if "entry_plan_override" in out.columns:
        out.loc[~prediction_mask, "entry_plan_override"] = False
    return out


def _build_signals(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    symbols = sorted(data["symbol"].dropna().astype(str).unique().tolist())
    return {
        symbol: data.loc[data["symbol"].astype(str) == symbol].copy()
        for symbol in symbols
    }


def _fold_dates(data: pd.DataFrame, fold: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    fold_numbers = pd.to_numeric(data["oos_fold"], errors="coerce")
    prediction_mask = data.get(
        "is_oos_prediction", pd.Series(True, index=data.index)
    ).fillna(False).astype(bool)
    dates = pd.to_datetime(data.loc[(fold_numbers == fold) & prediction_mask, "date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"fold {fold} has no OOS prediction rows")
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def select_fold_period(data: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame:
    """Keep one fold's entry signals plus a 60-business-day exit buffer."""
    if "oos_fold" not in data.columns:
        raise ValueError("OOS data must contain oos_fold for weight optimization")
    requested = sorted({int(fold) for fold in folds})
    if not requested:
        raise ValueError("at least one fold is required")
    starts_ends = [_fold_dates(data, fold) for fold in requested]
    start = min(pair[0] for pair in starts_ends)
    signal_end = max(pair[1] for pair in starts_ends)
    end = signal_end + pd.offsets.BDay(60)

    out = data.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.loc[out["date"].between(start, end)].copy()
    fold_numbers = pd.to_numeric(out["oos_fold"], errors="coerce")
    prediction_mask = out.get(
        "is_oos_prediction", pd.Series(True, index=out.index)
    ).fillna(False).astype(bool)
    allowed_signal = prediction_mask & fold_numbers.isin(requested)
    out.loc[~allowed_signal, "is_oos_prediction"] = False
    if "entry_plan_override" in out.columns:
        out.loc[~allowed_signal, "entry_plan_override"] = False
    return out


def _fold_metrics(
    data: pd.DataFrame,
    config: PipelineConfig,
    market_df: pd.DataFrame | None,
    fold: int,
) -> FoldMetrics:
    signals = _build_signals(data)
    _, _, metrics = run_portfolio_backtest(
        all_stock_signals=signals,
        config=config,
        market_df=market_df,
        expected_symbols=list(signals),
        investor_mode=True,
        save_snapshot=False,
    )
    annual_return = pd.to_numeric(pd.Series([metrics.get("cagr")]), errors="coerce").iloc[0]
    if pd.isna(annual_return):
        annual_return = pd.to_numeric(pd.Series([metrics.get("total_return", 0.0)]), errors="coerce").iloc[0]
    drawdown = pd.to_numeric(pd.Series([metrics.get("max_drawdown", 0.0)]), errors="coerce").iloc[0]
    trades = int(metrics.get("total_entries", 0) or 0)
    return FoldMetrics(
        net_annual_return=float(annual_return) if pd.notna(annual_return) else 0.0,
        max_drawdown=abs(float(drawdown)) if pd.notna(drawdown) else 0.0,
        trade_count=trades,
        fold=int(fold),
    )


class WalkForwardWeightEvaluator:
    """Evaluates one parameter set only on pre-selected OOS tuning folds."""

    def __init__(
        self,
        oos_df: pd.DataFrame,
        base_config: PipelineConfig,
        market_df: pd.DataFrame | None,
        tuning_folds: Sequence[int],
        objective_config: OptimizationConfig,
        target_weights: Sequence[float],
    ) -> None:
        self.oos_df = oos_df
        self.base_config = base_config
        self.market_df = market_df
        self.tuning_folds = tuple(int(fold) for fold in tuning_folds)
        self.objective_config = objective_config
        self.target_weights = np.asarray(target_weights, dtype=float)
        self._periods = {
            fold: select_fold_period(oos_df, [fold]) for fold in self.tuning_folds
        }

    def evaluate(self, parameters: DecisionParameters) -> tuple[EvaluationSummary, list[FoldMetrics]]:
        optimized_config = replace(
            self.base_config,
            backtest_entry_threshold=parameters.entry_threshold,
        )
        fold_metrics = [
            _fold_metrics(
                apply_decision_parameters(period, parameters),
                optimized_config,
                self.market_df,
                fold,
            )
            for fold, period in self._periods.items()
        ]
        returns = np.asarray([item.net_annual_return for item in fold_metrics])
        drawdowns = np.asarray([item.max_drawdown for item in fold_metrics])
        utilities = returns - self.objective_config.risk_penalty * drawdowns
        total_trades = int(sum(item.trade_count for item in fold_metrics))
        active_folds = int(sum(item.trade_count > 0 for item in fold_metrics))
        min_active_folds = min(4, len(fold_metrics))
        reasons = []
        if total_trades < self.objective_config.min_total_trades:
            reasons.append(f"total trades {total_trades} < {self.objective_config.min_total_trades}")
        if active_folds < min_active_folds:
            reasons.append(f"active folds {active_folds} < {min_active_folds}")
        regularization = self.objective_config.regularization * float(
            np.square(np.asarray(parameters.weights) - self.target_weights).sum()
        )
        utility_std = float(utilities.std(ddof=0))
        objective = float(utilities.mean() - self.objective_config.stability_penalty * utility_std - regularization)
        feasible = not reasons
        return EvaluationSummary(
            objective=objective if feasible else float("-inf"),
            mean_net_annual_return=float(returns.mean()),
            mean_max_drawdown=float(drawdowns.mean()),
            utility_std=utility_std,
            total_trades=total_trades,
            active_folds=active_folds,
            feasible=feasible,
            infeasible_reason="; ".join(reasons),
        ), fold_metrics


def _optimize_de(
    evaluator: WalkForwardWeightEvaluator,
    config: OptimizationConfig,
    *,
    generation_checkpoint_path: Path | None = None,
    checkpoint_context: dict | None = None,
) -> tuple[DecisionParameters, EvaluationSummary, list[FoldMetrics]]:
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError("scipy is required for Differential Evolution optimization") from exc

    generation = 0
    best_seen: dict[str, object] = {}

    def objective(vector: np.ndarray) -> float:
        try:
            parameters = normalize_parameters(vector[:4], float(vector[4]), config)
            summary, metrics = evaluator.evaluate(parameters)
            if summary.feasible:
                previous = best_seen.get("summary")
                if previous is None or summary.objective > previous.objective:
                    best_seen["parameters"] = parameters
                    best_seen["summary"] = summary
                    best_seen["metrics"] = metrics
                return -summary.objective
            return 1e6
        except (ValueError, FloatingPointError):
            return 1e6

    def progress(_vector: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        generation += 1
        print(f"DE generation {generation}/{config.maxiter}, convergence={convergence:.6f}", flush=True)
        if generation_checkpoint_path is not None:
            checkpoint = dict(checkpoint_context or {})
            checkpoint.update({
                "status": "de_search_in_progress",
                "generation": generation,
                "maxiter": config.maxiter,
                "convergence": float(convergence),
            })
            parameters = best_seen.get("parameters")
            summary = best_seen.get("summary")
            metrics = best_seen.get("metrics")
            if isinstance(parameters, DecisionParameters):
                checkpoint["optimized_parameters"] = parameters.as_dict()
            if isinstance(summary, EvaluationSummary):
                checkpoint["tuning_summary"] = summary.as_dict()
            if isinstance(metrics, list):
                checkpoint["tuning_fold_metrics"] = [item.as_dict() for item in metrics]
            _write_json(generation_checkpoint_path, checkpoint)
        return False

    result = differential_evolution(
        objective,
        bounds=[(1e-4, 1.0)] * 4 + [(config.entry_threshold_min, config.entry_threshold_max)],
        maxiter=config.maxiter,
        popsize=config.popsize,
        seed=config.random_seed,
        polish=False,
        workers=1,
        updating="immediate",
        callback=progress,
    )
    parameters = normalize_parameters(result.x[:4], float(result.x[4]), config)
    summary, metrics = evaluator.evaluate(parameters)
    if not summary.feasible:
        raise RuntimeError(f"DE did not find a feasible solution: {summary.infeasible_reason}")
    return parameters, summary, metrics


def _serialize(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    """Atomically persist an optimization checkpoint or final summary."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_serialize),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_claude_search_range_audit(
    *,
    output_dir: Path,
    baseline: DecisionParameters,
    optimized: DecisionParameters,
    tuning_summary: EvaluationSummary,
    tuning_metrics: Sequence[FoldMetrics],
    objective_config: OptimizationConfig,
) -> Path:
    """Write a non-binding Claude proposal for a *future* optimizer run.

    The prompt receives tuning-fold information only.  The response is never
    read by the current DE run, so an LLM cannot silently alter weights or use
    the held-out outer folds while this experiment is being evaluated.
    """
    from pipeline_ai import ClaudeClient, extract_json, load_env_file, ROOT_DIR

    load_env_file(ROOT_DIR / ".env")
    facts = {
        "baseline_parameters": baseline.as_dict(),
        "optimized_parameters": optimized.as_dict(),
        "tuning_summary": tuning_summary.as_dict(),
        "tuning_fold_metrics": [item.as_dict() for item in tuning_metrics],
        "current_search_bounds": {
            "max_weight": objective_config.max_weight,
            "entry_threshold_min": objective_config.entry_threshold_min,
            "entry_threshold_max": objective_config.entry_threshold_max,
        },
    }
    system = (
        "You are a research-method audit assistant. Return JSON only. You may "
        "recommend a search range for a FUTURE experiment, but you must not "
        "recommend trades, securities, or exact final score weights. Do not use "
        "or assume any holdout performance."
    )
    user = (
        "Review these walk-forward TUNING-only facts. Propose conservative bounds "
        "for a future DE run. JSON schema: "
        '{"proposal":{"max_weight":number,"entry_threshold_min":number,'
        '"entry_threshold_max":number},"rationale":[string],"warnings":[string]}. '
        "Limits: 0.25<=max_weight<=0.75, 0.40<=entry_threshold_min<=0.70, "
        "0.55<=entry_threshold_max<=0.90, and min < max. Facts:\n"
        + json.dumps(facts, ensure_ascii=False, default=_serialize)
    )
    client = ClaudeClient(
        model="claude-sonnet-5-5",
        max_tokens=700,
        timeout_sec=90,
        purpose="optimization_range_audit",
    )
    response = client.complete(system=system, user=user, tag="score_weight_de")
    parsed = extract_json(response.text)
    proposal = parsed.get("proposal", {}) if isinstance(parsed, dict) else {}
    try:
        max_weight = float(proposal["max_weight"])
        threshold_min = float(proposal["entry_threshold_min"])
        threshold_max = float(proposal["entry_threshold_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Claude range audit did not return the required proposal") from exc
    if not (
        0.25 <= max_weight <= 0.75
        and 0.40 <= threshold_min <= 0.70
        and 0.55 <= threshold_max <= 0.90
        and threshold_min < threshold_max
    ):
        raise ValueError("Claude proposal is outside the hard safety bounds")
    audit = {
        "binding": False,
        "scope": "tuning folds only; never applied to the optimizer that produced these facts",
        "proposal": {
            "max_weight": max_weight,
            "entry_threshold_min": threshold_min,
            "entry_threshold_max": threshold_max,
        },
        "rationale": parsed.get("rationale", []),
        "warnings": parsed.get("warnings", []),
        "llm_log_path": response.log_path,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    path = output_dir / f"claude_search_range_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_serialize), encoding="utf-8")
    return path


def run_score_weight_optimization(
    oos_df: pd.DataFrame,
    config: PipelineConfig,
    market_df: pd.DataFrame | None,
    *,
    tuning_folds: int = 8,
    maxiter: int = 12,
    popsize: int = 5,
    regularization: float = 0.05,
    min_total_trades: int = 30,
    ai_range_audit: bool = False,
    max_weight: float = 0.60,
    entry_threshold_min: float = 0.50,
    entry_threshold_max: float = 0.80,
) -> dict:
    """Run DE on early OOS folds and compare once on later frozen folds."""
    if oos_df.empty or "oos_fold" not in oos_df.columns:
        raise ValueError("score optimization requires non-empty OOS data with oos_fold")
    all_folds = sorted(
        pd.to_numeric(oos_df["oos_fold"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if len(all_folds) < 3:
        raise ValueError("at least three OOS folds are required")
    tuning_folds = min(int(tuning_folds), len(all_folds) - 1)
    tune = all_folds[:tuning_folds]
    outer = all_folds[tuning_folds:]
    profile = get_final_score_profile_info()
    profile_weights = profile["weights"]
    baseline = DecisionParameters(
        safety_weight=float(profile_weights["safety_score"]),
        return_weight=float(profile_weights["return_score"]),
        rule_weight=float(profile_weights["rule_filter_score"]),
        risk_reward_weight=float(profile_weights["risk_reward_score"]),
        entry_threshold=float(getattr(config, "backtest_entry_threshold", 0.55)),
    )
    objective_config = OptimizationConfig(
        tuning_folds=tuning_folds,
        maxiter=maxiter,
        popsize=popsize,
        regularization=regularization,
        min_total_trades=min_total_trades,
        max_weight=max_weight,
        entry_threshold_min=entry_threshold_min,
        entry_threshold_max=entry_threshold_max,
    )
    evaluator = WalkForwardWeightEvaluator(
        oos_df=oos_df,
        base_config=config,
        market_df=market_df,
        tuning_folds=tune,
        objective_config=objective_config,
        target_weights=baseline.weights,
    )
    output_dir = Path(config.output_dir) / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = output_dir / f"de_score_weights_{timestamp}"
    checkpoint_path = Path(f"{prefix}_checkpoint.json")
    checkpoint_context = {
        "method": "differential_evolution",
        "score_profile_baseline": profile["profile"],
        "tuning_folds": tune,
        "outer_folds": outer,
        "objective_config": asdict(objective_config),
        "baseline_parameters": baseline.as_dict(),
    }
    print(
        f"權重最佳化：前 {len(tune)} folds 調參（{tune}），"
        f"後 {len(outer)} folds 保留測試（{outer}）。"
    )
    best, best_summary, tuning_metrics = _optimize_de(
        evaluator,
        objective_config,
        generation_checkpoint_path=checkpoint_path,
        checkpoint_context=checkpoint_context,
    )

    checkpoint = {
        **checkpoint_context,
        "status": "de_completed_outer_pending",
        "optimized_parameters": best.as_dict(),
        "tuning_summary": best_summary.as_dict(),
        "tuning_fold_metrics": [item.as_dict() for item in tuning_metrics],
    }
    _write_json(checkpoint_path, checkpoint)
    print(f"DE checkpoint 已輸出：{checkpoint_path}")

    outer_data = select_fold_period(oos_df, outer)
    optimized_outer = apply_decision_parameters(outer_data, best)
    baseline_outer = apply_decision_parameters(outer_data, baseline)
    optimized_config = replace(config, backtest_entry_threshold=best.entry_threshold)
    baseline_config = replace(config, backtest_entry_threshold=baseline.entry_threshold)
    optimized_equity, optimized_trades, optimized_metrics = run_portfolio_backtest(
        _build_signals(optimized_outer), optimized_config, market_df=market_df,
        expected_symbols=sorted(optimized_outer["symbol"].dropna().astype(str).unique()),
        investor_mode=True, save_snapshot=False,
    )
    baseline_equity, baseline_trades, baseline_metrics = run_portfolio_backtest(
        _build_signals(baseline_outer), baseline_config, market_df=market_df,
        expected_symbols=sorted(baseline_outer["symbol"].dropna().astype(str).unique()),
        investor_mode=True, save_snapshot=False,
    )

    tuning_table = pd.DataFrame([item.as_dict() for item in tuning_metrics])
    tuning_table.to_csv(f"{prefix}_tuning_folds.csv", index=False, encoding="utf-8-sig")
    optimized_equity.to_csv(f"{prefix}_optimized_outer_equity.csv", index=False, encoding="utf-8-sig")
    optimized_trades.to_csv(f"{prefix}_optimized_outer_trades.csv", index=False, encoding="utf-8-sig")
    baseline_equity.to_csv(f"{prefix}_baseline_outer_equity.csv", index=False, encoding="utf-8-sig")
    baseline_trades.to_csv(f"{prefix}_baseline_outer_trades.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "complete",
        "method": "differential_evolution",
        "score_profile_baseline": profile["profile"],
        "tuning_folds": tune,
        "outer_folds": outer,
        "objective_config": asdict(objective_config),
        "baseline_parameters": baseline.as_dict(),
        "optimized_parameters": best.as_dict(),
        "tuning_summary": best_summary.as_dict(),
        "tuning_fold_metrics": [item.as_dict() for item in tuning_metrics],
        "baseline_outer_metrics": baseline_metrics,
        "optimized_outer_metrics": optimized_metrics,
        "artifacts": {
            "tuning_folds": f"{prefix}_tuning_folds.csv",
            "optimized_outer_equity": f"{prefix}_optimized_outer_equity.csv",
            "optimized_outer_trades": f"{prefix}_optimized_outer_trades.csv",
            "baseline_outer_equity": f"{prefix}_baseline_outer_equity.csv",
            "baseline_outer_trades": f"{prefix}_baseline_outer_trades.csv",
        },
    }
    if ai_range_audit:
        try:
            audit_path = write_claude_search_range_audit(
                output_dir=output_dir,
                baseline=baseline,
                optimized=best,
                tuning_summary=best_summary,
                tuning_metrics=tuning_metrics,
                objective_config=objective_config,
            )
            result["claude_search_range_audit"] = str(audit_path)
            print(f"Claude 搜尋範圍建議已輸出（不會套用到本次結果）：{audit_path}")
        except Exception as exc:
            result["claude_search_range_audit_error"] = str(exc)
            print(f"Claude 搜尋範圍建議失敗（不影響 DE 結果）：{exc}")
    summary_path = Path(f"{prefix}_summary.json")
    _write_json(summary_path, result)
    print(f"DE 最佳參數：{best.as_dict()}")
    print(f"調參目標值：{best_summary.objective:.6f}")
    print(f"保留 OOS baseline 績效：{baseline_metrics}")
    print(f"保留 OOS optimized 績效：{optimized_metrics}")
    print(f"最佳化比較結果：{summary_path}")
    return result


__all__ = [
    "DecisionParameters",
    "EvaluationSummary",
    "FoldMetrics",
    "OptimizationConfig",
    "apply_decision_parameters",
    "normalize_parameters",
    "run_score_weight_optimization",
    "select_fold_period",
    "write_claude_search_range_audit",
]
