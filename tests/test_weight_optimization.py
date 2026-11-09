import pandas as pd
import pytest
import numpy as np

from pipeline_weight_optimization import (
    DecisionParameters,
    OptimizationConfig,
    _write_json,
    apply_decision_parameters,
    normalize_parameters,
    select_fold_period,
)


def _parameters() -> DecisionParameters:
    return DecisionParameters(0.30, 0.20, 0.25, 0.25, 0.55)


def test_apply_parameters_does_not_turn_exit_rows_into_entries():
    data = pd.DataFrame(
        {
            "safety_score": [1.0, 1.0],
            "return_score": [1.0, 1.0],
            "rule_filter_score": [1.0, 1.0],
            "risk_reward_score": [1.0, 1.0],
            "is_oos_prediction": [True, False],
            "entry_plan_override": [False, True],
        }
    )

    result = apply_decision_parameters(data, _parameters())

    assert result.loc[0, "final_decision_score"] == pytest.approx(1.0)
    assert bool(result.loc[0, "optimized_trade_eligible"])
    assert result.loc[1, "final_decision_score"] == pytest.approx(0.0)
    assert not bool(result.loc[1, "optimized_trade_eligible"])
    assert not bool(result.loc[1, "entry_plan_override"])


def test_normalize_parameters_enforces_safety_at_least_return():
    config = OptimizationConfig()
    parameters = normalize_parameters([3.0, 2.0, 2.5, 2.5], 0.60, config)

    assert sum(parameters.weights) == pytest.approx(1.0)
    assert parameters.safety_weight >= parameters.return_weight

    with pytest.raises(ValueError, match="safety"):
        normalize_parameters([1.0, 2.0, 3.0, 3.0], 0.60, config)


def test_select_fold_period_disables_other_fold_signals_but_keeps_exit_rows():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "symbol": ["2330", "2330", "2330"],
            "oos_fold": [1, 2, None],
            "is_oos_prediction": [True, True, False],
            "entry_plan_override": [False, True, True],
        }
    )

    result = select_fold_period(data, [1])

    assert bool(result.loc[result["oos_fold"] == 1, "is_oos_prediction"].iloc[0])
    assert not bool(result.loc[result["oos_fold"] == 2, "is_oos_prediction"].iloc[0])
    assert not bool(result.loc[result["oos_fold"] == 2, "entry_plan_override"].iloc[0])
    assert len(result) == 3


def test_checkpoint_json_handles_numpy_values_and_replaces_atomically(tmp_path):
    path = tmp_path / "checkpoint.json"

    _write_json(path, {"weight": np.float64(0.3), "feasible": np.bool_(True)})

    assert path.exists()
    assert path.read_text(encoding="utf-8") == '{\n  "weight": 0.3,\n  "feasible": true\n}'
    assert not path.with_suffix(".json.tmp").exists()


def test_checkpoint_json_serializes_dataframe_metrics(tmp_path):
    path = tmp_path / "checkpoint.json"
    diagnostics = pd.DataFrame({"date": pd.to_datetime(["2026-10-07"]), "rejected": [2]})

    _write_json(path, {"entry_diagnostics_daily": diagnostics})

    assert '"entry_diagnostics_daily"' in path.read_text(encoding="utf-8")
