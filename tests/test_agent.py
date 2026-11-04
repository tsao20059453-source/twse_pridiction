"""決策代理測試（假 Claude、假分析結果，不花錢、不連網、不訓練模型）。執行：python tests/test_agent.py"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_agent as G  # noqa: E402

BASE = json.loads((ROOT / "tests" / "fixtures" / "row_2330_20260918.json").read_text(encoding="utf-8"))


def make_row(symbol, **kw):
    r = copy.deepcopy(BASE)
    r.update(symbol=symbol, **kw)
    return r


ROW_2330 = make_row("2330")                                            # 否決交易（R/R 不足）
POOL = pd.DataFrame([
    ROW_2330,
    make_row("2449", final_decision_score=0.70, trade_veto_flag=False, trade_veto_reason="", suggested_position_pct=0.08,
             final_decision_label="Bullish Watch", decision_action="Priority watch", entry_plan_override=False, close=262.5),
    make_row("8046", final_decision_score=0.49, trade_veto_flag=False, trade_veto_reason="", suggested_position_pct=0.0,
             final_decision_label="等待回測進場", entry_plan_override=True, close=1265.0,
             support_entry_zone_low=1180.0, support_entry_zone_high=1200.0),
])
MARKET = pd.DataFrame({"date": pd.bdate_range("2026-08-01", periods=40),
                       "market_close": [40000 + 50 * i for i in range(40)]})
CALLS = {"analyze": 0}


def fake_analyze(config, symbol, top_n=30):
    CALLS["analyze"] += 1
    row = POOL[POOL["symbol"] == symbol].iloc[0].to_dict()
    return row, POOL.copy()


def fake_market(config):
    return MARKET.copy()


class ScriptedClaude:
    """依序回傳預先寫好的回應；每個回應是一串 (工具名, 參數)。"""
    def __init__(self, turns):
        self.turns, self.i, self.bodies = turns, 0, []

    def __call__(self, body):
        self.bodies.append(body)
        calls = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        content = [{"type": "tool_use", "id": f"t{self.i}_{k}", "name": n, "input": a} for k, (n, a) in enumerate(calls)]
        return {"content": content, "usage": {"input_tokens": 1000, "output_tokens": 200}, "stop_reason": "tool_use"}


def run(post_fn, question="2330 現在可以買嗎？有沒有其他選擇？", dry=False):
    G.AGENT_SETTINGS["dry_run"] = dry
    CALLS["analyze"] = 0
    return G.answer_question(question, config=None, analyze_fn=fake_analyze, market_fn=fake_market, post_fn=post_fn)


def good_answer(s_facts_hint="F1"):
    return {"conclusion": "系統目前對 2330 的決策是「否決交易」，不建議以目前價格進場；股票池中 2449 目前有給倉位，可列入觀察。",
            "per_stock": ["2330：否決交易，第一止盈 R/R 只有 1.29，低於 1.5 的門檻。"],
            "reasons": ["大盤位於 20 日均線之上，大盤過濾通過。"],
            "triggers": ["價格回到可接受進場區間時，R/R 與停損條件才可能同時成立。"],
            "alternatives": ["2449：系統決策為穩健偏多，建議倉位 8.00%。"],
            "risks": ["價位每日更新，過去績效不代表未來。"]}


PLAN = [[("get_market_regime", {}), ("analyze_stock", {"symbol": "2330"})],
        [("explain_decision", {"symbol": "2330"}), ("what_if_entry", {"symbol": "2330"})],
        [("list_alternatives", {"max_n": 3})]]


def test_planner_happy_path():
    s = run(ScriptedClaude(PLAN + [[("finalize", {"answer": good_answer()})]]))
    assert s.mode == "agent" and s.final["conclusion"].startswith("系統目前"), (s.mode, s.steps[-3:])
    tools = [st["tool"] for st in s.steps]
    assert tools[:5] == ["get_market_regime", "analyze_stock", "explain_decision", "what_if_entry", "list_alternatives"], tools
    assert CALLS["analyze"] == 1
    print("✓ Claude 規劃：大盤 → 分析 → 解釋 → 試算 → 替代選項 → finalize，Critic 通過")


def test_critic_rejects_then_accepts():
    bad = good_answer()
    bad["conclusion"] = "2330 目前否決交易，但建議買進，目標價 2800。"
    s = run(ScriptedClaude(PLAN + [[("finalize", {"answer": bad})], [("finalize", {"answer": good_answer()})]]))
    rej = [st for st in s.steps if st["tool"] == "finalize_rejected"]
    assert rej and s.mode == "agent" and "2800" not in s.final["conclusion"], s.steps[-3:]
    issues = " ".join(rej[0]["result"]["issues"])
    assert "2800" in issues and "沒有給這檔股票倉位" in issues, issues
    print("✓ Critic 擋下「否決卻寫建議買進」與編造目標價，退回後重寫通過")


def test_finalize_before_analysis_is_blocked():
    s = run(ScriptedClaude([[("finalize", {"answer": good_answer()})]] + PLAN + [[("finalize", {"answer": good_answer()})]]))
    assert s.steps[0]["tool"] == "finalize_rejected" and s.mode == "agent"
    print("✓ 閘門：尚未分析任何股票就 finalize 會被退回")


def test_pool_reuse_no_retrain():
    s = run(ScriptedClaude([[("analyze_stock", {"symbol": "2330"})], [("analyze_stock", {"symbol": "2449"})],
                            [("get_market_regime", {})],
                            [("finalize", {"answer": {**good_answer(), "per_stock": ["2330：否決交易。", "2449：穩健偏多，優先觀察。"]}})]]))
    assert CALLS["analyze"] == 1 and "2449" in s.ctx, CALLS
    print("✓ 第二檔股票直接沿用同一次股票池計算，不重新訓練模型")


def test_what_if_math():
    r = ROW_2330
    stop = float(r.get("entry_stop_loss_price") or r.get("stop_loss_price_adjusted"))
    tgt = float(r.get("entry_take_profit_1") or r.get("take_profit_price"))
    entry = round(stop * 1.02, 2)
    G.AGENT_SETTINGS["dry_run"] = False
    s = G.Session(None, "試算", analyze_fn=fake_analyze, market_fn=fake_market)
    s.call("analyze_stock", {"symbol": "2330"}, "test")
    out = s.call("what_if_entry", {"symbol": "2330", "entry_price": entry}, "test")
    p_hi = min((tgt + 1.5 * stop) / 2.5, stop / 0.92)
    assert abs(out["acceptable_entry_high"] - round(p_hi, 2)) < 0.011, (out, p_hi)
    rr = (tgt - entry) / (entry - stop)
    assert abs(out["custom"]["rr"] - round(rr, 2)) < 0.011, out
    print(f"✓ 進場價試算：停損 {stop:.2f}、目標 {tgt:.2f} → 可接受上緣 {out['acceptable_entry_high']}；以 {entry} 進場 R/R {out['custom']['rr']}（與手算一致）")


def test_api_failure_falls_back():
    def boom(body):
        raise RuntimeError("HTTP 529 overloaded")
    s = run(boom)
    assert s.mode == "agent→fixed_pipeline" and s.final and not G.audit_answer(s.final, s), G.audit_answer(s.final, s)
    tools = [st["tool"] for st in s.steps]
    assert "analyze_stock" in tools and "list_alternatives" in tools and tools[-1] == "template_answer", tools
    print("✓ API 失敗：改用固定流程與確定性範本，Critic 通過")


def test_dry_run_and_render():
    s = run(None, dry=True)
    text = G.render(s)
    assert s.mode == "fixed_pipeline" and "【回答】" in text and "【決策軌跡】" in text and "否決交易" in text
    print("✓ dry-run：不呼叫 API，固定流程產出完整回答與決策軌跡")


def test_decision_not_mutated():
    before = copy.deepcopy(POOL)
    run(ScriptedClaude(PLAN + [[("finalize", {"answer": good_answer()})]]))
    pd.testing.assert_frame_equal(before, POOL)
    print("✓ 系統決策資料未被修改")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} 通過")
