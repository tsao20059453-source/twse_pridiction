"""AI 分析師報告測試（假 Claude，不花錢、不連網）。執行：python tests/test_ai.py

使用 tests/fixtures/row_2330_20260918.json（9/21 論文版 Mode 1 的 2330 最新一筆決策資料）。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_ai as A  # noqa: E402

ROW = json.loads((ROOT / "tests" / "fixtures" / "row_2330_20260918.json").read_text(encoding="utf-8"))
SUMMARY = {"ranking_spearman": 0.2377}


class FakeClaude:
    def __init__(self, payload=None, error: str | None = None):
        self.payload, self.error, self.calls = payload, error, 0

    def complete(self, system, user, tag=""):
        self.calls += 1
        if self.error:
            raise A.LLMError(self.error)
        return A.LLMResult(text="```json\n" + json.dumps(self.payload, ensure_ascii=False) + "\n```", model="fake",
                           input_tokens=1000, output_tokens=500)


def ctx():
    return A.build_facts(A._translated_row(ROW), summary=SUMMARY, symbol="2330")


def good_payload(c):
    f = {x["group"]: x["id"] for x in reversed(c["facts"])}          # 每類第一個編號
    ids = [x["id"] for x in c["facts"]]
    return {
        "headline": f"2330 目前遭系統否決交易，等待更好的風險報酬結構（{ids[0]}）",
        "decision_logic": (f"系統最終決策為「否決交易」（{ids[0]}），因為實務風險報酬比僅 0.53，低於 1.5 的門檻（{f['風險報酬']}），"
                           f"5 項硬條件只通過 3 項（{ids[5]}），最終分數 0.39 也低於 0.55（{ids[1]}），因此建議倉位為 0（{ids[4]}）。"),
        "supporting_signals": [f"大盤位於 20 日均線之上，距離 2.00%（{f['大盤']}）。",
                               f"RSI14 為 57.28，屬中性偏強區間（{f['技術']}）。"],
        "opposing_signals": [f"上方壓力 2505.00 只差 1.83%，最近支撐卻在 11.44% 之外（{f['價位']}）。"],
        "watch_triggers": [f"若價格回到回測支撐區間且第一止盈 R/R 達到 1.5，決策可能轉為可進場（{f['進場計畫']}）。"],
        "model_caveat": f"安全模型的排序有參考價值，但報酬模型幾乎沒有預測力，不宜作為主要依據（{f['模型']}）。",
    }


def test_template_path_passes_critic():
    c = ctx()
    res = A.write_analyst_report(c, client=None)
    assert not res.used_llm and res.critic.ok, res.critic.text()
    assert "否決交易" in res.report["decision_logic"]
    print("✓ 範本路徑：Critic 通過")


def test_good_llm_output_used():
    c = ctx()
    res = A.write_analyst_report(c, FakeClaude(good_payload(c)))
    assert res.used_llm and res.critic.ok and not res.fallback_sections, (res.fallback_sections, res.critic.text())
    assert "0.53" in res.report["decision_logic"]
    print("✓ 正常 Claude 輸出：全部採用，Critic 通過")


def test_bad_llm_sections_fall_back():
    c = ctx()
    p = good_payload(c)
    ids = [x["id"] for x in c["facts"]]
    p["headline"] = f"2330 目標價 2800 元，建議進場（{ids[0]}）"                     # 編造數字＋與否決相反
    p["decision_logic"] = f"系統否決交易（{ids[0]}），但長期保證獲利，建議逢低買進。（{ids[1]}）"  # 承諾用語＋相反建議＋缺標籤原文
    p["supporting_signals"] = ["大盤表現強勁，值得關注。"]                              # 沒有事實引用
    p["opposing_signals"] = [f"這个价位风险较高（{ids[2]}）。"]                           # 簡體字
    p["watch_triggers"] = [f"若出現 F999 所述情況（F999）。"]                          # 不存在的事實
    res = A.write_analyst_report(c, FakeClaude(p))
    bad = {s.split(":")[0] for s in res.fallback_sections}
    assert {"headline", "decision_logic", "supporting_signals", "opposing_signals", "watch_triggers"} <= bad, res.fallback_sections
    assert res.report["model_caveat"] == p["model_caveat"]                           # 合格的段落保留
    assert res.critic.ok, res.critic.text()                                          # 換成範本後再稽核通過
    assert "建議進場" not in res.report["headline"] and "2800" not in res.report["headline"]
    print(f"✓ 不合格段落改用範本：{sorted(bad)}，最終 Critic 通過")


def test_critic_negation_not_flagged():
    c = ctx()
    p = good_payload(c)
    ids = [x["id"] for x in c["facts"]]
    p["decision_logic"] = (f"系統最終決策為「否決交易」（{ids[0]}），目前不建議進場，也不是可以買進的時機；"
                           f"實務風險報酬比 0.53 低於 1.5（{ids[1]}）。")
    res = A.write_analyst_report(c, FakeClaude(p))
    assert "decision_logic:critic" not in res.fallback_sections, res.fallback_sections
    print("✓ 否定敘述（不建議進場）不會被誤判為建議進場")


def test_api_failure_uses_template():
    c = ctx()
    res = A.write_analyst_report(c, FakeClaude(error="HTTP 529 overloaded"))
    assert not res.used_llm and res.error and res.critic.ok
    print("✓ API 失敗：全部改用範本，報告仍完整產出")


def test_decision_inputs_not_mutated():
    row = copy.deepcopy(ROW)
    c = A.build_facts(A._translated_row(row), summary=SUMMARY, symbol="2330")
    A.write_analyst_report(c, FakeClaude(good_payload(c)))
    assert row == ROW
    print("✓ 決策資料未被修改")


def test_score_breakdown_matches_system():
    c = ctx()
    b = c["score"]
    assert b["consistent"], (b["recomputed"], b["final"])          # 重算結果與系統的最終分數一致
    assert abs(sum(x["contrib"] for x in b["components"]) - b["raw"]) < 1e-9
    assert b["ceiling"] and b["ceiling"]["cap"] == 0.39            # 否決 → 上限 0.39，低於門檻
    assert abs(b["gap"] - 0.16) < 0.006
    panel = "\n".join(A.render_score_panel(c))
    assert "離門檻還差 0.16" in panel and "只要交易被否決" in panel
    print("✓ 分數拆解：加權貢獻、扣分、上限與系統最終分數一致")


def test_score_plain_language_numbers_pass_critic():
    c = ctx()
    p = good_payload(c)
    sid = {f["text"].split("，")[0].split(" ")[0]: f["id"] for f in c["facts"] if f["group"] == "分數解讀"}
    p["decision_logic"] = (f"系統最終決策為「否決交易」（F1）。最終決策分數 0.39，距離 0.55 門檻還差 0.16（{sid['最終決策分數']}）；"
                           f"安全分數 0.69，代表模型預估它未來 10 天的跌幅比股票池中約 69% 的股票小，貢獻 0.21 分（{sid['安全分數']}）；"
                           f"但風險報酬分數只貢獻 0.00 分（{sid['風險報酬分數']}）。")
    res = A.write_analyst_report(c, FakeClaude(p))
    assert "decision_logic:critic" not in res.fallback_sections, (res.fallback_sections, getattr(res, "first_critic", None))
    print("✓ 引用分數解讀事實中的差距、百分比與貢獻值，Critic 放行")


def test_ceiling_not_binding_still_reported():
    row = dict(ROW)
    row.update(trade_veto_flag=False, rule_filter_score=0.05, technical_filter_result="技術條件偏弱", risk_reward_score=1.0,
               entry_plan_override=True, entry_rr_1=5.36, rule_penalty_points=1.0, safety_score=0.73, return_score=0.51,
               final_decision_score=0.4835)
    b = A.score_breakdown(A._translated_row(row))
    assert b["consistent"] and not b["binding_caps"]                 # 0.4835 < 0.49：上限沒有正在壓分數
    assert b["ceiling"]["cap"] == 0.49                               # 但仍是低於門檻的天花板
    assert b["drag"] == "規則分數"
    print("✓ 上限未實際壓分數時，仍提示「低於門檻的天花板」")


def test_rule_rr_profile_breakdown():
    import pipeline_risk as R
    try:
        R.set_final_score_profile("rule_rr_v2")
        row = dict(ROW)
        row.update(trade_veto_flag=False, rule_filter_score=0.80, risk_reward_score=0.70, rule_penalty_points=0.0,
                   technical_filter_result="pass", practical_risk_reward_ratio=2.5,
                   risk_warning="", entry_plan_override=False)
        import pandas as pd
        row["final_decision_score"] = float(R.build_final_decision(pd.Series(row))["final_decision_score"])   # 系統實際算出的分數
        c = A.build_facts(A._translated_row(row), summary=SUMMARY, symbol="2330")
        b = c["score"]
        assert abs(b["raw"] - 0.75) < 1e-9 and b["consistent"], (b["raw"], b["recomputed"], b["final"])
        assert "0.50×規則分數＋0.50×風險報酬分數" in c["facts"][1]["text"]
        assert "不計入最終分數" in "\n".join(f["text"] for f in c["facts"])
        assert "0.50×規則分數" in A.system_prompt()
        assert A.write_analyst_report(c, None).critic.ok
    finally:
        R.set_final_score_profile("low_model_weight_v1")
    print("✓ 規則＋風險報酬權重組合：分數拆解、公式文字與提示詞同步")


def test_override_satisfies_score_gate():
    row = dict(ROW)
    row.update(trade_veto_flag=False, rule_filter_score=0.05, technical_filter_result="技術條件偏弱", risk_reward_score=1.0,
               entry_plan_override=True, entry_rr_1=5.36, rule_penalty_points=1.0, safety_score=0.73, return_score=0.51,
               final_decision_score=0.4835, final_decision_label="等待回測進場", suggested_position_pct=0.0)
    c = A.build_facts(A._translated_row(row), summary=SUMMARY, symbol="2330")
    gate = c["decision"]["conditions"][-1]
    assert gate["pass"] is True and "回測支撐進場計畫成立" in gate["detail"]      # 與回測引擎、報告五之二一致
    panel = "\n".join(A.render_score_panel(c))
    assert "改走「回測支撐進場計畫」" in panel and "其他分數再好也進不了場" not in panel
    res = A.write_analyst_report(c, None)
    assert res.critic.ok and "回測支撐進場計畫成立" in res.report["decision_logic"], res.report["decision_logic"]
    print("✓ 回測支撐進場計畫成立時，分數門檻視為滿足，不再誤寫「進不了場」")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} 通過")
