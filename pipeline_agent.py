"""第四層：AI 決策代理（2026-10-26 新增）。

使用者用一句話提問（例如「2330 現在可以買嗎？有沒有更好的選擇？」），由 Claude 透過工具規劃分析流程：
    觀察（大盤、個股決策）→ 推理（哪些條件沒過）→ 規劃（還需要什麼證據）→ 行動（呼叫工具）→ 驗證（Critic）→ 回答

有邊界的自主（沿用 ETF 專案 orchestrator 的設計）：
    - 系統決策（標籤、分數、倉位）一律由既有的確定性模型與規則產生，代理不能改變任何數字。
    - 代理只能呼叫下列工具蒐集證據；所有回答中的數字都必須出現在工具結果中（Critic 檢查）。
    - 提交前硬性閘門：至少分析過一檔股票、Critic 通過；未通過可重寫 2 次，之後改用確定性範本。
    - 預算：步數與 token 上限；API 失敗或超出預算 → 以同一組工具執行固定流程，保證一定有答案。
紀錄：
    output/agent/<時間>_<代號>.md／_trace.json  回答與完整決策軌跡（每次工具呼叫、參數、結果、由誰決定）
    logs/llm_calls/<日期>/agent_*.json           每次規劃呼叫的完整內容（不含金鑰）
    data/agent_memory/episodes.jsonl            每次提問一行摘要（只用於回顧，不會自動改變任何參數）
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

import pipeline_ai as A

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "output" / "agent"
MEMORY_PATH = ROOT / "data" / "agent_memory" / "episodes.jsonl"
AGENT_SETTINGS = {"max_steps": 12, "token_budget": 150000, "max_rewrites": 2, "turn_max_tokens": 4000,
                  "model": None, "dry_run": False}
SYMBOL_RE = re.compile(r"(?<!\d)([1-9]\d{3})(?!\d)")
# 2026-10-26 誠實基準（固定基準股票池、as-of 10/25、預設參數；pools/README.md）
BASELINE = {"period": "2025-02-03～2026-10-07", "total_return": "211.06%", "sharpe": "3.20", "mdd": "−5.6%"}


# ============================================================
# 預設的分析函式（測試時可替換）
# ============================================================
def default_analyze(config, symbol: str, top_n: int = 30) -> tuple[dict, Optional[pd.DataFrame]]:
    """執行一次 Mode 1（個股深度分析），回傳（目標股票最新決策列, 股票池各股最新決策）。"""
    import pipeline_model_output as M
    cfg = replace(config, stock_no=symbol, mode="single_diagnostic", display_mode="presentation")
    ai_enabled = A.AI_SETTINGS.get("enabled")
    A.AI_SETTINGS["enabled"] = False                      # 分析過程中不另外產生 AI 分析師報告
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out, _ = M.run_single_stock_diagnostic(cfg, top_n=top_n, investor_mode=True)
    finally:
        A.AI_SETTINGS["enabled"] = ai_enabled
    row = out.sort_values("date").iloc[-1].to_dict()
    pool = M.LAST_POOL_DECISIONS.get("rows")
    return row, (pool.copy() if pool is not None else None)


def default_market(config) -> pd.DataFrame:
    import pipeline_data as D
    with contextlib.redirect_stdout(io.StringIO()):
        return D.get_market_index_data(replace(config, market="TWSE"))


# ============================================================
# Session：工具共用的狀態、事實與決策軌跡
# ============================================================
class Session:
    def __init__(self, config, question: str, top_n: int = 30,
                 analyze_fn: Optional[Callable] = None, market_fn: Optional[Callable] = None):
        self.config, self.question, self.top_n = config, question, top_n
        self.analyze_fn = analyze_fn or default_analyze
        self.market_fn = market_fn or default_market
        self.rows: dict[str, dict] = {}          # 代號 → 翻譯後的最新決策列
        self.ctx: dict[str, dict] = {}           # 代號 → pipeline_ai.build_facts 結果
        self.pool: Optional[pd.DataFrame] = None
        self.facts: list[dict] = []
        self.steps: list[dict] = []
        self.allowed_alternatives: set[str] = set()
        self.tokens = {"input": 0, "output": 0}
        self.rewrites = 0
        self.final: Optional[dict] = None
        self.mode = "agent"
        self.started = time.time()
        self.id = uuid.uuid4().hex[:8]

    def add_fact(self, tool: str, text: str) -> str:
        fid = f"F{len(self.facts) + 1}"
        self.facts.append({"id": fid, "tool": tool, "text": text})
        return fid

    def call(self, name: str, args: dict, decided_by: str) -> dict:
        t0 = time.time()
        spec = TOOLS.get(name)
        if spec is None:
            out = {"ok": False, "error": f"沒有這個工具：{name}"}
        else:
            try:
                out = spec["fn"](self, **(args or {}))
            except TypeError as e:
                out = {"ok": False, "error": f"參數錯誤：{e}"}
            except Exception as e:                                   # noqa: BLE001
                out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        self.steps.append({"n": len(self.steps) + 1, "tool": name, "args": args or {}, "ok": bool(out.get("ok")),
                           "decided_by": decided_by, "seconds": round(time.time() - t0, 2),
                           "result": _compact(out)})
        return out


def _compact(out: dict, limit: int = 1200) -> dict:
    s = json.dumps(out, ensure_ascii=False, default=str)
    return out if len(s) <= limit else {"ok": out.get("ok"), "summary": s[:limit] + "…"}


def _f(x, d=2) -> str:
    v = A._num(x)
    return "NA" if v is None else f"{v:.{d}f}"


def _pct(x, d=2) -> str:
    v = A._num(x)
    return "NA" if v is None else f"{v * 100:.{d}f}%"


# ============================================================
# 工具
# ============================================================
def tool_get_market_regime(s: Session) -> dict:
    m = s.market_fn(s.config)
    if m is None or m.empty:
        return {"ok": False, "error": "沒有大盤資料"}
    m = m.dropna(subset=["market_close"]).sort_values("date")
    c = pd.to_numeric(m["market_close"], errors="coerce")
    close, ma20 = float(c.iloc[-1]), float(c.tail(20).mean())
    dist = close / ma20 - 1
    r5 = close / float(c.iloc[-6]) - 1 if len(c) > 5 else float("nan")
    r20 = close / float(c.iloc[-21]) - 1 if len(c) > 20 else float("nan")
    date = str(pd.Timestamp(m["date"].iloc[-1]).date())
    above = dist >= 0
    fid = s.add_fact("get_market_regime",
                     f"加權指數 {date} 收盤 {close:.2f}，20 日均線 {ma20:.2f}，{'高於' if above else '低於'}均線 {abs(dist) * 100:.2f}%；"
                     f"近 5 日 {r5 * 100:.2f}%、近 20 日 {r20 * 100:.2f}%。大盤過濾（硬條件之一）：{'通過' if above else '未通過'}")
    return {"ok": True, "fact": fid, "date": date, "close": round(close, 2), "ma20": round(ma20, 2),
            "distance_pct": round(dist * 100, 2), "return_5d_pct": round(r5 * 100, 2), "return_20d_pct": round(r20 * 100, 2),
            "market_filter_pass": above}


def _register_row(s: Session, symbol: str, row: dict, source: str) -> dict:
    tr = A._translated_row(row)
    ctx = A.build_facts(tr, summary=None, symbol=symbol)
    s.rows[symbol], s.ctx[symbol] = tr, ctx
    ids = []
    for f in ctx["facts"]:                      # 改用本次對話的事實編號，範本與 Critic 才能共用
        f["id"] = s.add_fact(f"analyze_stock:{symbol}", f"【{A.stock_label(symbol)}】{f['text']}")
        ids.append(f["id"])
    d = ctx["decision"]
    failed = [c["name"] for c in d["conditions"] if c["pass"] is False]
    return {"ok": True, "symbol": symbol, "name": A.stock_label(symbol).replace(symbol, "").strip(), "source": source,
            "date": ctx["meta"]["date"], "close": ctx["meta"]["close"],
            "decision_label": d["label"], "decision_action": d["action"], "final_score": d["final_score"],
            "position_pct": round((d["position_pct"] or 0) * 100, 2), "veto": d["veto"],
            "hard_conditions_passed": f"{d['n_pass']}/5", "failed_conditions": failed,
            "entry_allowed_now": d["bullish_allowed"], "pullback_plan": bool(tr.get("entry_plan_override", False)),
            "fact_ids": f"{ids[0]}～{ids[-1]}"}


def tool_analyze_stock(s: Session, symbol: str) -> dict:
    symbol = str(symbol).strip()
    if not SYMBOL_RE.fullmatch(symbol):
        return {"ok": False, "error": "股票代碼必須是 4 位數字，例如 2330"}
    if symbol in s.rows:
        return {"ok": True, "symbol": symbol, "note": "本次已分析過，直接沿用", "decision_label": s.ctx[symbol]["decision"]["label"]}
    if s.pool is not None and symbol in set(s.pool["symbol"].astype(str)):
        row = s.pool[s.pool["symbol"].astype(str) == symbol].iloc[-1].to_dict()
        return _register_row(s, symbol, row, "同一次股票池計算（不需重新訓練）")
    row, pool = s.analyze_fn(s.config, symbol, s.top_n)
    if pool is not None and not pool.empty:
        s.pool = pool
    return _register_row(s, symbol, row, "系統 Mode 1 個股深度分析")


def tool_explain_decision(s: Session, symbol: str) -> dict:
    symbol = str(symbol).strip()
    if symbol not in s.ctx:
        return {"ok": False, "error": f"請先 analyze_stock({symbol})"}
    ctx, row = s.ctx[symbol], s.rows[symbol]
    panel = A.render_score_panel(ctx)
    conds = [f"{c['name']}：{'通過' if c['pass'] else '未通過' if c['pass'] is False else '無資料'}（{c['detail']}）"
             for c in ctx["decision"]["conditions"]]
    notes = A._txt(row.get("rule_filter_notes_zh") or row.get("rule_filter_notes"), "")
    warn = A._txt(row.get("risk_warning"), "")
    fid = s.add_fact(f"explain_decision:{symbol}", f"【{A.stock_label(symbol)}】分數解讀：" + " ".join(x.strip() for x in panel[1:]))
    return {"ok": True, "symbol": symbol, "fact": fid, "score_panel": panel, "hard_conditions": conds,
            "rule_notes": notes, "risk_warning": warn}


def tool_what_if_entry(s: Session, symbol: str, entry_price: Optional[float] = None) -> dict:
    """以今天的支撐、停損、目標價位，計算不同進場價的停損距離與風險報酬比（純計算，不改變決策）。"""
    symbol = str(symbol).strip()
    if symbol not in s.rows:
        return {"ok": False, "error": f"請先 analyze_stock({symbol})"}
    r = s.rows[symbol]
    close = A._num(r.get("close"))
    stop = A._num(r.get("entry_stop_loss_price")) or A._num(r.get("stop_loss_price_adjusted"))
    target = A._num(r.get("entry_take_profit_1")) or A._num(r.get("take_profit_price"))
    if not (close and stop and target) or target <= stop:
        return {"ok": False, "error": "缺少停損或目標價位，無法計算"}
    rr_min, s_lo, s_hi = A.HARD_RULES["rr_min"], A.HARD_RULES["stop_min"], A.HARD_RULES["stop_max"]
    p_rr = (target + rr_min * stop) / (1 + rr_min)            # R/R 剛好 1.5 的最高進場價
    p_stop_hi = stop / (1 - s_hi)                              # 停損距離 8% 的最高進場價
    p_stop_lo = stop / (1 - s_lo)                              # 停損距離 1.2% 的最低進場價
    upper, lower = min(p_rr, p_stop_hi), p_stop_lo
    out = {"ok": True, "symbol": symbol, "close": close, "stop": round(stop, 2), "target_1": round(target, 2),
           "basis": "以今天的支撐、停損與第一止盈價位估算；價位每日會隨行情更新"}
    if upper > lower:
        need = upper / close - 1
        out.update({"acceptable_entry_low": round(lower, 2), "acceptable_entry_high": round(upper, 2),
                    "close_vs_high_pct": round(need * 100, 2)})
        fid = s.add_fact(f"what_if_entry:{symbol}",
                         f"【{A.stock_label(symbol)}】以今天的價位估算：停損 {stop:.2f}、第一止盈 {target:.2f}；進場價在 {lower:.2f}～{upper:.2f} 之間時，"
                         f"停損距離 1.2%～8% 且 R/R ≥ 1.5 同時成立；收盤價 {close:.2f} 相對區間上緣 {need * 100:+.2f}%")
    else:
        fid = s.add_fact(f"what_if_entry:{symbol}", f"【{A.stock_label(symbol)}】以今天的價位估算：沒有任何進場價能同時滿足停損距離與 R/R 條件")
    out["fact"] = fid
    if entry_price is not None:
        p = float(entry_price)
        if p <= stop:
            out["custom"] = {"entry": p, "error": "進場價不高於停損價，交易計畫不成立"}
        else:
            sp, rr = (p - stop) / p, (target - p) / (p - stop)
            ok = s_lo <= sp <= s_hi and rr >= rr_min
            out["custom"] = {"entry": p, "stop_pct": round(sp * 100, 2), "rr": round(rr, 2), "conditions_pass": ok}
            out["custom_fact"] = s.add_fact(f"what_if_entry:{symbol}",
                                            f"【{A.stock_label(symbol)}】若以 {p:.2f} 進場：停損距離 {sp * 100:.2f}%、R/R {rr:.2f}，"
                                            f"停損與 R/R 兩項條件{'皆通過' if ok else '未全部通過'}")
    return out


def tool_list_alternatives(s: Session, max_n: int = 5) -> dict:
    if s.pool is None or s.pool.empty:
        return {"ok": False, "error": "請先 analyze_stock 任一檔股票（股票池決策會一起算出）"}
    p = s.pool.copy()
    p["symbol"] = p["symbol"].astype(str)
    p = p[~p["symbol"].isin(s.rows.keys())]
    out = {"ok": True, "entry_now": [], "wait_pullback": [], "date": str(pd.to_datetime(p["date"]).max().date())}
    max_n = int(max(1, min(int(max_n or 5), 8)))
    p["_score"] = pd.to_numeric(p.get("final_decision_score"), errors="coerce")
    for _, r in p.sort_values("_score", ascending=False).iterrows():
        tr = A._translated_row(r.to_dict())
        sym, label = tr["symbol"], A._txt(tr.get("final_decision_label_zh") or tr.get("final_decision_label"))
        veto = bool(tr.get("trade_veto_flag", False))
        pos = A._num(tr.get("suggested_position_pct")) or 0
        score = A._num(tr.get("final_decision_score"))
        close, zl, zh = A._num(tr.get("close")), A._num(tr.get("support_entry_zone_low")), A._num(tr.get("support_entry_zone_high"))
        item = {"symbol": sym, "stock": A.stock_label(sym), "label": label, "final_score": round(score, 2) if score is not None else None,
                "close": close, "support_zone": f"{_f(zl)}～{_f(zh)}" if zl else "NA"}
        if not veto and pos > 0 and len(out["entry_now"]) < max_n:
            out["entry_now"].append(item)
            s.allowed_alternatives.add(sym)
            s.add_fact("list_alternatives", f"【{A.stock_label(sym)}】系統決策「{label}」，最終分數 {_f(score)}，建議倉位 {pos * 100:.2f}%，收盤 {_f(close)}")
        elif bool(tr.get("entry_plan_override", False)) and len(out["wait_pullback"]) < max_n:
            dist = (close / zh - 1) if (close and zh) else None
            item["above_zone_pct"] = round(dist * 100, 2) if dist is not None else None
            out["wait_pullback"].append(item)
            where = ""
            if dist is not None:
                where = (f"，高於支撐區上緣 {dist * 100:.2f}%" if dist > 0 else
                         f"，收盤已在支撐區上緣以下（{abs(dist) * 100:.2f}%）")
            s.add_fact("list_alternatives", f"【{A.stock_label(sym)}】系統決策「{label}」，最終分數 {_f(score)}，回測支撐區 {item['support_zone']}，"
                                            f"收盤 {_f(close)}{where}")
    out["note"] = "entry_now＝系統目前給倉位；wait_pullback＝回測支撐進場計畫成立、等待價格回到支撐區"
    return out


def tool_get_chip_snapshot(s: Session, symbol: str) -> dict:
    """三大法人、融資、月營收近況。系統決策目前沒有使用這些資料，只作參考。"""
    import pipeline_chip as C
    symbol = str(symbol).strip()
    inst, marg, rev = C._load("institutional"), C._load("margin"), C._load("revenue")
    if inst.empty or symbol not in set(inst["code"]):
        return {"ok": False, "error": "data/chip 沒有這檔的資料（可執行 tools/fetch_chip_data.py --symbols 代號）"}
    asof = pd.Timestamp(s.rows[symbol]["date"]) if symbol in s.rows else None
    i = inst[inst["code"] == symbol].sort_values("date")
    if asof is not None:
        i = i[i["date"] <= asof]
    lots = lambda col, n: float(pd.to_numeric(i[col], errors="coerce").tail(n).sum()) / 1000   # noqa: E731
    out = {"ok": True, "symbol": symbol, "date": str(i["date"].iloc[-1].date()),
           "foreign_5d_lots": round(lots("foreign_net", 5)), "foreign_20d_lots": round(lots("foreign_net", 20)),
           "trust_5d_lots": round(lots("trust_net", 5)), "trust_20d_lots": round(lots("trust_net", 20))}
    m = marg[marg["code"] == symbol].sort_values("date")
    if asof is not None:
        m = m[m["date"] <= asof]
    if len(m) > 20:
        mb = pd.to_numeric(m["margin_balance"], errors="coerce")
        out["margin_change_20d_pct"] = round((mb.iloc[-1] / mb.iloc[-21] - 1) * 100, 2)
    rv = rev[rev["code"] == symbol].sort_values("period")
    if asof is not None and "available_date" in rv:
        rv = rv[rv["available_date"] <= asof]
    if not rv.empty:
        out["revenue_yoy_last3"] = [f"{r.period}：{_f(r.rev_yoy_pct, 1)}%" for r in rv.tail(3).itertuples()]
    txt = (f"【{A.stock_label(symbol)}】籌碼參考（截至 {out['date']}，系統決策未使用）：外資近 5 日 {out['foreign_5d_lots']:+,} 張、近 20 日 {out['foreign_20d_lots']:+,} 張；"
           f"投信近 5 日 {out['trust_5d_lots']:+,} 張、近 20 日 {out['trust_20d_lots']:+,} 張")
    if "margin_change_20d_pct" in out:
        txt += f"；融資餘額 20 日變化 {out['margin_change_20d_pct']:+.2f}%"
    if "revenue_yoy_last3" in out:
        txt += "；月營收年增率 " + "、".join(out["revenue_yoy_last3"])
    out["fact"] = s.add_fact("get_chip_snapshot", txt.replace("+", ""))
    out["note"] = "回測檢驗中，籌碼特徵加入模型後效果不穩定，目前不參與系統決策，只作參考"
    return out


def tool_get_track_record(s: Session, symbol: Optional[str] = None) -> dict:
    files = sorted(ROOT.glob("backtest_batch_*_perf.csv"), key=lambda p: p.stat().st_mtime)
    out = {"ok": True, "baseline": BASELINE}
    s.add_fact("get_track_record", f"系統誠實基準（固定股票池回測 {BASELINE['period']}）：總報酬 {BASELINE['total_return']}、"
                                   f"Sharpe {BASELINE['sharpe']}、最大回撤 {BASELINE['mdd']}；過去績效不代表未來")
    if symbol and files:
        perf = pd.read_csv(files[-1], dtype=str, encoding="utf-8-sig")
        col = perf.columns[0]
        hit = perf[perf[col].astype(str).str.strip() == str(symbol)]
        if not hit.empty:
            r = hit.iloc[0]
            out["symbol_record"] = {k: r.get(k) for k in ("總報酬", "勝率", "交易筆數", "profit_factor") if k in r}
            out["symbol_record_file"] = files[-1].name
            s.add_fact("get_track_record", f"【{A.stock_label(symbol)}】最近一次回測（{files[-1].name}）：交易 {r.get('交易筆數')} 筆、"
                                           f"勝率 {r.get('勝率')}、總報酬 {r.get('總報酬')}")
    return out


def tool_finalize(s: Session, answer: dict) -> dict:
    """由 orchestrator 處理（閘門＋Critic）；這裡只是登記。"""
    return {"ok": True}


TOOLS: dict[str, dict] = {
    "get_market_regime": {"fn": tool_get_market_regime, "params": {},
                          "description": "大盤狀態：加權指數與 20 日均線的距離、近 5／20 日報酬，以及大盤過濾（硬條件）是否通過。"},
    "analyze_stock": {"fn": tool_analyze_stock,
                      "params": {"symbol": {"type": "string", "description": "4 位數股票代碼（上市或上櫃）"}},
                      "description": "執行系統的個股深度分析（Mode 1），取得最新決策：標籤、最終分數、5 項硬條件、建議倉位。"
                                     "第一次約需數分鐘，會同時算出整個股票池的決策；同一檔不要重複分析。"},
    "explain_decision": {"fn": tool_explain_decision,
                         "params": {"symbol": {"type": "string"}},
                         "description": "分數拆解（各分數白話意義、權重貢獻、扣分與上限、離門檻多遠）與 5 項硬條件明細、規則判讀、風險警示。"},
    "what_if_entry": {"fn": tool_what_if_entry,
                      "params": {"symbol": {"type": "string"},
                                 "entry_price": {"type": "number", "description": "選填：要試算的進場價"}},
                      "description": "以今天的停損與目標價位，計算可同時滿足「停損距離 1.2%～8%」與「R/R ≥ 1.5」的進場價區間；"
                                     "可另外試算指定進場價。用來量化『價格到哪裡決策可能改變』。"},
    "list_alternatives": {"fn": tool_list_alternatives,
                          "params": {"max_n": {"type": "integer", "description": "每類最多幾檔，預設 5"}},
                          "description": "股票池中其他股票的最新決策：目前可進場（有倉位）與等待回測支撐的股票，依最終分數排序。需先 analyze_stock。"},
    "get_chip_snapshot": {"fn": tool_get_chip_snapshot,
                          "params": {"symbol": {"type": "string"}},
                          "description": "三大法人買賣超、融資變化、月營收年增率近況。系統決策沒有使用，只能作為參考資訊。"},
    "get_track_record": {"fn": tool_get_track_record,
                         "params": {"symbol": {"type": "string", "description": "選填：查詢該股在最近一次回測的交易紀錄"}},
                         "description": "系統的回測績效基準，以及個股在最近一次回測的交易筆數、勝率、報酬。"},
    "finalize": {"fn": tool_finalize,
                 "params": {"answer": {"type": "object", "description": "最終回答", "properties": {
                     "conclusion": {"type": "string", "description": "一段話直接回答使用者（40～200 字）"},
                     "per_stock": {"type": "array", "items": {"type": "string"}, "description": "每檔股票的系統決策與主因"},
                     "reasons": {"type": "array", "items": {"type": "string"}, "description": "支持結論的證據（引用事實編號）"},
                     "triggers": {"type": "array", "items": {"type": "string"}, "description": "決策可能改變的具體條件（價位、大盤）"},
                     "alternatives": {"type": "array", "items": {"type": "string"}, "description": "替代選項（沒有就給空陣列）"},
                     "risks": {"type": "array", "items": {"type": "string"}, "description": "主要風險與限制"}},
                     "required": ["conclusion", "per_stock", "reasons", "triggers", "risks"]}},
                 "description": "交出最終回答。系統會檢查：數字是否都來自工具結果、是否與系統決策一致、有無保證獲利用語；未通過會退回並說明原因。"},
}

SYSTEM_PROMPT = """你是台股量化決策支援系統的「決策代理」，用工具回答使用者的投資問題。

系統架構：第一層機器學習模型（安全分數、報酬分數）→ 第二層技術規則 → 第三層風險控管（停損、風險報酬比、倉位）→ 最終決策。
系統的決策（標籤、分數、倉位）由確定性的模型與規則產生，你不能改變任何決策或數字。你的工作是：
判斷需要哪些證據 → 呼叫工具 → 找出決策原因 → 量化「什麼條件下決策會改變」 → 必要時提出替代選項 → 用 finalize 回答。

工作規則：
1. 回答任何個股問題前，必須先 analyze_stock。每檔第一次分析約需數分鐘，不要重複分析同一檔。
2. 一定要呼叫 get_market_regime（大盤過濾是硬條件之一）。
3. 決策不是「可進場（有倉位）」時：用 explain_decision 找出未通過的條件，用 what_if_entry 算出可接受的進場價區間。
4. 該股不建議進場、或使用者問有沒有其他選擇時，呼叫 list_alternatives；只能推薦系統目前有給倉位的股票為「可進場」，
   等待回測的股票只能說「等待價格回到支撐區」。
5. 籌碼（get_chip_snapshot）只是參考資訊，系統決策沒有使用；提到時必須註明。
6. 回答中的所有數字都必須出現在工具結果中；可在句尾用（F12）標註事實編號。不得編造目標價、新聞或財報。
7. 不得保證獲利或預測股價必然漲跌；系統沒有給倉位的股票，不得寫「建議買進／可以進場」。
8. 一律使用繁體中文，用一般投資人看得懂的白話；分數要說明代表什麼意思；提到股票時寫成「名稱 代號」（例如「台積電 2330」）。
9. 工具呼叫越少越好，通常 4～7 次即可；最後一定要呼叫 finalize。"""


# ============================================================
# Critic：最終回答的稽核
# ============================================================
def _answer_texts(ans: dict) -> list[tuple[str, str]]:
    out = []
    for k in ("conclusion", "per_stock", "reasons", "triggers", "alternatives", "risks"):
        for t in A._as_list(ans.get(k)):
            out.append((k, t))
    return out


def audit_answer(ans: dict, s: Session) -> list[str]:
    issues = []
    if not isinstance(ans, dict) or not str(ans.get("conclusion", "")).strip():
        return ["缺少 conclusion"]
    fact_vals = []
    for f in s.facts:
        for m in A.NUM_RE.finditer(f["text"]):
            try:
                fact_vals.append(float(m.group(1)))
            except ValueError:
                pass
    q_vals = {float(m.group(1)) for m in A.NUM_RE.finditer(s.question)}
    fact_ids = {f["id"] for f in s.facts}
    allowed = {sym for sym, c in s.ctx.items() if c["decision"]["bullish_allowed"]} | s.allowed_alternatives
    disallowed = {sym for sym, c in s.ctx.items() if not c["decision"]["bullish_allowed"]}
    for key, t in _answer_texts(ans):
        body = A.REF_RE.sub("", t)
        for m in A.NUM_RE.finditer(body):
            v = float(m.group(1))
            if v in A.ALWAYS_OK_NUMBERS or v in q_vals or (1000 <= v <= 9999 and str(int(v)) in s.ctx) \
                    or any(abs(v - x) <= max(0.011, abs(x) * 0.001) for x in fact_vals):
                continue
            issues.append(f"{key}：數字「{m.group(0)}」不在任何工具結果中")
            break
        if any(c in A.SIMPLIFIED_ONLY for c in t):
            issues.append(f"{key}：出現簡體字")
        hit = [w for w in A.FORBIDDEN if A._affirmed(t, w)]
        if hit:
            issues.append(f"{key}：保證獲利用語 {hit}")
        missing = set(A.REF_RE.findall(t)) - fact_ids
        if missing:
            issues.append(f"{key}：引用不存在的事實 {sorted(missing)}")
        for sentence in re.split(r"[。；;！\n]", t):
            bull = [w for w in A.BULLISH_WORDS if A._affirmed(sentence, w)]
            if not bull:
                continue
            syms = set(SYMBOL_RE.findall(sentence))
            if syms & disallowed or (not syms & allowed and key == "conclusion" and not allowed):
                issues.append(f"{key}：「{sentence.strip()[:40]}」寫了 {bull}，但系統沒有給這檔股票倉位")
                break
    text_all = " ".join(t for _, t in _answer_texts(ans))
    for sym, c in s.ctx.items():
        if c["decision"]["label"] not in text_all:
            issues.append(f"沒有提到 {sym} 的實際系統決策「{c['decision']['label']}」")
    return issues


# ============================================================
# 確定性範本（備援）
# ============================================================
def template_answer(s: Session) -> dict:
    per, reasons, triggers, alts, risks = [], [], [], [], []
    for sym, ctx in s.ctx.items():
        t = A.template_report(ctx)
        per.append(t["headline"] if sym in t["headline"][:20] else f"{A.stock_label(sym)}：{t['headline']}")
        reasons.append(t["decision_logic"])
        triggers += [x for x in A._as_list(t["watch_triggers"])][:2]
        w = [f for f in s.facts if f["tool"] == f"what_if_entry:{sym}"]
        if w:
            triggers.append(f"{w[0]['text']}（{w[0]['id']}）")
    mk = [f for f in s.facts if f["tool"] == "get_market_regime"]
    if mk:
        reasons.append(f"{mk[0]['text']}（{mk[0]['id']}）")
    for f in s.facts:
        if f["tool"] == "list_alternatives":
            alts.append(f"{f['text']}（{f['id']}）")
    risks.append("系統的價位與分數每日更新，以上條件以最新一個交易日的資料計算；過去績效不代表未來。")
    labels = "、".join(f"{A.stock_label(sym)}「{c['decision']['label']}」" for sym, c in s.ctx.items()) or "（沒有分析任何股票）"
    concl = f"系統目前的決策為 {labels}。以下依系統的硬條件與分數說明原因，以及哪些條件改變時決策可能跟著改變。"
    return {"conclusion": concl, "per_stock": per, "reasons": reasons, "triggers": triggers[:5],
            "alternatives": alts[:5], "risks": risks}


# ============================================================
# 規劃者（Claude tool use）
# ============================================================
def _tools_spec() -> list[dict]:
    out = []
    for name, meta in TOOLS.items():
        props = meta.get("params", {})
        schema = {"type": "object", "properties": props, "additionalProperties": False}
        if name in ("analyze_stock", "explain_decision", "get_chip_snapshot"):
            schema["required"] = ["symbol"]
        if name == "what_if_entry":
            schema["required"] = ["symbol"]
        if name == "finalize":
            schema["required"] = ["answer"]
        out.append({"name": name, "description": meta["description"], "input_schema": schema})
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def _post(body: dict, timeout: int = 180) -> dict:
    A.load_env_file(ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("沒有設定 ANTHROPIC_API_KEY（請寫在 .env）")
    last = None
    for attempt in range(1, 4):
        try:
            r = requests.post(A.ANTHROPIC_URL, json=body, timeout=timeout,
                              headers={"x-api-key": key, "anthropic-version": A.ANTHROPIC_VERSION, "content-type": "application/json"})
        except requests.RequestException as e:
            last = e
            time.sleep(4 * attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 529):
            last = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(8 * attempt)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"規劃者呼叫連續失敗：{last}")


def _log_call(s: Session, turn: int, body: dict, resp: dict, elapsed: float) -> None:
    folder = A.LLM_LOG_DIR / A.now_taipei().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{A.now_taipei().strftime('%H%M%S')}_agent_{s.id}_turn{turn}.json"
    rec = {"purpose": "decision_agent", "session": s.id, "turn": turn, "model": body["model"], "elapsed_sec": round(elapsed, 2),
           "usage": resp.get("usage"), "stop_reason": resp.get("stop_reason"),
           "request_messages_tail": body["messages"][-2:], "response_content": resp.get("content")}
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def run_planner(s: Session, post_fn: Callable = _post) -> None:
    model = AGENT_SETTINGS.get("model") or A.AI_SETTINGS.get("model") or "claude-sonnet-5-5"
    messages = [{"role": "user", "content": f"使用者的問題：{s.question}\n請開始。"}]
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    tools = _tools_spec()
    used = 0
    for turn in range(1, AGENT_SETTINGS["max_steps"] + 1):
        body = {"model": model, "max_tokens": AGENT_SETTINGS["turn_max_tokens"], "system": system, "tools": tools, "messages": messages}
        t0 = time.time()
        resp = post_fn(body)
        try:
            _log_call(s, turn, body, resp, time.time() - t0)
        except Exception:                                   # noqa: BLE001
            pass
        u = resp.get("usage", {}) or {}
        s.tokens["input"] += int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0))
        s.tokens["output"] += int(u.get("output_tokens", 0))
        used += int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
        content = resp.get("content", []) or []
        messages.append({"role": "assistant", "content": content})
        calls = [b for b in content if b.get("type") == "tool_use"]
        if not calls:
            s.steps.append({"n": len(s.steps) + 1, "tool": "planner_stopped_without_finalize", "ok": False, "decided_by": "gate"})
            return
        results = []
        for b in calls:
            name, args = b.get("name"), b.get("input") or {}
            if name == "finalize":
                out = _finalize_gate(s, args.get("answer"))
                if out.get("ok") or s.rewrites > AGENT_SETTINGS["max_rewrites"]:
                    return
            else:
                out = s.call(name, args, decided_by="planner")
            results.append({"type": "tool_result", "tool_use_id": b["id"],
                            "content": json.dumps(out, ensure_ascii=False, default=str)[:3500]})
        messages.append({"role": "user", "content": results})
        if used > AGENT_SETTINGS["token_budget"]:
            s.steps.append({"n": len(s.steps) + 1, "tool": "budget_exceeded", "ok": False, "decided_by": "gate", "args": {"tokens": used}})
            return


def _finalize_gate(s: Session, answer) -> dict:
    if not s.ctx:
        s.steps.append({"n": len(s.steps) + 1, "tool": "finalize_rejected", "ok": False, "decided_by": "gate",
                        "result": {"reason": "尚未分析任何股票"}})
        return {"ok": False, "error": "還沒有分析任何股票，請先 analyze_stock"}
    issues = audit_answer(answer or {}, s)
    if issues:
        s.rewrites += 1
        s.steps.append({"n": len(s.steps) + 1, "tool": "finalize_rejected", "ok": False, "decided_by": "critic",
                        "result": {"issues": issues[:6]}})
        if s.rewrites > AGENT_SETTINGS["max_rewrites"]:
            return {"ok": False, "error": "Critic 多次未通過，系統將改用確定性範本", "issues": issues[:6]}
        return {"ok": False, "error": "Critic 未通過，請修正後再 finalize", "issues": issues[:6]}
    s.final = answer
    s.steps.append({"n": len(s.steps) + 1, "tool": "finalize", "ok": True, "decided_by": "planner"})
    return {"ok": True}


def run_fixed(s: Session, symbols: list[str]) -> None:
    """備援固定流程：同一組工具、確定性順序。"""
    s.call("get_market_regime", {}, decided_by="fallback")
    for sym in symbols:
        r = s.call("analyze_stock", {"symbol": sym}, decided_by="fallback")
        if not r.get("ok"):
            continue
        s.call("explain_decision", {"symbol": sym}, decided_by="fallback")
        if not s.ctx[sym]["decision"]["bullish_allowed"]:
            s.call("what_if_entry", {"symbol": sym}, decided_by="fallback")
    if s.ctx and not any(c["decision"]["bullish_allowed"] for c in s.ctx.values()):
        s.call("list_alternatives", {"max_n": 3}, decided_by="fallback")


def answer_question(question: str, config, symbols: Optional[list[str]] = None, top_n: int = 30,
                    analyze_fn: Optional[Callable] = None, market_fn: Optional[Callable] = None,
                    post_fn: Optional[Callable] = None) -> Session:
    s = Session(config, question, top_n=top_n, analyze_fn=analyze_fn, market_fn=market_fn)
    symbols = symbols or list(dict.fromkeys(SYMBOL_RE.findall(question)))
    if not AGENT_SETTINGS.get("dry_run"):
        try:
            run_planner(s, post_fn or _post)
        except Exception as e:                               # noqa: BLE001
            s.steps.append({"n": len(s.steps) + 1, "tool": "planner_error", "ok": False, "decided_by": "gate",
                            "result": {"error": str(e)[:300]}})
    if s.final is None:
        s.mode = "fixed_pipeline" if AGENT_SETTINGS.get("dry_run") else "agent→fixed_pipeline"
        if not s.ctx and symbols:
            run_fixed(s, symbols)
        elif s.ctx and not any(st["tool"] == "get_market_regime" for st in s.steps):
            s.call("get_market_regime", {}, decided_by="fallback")
        s.final = template_answer(s)
        issues = audit_answer(s.final, s)
        s.steps.append({"n": len(s.steps) + 1, "tool": "template_answer", "ok": not issues, "decided_by": "fallback",
                        "result": {"issues": issues}})
    return s


# ============================================================
# 輸出
# ============================================================
def render(s: Session) -> str:
    a = s.final or {}
    lines = [f"【問題】{s.question}", "", f"【回答】{a.get('conclusion', '')}", ""]
    for title, key in (("【各股系統決策】", "per_stock"), ("【依據】", "reasons"), ("【決策可能改變的條件】", "triggers"),
                       ("【替代選項】", "alternatives"), ("【風險與限制】", "risks")):
        items = A._as_list(a.get(key))
        if items:
            lines.append(title)
            lines += [f"・{t}" for t in items]
            lines.append("")
    lines.append("【引用事實】")
    lines += [f"{f['id']}　{f['text']}" for f in s.facts]
    lines += ["", "【決策軌跡】"]
    for st in s.steps:
        args = ", ".join(f"{k}={v}" for k, v in (st.get("args") or {}).items())
        who = {"planner": "Claude 規劃", "fallback": "固定流程", "critic": "Critic", "gate": "閘門"}.get(st.get("decided_by"), st.get("decided_by"))
        lines.append(f"{st['n']:>2}. {st['tool']}({args})　{'✓' if st.get('ok') else '✗'}　{who}"
                     + (f"　{st['seconds']} 秒" if st.get("seconds") is not None else ""))
    mode = {"agent": "Claude 規劃", "fixed_pipeline": "固定流程（未呼叫 API）",
            "agent→fixed_pipeline": "Claude 規劃未完成，改用固定流程"}[s.mode]
    lines += ["", f"產生方式：{mode}；規劃 token：輸入 {s.tokens['input']}／輸出 {s.tokens['output']}；耗時 {time.time() - s.started:.0f} 秒",
              "本回答只解讀系統既有決策，不改變任何數字或建議；不構成投資建議。"]
    return "\n".join(lines)


def save(s: Session, text: str) -> Path:
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(list(s.ctx)[:3]) or "general"
    base = AGENT_DIR / f"{A.now_taipei().strftime('%Y%m%d_%H%M%S')}_{tag}"
    md = base.with_suffix(".md")
    md.write_text(f"# 決策代理回答\n\n{text}\n", encoding="utf-8")
    trace = {"session": s.id, "question": s.question, "mode": s.mode, "answer": s.final, "facts": s.facts, "steps": s.steps,
             "decisions": {sym: c["decision"] for sym, c in s.ctx.items()}, "tokens": s.tokens,
             "created_at": A.now_taipei().isoformat()}
    Path(str(base) + "_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    try:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": A.now_taipei().isoformat(), "question": s.question, "mode": s.mode,
                                "decisions": {sym: c["decision"]["label"] for sym, c in s.ctx.items()},
                                "tools": [st["tool"] for st in s.steps]}, ensure_ascii=False) + "\n")
    except Exception:                                       # noqa: BLE001
        pass
    return md


def run_agent_cli(question: str, config, symbol: Optional[str] = None, top_n: int = 30,
                  dry_run: bool = False, model: Optional[str] = None) -> Session:
    AGENT_SETTINGS["dry_run"] = bool(dry_run)
    if model:
        AGENT_SETTINGS["model"] = model
    print("=" * 70)
    print("AI 決策代理" + ("（固定流程，不呼叫 API）" if dry_run else ""))
    print("=" * 70)
    print(f"問題：{question}")
    print("分析中（每檔股票第一次分析約需數分鐘）…")
    s = answer_question(question, config, symbols=[symbol] if symbol else None, top_n=top_n)
    text = render(s)
    print(text)
    print(f"\n回答與決策軌跡已輸出：{save(s, text)}")
    return s
