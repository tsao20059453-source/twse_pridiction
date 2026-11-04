"""第四層：AI 決策代理 — 階段 2A「Claude 分析師報告」（2026-10-26 新增）

定位
    決策（是否進場、倉位、停損停利）在這之前已由第一～三層（ML 評分 → 規則過濾 → 風險控管）決定。
    本模組只負責把「事實 → 規則 → 決策」寫成分析師可讀的推理，不改變任何數字或決策。

流程（與玉山 ETF 專案相同的四層防護）
    1. 事實層 build_facts()      ：把報告欄位整理成編號事實 F1、F2…（只用程式算出的數字）。
    2. 撰寫層 call Claude         ：每一段都必須以（F編號）引用事實；只輸出 JSON。
    3. 欄位檢核 _field_ok()       ：長度、繁體中文、引用的編號必須存在。
    4. Critic 稽核 audit()        ：數字必須來自事實、結論不得與實際決策相反、不得有保證獲利或預測股價的用語。
       不合格的段落改用確定性範本 template_report() 重寫；API 失敗或沒有金鑰時全部改用範本。

使用
    python Pipeline_clean.py --mode single_diagnostic --symbol 2330 --ai-report
    參數：--ai-report（啟用）、--ai-model（預設 claude-sonnet-5-5）、--ai-dry-run（不呼叫 API，只用範本並存下 prompt）
    API 金鑰：在專題資料夾的 .env 加一行 ANTHROPIC_API_KEY=…（程式只從環境變數讀取，不寫入任何紀錄）。
輸出
    報告第十二段「AI 分析師解讀」＋ output/ai_reports/<代號>_<日期>_<時間>.md／.json
    每次 API 呼叫的完整 prompt 與回應：logs/llm_calls/<日期>/…json
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ============================================================
# 設定
# ============================================================
AI_SETTINGS: dict = {
    "enabled": False,
    "dry_run": False,
    "model": "claude-sonnet-5-5",
    "max_tokens": 8000,
    "timeout_sec": 180,
}
ROOT_DIR = Path(__file__).resolve().parent
AI_REPORT_DIR = ROOT_DIR / "output" / "ai_reports"
LLM_LOG_DIR = ROOT_DIR / "logs" / "llm_calls"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TZ_TAIPEI = timezone(timedelta(hours=8))


def set_ai_mode(enabled: bool, model: Optional[str] = None, dry_run: bool = False) -> None:
    AI_SETTINGS["enabled"] = bool(enabled or dry_run)
    AI_SETTINGS["dry_run"] = bool(dry_run)
    if model:
        AI_SETTINGS["model"] = model
    if AI_SETTINGS["enabled"]:
        load_env_file(ROOT_DIR / ".env")
        mode = "dry-run（不呼叫 API，只用範本）" if dry_run else f"模型 {AI_SETTINGS['model']}"
        print(f"AI 分析師報告：開啟（{mode}）")


def load_env_file(path: Path) -> None:
    """讀取 .env 的 KEY=VALUE（不覆蓋已存在的環境變數，不印出任何內容）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)


# ============================================================
# LLM 呼叫層（改寫自玉山 ETF 專案 esun/llm/client.py，直接用 HTTP，不依賴 SDK）
# ============================================================
@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    log_path: Optional[str] = None


class LLMError(RuntimeError):
    pass


class ClaudeClient:
    def __init__(self, model: str, max_tokens: int, timeout_sec: int, purpose: str = "analyst") -> None:
        self.model, self.max_tokens, self.timeout, self.purpose = model, max_tokens, timeout_sec, purpose

    def complete(self, system: str, user: str, tag: str = "") -> LLMResult:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("沒有設定 ANTHROPIC_API_KEY（請在專題資料夾的 .env 加一行 ANTHROPIC_API_KEY=…）")
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}]}
        started, t0 = now_taipei(), time.time()
        data = self._post(body, key)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        res = LLMResult(text=text, model=self.model, input_tokens=int(usage.get("input_tokens", 0)),
                        output_tokens=int(usage.get("output_tokens", 0)), stop_reason=data.get("stop_reason"))
        res.log_path = str(self._log(system, user, res, tag, started, time.time() - t0))
        if res.stop_reason == "max_tokens":
            print(f"[AI] 警告：Claude 輸出達上限（max_tokens={self.max_tokens}）被截斷")
        return res

    def _post(self, body: dict, key: str) -> dict:
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
        last: Any = None
        for attempt in range(1, 4):
            try:
                r = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last = e
                time.sleep(5 * attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 529):
                last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(10 * attempt)
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        raise LLMError(f"Claude 呼叫連續失敗：{last}")

    def _log(self, system: str, user: str, res: LLMResult, tag: str, started: datetime, elapsed: float) -> Path:
        folder = LLM_LOG_DIR / started.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{started.strftime('%H%M%S')}_{self.purpose}_{tag}_{uuid.uuid4().hex[:8]}.json"
        rec = {**asdict(res), "started_at": started.isoformat(), "elapsed_sec": round(elapsed, 2),
               "max_tokens": self.max_tokens, "purpose": self.purpose, "tag": tag, "system": system, "user": user}
        rec.pop("log_path", None)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return path


def extract_json(text: str) -> dict:
    """從回應取出第一個 JSON 物件（容忍 ```json 包裝或前後說明文字）。"""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError("回應中找不到 JSON")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 括號不完整")


# ============================================================
# 事實層
# ============================================================
def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _pct(v, d: int = 2) -> str:
    f = _num(v)
    return "NA" if f is None else f"{f * 100:.{d}f}%"


def _px(v) -> str:
    f = _num(v)
    return "NA" if f is None else f"{f:.2f}"


def _sc(v) -> str:
    f = _num(v)
    return "NA" if f is None else f"{f:.2f}"


def _txt(v, default: str = "NA") -> str:
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none") else default


def _name_of(symbol) -> str:
    try:
        from pipeline_data import get_stock_name
        return get_stock_name(symbol)
    except Exception:
        return ""


def stock_label(symbol, name: str = "") -> str:
    """「名稱 代號」，例如「台積電 2330」；查不到名稱時只顯示代號。"""
    sym = str(symbol or "").strip()
    name = (name or _name_of(sym)).rstrip("*").strip()
    return f"{name} {sym}" if name else sym


def _summary_get(summary, keys, default=None):
    keys = [keys] if isinstance(keys, str) else keys
    for k in keys:
        try:
            if isinstance(summary, dict) and k in summary:
                return summary[k]
            if hasattr(summary, "get"):
                v = summary.get(k)
                if v is not None:
                    return v
        except Exception:
            continue
    return default


HARD_RULES = {"stop_min": 0.012, "stop_max": 0.08, "rr_min": 1.5, "score_min": 0.55}


def hard_conditions(row) -> list[dict]:
    """回測引擎的 5 項進場硬條件（與報告「五之二」相同的判斷）。"""
    out = []
    above = _num(row.get("market_above_ma20"))
    out.append({"name": "大盤過濾", "pass": None if above is None else bool(above >= 0.5),
                "detail": "TAIEX 高於 MA20" if above and above >= 0.5 else "TAIEX 低於 MA20" if above is not None else "無資料"})
    path = _txt(row.get("entry_support_source") or row.get("support_entry_path"), "")
    zone_low = _num(row.get("support_entry_zone_low"))
    out.append({"name": "支撐進場路徑", "pass": zone_low is not None, "detail": path or ("有支撐進場區間" if zone_low else "無")})
    sl = _num(row.get("entry_stop_loss_pct"))
    out.append({"name": "停損距離 1.2%～8%", "pass": None if sl is None else HARD_RULES["stop_min"] <= sl <= HARD_RULES["stop_max"],
                "detail": f"進場停損距離 {_pct(sl)}"})
    rr = _num(row.get("entry_rr_1"))
    out.append({"name": "R/R ≥ 1.5", "pass": None if rr is None else rr >= HARD_RULES["rr_min"], "detail": f"第一止盈 R/R {_sc(rr)}"})
    fs = _num(row.get("final_decision_score"))
    ov = bool(row.get("entry_plan_override", False))
    # 與回測引擎、報告「五之二」一致：分數 ≥ 0.55，或回測支撐進場計畫成立（entry_plan_override），即滿足分數門檻
    ok = None if fs is None else (fs >= HARD_RULES["score_min"] or ov)
    detail = f"最終決策分數 {_sc(fs)}" + ("，未達 0.55 但回測支撐進場計畫成立" if ov and fs is not None and fs < HARD_RULES["score_min"] else "")
    out.append({"name": "分數門檻（≥ 0.55 或回測進場計畫）", "pass": ok, "detail": detail})
    return out


# ============================================================
# 分數解讀（把 0～1 的分數翻成白話；所有數字由程式計算）
# ============================================================
FINAL_WEIGHTS = {"safety_score": 0.30, "return_score": 0.20, "rule_filter_score": 0.25, "risk_reward_score": 0.25}   # 論文版（備用）
SCORE_NAMES_ZH = {"safety_score": "安全分數", "return_score": "報酬分數", "rule_filter_score": "規則分數", "risk_reward_score": "風險報酬分數"}


def live_weights() -> dict:
    """目前實際使用的最終分數權重（跟著 --score-profile）。"""
    try:
        import pipeline_risk
        return dict(pipeline_risk.FINAL_SCORE_WEIGHTS)
    except Exception:
        return dict(FINAL_WEIGHTS)


def formula_zh(weights: Optional[dict] = None) -> str:
    w = weights or live_weights()
    return "＋".join(f"{w[k]:.2f}×{SCORE_NAMES_ZH[k]}" for k in SCORE_NAMES_ZH if w.get(k, 0) > 0)


def _bar(v: Optional[float], width: int = 20, mark: Optional[float] = None) -> str:
    """文字長條：每格 0.05；mark 為門檻位置（以 ┃ 標示）。"""
    if v is None:
        return "░" * width
    filled = max(0, min(width, int(round(v * width))))
    cells = ["█" if i < filled else "░" for i in range(width)]
    if mark is not None:
        k = max(0, min(width - 1, int(round(mark * width))))
        cells.insert(k, "┃")
    return "".join(cells)


def _level(v: Optional[float], hi: str, mid_hi: str, mid: str, mid_lo: str, lo: str) -> str:
    if v is None:
        return "無資料"
    return hi if v >= 0.65 else mid_hi if v >= 0.55 else mid if v >= 0.45 else mid_lo if v >= 0.35 else lo


def _caps(row) -> list[dict]:
    """重現 pipeline_risk.build_final_decision 的分數上限規則（只讀取，不改變決策）。"""
    caps = []
    rule = _num(row.get("rule_filter_score"))
    tfr = str(row.get("technical_filter_result", "")).strip().lower()
    if tfr in ("fail", "技術條件偏弱") or (rule is not None and rule < 0.35):
        caps.append({"cap": 0.49, "cond": "技術規則分數低於 0.35（技術面偏弱）", "reason": "技術規則分數低於 0.35（技術面偏弱）時，最終分數最高 0.49"})
    sl = _num(row.get("stop_loss_pct"))
    if sl is not None and sl > 0.06:
        caps.append({"cap": 0.69, "cond": "以收盤價進場的停損距離超過 6%", "reason": "以收盤價進場的停損距離超過 6%時，最終分數最高 0.69"})
    rr = _num(row.get("practical_risk_reward_ratio"))
    rr = rr if rr is not None else _num(row.get("risk_reward_ratio"))
    if rr is not None and rr < 2.0:
        caps.append({"cap": 0.69, "cond": "以收盤價進場的風險報酬比低於 2.0", "reason": "以收盤價進場的風險報酬比低於 2.0時，最終分數最高 0.69"})
    if str(row.get("stop_loss_too_close_flag", False)).strip().lower() in {"true", "1", "yes", "y"}:
        caps.append({"cap": 0.59, "cond": "停損距離過近", "reason": "停損距離過近時，最終分數最高 0.59"})
    res = _num(row.get("nearest_resistance_level"))
    if res is None or res <= 0:
        caps.append({"cap": 0.69, "cond": "找不到有效壓力位", "reason": "找不到有效壓力位時，最終分數最高 0.69"})
    try:
        from pipeline_risk import _has_material_warning
        if _has_material_warning(row.get("risk_warning", "")):
            caps.append({"cap": 0.69, "cond": "有系統風險警示", "reason": "有系統風險警示時，最終分數最高 0.69"})
    except Exception:
        pass
    if bool(row.get("trade_veto_flag", False)):
        caps.append({"cap": 0.39, "cond": "交易被否決", "reason": "交易被否決時，最終分數最高 0.39"})
    return caps


def score_breakdown(row) -> dict:
    """把最終決策分數拆成：各分數白話意義 → 加權貢獻 → 規則扣分 → 分數上限 → 與門檻的差距。"""
    W = live_weights()
    v = {k: _num(row.get(k)) for k in W}
    final = _num(row.get("final_decision_score"))
    override = bool(row.get("entry_plan_override", False))
    rr_basis = _num(row.get("entry_rr_1")) if override else _num(row.get("practical_risk_reward_ratio"))
    rr_where = "回測支撐進場" if override else "以收盤價進場"

    s, r, ru, rk = v["safety_score"], v["return_score"], v["rule_filter_score"], v["risk_reward_score"]
    comps = [
        {"key": "safety_score", "name": "安全分數",
         "meaning": (f"模型預估未來 10 天的最大跌幅，比同一天股票池中約 {round(s * 100)}% 的股票小（" +
                     _level(s, "明顯比平均安全", "略比平均安全", "與平均相當", "略比平均危險", "明顯比平均危險") + "）")
         if s is not None else "無資料"},
        {"key": "return_score", "name": "報酬分數",
         "meaning": (f"模型預估未來 10 天漲幅勝過股票池中約 {round(r * 100)}% 的股票；但此模型幾乎沒有預測力，接近 0.50 時等於沒有資訊")
         if r is not None else "無資料"},
        {"key": "rule_filter_score", "name": "規則分數",
         "meaning": (f"技術面體檢（0.50 起算，利多指標加分、利空指標扣分；0.65 以上良好、0.45 以下偏弱）→ " +
                     ("良好" if ru >= 0.65 else "中性" if ru >= 0.45 else "偏弱" if ru >= 0.25 else "扣分項遠多於加分項，明顯偏弱"))
         if ru is not None else "無資料"},
        {"key": "risk_reward_score", "name": "風險報酬分數",
         "meaning": ((f"{rr_where}時，潛在獲利約為潛在虧損的 {rr_basis:.2f} 倍 → " if rr_basis is not None else "") +
                     ("低於 1.5 倍，不給分" if (rk or 0) <= 0 else "1.5～2 倍，給 0.40" if rk <= 0.4 + 1e-9
                      else "滿分" if rk >= 0.999 else "2 倍以上，分數隨倍數增加"))
         if rk is not None else "無資料"},
    ]
    raw = 0.0
    complete = all(v[k] is not None for k in W if W[k] > 0)
    for c in comps:
        val = v[c["key"]]
        c["value"], c["weight"] = val, W[c["key"]]
        c["contrib"] = None if val is None else val * c["weight"]
        if c["weight"] == 0 and val is not None:
            c["meaning"] += "（目前不計入最終分數，僅作風險參考）"
        raw += c["contrib"] or 0.0
    pen_pts = _num(row.get("rule_penalty_points")) or 0.0
    penalty = min(0.10, max(0.0, min(1.0, pen_pts)) * 0.10)
    after_pen = max(0.0, min(1.0, raw - penalty))
    caps = _caps(row)
    cap_val = min((c["cap"] for c in caps), default=None)
    recomputed = min(after_pen, cap_val) if cap_val is not None else after_pen
    consistent = complete and final is not None and abs(recomputed - final) <= 0.006
    binding = [c for c in caps if c["cap"] == cap_val] if (cap_val is not None and after_pen > cap_val) else []
    # 拉低分數最多的項目：與「每項都拿 0.5 平均水準」相比，少掉最多的那一項
    drag = None
    if complete:
        drag = min((c for c in comps if c["weight"] > 0), key=lambda c: c["contrib"] - 0.5 * c["weight"])
    th = HARD_RULES["score_min"]
    # 低於門檻的分數上限＝天花板：不論是否正在壓分數，只要存在就不可能達到門檻
    ceiling = min((c for c in caps if c["cap"] < th), key=lambda c: c["cap"], default=None)
    return {"components": comps, "raw": raw, "penalty": penalty, "after_penalty": after_pen, "caps": caps,
            "binding_caps": binding, "cap": cap_val, "final": final, "recomputed": recomputed, "consistent": consistent,
            "threshold": th, "gap": None if final is None else th - final, "ceiling": ceiling,
            "drag": drag["name"] if drag else None, "override": override}


def score_facts_text(b: dict) -> list[str]:
    """分數解讀類事實（給 Claude 引用；數字都在這裡出現，Critic 才會放行）。"""
    out = []
    fin, gap = b["final"], b["gap"]
    if fin is not None:
        where = f"還差 {gap:.2f}" if gap > 0 else f"高出 {-gap:.2f}"
        out.append(f"最終決策分數 {fin:.2f} 分（滿分 1 分），距離進場門檻 0.55 {where}")
    for c in b["components"]:
        if c["value"] is not None:
            if c["weight"] > 0:
                out.append(f"{c['name']} {c['value']:.2f}，佔最終分數權重 {round(c['weight'] * 100)}%，貢獻 {c['contrib']:.2f} 分；白話：{c['meaning']}")
            else:
                out.append(f"{c['name']} {c['value']:.2f}，目前權重 0%，不計入最終分數；白話：{c['meaning']}")
    if b["consistent"]:
        n_used = sum(1 for c in b["components"] if c["weight"] > 0)
        chain = f"{'四' if n_used == 4 else '兩' if n_used == 2 else n_used}項加權合計 {b['raw']:.2f}"
        if b["penalty"] > 0:
            chain += f"，技術規則扣分 {b['penalty']:.2f} 後為 {b['after_penalty']:.2f}"
        if b["binding_caps"]:
            chain += f"，再受分數上限 {b['cap']:.2f} 限制（{b['binding_caps'][0]['reason']}）"
        chain += f"，得到最終分數 {b['final']:.2f}"
        if b["drag"]:
            chain += f"；與各項都在平均水準 0.50 相比，拉低分數最多的是{b['drag']}"
        out.append(chain)
    if b["override"] and fin is not None and fin < b["threshold"]:
        out.append(f"回測支撐進場計畫成立：最終分數 {fin:.2f} 雖未達 0.55，但依回測引擎規則，分數門檻改由此計畫滿足；"
                   f"系統不追目前價格，價格回到支撐進場區間且符合進場條件時才會進場")
    elif b["ceiling"]:
        out.append(f"關鍵：只要{b['ceiling']['cond']}，最終分數最高只有 {b['ceiling']['cap']:.2f}，到不了 0.55 門檻；"
                   f"這個條件解除前，其他分數再高也無法以分數進場")
    out.append("安全分數與報酬分數是模型對「在股票池中排第幾」的預估，不是上漲機率，也不是預估漲跌幅")
    return out


def render_score_panel(ctx: dict) -> list[str]:
    """確定性的「分數怎麼看」區塊，不經過 Claude，永遠與系統數字一致。"""
    b = ctx.get("score")
    if not b or b["final"] is None:
        return []
    fin, th, gap = b["final"], b["threshold"], b["gap"]
    lines = ["【分數怎麼看】（每項 0～1 分；0.50 ≈ 股票池平均水準；進場需最終分數 ≥ 0.55，或「回測支撐進場計畫」成立）"]
    status = f"離門檻還差 {gap:.2f}" if gap > 0 else f"已超過門檻 {-gap:.2f}"
    if gap > 0 and b["override"]:
        status += "，但回測支撐進場計畫成立"
    lines.append(f"最終決策分數 {fin:.2f}  {_bar(fin, mark=th)}  ┃＝門檻 {th:.2f}，{status}")
    lines.append("組成（分數 × 權重 = 貢獻）：")
    for i, c in enumerate(b["components"]):
        tree = "└" if i == len(b["components"]) - 1 else "├"
        if c["value"] is None:
            lines.append(f"  {tree} {c['name']}：無資料")
            continue
        lines.append(f"  {tree} {c['name'] + '　' * (6 - len(c['name']))} {c['value']:.2f} × {round(c['weight'] * 100):>2}% = {c['contrib']:.2f}  {_bar(c['value'], 10)}  {c['meaning']}")
    if b["consistent"]:
        chain = f"  合計 {b['raw']:.2f}"
        if b["penalty"] > 0:
            chain += f" → 技術扣分 −{b['penalty']:.2f} = {b['after_penalty']:.2f}"
        if b["binding_caps"]:
            chain += f" → 分數上限 {b['cap']:.2f}"
        chain += f" → 最終 {fin:.2f}"
        lines.append(chain)
        for c in b["binding_caps"]:
            lines.append(f"  ※ 分數上限：{c['reason']}")
    else:
        lines.append(f"  （最終分數另經系統風控調整，上列為各項分數的加權貢獻，合計 {b['raw']:.2f}）")
    if gap > 0 and b["override"]:
        lines.append("  ⚠ 關鍵：分數未達 0.55，系統改走「回測支撐進場計畫」——不追目前價格，"
                     "等價格回到支撐進場區間、且符合量能等進場條件時才進場。")
        if b["ceiling"]:
            lines.append(f"  　（若沒有這個計畫：只要{b['ceiling']['cond']}，最終分數最高只有 {b['ceiling']['cap']:.2f}，無法以分數進場）")
    elif b["ceiling"] and gap > 0:
        lines.append(f"  ⚠ 關鍵：只要{b['ceiling']['cond']}，最終分數最高只有 {b['ceiling']['cap']:.2f}，到不了 {th:.2f} 門檻"
                     f"——這個條件解除前，其他分數再好也進不了場。")
    elif b["consistent"] and b["drag"] and gap > 0:
        lines.append(f"  ⚠ 拉低分數最多的是「{b['drag']}」（與各項都拿 0.50 平均水準相比），它是未達門檻的主因。")
    lines.append("  註：安全／報酬分數是「排名」的預估（例如 0.73 ≈ 贏過約七成股票），不是上漲機率，也不是漲跌幅。")
    return lines


def build_facts(row, summary=None, symbol: str = "", stock_name: str = "") -> dict:
    """回傳 {'facts': [{'id','group','text'}], 'decision': {...}, 'meta': {...}}。所有數字都來自程式計算結果。"""
    facts: list[dict] = []

    def add(group: str, text: str) -> str:
        fid = f"F{len(facts) + 1}"
        facts.append({"id": fid, "group": group, "text": text})
        return fid

    veto = bool(row.get("trade_veto_flag", False))
    label = _txt(row.get("final_decision_label_zh") or row.get("final_decision_label"))
    action = _txt(row.get("decision_action_zh") or row.get("decision_action"))
    pos_pct = _num(row.get("suggested_position_pct"))
    conds = hard_conditions(row)
    n_pass = sum(1 for c in conds if c["pass"])

    # 決策
    add("決策", f"系統最終決策標籤為「{label}」，決策動作為「{action}」")
    add("決策", f"最終決策分數 {_sc(row.get('final_decision_score'))}（公式 {formula_zh()}；進場門檻 0.55）")
    add("決策", f"安全分數 {_sc(row.get('safety_score'))}、報酬機會分數 {_sc(row.get('return_score'))}、第一層模型綜合分數 {_sc(row.get('combined_decision_score'))}")
    add("決策", f"交易否決：{'是' if veto else '否'}" + (f"，原因：{_txt(row.get('trade_veto_reason_zh') or row.get('trade_veto_reason'))}" if veto else ""))
    add("決策", f"建議倉位 {_pct(pos_pct)}（{_txt(row.get('position_sizing_label'))}）")
    add("決策", f"回測引擎 5 項進場硬條件通過 {n_pass} 項：" + "；".join(
        f"{c['name']}{'通過' if c['pass'] else '未通過' if c['pass'] is not None else '無資料'}（{c['detail']}）" for c in conds))
    # 分數解讀
    breakdown = score_breakdown(row)
    for t in score_facts_text(breakdown):
        add("分數解讀", t)
    # 大盤
    dist = _num(row.get("market_ma20_distance"))
    above = _num(row.get("market_above_ma20"))
    if above is not None:
        add("大盤", f"加權指數{'高於' if above >= 0.5 else '低於'} 20 日均線，距離 {_pct(dist)}")
    # 價格與技術
    _sym = symbol or _txt(row.get('symbol'))
    stock_name = (stock_name or _name_of(_sym)).rstrip("*").strip()
    add("價格", f"股票 {stock_label(_sym, stock_name)}，最新收盤價 {_px(row.get('close'))}，資料截止日 {_txt(str(row.get('date'))[:10])}")
    add("技術", f"RSI14 為 {_sc(row.get('RSI14'))}，MACD 為 {_sc(row.get('MACD'))}")
    add("技術", f"KD 指標 K={_sc(row.get('K'))}、D={_sc(row.get('D'))}、J={_sc(row.get('J'))}")
    add("技術", f"24 日乖離率 {_txt(_sc(row.get('BIAS24')))}%，VR_14 為 {_sc(row.get('VR_14'))}")
    add("技術", f"近 5 日報酬 {_pct(row.get('return_5d'))}，近 10 日報酬 {_pct(row.get('return_10d'))}")
    add("價位", f"最近支撐 {_px(row.get('nearest_support_level'))}（距離 {_pct(row.get('support_distance'))}），"
                f"最近壓力 {_px(row.get('nearest_resistance_level'))}（距離 {_pct(row.get('resistance_distance'))}）")
    add("價位", f"支撐品質分數 {_sc(row.get('support_quality_score'))}（0.7 以上為強支撐、0.4 以下為弱支撐）")
    add("價位", f"Fibonacci 位置 {_sc(row.get('fib_position'))}（{_txt(row.get('fib_zone'))}），ATR 佔股價 {_pct(row.get('atr_pct'))}")
    # 規則與風險報酬
    add("規則", f"第二層規則分數 {_sc(row.get('rule_filter_score'))}，判讀：{_txt(row.get('technical_filter_result_zh') or row.get('technical_filter_result'))}")
    notes = _txt(row.get("rule_filter_notes_zh") or row.get("rule_filter_notes"), "")
    if notes:
        add("規則", f"規則判讀細項：{notes}")
    add("風險報酬", f"以收盤價進場：停損 {_px(row.get('stop_loss_price_adjusted'))}（距離 {_pct(row.get('stop_loss_pct'))}），"
                    f"目標 {_px(row.get('take_profit_price'))}，實務風險報酬比 {_sc(row.get('practical_risk_reward_ratio'))}（門檻 1.5）")
    if _num(row.get("support_entry_zone_low")) is not None:
        add("進場計畫", f"回測支撐進場區間 {_px(row.get('support_entry_zone_low'))}～{_px(row.get('support_entry_zone_high'))}，"
                        f"參考價 {_px(row.get('support_entry_price'))}")
    if _num(row.get("breakout_entry_zone_low")) is not None:
        add("進場計畫", f"突破確認進場區間 {_px(row.get('breakout_entry_zone_low'))}～{_px(row.get('breakout_entry_zone_high'))}")
    if _num(row.get("entry_stop_loss_price")) is not None:
        add("進場計畫", f"進場計畫停損 {_px(row.get('entry_stop_loss_price'))}（距離 {_pct(row.get('entry_stop_loss_pct'))}），"
                        f"第一止盈 {_px(row.get('entry_take_profit_1'))}（R/R {_sc(row.get('entry_rr_1'))}），"
                        f"第二止盈 {_px(row.get('entry_take_profit_2'))}（R/R {_sc(row.get('entry_rr_2'))}）")
    cond_text = _txt(row.get("entry_condition_text"), "")
    if cond_text:
        add("進場計畫", f"進場條件說明：{cond_text}")
    warn = _txt(row.get("risk_warning"), "")
    if warn:
        add("風險", f"系統風險警示：{warn}")
    # 模型可信度
    sp = _num(_summary_get(summary, ["ranking_spearman", "spearman_corr"]))
    if sp is not None:
        add("模型", f"安全模型 walk-forward 驗證 Spearman 為 {sp:.4f}（排序能力有參考價值）")
    add("模型", "報酬模型在加入清洗期的樣本外驗證中 IC 約 0.003，幾乎沒有預測力，系統只把它當輔助訊號")

    decision = {"veto": veto, "label": label, "action": action, "position_pct": pos_pct or 0.0,
                "final_score": _num(row.get("final_decision_score")), "conditions": conds, "n_pass": n_pass,
                "bullish_allowed": (not veto) and (pos_pct or 0) > 0}
    meta = {"symbol": symbol or _txt(row.get("symbol")), "name": stock_name, "date": _txt(str(row.get("date"))[:10]),
            "close": _num(row.get("close"))}
    return {"facts": facts, "decision": decision, "meta": meta, "score": breakdown}


# ============================================================
# 範本（確定性，永遠可用）
# ============================================================
def template_report(ctx: dict) -> dict:
    d, m = ctx["decision"], ctx["meta"]
    groups = {}
    for f in ctx["facts"]:
        groups.setdefault(f["group"], []).append(f["id"])
    g = lambda name, i=0, fb="決策": (groups.get(name) or groups[fb])[min(i, len(groups.get(name) or groups[fb]) - 1)]
    f_label, f_score, f_veto, f_pos, f_cond = g("決策", 0), g("決策", 1), g("決策", 3), g("決策", 4), g("決策", 5)
    passed = [c["name"] for c in d["conditions"] if c["pass"]]
    failed = [c["name"] for c in d["conditions"] if c["pass"] is False]

    sc_ids = groups.get("分數解讀", [])
    b = ctx.get("score") or {}
    score_sentence = f"最終決策分數與 0.55 門檻的比較見（{f_score}）。"
    if sc_ids and b.get("final") is not None:
        fin, gap = b["final"], b["gap"]
        chain_id = next((f["id"] for f in ctx["facts"] if f["group"] == "分數解讀" and "項加權合計" in f["text"][:6]), None)
        key_id = next((f["id"] for f in ctx["facts"] if f["group"] == "分數解讀" and f["text"].startswith("關鍵：")), None)
        ov_id = next((f["id"] for f in ctx["facts"] if f["group"] == "分數解讀" and f["text"].startswith("回測支撐進場計畫成立")), None)
        if gap > 0:
            score_sentence = f"最終決策分數 {fin:.2f}，距離 0.55 的進場門檻還差 {gap:.2f}（{sc_ids[0]}）。"
        else:
            score_sentence = f"最終決策分數 {fin:.2f}，已達 0.55 的進場門檻（{sc_ids[0]}）。"
        if ov_id and gap > 0:
            score_sentence += f"不過回測支撐進場計畫成立，系統不追目前價格，改為等待價格回到支撐區再進場（{ov_id}）。"
        elif key_id and gap > 0:
            score_sentence += f"關鍵在於觸發了低於門檻的分數上限，這個條件解除前，其他分數再高也無法進場（{key_id}）。"
        elif chain_id and b.get("drag") and gap > 0:
            score_sentence += f"拉低分數最多的是{b['drag']}（{chain_id}）。"

    if d["veto"]:
        reason = f"交易被風險控管否決（{f_veto}）"
    elif not d["bullish_allowed"]:
        reason = f"建議倉位為 0（{f_pos}）"
    else:
        reason = ""
    if reason:
        headline = f"{stock_label(m['symbol'], m.get('name', ''))} 目前不符合進場條件，系統決策為「{d['label']}」（{f_label}）"
        logic = (f"系統的最終決策是「{d['label']}」（{f_label}）。直接原因是{reason}；"
                 f"回測引擎的 5 項進場硬條件中，未通過的是{'、'.join(failed) if failed else '（無）'}（{f_cond}）。"
                 f"{score_sentence}在這些條件改善之前，系統不建議以目前價格進場。")
    else:
        headline = f"{stock_label(m['symbol'], m.get('name', ''))} 符合系統的進場條件，系統決策為「{d['label']}」（{f_label}）"
        logic = (f"系統的最終決策是「{d['label']}」（{f_label}），5 項進場硬條件與最終分數均達標（{f_cond}、{f_score}），"
                 f"倉位依風險控管規則配置（{f_pos}），進場後須遵守停損（{g('風險報酬', 0)}）。")

    support = []
    if passed:
        support.append(f"5 項硬條件中已通過：{'、'.join(passed)}（{f_cond}）。")
    safety_id = next((f["id"] for f in ctx["facts"] if f["group"] == "分數解讀" and f["text"].startswith("安全分數")), None)
    sv = next((c["value"] for c in (b.get("components") or []) if c["key"] == "safety_score"), None)
    if safety_id and sv is not None and sv >= 0.55:
        support.append(f"安全分數 {sv:.2f}：模型預估它未來 10 天的最大跌幅比股票池中約 {round(sv * 100)}% 的股票小（{safety_id}）。")
    support.append(f"技術指標現況見（{g('技術', 0, '價格')}、{g('技術', 1, '價格')}）。")
    oppose = []
    if failed:
        oppose.append(f"5 項硬條件中未通過：{'、'.join(failed)}（{f_cond}）。")
    if "風險" in groups:
        oppose.append(f"系統風險警示（{g('風險', 0)}）。")
    oppose.append(f"支撐與壓力的相對位置影響風險報酬是否划算（{g('價位', 0, '價格')}、{g('風險報酬', 0)}）。")

    plan = groups.get("進場計畫")
    trig_map = {
        "大盤過濾": f"加權指數重新站上 20 日均線，大盤過濾才會解除（{g('大盤', 0)}）。",
        "支撐進場路徑": f"價格出現有效支撐、形成支撐進場路徑（{g('價位', 0, '價格')}）。",
        "停損距離 1.2%～8%": f"進場停損距離回到 1.2%～8% 的合理區間（{plan[-1] if plan else g('風險報酬', 0)}）。",
        "R/R ≥ 1.5": f"價格回到支撐進場區間，使第一止盈的風險報酬比達到 1.5 以上（{plan[0] if plan else g('風險報酬', 0)}）。",
        "分數門檻（≥ 0.55 或回測進場計畫）": f"最終決策分數回升到 0.55 以上，或出現回測支撐進場計畫（{f_score}）。",
    }
    watch = [trig_map[n] for n in failed if n in trig_map][:3] or [
        f"若停損被觸發或風險警示升高，系統會重新評估（{g('風險報酬', 0)}）。"]
    model_ids = groups.get("模型", [f_label])
    caveat = (f"安全模型的排序能力有參考價值（{model_ids[0]}），但報酬模型幾乎沒有預測力（{model_ids[-1]}），"
              f"因此決策主要依據規則過濾與風險報酬比。")
    return {"headline": headline, "decision_logic": logic, "supporting_signals": support[:4], "opposing_signals": oppose[:4],
            "watch_triggers": watch, "model_caveat": caveat}


# ============================================================
# 欄位檢核與 Critic 稽核
# ============================================================
SIMPLIFIED_ONLY = set("这个们来时说对过还没为会经发问题么样进关应动现长东学实业场点产简历见体转处总统战权变认义党车电单么让该从间无边达观务价议华开门书语读识级运复视资币择与亚岁协缩汇础标网络结构线脑调节环观该虑审轮涨态贷财币兑确")
FORBIDDEN = ("保證", "必漲", "穩賺", "一定會漲", "必然上漲", "無風險", "肯定上漲", "保本", "包賺", "必賺")
BULLISH_WORDS = ("建議進場", "建議買進", "可以進場", "立即買進", "可進場", "買進訊號", "建議布局", "逢低買進", "可布局")
NEGATIONS = ("不是", "並非", "不代表", "並不", "不會", "沒有", "非", "不", "無", "未", "暫不", "尚未", "等待", "待")
NUM_RE = re.compile(r"(?<![A-Za-z\d.])(\d{1,6}(?:\.\d{1,4})?)\s*(%?)")
REF_RE = re.compile(r"F\d{1,3}")
SECTION_KEYS = ("headline", "decision_logic", "supporting_signals", "opposing_signals", "watch_triggers", "model_caveat")
ALWAYS_OK_NUMBERS = {0, 1, 2, 3, 4, 5, 10, 20, 60, 14, 24, 26, 0.55, 1.5, 1.2, 8, 0.7, 0.4, 0.3, 0.2, 0.25, 30, 50}


def _affirmed(text: str, word: str, window: int = 8) -> bool:
    for m in re.finditer(re.escape(word), text):
        pre = re.split(r"[，。；、,;：:]", text[max(0, m.start() - window):m.start()])[-1]
        if not any(n in pre for n in NEGATIONS):
            return True
    return False


def _as_list(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def _field_ok(v, fact_ids: set[str], lo: int, hi: int, need_ref: bool = True) -> bool:
    items = _as_list(v)
    if not items:
        return False
    for t in items:
        if not (lo <= len(t) <= hi) or any(c in SIMPLIFIED_ONLY for c in t):
            return False
        refs = set(REF_RE.findall(t))
        if refs - fact_ids or (need_ref and not refs):
            return False
    return True


@dataclass
class Issue:
    code: str
    section: str
    message: str


@dataclass
class CriticReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def failed_sections(self) -> set[str]:
        return {i.section for i in self.issues}

    def text(self) -> str:
        if self.ok:
            return "Critic 稽核：全部通過（數字皆有事實依據、結論與實際決策一致、無保證獲利用語）"
        return "Critic 稽核：\n" + "\n".join(f"  ✗ [{i.code}] {i.section}：{i.message}" for i in self.issues)


def _fact_numbers(ctx: dict) -> list[float]:
    vals = []
    for f in ctx["facts"]:
        for m in NUM_RE.finditer(f["text"]):
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    return vals


def audit(report: dict, ctx: dict) -> CriticReport:
    rep = CriticReport()
    fact_ids = {f["id"] for f in ctx["facts"]}
    fact_vals = _fact_numbers(ctx)
    d = ctx["decision"]
    for key in SECTION_KEYS:
        for t in _as_list(report.get(key)):
            body = REF_RE.sub("", t)                      # 事實編號本身不算數字
            # 1. 數字必須來自事實
            for m in NUM_RE.finditer(body):
                v = float(m.group(1))
                if v in ALWAYS_OK_NUMBERS or any(abs(v - x) <= max(0.011, abs(x) * 0.001) for x in fact_vals):
                    continue
                rep.issues.append(Issue("NUMBER_SUPPORT", key, f"「{m.group(0)}」找不到對應事實"))
                break
            # 2. 保證獲利／預測股價用語
            hit = [w for w in FORBIDDEN if _affirmed(t, w)]
            if hit:
                rep.issues.append(Issue("PROMISE", key, f"出現不當承諾用語 {hit}"))
            # 3. 結論不得與實際決策相反（否決時不得肯定地建議進場）
            if not d["bullish_allowed"] and key in ("headline", "decision_logic"):
                bad = [w for w in BULLISH_WORDS if _affirmed(t, w)]
                if bad:
                    rep.issues.append(Issue("DECISION_CONFLICT", key, f"實際決策為「{d['label']}」，文字卻寫 {bad}"))
            # 4. 引用的事實編號必須存在
            missing = set(REF_RE.findall(t)) - fact_ids
            if missing:
                rep.issues.append(Issue("BAD_REF", key, f"引用不存在的事實 {sorted(missing)}"))
    # 5. 決策段必須提到實際決策標籤
    logic = " ".join(_as_list(report.get("decision_logic")))
    if d["label"] not in logic and d["label"] != "NA":
        rep.issues.append(Issue("DECISION_MISSING", "decision_logic", f"沒有提到實際決策「{d['label']}」"))
    return rep


# ============================================================
# 撰寫層
# ============================================================
SYSTEM_PROMPT = """你是一位台股量化研究團隊的資深分析師，負責替「已經由系統決定好的投資決策」撰寫解讀。
系統架構：第一層機器學習模型（安全分數、報酬機會分數）→ 第二層技術規則過濾 → 第三層風險控管（停損、風險報酬比、倉位）。
最終決策分數 = {FORMULA}；進場還必須通過 5 項硬條件。
你的任務是解釋「為什麼系統得到這個決策」，以及「哪些條件改變時決策會改變」，不是重新判斷要不要買。

寫作規則：
1. 一律使用繁體中文與台灣金融業慣用語。
2. 只能使用提供的事實。每一句引用事實時，在句中以括號註明事實編號，例如（F3）。不得自行編造任何數字、新聞、財報或目標價。
3. 結論必須與系統決策一致：系統否決或倉位為 0 時，不得寫「建議進場、可以買進」之類的肯定建議。
4. 不得使用保證獲利、預測股價必然漲跌的用語。報酬模型幾乎沒有預測力，不得把它當成主要依據。
5. 要同時寫支持與反對的訊號，讓讀者看到兩面。
6. 讀者不是量化專家。提到任何分數時，必須在同一句用白話說明它代表什麼（引用「分數解讀」類事實），
   例如「安全分數 0.73，代表模型預估它未來 10 天的跌幅比股票池中約七成股票小（F9）」；
   不要連續羅列多個沒有解釋的分數。解釋最終分數為何未達門檻時，先說「還差多少」，再指出拉低最多的項目或分數上限。
7. 提到股票時寫成「名稱 代號」，例如「台積電 2330」（名稱見事實中的「股票」欄）。
8. 只輸出一個 JSON 物件，不要其他文字。"""

def system_prompt() -> str:
    return SYSTEM_PROMPT.replace("{FORMULA}", formula_zh())


OUTPUT_SCHEMA = """{
  "headline": "一句話總結（20～60 字）",
  "decision_logic": "決策推理：哪些事實 → 依哪條規則 → 得到這個決策（80～320 字）",
  "supporting_signals": ["有利於進場的訊號，每點 20～120 字，可 1～4 點"],
  "opposing_signals": ["不利於進場的訊號，每點 20～120 字，可 1～4 點"],
  "watch_triggers": ["哪些具體條件出現時決策可能改變，每點 20～120 字，1～3 點"],
  "model_caveat": "模型可信度與使用限制（30～160 字）"
}"""

FIELD_LIMITS = {"headline": (10, 90), "decision_logic": (60, 420), "supporting_signals": (12, 160),
                "opposing_signals": (12, 160), "watch_triggers": (12, 160), "model_caveat": (20, 220)}


@dataclass
class AnalystResult:
    report: dict
    used_llm: bool
    fallback_sections: list[str]
    critic: CriticReport
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    log_path: Optional[str] = None
    error: Optional[str] = None
    prompt: Optional[str] = None


def build_user_prompt(ctx: dict) -> str:
    payload = {"股票": ctx["meta"], "事實": [{"編號": f["id"], "類別": f["group"], "內容": f["text"]} for f in ctx["facts"]]}
    return ("請依下列事實撰寫這檔股票的分析師解讀。輸出格式：\n" + OUTPUT_SCHEMA +
            "\n\n資料：\n" + json.dumps(payload, ensure_ascii=False, indent=1))


def write_analyst_report(ctx: dict, client: Optional[ClaudeClient] = None) -> AnalystResult:
    template = template_report(ctx)
    fact_ids = {f["id"] for f in ctx["facts"]}
    user = build_user_prompt(ctx)
    res = AnalystResult(report=dict(template), used_llm=False, fallback_sections=list(SECTION_KEYS),
                        critic=CriticReport(), model="template", prompt=user)
    if client is None:
        res.critic = audit(res.report, ctx)
        return res
    try:
        r = client.complete(system_prompt(), user, tag="analyst_report")
        out = extract_json(r.text)
        res.used_llm, res.model = True, r.model
        res.input_tokens, res.output_tokens, res.log_path = r.input_tokens, r.output_tokens, r.log_path
    except Exception as e:                                     # 沒有金鑰、API 失敗、格式錯 → 全部用範本
        res.error = str(e)
        res.critic = audit(res.report, ctx)
        return res
    res.fallback_sections = []
    for key in SECTION_KEYS:
        lo, hi = FIELD_LIMITS[key]
        need_ref = key != "headline"
        if _field_ok(out.get(key), fact_ids, lo, hi, need_ref=need_ref):
            res.report[key] = out[key]
        else:
            res.fallback_sections.append(f"{key}:format")
    critic = audit(res.report, ctx)
    if not critic.ok:                                          # Critic 不通過的段落改用範本，再稽核一次
        for key in critic.failed_sections():
            res.report[key] = template[key]
            res.fallback_sections.append(f"{key}:critic")
        res.first_critic = critic                              # type: ignore[attr-defined]
        critic = audit(res.report, ctx)
    res.critic = critic
    return res


# ============================================================
# 輸出
# ============================================================
def render_text(res: AnalystResult, ctx: dict) -> str:
    r = res.report
    lines = [f"【結論】{r['headline']}", ""]
    panel = render_score_panel(ctx)
    if panel:
        lines += panel + [""]
    lines += ["【決策推理】", r["decision_logic"], ""]
    for title, key in (("【支持進場的訊號】", "supporting_signals"), ("【不利進場的訊號】", "opposing_signals"),
                       ("【決策可能改變的條件】", "watch_triggers")):
        lines.append(title)
        lines += [f"・{t}" for t in _as_list(r[key])]
        lines.append("")
    lines += ["【模型可信度】", r["model_caveat"], "", "【引用事實】"]
    lines += [f"{f['id']}　{f['text']}" for f in ctx["facts"]]
    src = (f"Claude（{res.model}，輸入 {res.input_tokens}／輸出 {res.output_tokens} tokens）" if res.used_llm
           else "確定性範本" + (f"（{res.error}）" if res.error else ""))
    lines += ["", f"產生方式：{src}"]
    if res.used_llm and res.fallback_sections:
        lines.append(f"改用範本的段落：{', '.join(res.fallback_sections)}")
    lines.append(res.critic.text())
    lines.append("本段文字只解讀系統既有決策，不改變任何數字或建議。")
    return "\n".join(lines)


def save_outputs(res: AnalystResult, ctx: dict, text: str) -> tuple[Path, Path]:
    AI_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now_taipei().strftime("%Y%m%d_%H%M%S")
    base = AI_REPORT_DIR / f"{ctx['meta']['symbol']}_{ctx['meta']['date']}_{stamp}"
    md_path, js_path = base.with_suffix(".md"), base.with_suffix(".json")
    md_path.write_text(f"# {stock_label(ctx['meta']['symbol'], ctx['meta'].get('name', ''))} AI 分析師解讀（資料截止 {ctx['meta']['date']}）\n\n" + text + "\n", encoding="utf-8")
    rec = {"meta": ctx["meta"], "decision": ctx["decision"], "score_breakdown": ctx.get("score"), "facts": ctx["facts"], "report": res.report,
           "used_llm": res.used_llm, "model": res.model, "fallback_sections": res.fallback_sections,
           "critic": [asdict(i) for i in res.critic.issues], "critic_ok": res.critic.ok,
           "first_critic": [asdict(i) for i in getattr(res, "first_critic", CriticReport()).issues],
           "tokens": {"input": res.input_tokens, "output": res.output_tokens}, "llm_log": res.log_path, "error": res.error,
           "created_at": now_taipei().isoformat()}
    if AI_SETTINGS.get("dry_run"):
        rec["prompt"] = {"system": system_prompt(), "user": res.prompt}
    js_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return md_path, js_path


def pick_rows_for_ai(rows: "pd.DataFrame", max_n: int = 3):
    """多檔模式挑選要解讀的股票：先取系統有給倉位的，再依最終分數補足。"""
    import pandas as pd
    if rows is None or len(rows) == 0:
        return []
    df = rows.copy()
    df["_score"] = pd.to_numeric(df.get("final_decision_score"), errors="coerce")
    df["_pos"] = pd.to_numeric(df.get("suggested_position_pct"), errors="coerce").fillna(0)
    df["_veto"] = df.get("trade_veto_flag", False)
    df["_veto"] = df["_veto"].fillna(False).astype(bool)
    df["_rank"] = ((df["_pos"] > 0) & ~df["_veto"]).astype(int)
    df = df.sort_values(["_rank", "_score"], ascending=[False, False])
    return [r for _, r in df.head(max_n).drop(columns=["_score", "_pos", "_veto", "_rank"]).iterrows()]


def run_ai_for_rows(rows, config=None, title_prefix: str = "十二、AI 分析師解讀", max_n: int = 3) -> int:
    """對多檔股票各產生一份 AI 分析師解讀；回傳產生份數。"""
    if not AI_SETTINGS.get("enabled"):
        return 0
    picked = pick_rows_for_ai(rows, max_n=max_n)
    for i, r in enumerate(picked, 1):
        sym = _txt(r.get("symbol"), "")
        try:
            maybe_run_ai_analyst(r, config, None, title=f"{title_prefix}（{i}/{len(picked)}）：{stock_label(sym)}")
        except Exception as e:                                   # noqa: BLE001
            print(f"[AI] {sym} 分析師報告產生失敗（不影響上方結果）：{e}")
    return len(picked)


def _translated_row(latest_row) -> dict:
    """套用報告原有的翻譯函式，確保標籤、否決原因等欄位為中文（與第一～十一段一致）。"""
    row = dict(latest_row.items()) if hasattr(latest_row, "items") else dict(latest_row)
    try:
        from pipeline_risk import (translate_decision_action, translate_decision_label,
                                   translate_rule_filter_notes, translate_technical_filter_result,
                                   translate_veto_reason)
        row["final_decision_label_zh"] = translate_decision_label(row.get("final_decision_label", "中性觀察"))
        row["decision_action_zh"] = translate_decision_action(row.get("decision_action", "NA"))
        row["trade_veto_reason_zh"] = translate_veto_reason(row.get("trade_veto_reason", ""))
        row["technical_filter_result_zh"] = translate_technical_filter_result(row.get("technical_filter_result", "NA"))
        row["rule_filter_notes_zh"] = translate_rule_filter_notes(row.get("rule_filter_notes", ""))
    except Exception:
        pass
    return row


def maybe_run_ai_analyst(latest_row, config=None, summary=None, title: Optional[str] = None) -> Optional[AnalystResult]:
    """由各模式的決策輸出呼叫；未開啟 AI 分析師時什麼都不做。"""
    if not AI_SETTINGS.get("enabled"):
        return None
    symbol = _txt(latest_row.get("symbol"), getattr(config, "stock_no", ""))
    row = _translated_row(latest_row)
    ctx = build_facts(row, summary=summary, symbol=symbol, stock_name=_txt(latest_row.get("stock_name"), "") or _name_of(symbol))
    client = None if AI_SETTINGS.get("dry_run") else ClaudeClient(
        AI_SETTINGS["model"], AI_SETTINGS["max_tokens"], AI_SETTINGS["timeout_sec"])
    res = write_analyst_report(ctx, client)
    text = render_text(res, ctx)
    print()
    print("-" * 70)
    print(title or "十二、AI 分析師解讀（第四層：AI 決策代理）")
    print("-" * 70)
    print(text)
    try:
        md_path, js_path = save_outputs(res, ctx, text)
        print(f"AI 報告已輸出：{md_path}")
    except Exception as e:
        print(f"[AI] 報告存檔失敗：{e}")
    return res
