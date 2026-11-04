"""籌碼與基本面特徵：三大法人買賣超、融資融券、月營收（2026-10-26 新增）。

資料：data/chip/{institutional,margin,revenue}.csv（由 tools/fetch_chip_data.py 從 FinMind 取得，
欄位與 ETF 專案相同）。以 --chip-features 開啟；未開啟時本模組不影響任何結果。

避免偷看未來：
- 三大法人（當日 15:00 後公布）、融資融券（當日 21:00 後公布）一律延後 1 個交易日才使用。
- 月營收以「次月 11 日」作為可得日（法規要求次月 10 日前公告），用可得日對齊交易日。

單位一致：價量已做除權／面額變更還原（成交量 ÷ adj_factor），法人買賣超股數與融資餘額也同樣 ÷ adj_factor，
避免面額變更前後的比例失真。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHIP_DIR = Path(__file__).resolve().parent / "data" / "chip"
CHIP_SETTINGS: dict = {"enabled": False}
_CACHE: dict = {}
_WARNED: set = set()

CHIP_FEATURES = [
    "chip_foreign_ratio_5d", "chip_foreign_ratio_20d",
    "chip_trust_ratio_5d", "chip_trust_ratio_20d",
    "chip_dealer_ratio_5d", "chip_total_ratio_20d",
    "chip_foreign_streak", "chip_trust_streak",
    "chip_margin_chg_5d", "chip_margin_chg_20d", "chip_short_chg_5d",
    "chip_short_to_margin", "chip_margin_to_volume",
    "chip_rev_yoy", "chip_rev_mom", "chip_rev_yoy_3m", "chip_rev_yoy_accel", "chip_rev_to_12m_high",
]
CHIP_CS_RANK_SOURCES = ["chip_foreign_ratio_20d", "chip_trust_ratio_20d", "chip_total_ratio_20d",
                        "chip_margin_chg_20d", "chip_rev_yoy", "chip_rev_yoy_accel"]


def set_chip_features(enabled: bool = True) -> None:
    CHIP_SETTINGS["enabled"] = bool(enabled)


def chip_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = CHIP_FEATURES + [f"{c}_cs_rank" for c in CHIP_CS_RANK_SOURCES]
    return [c for c in cols if c in df.columns]


def _load(name: str) -> pd.DataFrame:
    if name not in _CACHE:
        path = CHIP_DIR / f"{name}.csv"
        if not path.exists():
            _CACHE[name] = pd.DataFrame()
        else:
            df = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            for c in ("date", "available_date"):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            _CACHE[name] = df
    return _CACHE[name]


def _streak(x: pd.Series) -> pd.Series:
    """連續買超（正）／賣超（負）天數，上限 ±20。"""
    sign = np.sign(x.fillna(0.0)).to_numpy()
    out = np.zeros(len(sign))
    run = 0.0
    for i, s in enumerate(sign):
        if s == 0:
            run = 0.0
        elif run == 0 or np.sign(run) != s:
            run = s
        else:
            run += s
        out[i] = run
    return pd.Series(np.clip(out, -20, 20), index=x.index)


def _monthly_revenue_features(rev: pd.DataFrame) -> pd.DataFrame:
    r = rev.sort_values("period").reset_index(drop=True).copy()
    r["chip_rev_yoy"] = pd.to_numeric(r.get("rev_yoy_pct"), errors="coerce")
    r["chip_rev_mom"] = pd.to_numeric(r.get("rev_mom_pct"), errors="coerce")
    r["chip_rev_yoy_3m"] = r["chip_rev_yoy"].rolling(3, min_periods=2).mean()
    r["chip_rev_yoy_accel"] = r["chip_rev_yoy"] - r["chip_rev_yoy"].shift(3)
    revenue = pd.to_numeric(r["revenue"], errors="coerce")
    r["chip_rev_to_12m_high"] = revenue / revenue.rolling(12, min_periods=6).max()
    return r[["available_date", "chip_rev_yoy", "chip_rev_mom", "chip_rev_yoy_3m",
              "chip_rev_yoy_accel", "chip_rev_to_12m_high"]].dropna(subset=["available_date"])


def add_chip_features(df: pd.DataFrame) -> pd.DataFrame:
    """對單一股票的日資料加入籌碼與營收特徵；未開啟或沒有資料時原樣回傳（缺值補 0）。"""
    if not CHIP_SETTINGS["enabled"] or df is None or df.empty or "symbol" not in df.columns:
        return df
    sym = str(df["symbol"].iloc[0])
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["date"] = pd.to_datetime(out["date"])
    adj = pd.to_numeric(out.get("adj_factor", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    vol = pd.to_numeric(out["volume"], errors="coerce")

    inst = _load("institutional")
    inst = inst[inst["code"] == sym] if not inst.empty else inst
    marg = _load("margin")
    marg = marg[marg["code"] == sym] if not marg.empty else marg
    rev = _load("revenue")
    rev = rev[rev["code"] == sym] if not rev.empty else rev
    if (inst.empty or marg.empty or rev.empty) and sym not in _WARNED:
        _WARNED.add(sym)
        print(f"WARNING: {sym} 缺少籌碼／營收資料（請執行 tools/fetch_chip_data.py），相關特徵以 0 代替")

    # ---- 三大法人（延後 1 日）----
    if not inst.empty:
        m = out[["date"]].merge(inst[["date", "foreign_net", "trust_net", "dealer_net", "total_net"]], on="date", how="left")
        for c in ("foreign_net", "trust_net", "dealer_net", "total_net"):
            out[f"_{c}"] = (pd.to_numeric(m[c], errors="coerce") / adj).shift(1)
        vol_lag = vol.shift(1)
        for w in (5, 20):
            denom = vol_lag.rolling(w, min_periods=w).sum()
            out[f"chip_foreign_ratio_{w}d"] = out["_foreign_net"].rolling(w, min_periods=w).sum() / denom
            out[f"chip_trust_ratio_{w}d"] = out["_trust_net"].rolling(w, min_periods=w).sum() / denom
        out["chip_dealer_ratio_5d"] = out["_dealer_net"].rolling(5, min_periods=5).sum() / vol_lag.rolling(5, min_periods=5).sum()
        out["chip_total_ratio_20d"] = out["_total_net"].rolling(20, min_periods=20).sum() / vol_lag.rolling(20, min_periods=20).sum()
        out["chip_foreign_streak"] = _streak(out["_foreign_net"])
        out["chip_trust_streak"] = _streak(out["_trust_net"])
        out = out.drop(columns=[c for c in out.columns if c.startswith("_") and c.endswith("_net")])

    # ---- 融資融券（延後 1 日）----
    if not marg.empty:
        m = out[["date"]].merge(marg[["date", "margin_balance", "short_balance"]], on="date", how="left")
        mb = (pd.to_numeric(m["margin_balance"], errors="coerce") / adj).shift(1)
        sb = (pd.to_numeric(m["short_balance"], errors="coerce") / adj).shift(1)
        out["chip_margin_chg_5d"] = mb / mb.shift(5) - 1
        out["chip_margin_chg_20d"] = mb / mb.shift(20) - 1
        out["chip_short_chg_5d"] = (sb - sb.shift(5)) / mb.replace(0, np.nan)
        out["chip_short_to_margin"] = sb / mb.replace(0, np.nan)
        out["chip_margin_to_volume"] = mb * 1000 / vol.shift(1).rolling(20, min_periods=20).mean()

    # ---- 月營收（依可得日對齊）----
    if not rev.empty:
        feats = _monthly_revenue_features(rev).sort_values("available_date")
        merged = pd.merge_asof(out[["date"]].reset_index(), feats, left_on="date", right_on="available_date",
                               direction="backward").set_index("index").sort_index()
        for c in ("chip_rev_yoy", "chip_rev_mom", "chip_rev_yoy_3m", "chip_rev_yoy_accel", "chip_rev_to_12m_high"):
            out[c] = merged[c].to_numpy()

    for c in CHIP_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # 暖身期與缺資料以 0 代替（0 ＝ 沒有法人買賣、融資不變、營收持平），避免整列被模型剔除
    out[CHIP_FEATURES] = out[CHIP_FEATURES].fillna(0.0)
    return out


def add_chip_cs_ranks(df: pd.DataFrame) -> pd.DataFrame:
    if not CHIP_SETTINGS["enabled"] or df is None or df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    for c in CHIP_CS_RANK_SOURCES:
        if c in out.columns:
            out[f"{c}_cs_rank"] = out.groupby("date")[c].rank(pct=True)
    return out
