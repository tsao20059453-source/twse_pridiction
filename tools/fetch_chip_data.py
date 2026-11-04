"""抓取籌碼與基本面資料（三大法人、融資融券、月營收、除權息），存到 data/chip/*.csv。

資料來源：FinMind（與 ETF 專案 esun/data/finmind.py 相同的資料集與正規化方式）。
本程式只寫入「專題」自己的 data/chip/，不讀寫 ETF 專案。

用法（在「專題」資料夾）：
    python tools/fetch_chip_data.py                 # 補齊兩個固定股票池中缺資料的股票
    python tools/fetch_chip_data.py --update        # 另外把所有已有股票更新到最新
    python tools/fetch_chip_data.py --symbols 2330 2317

額度：未登入每小時 300 次；在 .env 設 FINMIND_TOKEN 可提高到 600 次。每檔每個資料集 1 次請求。
程式不會印出 token。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CHIP_DIR = ROOT / "data" / "chip"
API = "https://api.finmindtrade.com/api/v4/data"
KEYS = {"institutional": ["date", "code"], "margin": ["date", "code"],
        "revenue": ["code", "period"], "dividends": ["date", "code"]}
DATASETS = {"institutional": "TaiwanStockInstitutionalInvestorsBuySell",
            "margin": "TaiwanStockMarginPurchaseShortSale",
            "revenue": "TaiwanStockMonthRevenue",
            "dividends": "TaiwanStockDividendResult"}
START = {"institutional": "2019-01-01", "margin": "2019-01-01", "revenue": "2018-01-01", "dividends": "2019-01-01"}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class FinMind:
    def __init__(self) -> None:
        self.token = os.environ.get("FINMIND_TOKEN", "")
        self.interval = 3600 / (600 if self.token else 300) * 1.1
        self.last = 0.0
        self.calls = 0

    def get(self, dataset: str, code: str, start: str) -> pd.DataFrame:
        params = {"dataset": dataset, "data_id": code, "start_date": start}
        if self.token:
            params["token"] = self.token
        for attempt in range(1, 4):
            wait = self.interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = requests.get(API, params=params, timeout=60)
                payload = r.json()
            except Exception as e:                      # 網路錯誤：稍後重試
                print(f"  連線失敗（第 {attempt} 次）：{type(e).__name__}")
                time.sleep(10 * attempt)
                continue
            self.calls += 1
            status = payload.get("status")
            if status == 402:
                print("  FinMind 額度用完，等待 10 分鐘…")
                time.sleep(600)
                continue
            if status not in (200, None):
                raise RuntimeError(f"FinMind {dataset} {code}: status={status} msg={payload.get('msg')}")
            return pd.DataFrame(payload.get("data", []))
        return pd.DataFrame()


# ---- 正規化（與 ETF 專案 finmind.py 相同）----
INSTI_MAP = {"Foreign_Investor": "foreign", "Foreign_Dealer_Self": "foreign", "Investment_Trust": "trust",
             "Dealer_self": "dealer", "Dealer_Hedging": "dealer", "Dealer": "dealer"}


def norm_institutional(raw: pd.DataFrame) -> pd.DataFrame:
    w = raw.copy()
    w["group"] = w["name"].map(INSTI_MAP)
    w = w.dropna(subset=["group"])
    w["net"] = w["buy"].astype(float) - w["sell"].astype(float)
    p = w.pivot_table(index=["date", "stock_id"], columns="group", values="net", aggfunc="sum").reset_index()
    for g in ("foreign", "trust", "dealer"):
        if g not in p.columns:
            p[g] = 0.0
    df = pd.DataFrame({"date": pd.to_datetime(p["date"]), "code": p["stock_id"].astype(str),
                       "foreign_net": p["foreign"], "trust_net": p["trust"], "dealer_net": p["dealer"]})
    df["total_net"] = df[["foreign_net", "trust_net", "dealer_net"]].sum(axis=1)
    return df


def norm_margin(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(raw["date"]), "code": raw["stock_id"].astype(str),
                         "margin_prev": raw["MarginPurchaseYesterdayBalance"].astype(float),
                         "margin_balance": raw["MarginPurchaseTodayBalance"].astype(float),
                         "short_prev": raw["ShortSaleYesterdayBalance"].astype(float),
                         "short_balance": raw["ShortSaleTodayBalance"].astype(float)})


def norm_revenue(raw: pd.DataFrame) -> pd.DataFrame:
    period = pd.to_datetime(dict(year=raw["revenue_year"].astype(int), month=raw["revenue_month"].astype(int), day=1))
    df = pd.DataFrame({"code": raw["stock_id"].astype(str), "period": period.dt.strftime("%Y-%m"),
                       "revenue": raw["revenue"].astype(float)}).sort_values(["code", "period"]).reset_index(drop=True)
    # 保守的可得日：次月 11 日（法規要求次月 10 日前公告），回測不會提早看到營收
    df["available_date"] = pd.to_datetime(df["period"] + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
    return df


def norm_dividends(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "code": raw["stock_id"].astype(str),
                       "before_price": raw["before_price"].astype(float), "after_price": raw["after_price"].astype(float)})
    for c in ("stock_and_cache_dividend", "stock_or_cache_dividend"):
        if c in raw.columns:
            df[c] = raw[c].values
    return df[(df["before_price"] > 0) & (df["after_price"] > 0)].reset_index(drop=True)


NORM = {"institutional": norm_institutional, "margin": norm_margin, "revenue": norm_revenue, "dividends": norm_dividends}


def add_revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    """以同代號、指定月份對齊上月與去年同月（月份缺漏時自然為缺值）。"""
    df = df.sort_values(["code", "period"]).reset_index(drop=True)
    pdt = pd.to_datetime(df["period"] + "-01")
    lookup = df.set_index(["code", "period"])["revenue"]

    def at(offset):
        keys = pd.MultiIndex.from_arrays([df["code"], (pdt - offset).dt.strftime("%Y-%m")])
        return pd.Series(lookup.reindex(keys).values, index=df.index)

    prev = at(pd.DateOffset(months=1))
    df["revenue_last_year"] = at(pd.DateOffset(years=1))
    df["rev_mom_pct"] = (df["revenue"] / prev - 1) * 100
    df["rev_yoy_pct"] = (df["revenue"] / df["revenue_last_year"] - 1) * 100
    return df


def read_table(name: str) -> pd.DataFrame:
    p = CHIP_DIR / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"code": str})
    return df


def write_table(name: str, df: pd.DataFrame) -> None:
    CHIP_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in ("date", "available_date"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates(KEYS[name], keep="last").sort_values(KEYS[name]).reset_index(drop=True)
    tmp = CHIP_DIR / f"{name}.csv.tmp"
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CHIP_DIR / f"{name}.csv")


def pool_symbols() -> list[str]:
    syms: list[str] = []
    for p in sorted((ROOT / "pools").glob("pool_top30_*.csv")):
        col = pd.read_csv(p, dtype=str).iloc[:, 0]
        syms += [s for s in col.astype(str).str.strip() if s and s not in syms]
    return syms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", help="指定股票代碼；預設為 pools/ 中所有固定股票池")
    ap.add_argument("--update", action="store_true", help="已有資料的股票也更新到最新（從最後日期往前 40 天重抓）")
    args = ap.parse_args()
    load_env()
    fm = FinMind()
    symbols = args.symbols or pool_symbols()
    print(f"股票共 {len(symbols)} 檔；FinMind {'已登入' if fm.token else '未登入'}（每次請求間隔 {fm.interval:.1f} 秒）")

    for name, dataset in DATASETS.items():
        table = read_table(name)
        have = set(table["code"].astype(str)) if not table.empty else set()
        todo = []
        for s in symbols:
            if s not in have:
                todo.append((s, START[name]))
            elif args.update:
                col = "date" if name != "revenue" else "available_date"
                last = pd.to_datetime(table.loc[table["code"] == s, col]).max()
                start = (last - pd.Timedelta(days=40)).strftime("%Y-%m-%d") if name != "revenue" else \
                        (last - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
                todo.append((s, start))
        if not todo:
            print(f"[{name}] 無需更新")
            continue
        print(f"[{name}] 需要抓取 {len(todo)} 檔")
        frames = []
        for i, (s, start) in enumerate(todo, 1):
            raw = fm.get(dataset, s, start)
            n = 0
            if not raw.empty:
                df = NORM[name](raw)
                frames.append(df)
                n = len(df)
            print(f"  {i}/{len(todo)} {s} 自 {start}：{n} 筆")
        if frames:
            new = pd.concat(frames, ignore_index=True)
            if name == "revenue":
                base = table[["code", "period", "revenue", "available_date"]] if not table.empty else pd.DataFrame()
                merged = pd.concat([base, new], ignore_index=True).drop_duplicates(["code", "period"], keep="last")
                merged = add_revenue_growth(merged)
            else:
                merged = pd.concat([table, new], ignore_index=True) if not table.empty else new
            write_table(name, merged)
            print(f"[{name}] 已寫入 {CHIP_DIR / (name + '.csv')}")
    print(f"完成，共呼叫 FinMind {fm.calls} 次。")


if __name__ == "__main__":
    sys.exit(main())
