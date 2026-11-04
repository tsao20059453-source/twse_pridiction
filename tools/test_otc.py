"""上櫃資料連線測試（在「專題」資料夾執行：python tools/test_otc.py）。

依序測試：
1. FinMind 上市／上櫃對照表（data/cache/market_map.csv）
2. FinMind 上櫃個股日線（6274 台燿最近 10 個交易日）
3. 櫃買中心當日成交金額排行（前 5 檔）
只讀取網路資料，唯一會寫入的檔案是 data/cache/market_map.csv。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_data as D  # noqa: E402


def main() -> int:
    ok = True
    print("1. 上市／上櫃對照表")
    try:
        mapping = D.refresh_market_map()
        for s in ("2330", "6274", "6488", "3105"):
            print(f"   {s} → {mapping.get(s, '查無')}")
    except Exception as e:
        ok = False
        print(f"   失敗：{e}")

    print("2. 上櫃個股日線（6274）")
    try:
        start = (datetime.today() - timedelta(days=20)).strftime("%Y-%m-%d")
        df = D.fetch_finmind_price_range("6274", start)
        print(df.tail(10)[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))
        ok &= not df.empty
    except Exception as e:
        ok = False
        print(f"   失敗：{e}")

    print("3. 櫃買中心成交金額排行")
    try:
        top = D.fetch_tpex_top_turnover_stocks(limit=5)
        print(top.to_string(index=False))
        ok &= not top.empty
    except Exception as e:
        ok = False
        print(f"   失敗：{e}")

    print("\n結果：" + ("全部成功，上櫃資料可正常使用" if ok else "有項目失敗，請把上面的訊息貼給 Claude"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
