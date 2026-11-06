"""Point-in-time daily TWSE universe utilities for historical backtests.

The legacy historical backtest freezes the stocks that are liquid today and
reuses them for every historical event.  This module provides a separate,
cached daily Top-N universe so that an event backtest can instead use only
the stocks that were liquid on each historical trading day.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from pipeline_data import RAW_DATA_DIR, SESSION, clean_numeric


# The legacy exchangeReport endpoint intermittently returns HTTP 428 for older
# dates.  The current RWD endpoint serves the same MI_INDEX payload reliably.
TWSE_MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
HISTORICAL_UNIVERSE_CACHE_DIR = RAW_DATA_DIR / "twse_daily_universe"
HISTORICAL_UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SHARED_SNAPSHOT_CACHE_DIR: Path | None = None


def _snapshot_cache_path(source_date) -> Path:
    date_text = pd.Timestamp(source_date).strftime("%Y%m%d")
    return HISTORICAL_UNIVERSE_CACHE_DIR / f"twse_market_{date_text}.csv.gz"


def set_shared_snapshot_cache_dir(path: str | Path | None) -> None:
    """Use an existing compatible snapshot cache as a read-only source."""
    global SHARED_SNAPSHOT_CACHE_DIR
    if path is None:
        SHARED_SNAPSHOT_CACHE_DIR = None
        return
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"shared snapshot cache directory not found: {candidate}")
    SHARED_SNAPSHOT_CACHE_DIR = candidate


def load_shared_market_index(
    start_date,
    end_date,
    burn_in_months: int = 12,
) -> pd.DataFrame:
    """Load a compatible TAIEX cache beside the shared daily snapshots.

    The shared cache is read-only.  An empty frame tells the caller to use
    its normal network-backed market-index source instead.
    """
    if SHARED_SNAPSHOT_CACHE_DIR is None:
        return pd.DataFrame(columns=["date", "market_close"])
    cache_path = SHARED_SNAPSHOT_CACHE_DIR.parent / "TWSE_TAIEX_raw.csv"
    if not cache_path.exists():
        return pd.DataFrame(columns=["date", "market_close"])
    try:
        market = pd.read_csv(cache_path, usecols=["date", "market_close"])
        market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
        market["market_close"] = pd.to_numeric(market["market_close"], errors="coerce")
        start = pd.Timestamp(start_date).normalize() - pd.DateOffset(months=int(burn_in_months))
        end = pd.Timestamp(end_date).normalize()
        return (
            market.loc[market["date"].between(start, end)]
            .dropna(subset=["date", "market_close"])
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return pd.DataFrame(columns=["date", "market_close"])


def _read_cached_snapshot(source_date) -> pd.DataFrame:
    local_path = _snapshot_cache_path(source_date)
    candidates = [local_path]
    if SHARED_SNAPSHOT_CACHE_DIR is not None:
        candidates.append(SHARED_SNAPSHOT_CACHE_DIR / local_path.name)
    for path in candidates:
        if not path.exists():
            continue
        cached = pd.read_csv(path, dtype={"symbol": str})
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
        if not cached.empty:
            return cached
        if path == local_path:
            # Do not alter a user-provided read-only cache, only invalidate an
            # empty local cache written by an interrupted previous run.
            path.unlink()
    return pd.DataFrame()


def snapshot_cache_available(source_date) -> bool:
    return not _read_cached_snapshot(source_date).empty


def _stock_table(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        status = payload.get("stat") if isinstance(payload, dict) else None
        raise ValueError(f"TWSE MI_INDEX response invalid: {status or 'unknown'}")

    required = {"證券代號", "證券名稱", "成交股數", "成交金額"}
    for table in payload.get("tables", []):
        fields = table.get("fields") or []
        if required.issubset(fields):
            frame = pd.DataFrame(table.get("data") or [], columns=fields)
            if not frame.empty:
                return frame
    raise ValueError("TWSE MI_INDEX did not include the listed-equity table")


def parse_twse_daily_snapshot(payload: dict, source_date) -> pd.DataFrame:
    """Normalize one full listed-equity snapshot from TWSE MI_INDEX."""
    table = _stock_table(payload)
    required_prices = ["開盤價", "最高價", "最低價", "收盤價"]
    missing = [field for field in required_prices if field not in table.columns]
    if missing:
        raise ValueError(f"TWSE MI_INDEX missing OHLC fields: {missing}")

    out = pd.DataFrame({
        "date": pd.Timestamp(source_date).normalize(),
        "symbol": table["證券代號"].astype(str).str.strip(),
        "stock_name": table["證券名稱"].astype(str).str.strip(),
        "volume": table["成交股數"].map(clean_numeric),
        "turnover": table["成交金額"].map(clean_numeric),
        "open": table["開盤價"].map(clean_numeric),
        "high": table["最高價"].map(clean_numeric),
        "low": table["最低價"].map(clean_numeric),
        "close": table["收盤價"].map(clean_numeric),
        "market": "TWSE",
    })
    for column in ["volume", "turnover", "open", "high", "low", "close"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return (
        out[out["symbol"].str.fullmatch(r"[1-9]\d{3}", na=False)]
        .drop_duplicates(subset=["date", "symbol"], keep="first")
        .sort_values("symbol")
        .reset_index(drop=True)
    )


def fetch_twse_daily_snapshot(source_date, retries: int = 2) -> pd.DataFrame:
    """Load one market day, using a resumable local cache on subsequent runs."""
    source_date = pd.Timestamp(source_date).normalize()
    cache_path = _snapshot_cache_path(source_date)
    cached = _read_cached_snapshot(source_date)
    if not cached.empty:
        return cached

    params = {
        "response": "json",
        "date": source_date.strftime("%Y%m%d"),
        "type": "ALLBUT0999",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.twse.com.tw/zh/trading/historical/mi-index.html",
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = SESSION.get(
                TWSE_MI_INDEX_URL,
                params=params,
                headers=headers,
                timeout=30,
                verify=False,
            )
            response.raise_for_status()
            snapshot = parse_twse_daily_snapshot(response.json(), source_date)
            snapshot.to_csv(cache_path, index=False, encoding="utf-8-sig", compression="gzip")
            return snapshot
        except Exception as exc:  # HTTP errors and temporary TWSE throttling are retried.
            last_error = exc
            if attempt < retries:
                time.sleep(float(2 ** (attempt - 1)))
    # TWSE intermittently gives Python HTTP clients an empty or 428 response
    # for older daily files while serving the same URL to curl.  Use curl only
    # as a transport fallback; parsing and cache validation remain identical.
    try:
        url = f"{TWSE_MI_INDEX_URL}?{urlencode(params)}"
        result = subprocess.run(
            [
                "curl", "-sS", "--max-time", "30",
                "-A", headers["User-Agent"],
                "-e", headers["Referer"],
                url,
            ],
            capture_output=True,
            text=True,
            timeout=35,
            check=True,
        )
        snapshot = parse_twse_daily_snapshot(json.loads(result.stdout), source_date)
        if snapshot.empty:
            raise ValueError("curl returned an empty listed-equity snapshot")
        snapshot.to_csv(cache_path, index=False, encoding="utf-8-sig", compression="gzip")
        return snapshot
    except Exception as curl_error:
        raise ValueError(
            f"{source_date.date()} daily market snapshot failed: requests={last_error}; curl={curl_error}"
        ) from curl_error


def build_daily_top_n_membership(
    trading_dates,
    top_n: int = 30,
    rank_metric: str = "turnover",
    sleep_sec: float = 0.8,
) -> pd.DataFrame:
    """Build per-date Top-N membership from information available at that close.

    ``date`` is deliberately the selection date.  The external project's
    existing engine assumes a same-close fill, so this preserves its execution
    convention and isolates the universe change for an apples-to-apples test.
    It must not be represented as a next-open simulation.
    """
    metric = str(rank_metric)
    if metric not in {"turnover", "volume"}:
        raise ValueError("rank_metric must be 'turnover' or 'volume'")
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(trading_dates), errors="coerce").dropna())
    dates = pd.DatetimeIndex(sorted(dates.normalize().unique()))
    if not len(dates):
        return pd.DataFrame(columns=["date", "symbol", "stock_name", "rank_metric", "rank_value", "universe_rank", "historical_universe_eligible"])

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for position, date in enumerate(dates, start=1):
        cache_hit = snapshot_cache_available(date)
        try:
            snapshot = fetch_twse_daily_snapshot(date)
            ranked = snapshot[["symbol", "stock_name", metric]].copy()
            ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
            ranked = ranked[ranked[metric] > 0].sort_values([metric, "symbol"], ascending=[False, True]).head(int(top_n))
            if len(ranked) != int(top_n):
                raise ValueError(f"expected Top {top_n}, found {len(ranked)}")
            ranked = ranked.rename(columns={metric: "rank_value"})
            ranked.insert(0, "date", date)
            ranked["rank_metric"] = metric
            ranked["universe_rank"] = range(1, len(ranked) + 1)
            ranked["historical_universe_eligible"] = True
            frames.append(ranked)
        except Exception as exc:
            failures.append(f"{date.date()}: {exc}")
        if not cache_hit and sleep_sec > 0:
            time.sleep(float(sleep_sec))
        if position % 50 == 0 or position == len(dates):
            print(f"[DAILY-UNIVERSE] progress: {position}/{len(dates)}")

    if failures:
        raise ValueError(
            f"daily Top-{top_n} universe incomplete for {len(failures)} trading days; "
            f"retry resumes from cache. First errors: {'; '.join(failures[:3])}"
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "universe_rank", "symbol"]).reset_index(drop=True)


def load_daily_snapshot_prices(
    trading_dates,
    symbols: list[str],
    sleep_sec: float = 0.8,
) -> pd.DataFrame:
    """Return cached/fetched daily OHLCV for a universe union in one pass.

    Snapshot reads are shared by the frozen control and the dynamic universe,
    so a universe comparison is not confounded by different monthly endpoints.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(trading_dates), errors="coerce").dropna())
    dates = pd.DatetimeIndex(sorted(dates.normalize().unique()))
    wanted = {str(symbol) for symbol in symbols}
    frames: list[pd.DataFrame] = []
    for position, date in enumerate(dates, start=1):
        cache_hit = snapshot_cache_available(date)
        snapshot = fetch_twse_daily_snapshot(date)
        daily = snapshot[snapshot["symbol"].astype(str).isin(wanted)]
        if not daily.empty:
            frames.append(daily)
        if not cache_hit and sleep_sec > 0:
            # TWSE returns HTTP 428 when historical MI_INDEX calls are bursty.
            time.sleep(float(sleep_sec))
        if position % 50 == 0 or position == len(dates):
            print(f"[DAILY-UNIVERSE] price progress: {position}/{len(dates)}")
    if not frames:
        raise ValueError("daily market snapshots contained no rows for the requested universe")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def attach_daily_universe_membership(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Mark entry-eligible rows without discarding price rows needed for exits."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    cols = ["date", "symbol", "rank_metric", "rank_value", "universe_rank", "historical_universe_eligible"]
    member = membership[cols].copy()
    member["date"] = pd.to_datetime(member["date"], errors="coerce").dt.normalize()
    member["symbol"] = member["symbol"].astype(str)
    out = out.merge(member, on=["date", "symbol"], how="left")
    out["historical_universe_eligible"] = out["historical_universe_eligible"].fillna(False).astype(bool)
    return out


def append_unscored_execution_rows(oos_frame: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    """Restore non-model rows so an open position can still be exited.

    Only daily-universe members are model candidates.  Securities can leave the
    Top-N after entry, but their later OHLC rows remain necessary for stop and
    take-profit handling.  These restored rows have no decision score and thus
    cannot create a new entry in the existing engine.
    """
    predicted = oos_frame.copy()
    predicted["date"] = pd.to_datetime(predicted["date"], errors="coerce").dt.normalize()
    predicted["symbol"] = predicted["symbol"].astype(str)
    predicted["is_oos_prediction"] = True

    source = source_frame.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    source["symbol"] = source["symbol"].astype(str)
    known = pd.MultiIndex.from_frame(predicted[["date", "symbol"]])
    extra_mask = ~pd.MultiIndex.from_frame(source[["date", "symbol"]]).isin(known)
    extra = source.loc[extra_mask].copy()
    extra["is_oos_prediction"] = False
    for column in predicted.columns:
        if column not in extra.columns:
            extra[column] = pd.NA
    for column in extra.columns:
        if column not in predicted.columns:
            predicted[column] = pd.NA
    return (
        pd.concat([predicted, extra[predicted.columns]], ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
