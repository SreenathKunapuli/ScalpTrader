"""Build the runner-day corpus from free Alpaca historical SIP data.

Two stages, both resumable:

  --scan          Fetch RAW daily bars for the full US-equity asset list
                  (batch-cached under data/daily_raw/), then select runner
                  days per corpus.RunnerCriteria into data/runner_index.parquet.
  --fetch N       For the top-N unfetched stock-days by score (interleaved
                  across price buckets for diversity), download tick trades +
                  quotes and store 1-second bars with NBBO under
                  data/corpus/1s/<SYMBOL>_<DATE>.parquet. Progress in
                  data/corpus/manifest.csv (status: ok / empty / error).

Survivorship note: the asset list includes inactive (delisted) symbols when
Alpaca exposes them; coverage is logged at scan time so the viability study
can report how much of the runner population is reachable. Do not quote
corpus-wide numbers without that context.

Usage:
  .venv/bin/python research/scripts/build_runner_corpus.py --scan
  .venv/bin/python research/scripts/build_runner_corpus.py --fetch 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp import corpus  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DAILY_DIR = DATA / "daily_raw"
INDEX_PATH = DATA / "runner_index.parquet"
CORPUS_DIR = DATA / "corpus" / "1s"
MANIFEST = DATA / "corpus" / "manifest.csv"

SCAN_START = datetime(2019, 1, 1)
BATCH = 100
REQ_SLEEP = 0.35          # ~170 req/min worst case, under the 200/min cap
PRICE_BUCKETS = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def all_symbols() -> list[str]:
    """Every US equity Alpaca knows, active AND inactive (best-effort on
    delisted names — see module docstring)."""
    cache = DATA / "all_symbols.csv"
    if cache.exists():
        return pd.read_csv(cache)["symbol"].tolist()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    from alpaca.trading.requests import GetAssetsRequest
    tc = TradingClient(os.environ["ALPACA_API_KEY"],
                       os.environ["ALPACA_SECRET_KEY"], paper=True)
    assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    active = sum(a.status == "active" for a in assets)
    syms = sorted({a.symbol for a in assets
                   if a.symbol.isascii() and "/" not in a.symbol})
    print(f"asset list: {len(syms)} symbols ({active} active, "
          f"{len(syms) - active} inactive)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": syms}).to_csv(cache, index=False)
    return syms


def scan() -> None:
    """Stage 1: daily bars (RAW adjustment) -> runner index."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    symbols = all_symbols()
    client = _client()
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    n_batches = (len(symbols) + BATCH - 1) // BATCH
    for i in range(0, len(symbols), BATCH):
        part = DAILY_DIR / f"batch_{i // BATCH:04d}.parquet"
        if part.exists():
            continue
        chunk = symbols[i: i + BATCH]
        try:
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                   start=SCAN_START, adjustment="raw", feed="sip")
            df = client.get_stock_bars(req).df
        except Exception as exc:  # noqa: BLE001 — log and move on, resumable
            print(f"batch {i // BATCH}: ERROR {exc}", flush=True)
            time.sleep(5)
            continue
        if len(df):
            df = df[["open", "high", "low", "close", "volume", "vwap"]]
            df.index.names = ["symbol", "ts"]
            df.to_parquet(part)
        else:
            df = pd.DataFrame()
            part.touch()  # empty marker so we don't refetch
        print(f"daily batch {i // BATCH + 1}/{n_batches}: {len(df):,} rows",
              flush=True)
        time.sleep(REQ_SLEEP)

    parts = [p for p in sorted(DAILY_DIR.glob("batch_*.parquet"))
             if p.stat().st_size > 0]
    daily = pd.concat([pd.read_parquet(p) for p in parts])
    print(f"daily panel: {len(daily):,} rows, "
          f"{daily.index.get_level_values('symbol').nunique():,} symbols")
    idx = corpus.select_runner_days(daily)
    idx.to_parquet(INDEX_PATH)
    by_year = idx.groupby(idx.index.get_level_values("ts").year).size()
    print(f"runner days selected: {len(idx):,}\nby year:\n{by_year}")


def _manifest() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST)
    return pd.DataFrame(columns=["symbol", "date", "status", "rows", "n_trades"])


def _pick_next(index: pd.DataFrame, done: set[tuple[str, str]],
               n: int) -> list[tuple[str, str, float]]:
    """Top-N unfetched stock-days, round-robin across price buckets so cheap
    names are represented, not just the biggest dollar-volume runners."""
    per_bucket: list[list[tuple[str, str, float]]] = []
    for lo, hi in PRICE_BUCKETS:
        rows = index[(index["open"] >= lo) & (index["open"] < hi)]
        picks = []
        for (sym, ts), row in rows.iterrows():
            key = (sym, str(pd.Timestamp(ts).date()))
            if key not in done:
                picks.append((sym, key[1], float(row["score"])))
        per_bucket.append(picks)  # already score-sorted from the index
    out: list[tuple[str, str, float]] = []
    k = 0
    while len(out) < n and any(per_bucket):
        b = per_bucket[k % len(per_bucket)]
        if b:
            out.append(b.pop(0))
        k += 1
        if all(not b for b in per_bucket):
            break
    return out


FETCH_DAY_TIMEOUT_S = 240  # observed live hang: one stuck API call froze the
#                            whole run for 74 min — bound each stock-day hard


def _fetch_one(client, sym: str, start, end):  # noqa: ANN001 — alpaca types
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest, StockTradesRequest
    tr = client.get_stock_trades(StockTradesRequest(
        symbol_or_symbols=sym, start=start.to_pydatetime(),
        end=end.to_pydatetime(), feed=DataFeed.SIP)).df
    qu = client.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=start.to_pydatetime(),
        end=end.to_pydatetime(), feed=DataFeed.SIP)).df
    return tr, qu


def fetch(n: int) -> None:
    """Stage 2: tick trades+quotes -> 1s bar parquet per stock-day."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutTimeout

    index = pd.read_parquet(INDEX_PATH)
    man = _manifest()
    settled = man[man["status"].isin(["ok", "empty"])]  # errors stay retryable
    done = set(zip(settled["symbol"], settled["date"], strict=False))
    todo = _pick_next(index, done, n)
    print(f"fetching {len(todo)} stock-days "
          f"({len(man[man.status == 'ok'])} already in corpus)")
    client = _client()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    pool = ThreadPoolExecutor(max_workers=1)
    for j, (sym, date, score) in enumerate(todo, 1):
        day = pd.Timestamp(date, tz="America/New_York")
        start = (day + pd.Timedelta(hours=9, minutes=25)).tz_convert("UTC")
        end = (day + pd.Timedelta(hours=16, minutes=5)).tz_convert("UTC")
        t0 = time.time()
        status, rows, ntr = "ok", 0, 0
        try:
            fut = pool.submit(_fetch_one, client, sym, start, end)
            try:
                tr, qu = fut.result(timeout=FETCH_DAY_TIMEOUT_S)
            except FutTimeout:
                # abandon the stuck call; its worker thread is unusable now,
                # so replace the pool and move on
                pool = ThreadPoolExecutor(max_workers=1)
                raise TimeoutError(f"day fetch exceeded {FETCH_DAY_TIMEOUT_S}s")
            if tr.empty:
                status = "empty"
            else:
                tr = tr.droplevel("symbol")[["price", "size"]]
                qu = (qu.droplevel("symbol")
                      [["bid_price", "ask_price", "bid_size", "ask_size"]]
                      if not qu.empty else pd.DataFrame())
                bars = corpus.second_bars(tr, qu, interval_s=1)
                rows, ntr = len(bars), len(tr)
                bars.to_parquet(CORPUS_DIR / f"{sym}_{date}.parquet")
        except Exception as exc:  # noqa: BLE001 — mark and continue
            status = f"error:{type(exc).__name__}"
            print(f"  {sym} {date}: {exc}", flush=True)
        man = pd.concat([man, pd.DataFrame([{
            "symbol": sym, "date": date, "status": status,
            "rows": rows, "n_trades": ntr}])], ignore_index=True)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        man.to_csv(MANIFEST, index=False)
        print(f"[{j}/{len(todo)}] {sym} {date}: {status} "
              f"({ntr:,} trades -> {rows:,} bars, {time.time() - t0:.1f}s)",
              flush=True)
        time.sleep(REQ_SLEEP)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--fetch", type=int, metavar="N", default=0)
    args = p.parse_args()
    _load_env()
    if args.scan:
        scan()
    if args.fetch:
        fetch(args.fetch)
    if not args.scan and not args.fetch:
        p.print_help()


if __name__ == "__main__":
    main()
