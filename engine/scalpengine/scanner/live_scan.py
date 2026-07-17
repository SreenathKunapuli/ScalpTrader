"""Live morning scanner — turns the trained ranker into the day's watchlist.

Called once at ~10:01 ET (after the 09:30-09:45 window closes PLUS the
free-tier 15-minute SIP embargo PLUS a few seconds of slack).  Accepts up
to ~40 candidate symbols (sourced by screener.scan_candidates /
morning_scan.build_daily_candidates in the caller) and:

  1. Fetches the morning's 09:30-09:45 SIP tick trades + quotes via
     alpaca-py historical REST, mirroring build_runner_corpus.py pagination.
  2. Builds 1s bars by calling corpus.second_bars from the research package
     (zero train/serve skew: same function the corpus was built with).
  3. Computes morning features via scanner.rank.build_scanner_features (same
     function the model was trained against).  Daily context fields — open,
     prev_close, prev_day_dollar_vol, prev_day_volume — come from the prior
     RAW daily bars fetched via REST: prev_day_dollar_vol = close * volume
     and prev_day_volume = volume of the true prior CALENDAR session, exactly
     matching train_scanner._daily_lookup which uses data/daily_raw batches.
  4. Scores with the newest runs/scanner/<ts>/model.joblib artifact, ranks
     descending, and truncates to top_n (default 30).
  5. Persists via repo.replace_watchlist and returns the ranked list as the
     day's plan_focus.

Design invariants:
  - Pure-logic functions (build_bars_from_ticks, build_daily_context,
    score_candidates) accept plain DataFrames / dicts — no network inside.
    I/O is isolated to fetch_ticks_and_quotes, fetch_daily_context, and
    run_morning_scan so tests run on canned fixtures with zero network.
  - An injected rate_limiter callable (token-bucket or plain sleep) is
    called before every HTTP request.  Symbols are capped to
    max_symbols_for_budget (default 40) before the fetch loop; excess
    symbols are logged and skipped.
  - Missing artifact, missing viability label data, or 0-bar fetch for a
    symbol are handled gracefully: the symbol is scored NaN and excluded
    from the watchlist (never crashes the scan run).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Research package bootstrap (same pattern as scalp_gbt.py)
# -------------------------------------------------------------------------
_RESEARCH = Path(__file__).resolve().parents[3] / "research"
if not (_RESEARCH / "scalp" / "corpus.py").exists():
    raise ImportError(
        f"research package not found at {_RESEARCH} — live_scan requires "
        "the monorepo layout (engine/ and research/ side by side)")
if str(_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_RESEARCH))

from scalp.corpus import second_bars  # noqa: E402
from scanner.rank import (  # noqa: E402
    FEATURES,
    build_scanner_features,
    slice_early_bars,
)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
_NY = "America/New_York"

# Runs artifact directory, relative to repo root.
_RUNS_DIR = Path(__file__).resolve().parents[3] / "runs" / "scanner"

# Watchlist column name for the ranker score.
_SCORE_COL = "score"


# -------------------------------------------------------------------------
# Public return type
# -------------------------------------------------------------------------
@dataclass
class ScanResult:
    """Outcome of a single morning scan cycle."""
    plan_focus: list[str]                   # symbols, best-first (len <= top_n)
    rows: list[dict[str, Any]]              # full per-symbol detail (all scored)
    dropped_budget: list[str] = field(default_factory=list)  # over-budget symbols
    artifact_dir: Path | None = None


# -------------------------------------------------------------------------
# Artifact discovery
# -------------------------------------------------------------------------

def _find_latest_artifact(runs_dir: Path = _RUNS_DIR) -> Path | None:
    """Return the newest runs/scanner/<ts>/ directory that has a model.joblib."""
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir() and (d / "model.joblib").exists()),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_artifact(artifact_dir: Path | None = None):
    """Load model + feature order from the artifact directory.

    Returns (model, feature_names: list[str]) or raises FileNotFoundError.
    """
    import joblib

    d = artifact_dir or _find_latest_artifact()
    if d is None:
        raise FileNotFoundError(
            f"No scanner artifact with model.joblib found under {_RUNS_DIR}. "
            "Run: .venv/bin/python research/scripts/train_scanner.py")
    cfg = json.loads((d / "config.json").read_text())
    model = joblib.load(d / "model.joblib")
    features: list[str] = cfg["features"]
    log.info("scanner.artifact_loaded dir=%s features=%d", d, len(features))
    return model, features, d


# -------------------------------------------------------------------------
# Daily context (open, prev_close, prev_day_dollar_vol, prev_day_volume)
# -------------------------------------------------------------------------

def build_daily_context(raw_daily_bars: dict[str, list[dict]]) -> dict[str, dict]:
    """Convert raw per-symbol daily bar payloads into the settled-by-open fields.

    Each entry in raw_daily_bars[sym] is a dict with keys:
        date (str YYYY-MM-DD), open, high, low, close, volume, vwap
    at RAW adjustment, sorted ascending by date (the caller ensures this).

    Returns {sym: {"open": float, "prev_close": float,
                   "prev_day_dollar_vol": float, "prev_day_volume": float}}
    for today's session.  When fewer than 2 bars are present the prev-day
    fields come back as NaN (the ranker handles missing natively).

    This mirrors train_scanner._daily_lookup semantics exactly:
    prev_day_dollar_vol = close[-2]*volume[-2] (true prior CALENDAR session,
    not the prior runner-index row which could be months stale).
    prev_close = close[-2].
    """
    ctx: dict[str, dict] = {}
    for sym, bars in raw_daily_bars.items():
        if not bars:
            ctx[sym] = {
                "open": np.nan, "prev_close": np.nan,
                "prev_day_dollar_vol": np.nan, "prev_day_volume": np.nan,
            }
            continue
        today_bar = bars[-1]
        if len(bars) >= 2:
            prev = bars[-2]
            prev_close = float(prev["close"])
            prev_vol = float(prev["volume"])
            prev_dv = prev_close * prev_vol
        else:
            prev_close = np.nan
            prev_vol = np.nan
            prev_dv = np.nan
        ctx[sym] = {
            "open": float(today_bar["open"]),
            "prev_close": prev_close,
            "prev_day_dollar_vol": prev_dv,
            "prev_day_volume": prev_vol,
        }
    return ctx


# -------------------------------------------------------------------------
# Bar building from canned tick payloads (pure, testable)
# -------------------------------------------------------------------------

def build_bars_from_ticks(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build 1s bars from raw SIP tick DataFrames.

    Delegates directly to corpus.second_bars (zero train/serve skew).

    trades_df: DatetimeIndex (UTC), columns [price, size].
    quotes_df: DatetimeIndex (UTC), columns [bid_price, ask_price, bid_size, ask_size].
               May be empty.

    Returns the 1s bar DataFrame (RTH only) or an empty DataFrame.
    """
    if trades_df.empty:
        return pd.DataFrame()
    return second_bars(trades_df, quotes_df, interval_s=1)


# -------------------------------------------------------------------------
# Feature + scoring (pure)
# -------------------------------------------------------------------------

def score_candidates(
    bars_map: dict[str, pd.DataFrame],
    daily_ctx: dict[str, dict],
    model: Any,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Score each symbol's morning 1s bars with the ranker model.

    Returns one dict per symbol with keys:
        symbol, score (float or NaN), and the 8 scanner features.
    Symbols with 0 bars or where build_scanner_features raises are given
    score=NaN and excluded from the watchlist ranking but included in the
    return list so callers can audit them.
    """
    rows: list[dict[str, Any]] = []
    for sym, bars in bars_map.items():
        ctx = daily_ctx.get(sym, {})
        early = slice_early_bars(bars) if not bars.empty else bars
        feat: dict[str, Any] = {}
        score = np.nan
        try:
            feat = build_scanner_features(ctx, early)
            x = pd.DataFrame([feat]).reindex(columns=feature_names)
            score = float(model.predict(x)[0])
        except Exception as exc:
            log.warning("scanner.feature_error sym=%s err=%s", sym, exc)
        row: dict[str, Any] = {"symbol": sym, _SCORE_COL: score}
        row.update(feat)
        rows.append(row)
    return rows


# -------------------------------------------------------------------------
# Network I/O helpers (real calls, isolated here for testability)
# -------------------------------------------------------------------------

def fetch_ticks_and_quotes(
    client: Any,
    sym: str,
    session_date: date,
    rate_limiter: Callable[[], None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 09:30-09:45 ET SIP trades + quotes for one symbol.

    Mirrors build_runner_corpus._fetch_one's request pattern.
    Returns (trades_df, quotes_df) — either may be empty on error.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest, StockTradesRequest

    day = pd.Timestamp(session_date, tz=_NY)
    start = (day + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    end = (day + pd.Timedelta(hours=9, minutes=45)).tz_convert("UTC")

    trades_df = pd.DataFrame()
    quotes_df = pd.DataFrame()

    try:
        rate_limiter()
        resp = client.get_stock_trades(StockTradesRequest(
            symbol_or_symbols=sym,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            feed=DataFeed.SIP,
        ))
        df = resp.df
        if not df.empty:
            trades_df = df.droplevel("symbol")[["price", "size"]]
    except Exception as exc:
        log.warning("scanner.trades_fetch_error sym=%s err=%s", sym, exc)

    try:
        rate_limiter()
        resp = client.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=sym,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            feed=DataFeed.SIP,
        ))
        df = resp.df
        if not df.empty:
            quotes_df = (df.droplevel("symbol")
                         [["bid_price", "ask_price", "bid_size", "ask_size"]])
    except Exception as exc:
        log.warning("scanner.quotes_fetch_error sym=%s err=%s", sym, exc)

    return trades_df, quotes_df


def fetch_daily_context(
    client: Any,
    symbols: list[str],
    session_date: date,
    rate_limiter: Callable[[], None],
) -> dict[str, dict]:
    """Fetch 2 prior RAW daily bars per symbol for the daily context fields.

    Uses a batched request (up to 200 symbols per call) to minimise
    request budget.  RAW adjustment — same as train_scanner's runner_index
    and build_runner_corpus daily scan.  We fetch [session_date - 5cd,
    session_date) so we always capture prev_close and prev_day_dollar_vol
    regardless of intervening holidays.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    raw_daily: dict[str, list[dict]] = {s: [] for s in symbols}

    day = pd.Timestamp(session_date, tz=_NY)
    end_dt = day.tz_convert("UTC").to_pydatetime()
    start_dt = (day - pd.Timedelta(days=7)).tz_convert("UTC").to_pydatetime()

    for i in range(0, len(symbols), 200):
        chunk = symbols[i: i + 200]
        try:
            rate_limiter()
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment="raw",
                feed="sip",
            )
            resp = client.get_stock_bars(req)
            for sym in chunk:
                bars_list = resp.data.get(sym, [])
                raw_daily[sym] = [
                    {
                        "date": b.timestamp.date().isoformat(),
                        "open": float(b.open),
                        "close": float(b.close),
                        "volume": float(b.volume),
                    }
                    for b in bars_list[-3:]   # at most 3 trailing sessions
                ]
        except Exception as exc:
            log.warning("scanner.daily_fetch_error chunk=%s err=%s",
                        chunk[:3], exc)

    return build_daily_context(raw_daily)


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------

def run_morning_scan(
    *,
    symbols: list[str],
    session_date: date,
    client: Any,
    repo: Any,
    rate_limiter: Callable[[], None] | None = None,
    top_n: int = 30,
    max_symbols_for_budget: int = 40,
    artifact_dir: Path | None = None,
) -> ScanResult:
    """Full morning scan pipeline: fetch -> build bars -> score -> persist.

    Parameters
    ----------
    symbols:
        Candidate symbols to evaluate (passed by the caller; sourced from
        screener.scan_candidates / morning_scan.build_daily_candidates).
    session_date:
        The trading date whose 09:30-09:45 window to fetch.
    client:
        Alpaca StockHistoricalDataClient (injected so tests can stub it).
    repo:
        Repo instance for watchlist persistence.  Must expose replace_watchlist.
    rate_limiter:
        Zero-arg callable invoked before every HTTP request.  Defaults to
        a no-op when None (tests / manual callers that throttle externally).
    top_n:
        How many top-ranked symbols to keep in the watchlist.
    max_symbols_for_budget:
        Hard cap on symbols before the fetch loop starts.  Excess are
        logged and returned in ScanResult.dropped_budget.
    artifact_dir:
        Override the artifact directory (for tests / re-scoring older runs).
        When None, the newest runs/scanner/<ts>/ is used.
    """
    rl = rate_limiter or (lambda: None)

    # Load model artifact.
    try:
        model, feature_names, art_dir = load_artifact(artifact_dir)
    except FileNotFoundError as exc:
        log.error("scanner.no_artifact err=%s", exc)
        return ScanResult(plan_focus=[], rows=[], artifact_dir=None)

    # Budget cap.
    dropped: list[str] = []
    if len(symbols) > max_symbols_for_budget:
        dropped = symbols[max_symbols_for_budget:]
        symbols = symbols[:max_symbols_for_budget]
        log.warning("scanner.budget_cap keeping=%d dropped=%d dropped_sample=%s",
                    len(symbols), len(dropped), dropped[:5])

    # Fetch daily context (batched, 1-2 requests total).
    log.info("scanner.daily_ctx symbols=%d date=%s", len(symbols), session_date)
    daily_ctx = fetch_daily_context(client, symbols, session_date, rl)

    # Fetch per-symbol ticks and build 1s bars.
    bars_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        log.debug("scanner.fetch_ticks sym=%s", sym)
        trades_df, quotes_df = fetch_ticks_and_quotes(client, sym, session_date, rl)
        bars_map[sym] = build_bars_from_ticks(trades_df, quotes_df)
        n_bars = len(bars_map[sym])
        log.info("scanner.bars sym=%s n_bars=%d", sym, n_bars)

    # Score all candidates.
    rows = score_candidates(bars_map, daily_ctx, model, feature_names)

    # Rank: sort by score descending, drop NaN-scored.
    valid = [r for r in rows if not np.isnan(r[_SCORE_COL])]
    valid.sort(key=lambda r: r[_SCORE_COL], reverse=True)
    top = valid[:top_n]

    plan_focus = [r["symbol"] for r in top]
    log.info("scanner.ranked top=%d plan_focus=%s", len(plan_focus), plan_focus[:5])

    # Persist watchlist (replace-all).
    # NaN floats are coerced to 0.0: SQLite/Postgres disallow NaN in NOT NULL
    # numeric columns and WatchlistEntry has no nullable=True on these fields.
    def _f(v: Any) -> float:
        x = float(v) if v is not None else 0.0
        return 0.0 if (x != x) else x   # NaN check without importing math

    now_utc = datetime.now(UTC)
    watchlist_rows: list[dict[str, Any]] = []
    for r in top:
        watchlist_rows.append({
            "ts": now_utc,
            "symbol": r["symbol"],
            "score": _f(r[_SCORE_COL]),
            "price": _f(r.get("prev_close")),
            "gain_pct": _f(r.get("gap_pct")),
            "relvol": _f(r.get("relvol_at_open")),
            "spread_bps": _f(r.get("first15_spread_bps_med")),
            "streamed": False,
        })
    if repo is not None:
        repo.replace_watchlist(watchlist_rows)
        log.info("scanner.watchlist_written n=%d", len(watchlist_rows))

    return ScanResult(
        plan_focus=plan_focus,
        rows=rows,
        dropped_budget=dropped,
        artifact_dir=art_dir,
    )
