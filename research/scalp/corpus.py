"""Runner-day corpus: pure transforms from raw ticks to second bars, and
runner-day selection from daily bars.

Design constraints (see plan):
- Price-band filtering uses RAW (unadjusted) daily bars — scalping cares about
  the actual traded price that day, and split adjustment would push historical
  prices of split stocks out of the $0.5–$10 band they really traded in.
- Second bars carry the NBBO as of each bar close so the viability oracle can
  charge real spread. Quotes are forward-filled at most QUOTE_STALE_LIMIT_S
  seconds; beyond that bid/ask are NaN and the oracle must not trade.
- Everything here is pure pandas (no network) so it is perturbation-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RTH_START = "09:30"
RTH_END = "16:00"
NY = "America/New_York"
QUOTE_STALE_LIMIT_S = 60


@dataclass(frozen=True)
class RunnerCriteria:
    price_min: float = 0.5
    price_max: float = 10.0
    min_gain: float = 0.15        # close/prev_close - 1
    min_gap: float = 0.10         # open/prev_close - 1 (alternative trigger)
    min_relvol: float = 3.0       # volume vs 20d median
    min_dollar_vol: float = 5e6   # that day, so it is actually scalpable
    relvol_window: int = 20


def select_runner_days(daily: pd.DataFrame,
                       crit: RunnerCriteria = RunnerCriteria()) -> pd.DataFrame:
    """Select runner stock-days from a raw daily panel.

    `daily`: long format indexed by (symbol, ts) with columns
    open/high/low/close/volume/vwap — RAW adjustment (see module docstring).
    Returns one row per selected stock-day with the selection features.
    Uses only same-day and PRIOR data per row (prev_close, trailing median
    volume) — no lookahead.
    """
    df = daily.sort_index()
    g = df.groupby(level="symbol", group_keys=False, sort=False)
    prev_close = g["close"].shift(1)
    # trailing relvol: median over the *prior* `relvol_window` sessions
    med_vol = g["volume"].apply(
        lambda s: s.shift(1).rolling(crit.relvol_window,
                                     min_periods=crit.relvol_window).median())

    gain = df["close"] / prev_close - 1.0
    gap = df["open"] / prev_close - 1.0
    relvol = df["volume"] / med_vol
    dollar_vol = df["volume"] * df["vwap"].fillna(df["close"])

    in_band = (df["open"] >= crit.price_min) & (df["open"] <= crit.price_max) \
        & (prev_close >= crit.price_min) & (prev_close <= crit.price_max)
    trigger = (gain >= crit.min_gain) | (gap >= crit.min_gap)
    liquid = (relvol >= crit.min_relvol) & (dollar_vol >= crit.min_dollar_vol)

    sel = df[in_band & trigger & liquid].copy()
    sel["prev_close"] = prev_close[sel.index]
    sel["gain"] = gain[sel.index]
    sel["gap"] = gap[sel.index]
    sel["relvol"] = relvol[sel.index]
    sel["dollar_vol"] = dollar_vol[sel.index]
    # Fetch priority: liquid big movers first; cap dollar_vol's influence so a
    # single mega-runner doesn't crowd out price-bucket diversity.
    sel["score"] = np.minimum(sel["dollar_vol"], 200e6) * (1.0 + sel["gain"].clip(0, 3))
    return sel.sort_values("score", ascending=False)


def _rth_mask(ts: pd.DatetimeIndex) -> np.ndarray:
    local = ts.tz_convert(NY)
    t = local.time
    lo = pd.Timestamp(RTH_START).time()
    hi = pd.Timestamp(RTH_END).time()
    return (t >= lo) & (t < hi)


def second_bars(trades: pd.DataFrame, quotes: pd.DataFrame,
                interval_s: int = 1) -> pd.DataFrame:
    """Aggregate raw ticks into fixed-interval bars with NBBO-at-close.

    `trades`: DatetimeIndex (UTC), columns price,size.
    `quotes`: DatetimeIndex (UTC), columns bid_price,ask_price,bid_size,ask_size.
    Returns bars indexed by UTC bar-close time, RTH only, with columns:
    open,high,low,close,volume,vwap,n_trades,bid,ask,bid_size,ask_size,spread.
    A bar labeled T covers (T - interval, T]; NBBO is the last quote at or
    before T (forward-filled at most QUOTE_STALE_LIMIT_S; NaN beyond that).
    Seconds with no trades carry NaN OHLC (no phantom prices) but do carry NBBO.
    """
    if trades.empty:
        return pd.DataFrame()
    freq = f"{interval_s}s"
    tr = trades.sort_index()
    px, sz = tr["price"], tr["size"]
    grp = tr.groupby(pd.Grouper(freq=freq, label="right", closed="right"))
    bars = pd.DataFrame({
        "open": grp["price"].first(),
        "high": grp["price"].max(),
        "low": grp["price"].min(),
        "close": grp["price"].last(),
        "volume": grp["size"].sum(),
        "n_trades": grp["price"].count(),
    })
    notional = (px * sz).groupby(pd.Grouper(freq=freq, label="right",
                                            closed="right")).sum()
    with np.errstate(invalid="ignore"):
        bars["vwap"] = notional / bars["volume"].replace(0, np.nan)

    # continuous second grid across the trading day so quote-only seconds exist
    # (endpoints carry tz already; passing tz= too trips pandas' identity
    # assertion when the source tz object differs, e.g. alpaca-py's UTC)
    full = pd.date_range(bars.index.min(), bars.index.max(), freq=freq)
    bars = bars.reindex(full)
    bars["volume"] = bars["volume"].fillna(0)
    bars["n_trades"] = bars["n_trades"].fillna(0)

    if not quotes.empty:
        q = quotes.sort_index()
        # sanity: drop crossed/degenerate quotes (bad prints happen on runners)
        q = q[(q["ask_price"] > 0) & (q["bid_price"] > 0)
              & (q["ask_price"] >= q["bid_price"])]
        snap = q.groupby(pd.Grouper(freq=freq, label="right", closed="right")).last()
        snap = snap.reindex(full)
        limit = max(1, QUOTE_STALE_LIMIT_S // interval_s)
        snap = snap.ffill(limit=limit)
        bars["bid"] = snap["bid_price"]
        bars["ask"] = snap["ask_price"]
        bars["bid_size"] = snap["bid_size"]
        bars["ask_size"] = snap["ask_size"]
        bars["spread"] = bars["ask"] - bars["bid"]
    else:
        bars[["bid", "ask", "bid_size", "ask_size", "spread"]] = np.nan

    return bars[_rth_mask(bars.index)]
