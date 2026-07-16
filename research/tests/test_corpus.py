"""Corpus transforms: aggregation correctness + no-lookahead perturbation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp import corpus


def make_daily(n_days: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    rows = []
    for sym, base in [("AAA", 3.0), ("BBB", 6.0), ("EXP", 500.0)]:
        close = base * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        vol = rng.integers(1_000_000, 2_000_000, n_days).astype(float)
        for i, t in enumerate(ts):
            o = close[i] * (1 + rng.normal(0, 0.002))
            rows.append({"symbol": sym, "ts": t, "open": o,
                         "high": max(o, close[i]) * 1.01,
                         "low": min(o, close[i]) * 0.99,
                         "close": close[i], "volume": vol[i],
                         "vwap": (o + close[i]) / 2})
    return pd.DataFrame(rows).set_index(["symbol", "ts"]).sort_index()


def _spike(daily: pd.DataFrame, sym: str, i: int, gain: float = 0.30,
           relvol: float = 10.0) -> pd.DataFrame:
    """Turn session i of `sym` into a runner day."""
    df = daily.copy()
    idx = df.loc[sym].index[i]
    prev = df.loc[(sym, df.loc[sym].index[i - 1]), "close"]
    df.loc[(sym, idx), "open"] = prev * 1.02
    df.loc[(sym, idx), "close"] = prev * (1 + gain)
    df.loc[(sym, idx), "high"] = prev * (1 + gain) * 1.02
    med = df.loc[sym, "volume"].iloc[max(0, i - 20):i].median()
    df.loc[(sym, idx), "volume"] = med * relvol
    df.loc[(sym, idx), "vwap"] = prev * 1.15
    return df


def test_select_runner_days_picks_the_spike():
    daily = _spike(make_daily(), "AAA", 40)
    sel = corpus.select_runner_days(daily)
    assert len(sel) == 1
    (sym, ts) = sel.index[0]
    assert sym == "AAA"
    assert sel.iloc[0]["gain"] == pytest.approx(0.30, abs=1e-9)
    assert sel.iloc[0]["relvol"] == pytest.approx(10.0, rel=0.01)


def test_select_excludes_out_of_band_and_illiquid():
    daily = make_daily()
    # same spike on a $500 stock -> excluded by price band
    sel = corpus.select_runner_days(_spike(daily, "EXP", 40))
    assert sel.empty
    # spike with low relative volume -> excluded
    sel = corpus.select_runner_days(_spike(daily, "AAA", 40, relvol=1.5))
    assert sel.empty


def test_select_ignores_future_perturbation():
    """Rewriting bars AFTER day i must not change day i's selection row."""
    daily = _spike(make_daily(), "AAA", 40)
    before = corpus.select_runner_days(daily)
    fut = daily.copy()
    aaa_idx = fut.loc["AAA"].index
    for t in aaa_idx[41:]:
        fut.loc[("AAA", t), ["open", "high", "low", "close"]] *= 7.0
        fut.loc[("AAA", t), "volume"] *= 100
    after = corpus.select_runner_days(fut)
    row_b = before.loc[[("AAA", aaa_idx[40])]]
    row_a = after.loc[[("AAA", aaa_idx[40])]]
    pd.testing.assert_frame_equal(row_b, row_a)


def _ticks():
    """One minute of synthetic RTH ticks (2024-06-03 09:35 ET)."""
    base = pd.Timestamp("2024-06-03 13:35:00", tz="UTC")
    t = [base + pd.Timedelta(ms, "ms") for ms in
         [100, 400, 900, 1200, 1800, 4500]]
    trades = pd.DataFrame({"price": [3.00, 3.02, 3.01, 3.05, 3.04, 3.10],
                           "size": [100, 200, 100, 300, 100, 500]}, index=t)
    q = [base + pd.Timedelta(ms, "ms") for ms in [50, 1000, 4000]]
    quotes = pd.DataFrame({"bid_price": [2.99, 3.03, 3.08],
                           "ask_price": [3.01, 3.05, 3.11],
                           "bid_size": [5, 8, 2], "ask_size": [7, 3, 4]},
                          index=q)
    return trades, quotes


def test_second_bars_ohlcv_and_nbbo():
    trades, quotes = _ticks()
    bars = corpus.second_bars(trades, quotes)
    b1 = bars.iloc[0]  # covers (13:35:00, 13:35:01]: ticks at .1 .4 .9
    assert (b1["open"], b1["high"], b1["low"], b1["close"]) == (3.00, 3.02, 3.00, 3.01)
    assert b1["volume"] == 400
    assert b1["vwap"] == pytest.approx((3.00 * 100 + 3.02 * 200 + 3.01 * 100) / 400)
    assert (b1["bid"], b1["ask"]) == (3.03, 3.05)  # quote at exactly 13:35:01
    b2 = bars.iloc[1]  # tick at 1.2s and 1.8s; NBBO ffilled from 1.0s
    assert b2["close"] == 3.04 and b2["bid"] == 3.03
    b3 = bars.iloc[2]  # no trades in (2s,3s]: NaN OHLC, volume 0, NBBO carried
    assert np.isnan(b3["close"]) and b3["volume"] == 0 and b3["bid"] == 3.03
    assert bars.iloc[4]["spread"] == pytest.approx(0.03)  # 3.11-3.08 at 5s


def test_second_bars_quote_staleness_limit():
    trades, quotes = _ticks()
    # add one trade far in the future with no fresh quote
    late = trades.index[-1] + pd.Timedelta(seconds=200)
    trades = pd.concat([trades, pd.DataFrame(
        {"price": [3.20], "size": [100]}, index=[late])])
    bars = corpus.second_bars(trades, quotes)
    assert np.isnan(bars.iloc[-1]["bid"])  # stale > 60s -> no NBBO, oracle can't trade


def test_second_bars_drops_crossed_quotes_and_rth_only():
    trades, quotes = _ticks()
    crossed = pd.DataFrame({"bid_price": [3.50], "ask_price": [3.40],
                            "bid_size": [1], "ask_size": [1]},
                           index=[trades.index[0] + pd.Timedelta("2.5s")])
    bars = corpus.second_bars(trades, pd.concat([quotes, crossed]).sort_index())
    assert bars.iloc[2]["bid"] == 3.03  # crossed quote ignored, prior NBBO carried
    # pre-market tick is excluded
    pre = pd.Timestamp("2024-06-03 12:00:00", tz="UTC")  # 08:00 ET
    trades2 = pd.concat([pd.DataFrame({"price": [2.5], "size": [100]},
                                      index=[pre]), trades])
    bars2 = corpus.second_bars(trades2, quotes)
    assert bars2.index.min().tz_convert("America/New_York").time() >= \
        pd.Timestamp("09:30").time()
