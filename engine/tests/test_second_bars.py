"""SecondBarBuilder parity with research corpus.second_bars.

The parity test IS the spec: the same synthetic tick tape must produce
bit-equal bars through the offline corpus transform and the incremental
builder — any drift here is train/serve skew on the money path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalpengine.data.second_bars import SecondBarBuilder

_RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(_RESEARCH))
from scalp.corpus import second_bars  # noqa: E402

# 2025-01-06 is a Monday in EST: 14:33 UTC = 09:33 ET, inside RTH.
START = pd.Timestamp("2025-01-06 14:33:00.500", tz="UTC")


def synth_tape() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random tape with the nasty bits: crossed quotes, a >60s quote gap,
    a pre-grid quote, quote-only trailing seconds."""
    rng = np.random.default_rng(42)
    n = 400
    # gaps up to 4s guarantee some trade-empty seconds, so the parity test
    # also covers NaN-OHLC rows (and corpus n_trades comes out float)
    tts = START + pd.to_timedelta(rng.uniform(0.1, 4.0, n).cumsum(), unit="s")
    px = np.round(3.0 + rng.normal(0, 0.01, n).cumsum(), 2).clip(0.5)
    trades = pd.DataFrame(
        {"price": px, "size": rng.integers(50, 500, n).astype(float)},
        index=pd.DatetimeIndex(tts))

    span = (tts[-1] - START).total_seconds()
    q_off = np.sort(rng.uniform(0.0, span, 300))
    # carve a 75s quote-silent window mid-tape -> NBBO must go NaN there
    gap_lo, gap_hi = span * 0.5, span * 0.5 + 75
    q_off = q_off[(q_off < gap_lo) | (q_off > gap_hi)]
    q_ts = START + pd.to_timedelta(q_off, unit="s")
    mid = np.interp(q_off, (tts - START).total_seconds(), px)
    bid = np.round(mid - 0.01, 4)
    ask = np.round(mid + 0.01, 4)
    # poison a handful with crossed/degenerate quotes: both paths must drop them
    bad = rng.choice(len(q_off), 8, replace=False)
    ask[bad[:4]] = bid[bad[:4]] - 0.02
    bid[bad[4:]] = -1.0
    quotes = pd.DataFrame(
        {"bid_price": bid, "ask_price": ask,
         "bid_size": rng.integers(1, 50, len(q_off)).astype(float),
         "ask_size": rng.integers(1, 50, len(q_off)).astype(float)},
        index=pd.DatetimeIndex(q_ts))
    # pre-grid quote (before the first trade second): corpus drops it on
    # reindex, the builder must mirror that
    pre = pd.DataFrame({"bid_price": [2.9], "ask_price": [2.95],
                        "bid_size": [10.0], "ask_size": [10.0]},
                       index=pd.DatetimeIndex([START - pd.Timedelta(seconds=10)]))
    return trades, pd.concat([pre, quotes]).sort_index()


def feed(builder: SecondBarBuilder, trades: pd.DataFrame,
         quotes: pd.DataFrame, symbol: str = "TEST") -> None:
    events = []
    for ts, r in trades.iterrows():
        events.append((ts, "t", r))
    for ts, r in quotes.iterrows():
        events.append((ts, "q", r))
    events.sort(key=lambda e: e[0])
    for ts, kind, r in events:
        if kind == "t":
            builder.add_trade(symbol, float(r["price"]), float(r["size"]), ts)
        else:
            builder.add_quote(symbol, float(r["bid_price"]), float(r["ask_price"]),
                              float(r["bid_size"]), float(r["ask_size"]), ts)


def test_parity_with_corpus():
    trades, quotes = synth_tape()
    expected = second_bars(trades, quotes)
    assert len(expected) > 300 and expected["bid"].notna().sum() > 100

    b = SecondBarBuilder(window_s=5000)
    feed(b, trades, quotes)
    b.poll(trades.index[-1] + pd.Timedelta(seconds=5))
    frame = b.get_frame("TEST")

    got = frame.loc[expected.index]
    # dtype parity is data-dependent in the corpus (int counts via Grouper);
    # value parity is what train/serve skew cares about — that stays strict
    pd.testing.assert_frame_equal(got, expected, check_freq=False,
                                  check_names=False, check_dtype=False)


def test_quote_gap_goes_nan_and_recovers():
    trades, quotes = synth_tape()
    b = SecondBarBuilder(window_s=5000)
    feed(b, trades, quotes)
    b.poll(trades.index[-1] + pd.Timedelta(seconds=5))
    frame = b.get_frame("TEST")
    assert frame["bid"].isna().any()          # the 75s silence went stale
    assert frame["bid"].notna().sum() > 100   # and quotes resumed after


def test_boundary_trade_belongs_to_that_second():
    b = SecondBarBuilder()
    t0 = pd.Timestamp("2025-01-06 15:00:00", tz="UTC")  # exactly on boundary
    b.add_trade("X", 3.0, 100, t0)
    b.add_trade("X", 3.05, 100, t0 + pd.Timedelta(seconds=1, milliseconds=200))
    out = b.poll(t0 + pd.Timedelta(seconds=5))
    labels = [ts for sym, ts, _ in out]
    assert labels[0] == t0                    # (T-1s, T] closed-right
    first = out[0][2]
    assert first["close"] == 3.0 and first["volume"] == 100


def test_late_trade_dropped_never_revised():
    b = SecondBarBuilder()
    t0 = pd.Timestamp("2025-01-06 15:00:00.300", tz="UTC")
    b.add_trade("X", 3.0, 100, t0)
    b.poll(t0 + pd.Timedelta(seconds=10))
    frame_before = b.get_frame("X")
    b.add_trade("X", 9.9, 500, t0 + pd.Timedelta(seconds=2))  # already final
    assert b.late_events("X") == 1
    pd.testing.assert_frame_equal(b.get_frame("X"), frame_before)


def test_rth_mask_and_reset():
    b = SecondBarBuilder()
    pre = pd.Timestamp("2025-01-06 14:00:00.100", tz="UTC")  # 09:00 ET premarket
    b.add_trade("X", 3.0, 100, pre)
    rth = pd.Timestamp("2025-01-06 14:30:00.100", tz="UTC")  # 09:30:00.1 ET
    b.add_trade("X", 3.1, 100, rth)
    out = b.poll(rth + pd.Timedelta(seconds=5))
    assert all(ts.tz_convert("America/New_York").time()
               >= pd.Timestamp("09:30").time() for _, ts, _ in out)
    b.reset("X")
    assert b.get_frame("X").empty


def test_crossed_quote_dropped():
    b = SecondBarBuilder()
    t0 = pd.Timestamp("2025-01-06 15:00:00.100", tz="UTC")
    b.add_trade("X", 3.0, 100, t0)
    b.add_quote("X", 3.05, 3.01, 5, 5, t0)     # crossed: must be ignored
    out = b.poll(t0 + pd.Timedelta(seconds=3))
    assert np.isnan(out[0][2]["bid"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
