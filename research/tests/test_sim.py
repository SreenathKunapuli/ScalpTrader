"""Fill simulator: capacity floors, barrier semantics, fee math, no lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.sim import SimConfig, simulate
from scalp.viability import FeeModel


def mk_bars(bid: np.ndarray, spread: float = 0.01,
            low: np.ndarray | None = None,
            high: np.ndarray | None = None,
            volume: float = 1e6,
            ask_size: float = 10.0) -> pd.DataFrame:
    n = len(bid)
    idx = pd.date_range("2024-06-03 14:00:00", periods=n, freq="1s", tz="UTC")
    bid = np.asarray(bid, dtype=float)
    mid = bid + spread / 2
    return pd.DataFrame({
        "open": mid, "high": mid if high is None else high,
        "low": mid if low is None else low, "close": mid,
        "volume": np.full(n, float(volume)), "vwap": mid,
        "n_trades": np.full(n, 5.0),
        "bid": bid, "ask": bid + spread,
        "bid_size": np.full(n, 10.0), "ask_size": np.full(n, float(ask_size)),
        "spread": np.full(n, float(spread)),
    }, index=idx)


def mk_entries(bars: pd.DataFrame, rows: list[tuple]) -> pd.DataFrame:
    """rows: (t, qty, target_px, stop_px, deadline_s), offsets in bars."""
    return pd.DataFrame({
        "qty": [r[1] for r in rows],
        "target_px": [r[2] for r in rows],
        "stop_px": [r[3] for r in rows],
        "deadline": [bars.index[r[0]] + pd.Timedelta(seconds=r[4])
                     for r in rows],
    }, index=bars.index[[r[0] for r in rows]])


def test_taker_fill_capacity_floors():
    bid = np.full(200, 3.0)
    # participation cap binds: 60s trailing volume = 6000, 5% -> 300 shares
    bars = mk_bars(bid, volume=100.0, ask_size=10.0)   # displayed = 1000
    ent = mk_entries(bars, [(100, 1000, 99.0, 0.01, 30)])
    res = simulate(bars, ent, SimConfig())
    assert res.iloc[0]["filled"] and res.iloc[0]["fill_qty"] == 300
    assert res.iloc[0]["entry_px"] == pytest.approx(3.01)   # crossed at ask

    # displayed size binds: 2 lots = 200 shares < participation budget
    bars = mk_bars(bid, volume=1e6, ask_size=2.0)
    res = simulate(bars, ent, SimConfig())
    assert res.iloc[0]["fill_qty"] == 200

    # zero displayed size: cannot buy from an empty book
    bars = mk_bars(bid, volume=1e6, ask_size=0.0)
    res = simulate(bars, ent, SimConfig())
    assert not res.iloc[0]["filled"]
    assert res.iloc[0]["exit_reason"] == "unfilled"
    assert res.iloc[0]["fill_qty"] == 0


def test_maker_entry_requires_trade_through():
    cfg = SimConfig(entry_mode="maker", maker_wait_s=30)
    bid = np.full(120, 3.0)
    ent_rows = [(0, 500, 3.05, 2.90, 60)]
    # nothing ever prints below the bid -> no entry
    bars = mk_bars(bid)
    ent = mk_entries(bars, ent_rows)
    res = simulate(bars, ent, cfg)
    assert not res.iloc[0]["filled"]
    assert res.iloc[0]["exit_reason"] == "unfilled"

    # a print AT the bid is not enough (strictly-below trade-through)
    low = np.full(120, 3.005)
    low[3] = 3.0
    res = simulate(mk_bars(bid, low=low), ent, cfg)
    assert not res.iloc[0]["filled"]

    # a print strictly below fills at our posted bid, at the print second
    low[3] = 2.995
    bars = mk_bars(bid, low=low)
    res = simulate(bars, ent, cfg)
    row = res.iloc[0]
    assert row["filled"] and row["entry_px"] == pytest.approx(3.0)
    assert row["entry_t"] == bars.index[3]


def test_target_needs_bid_crossing_not_touch():
    bid = np.full(100, 3.0)
    bid[20:] = 3.049                        # = target - 0.001: only a touch
    high = np.full(100, 3.10)               # prints ABOVE the target all day
    bars = mk_bars(bid, high=high)
    ent = mk_entries(bars, [(0, 100, 3.05, 2.0, 60)])
    res = simulate(bars, ent, SimConfig())
    row = res.iloc[0]
    assert row["exit_reason"] == "timeout"   # offer never lifted by the bid
    assert row["exit_px"] == pytest.approx(3.049)
    assert row["exit_t"] == bars.index[60]

    bid[40:] = 3.05                          # now the bid actually crosses
    bars = mk_bars(bid, high=high)
    res = simulate(bars, ent, SimConfig())
    row = res.iloc[0]
    assert row["exit_reason"] == "target"
    assert row["exit_px"] == pytest.approx(3.05)
    assert row["exit_t"] == bars.index[40]
    assert row["slippage_ps"] == 0.0


def test_stop_gap_through_pays_slippage():
    bid = np.full(100, 3.0)
    bid[10:] = 2.80                          # gaps straight through the stop
    bars = mk_bars(bid)
    ent = mk_entries(bars, [(0, 100, 99.0, 2.90, 60)])
    res = simulate(bars, ent, SimConfig())
    row = res.iloc[0]
    assert row["exit_reason"] == "stop"
    # market exit: next valid bid minus half the then-current spread
    assert row["exit_px"] == pytest.approx(2.80 - 0.005)
    assert row["slippage_ps"] == pytest.approx(2.90 - 2.795)
    assert row["slippage_ps"] > 0
    assert row["exit_t"] == bars.index[10]


def test_stop_wins_same_second_as_target():
    # degenerate barriers where one print satisfies both: stop must win
    bid = np.full(50, 3.05)
    bars = mk_bars(bid)
    ent = mk_entries(bars, [(0, 100, 3.00, 3.10, 30)])
    res = simulate(bars, ent, SimConfig())
    row = res.iloc[0]
    assert row["exit_reason"] == "stop"
    assert row["exit_px"] == pytest.approx(3.05 - 0.005)
    assert row["exit_t"] == bars.index[1]


def test_one_position_at_a_time_skips_overlaps():
    bars = mk_bars(np.full(100, 3.0))
    ent = mk_entries(bars, [
        (0, 100, 99.0, 0.01, 30),     # times out at t=30
        (10, 100, 99.0, 0.01, 30),    # decided while open -> skipped
        (30, 100, 99.0, 0.01, 30),    # exit second still occupied -> skipped
        (31, 100, 99.0, 0.01, 30),    # book is free again -> fills
    ])
    res = simulate(bars, ent, SimConfig())
    assert list(res["exit_reason"]) == ["timeout", "unfilled",
                                        "unfilled", "timeout"]
    assert list(res["filled"]) == [True, False, False, True]
    assert res.iloc[0]["exit_t"] == bars.index[30]
    assert res.iloc[3]["entry_t"] == bars.index[31]


def test_eod_exit_at_last_valid_bid():
    bars = mk_bars(np.full(80, 3.0))
    bars.iloc[75:, bars.columns.get_loc("bid")] = np.nan  # quotes go stale
    ent = mk_entries(bars, [(50, 100, 99.0, 0.01, 600)])  # deadline past data
    res = simulate(bars, ent, SimConfig())
    row = res.iloc[0]
    assert row["exit_reason"] == "eod"
    assert row["exit_px"] == pytest.approx(3.0)
    assert row["exit_t"] == bars.index[74]                # last valid bid


def test_exact_fee_arithmetic_one_trade():
    fees = FeeModel()
    bid = np.full(100, 3.0)
    bid[10:] = 3.06
    bars = mk_bars(bid)
    ent = mk_entries(bars, [(0, 500, 3.05, 2.0, 60)])
    res = simulate(bars, ent, SimConfig())
    row = res.iloc[0]
    assert row["exit_reason"] == "target" and row["fill_qty"] == 500
    # SEC on sell notional + per-share TAF (uncapped at 500 shares)
    fee_ps = 27.80e-6 * 3.05 + min(0.000166 * 500, 8.30) / 500
    assert fee_ps == pytest.approx(fees.sell_cost_per_share(3.05, 500))
    assert row["pnl"] == pytest.approx((3.05 - 3.01 - fee_ps) * 500)


def test_no_lookahead_future_perturbation():
    bid = np.full(200, 3.0)
    bid[40:] = 3.06                          # target fills at t=40
    bars = mk_bars(bid)
    ent = mk_entries(bars, [(10, 300, 3.05, 2.0, 120)])
    base = simulate(bars, ent, SimConfig())
    assert base.iloc[0]["exit_t"] == bars.index[40]

    # rewrite the tape strictly AFTER exit_t: crash it through the stop,
    # dry up the book, blow out the spread — the trade must not change
    mutated = bars.copy()
    for col, val in [("bid", 1.0), ("ask", 1.01), ("low", 0.5),
                     ("volume", 1e9), ("spread", 0.5)]:
        mutated.iloc[41:, mutated.columns.get_loc(col)] = val
    pd.testing.assert_frame_equal(simulate(mutated, ent, SimConfig()), base)
