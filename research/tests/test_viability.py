"""Oracle viability core: exact cost math, non-overlap, fill realism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.viability import FeeModel, OracleConfig, frictionless_oracle, maker_oracle, taker_oracle


def mk_bars(mid: np.ndarray, spread: float = 0.01,
            low: np.ndarray | None = None,
            high: np.ndarray | None = None) -> pd.DataFrame:
    n = len(mid)
    idx = pd.date_range("2024-06-03 14:00:00", periods=n, freq="1s", tz="UTC")
    bid, ask = mid - spread / 2, mid + spread / 2
    return pd.DataFrame({
        "open": mid, "high": mid if high is None else high,
        "low": mid if low is None else low, "close": mid,
        "volume": np.full(n, 1000.0), "vwap": mid,
        "n_trades": np.full(n, 5.0),
        "bid": bid, "ask": ask,
        "bid_size": np.full(n, 10.0), "ask_size": np.full(n, 10.0),
        "spread": np.full(n, float(spread)),
    }, index=idx)


def test_taker_oracle_rising_tape_exact_math():
    mid = 3.0 + 0.001 * np.arange(300)          # +6c per 60s
    bars = mk_bars(mid, spread=0.01)
    cfg = OracleConfig(horizon_s=60, clip_shares=1000)
    tr = taker_oracle(bars, cfg)
    assert len(tr) == 4                          # t=0,60,120,180 (240 has no exit)
    # exact edge: bid[t+60] - ask[t] - sell fees
    t0_edge = (bars["bid"].iloc[60] - bars["ask"].iloc[0]
               - cfg.fees.sell_cost_per_share(bars["bid"].iloc[60], 1000))
    assert tr.iloc[0]["edge_ps"] == pytest.approx(t0_edge)
    assert tr.iloc[0]["clip_pnl"] == pytest.approx(t0_edge * 1000)
    # non-overlapping: entries at least horizon apart
    gaps = tr["t"].diff().dt.total_seconds().dropna()
    assert (gaps >= 60).all()
    # displayed capacity: 10 lots = 1000 shares -> full clip fill
    assert tr.iloc[0]["displayed_pnl"] == pytest.approx(tr.iloc[0]["clip_pnl"])


def test_taker_oracle_wide_spread_kills_it():
    mid = 3.0 + 0.001 * np.arange(300)
    bars = mk_bars(mid, spread=0.10)             # 10c spread vs 6c move
    assert taker_oracle(bars, OracleConfig(horizon_s=60)).empty


def test_taker_oracle_fees_block_marginal_trades():
    n = 200
    mid = np.full(n, 3.0)
    mid[100:] = 3.0102                            # +1.02c step, spread 1c
    bars = mk_bars(mid, spread=0.01)
    fee_free = OracleConfig(horizon_s=100, fees=FeeModel(0.0, 0.0, 0.0))
    assert len(taker_oracle(bars, fee_free)) >= 1   # 0.02c gross edge exists
    heavy = OracleConfig(horizon_s=100,
                         fees=FeeModel(sec_rate=0.0, taf_per_share=0.0005,
                                       taf_cap=8.30))  # 0.05c/share > edge
    assert taker_oracle(bars, heavy).empty


def test_taker_oracle_skips_nan_nbbo():
    mid = 3.0 + 0.001 * np.arange(300)
    bars = mk_bars(mid, spread=0.01)
    bars.iloc[:150, bars.columns.get_loc("bid")] = np.nan  # stale first half
    tr = taker_oracle(bars, OracleConfig(horizon_s=60))
    assert len(tr) and (tr["t"] >= bars.index[150]).all()


def test_maker_oracle_requires_trade_through():
    mid = np.full(400, 3.0)
    # no trade ever prints below the bid -> no entry fills
    assert maker_oracle(mk_bars(mid), OracleConfig(horizon_s=60)).empty

    low = np.full(400, 3.0)
    low[5] = 2.99                                 # prints through the bid at t=5
    high = np.full(400, 3.0)
    high[70] = 3.01                               # prints through the ask later
    bars = mk_bars(mid, spread=0.01, low=low, high=high)
    tr = maker_oracle(bars, OracleConfig(horizon_s=60, maker_wait_s=30))
    assert len(tr) == 1
    # entry at bid 2.995, exit at ask 3.005: earns the spread, fee-adjusted
    exp = 3.005 - 2.995 - OracleConfig().fees.sell_cost_per_share(3.005, 1000)
    assert tr.iloc[0]["edge_ps"] == pytest.approx(exp)
    assert not tr.iloc[0]["exit_crossed"]


def test_maker_oracle_crossed_exit_pays_spread():
    mid = np.full(400, 3.0)
    low = np.full(400, 3.0)
    low[5] = 2.99                                 # entry fills; exit never does
    bars = mk_bars(mid, spread=0.01, low=low)
    tr = maker_oracle(bars, OracleConfig(horizon_s=60, maker_wait_s=30))
    # exit crossed at bid 2.995 == entry 2.995 -> edge <= 0 after fees: no trade kept
    assert tr.empty


def test_frictionless_sums_positive_moves_only():
    mid = np.concatenate([np.linspace(3.0, 3.1, 100),   # +0.1
                          np.linspace(3.1, 2.9, 100)])  # -0.2
    bars = mk_bars(mid, spread=0.01)
    val = frictionless_oracle(bars, OracleConfig(horizon_s=10))
    assert val > 0
    flat = frictionless_oracle(mk_bars(np.full(50, 3.0)), OracleConfig(horizon_s=10))
    assert flat == 0.0
