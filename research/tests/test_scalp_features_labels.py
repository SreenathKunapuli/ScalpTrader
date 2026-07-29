"""Tests for bars_features.py and triple_barrier.py.

Patterns:
- Exact value checks on hand-computed tapes.
- Perturbation / no-lookahead: mutate bars AFTER a cut-point t0 and assert
  every feature/label row AT OR BEFORE t0 is EXACTLY unchanged
  (pd.testing.assert_frame_equal).
- Edge-case guards (zero-volume, NaN NBBO, fee boundary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.bars_features import build_features
from scalp.triple_barrier import BarrierConfig, label_scalps
from scalp.viability import FeeModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    n: int = 400,
    price: float | np.ndarray = 5.0,
    volume: float | np.ndarray = 1000.0,
    n_trades: float | np.ndarray = 10.0,
    spread: float = 0.02,
    start: str = "2024-06-03 13:30:00",  # 09:30 ET
    tz: str = "UTC",
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a minimal contiguous-second bar DataFrame."""
    idx = pd.date_range(start, periods=n, freq="1s", tz=tz)
    px = np.full(n, float(price)) if np.isscalar(price) else np.asarray(price, float)
    vol = np.full(n, float(volume)) if np.isscalar(volume) else np.asarray(volume, float)
    nt = np.full(n, float(n_trades)) if np.isscalar(n_trades) else np.asarray(n_trades, float)

    if bid is None:
        bid_arr = px - spread / 2
    else:
        bid_arr = np.asarray(bid, float)
    if ask is None:
        ask_arr = px + spread / 2
    else:
        ask_arr = np.asarray(ask, float)

    return pd.DataFrame(
        {
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": vol,
            "vwap": px,
            "n_trades": nt,
            "bid": bid_arr,
            "ask": ask_arr,
            "bid_size": np.full(n, 10.0),
            "ask_size": np.full(n, 10.0),
            "spread": np.full(n, float(spread)),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# bars_features: causality (perturbation test)
# ---------------------------------------------------------------------------

class TestBuildFeaturesNolookahead:
    """Rewriting bars AFTER t0 must not change any feature row at/before t0."""

    def test_causality_all_features(self):
        bars = _make_bars(n=400, price=5.0)
        feats_before = build_features(bars)

        # Perturb everything after index 200
        t0 = 200
        bars_perturbed = bars.copy()
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("close")] *= 5.0
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("open")] *= 5.0
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("high")] *= 5.0
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("low")] *= 5.0
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("volume")] *= 5.0
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("vwap")] *= 5.0
        bars_perturbed.iloc[t0 + 1:, bars_perturbed.columns.get_loc("n_trades")] *= 5.0

        feats_after = build_features(bars_perturbed)

        # Rows at and before t0 must be identical
        pd.testing.assert_frame_equal(
            feats_before.iloc[: t0 + 1],
            feats_after.iloc[: t0 + 1],
        )


# ---------------------------------------------------------------------------
# bars_features: exact value checks
# ---------------------------------------------------------------------------

class TestVwapDistExact:
    """Hand-computed vwap_dist on a simple tape."""

    def test_vwap_dist_flat_price(self):
        # Constant price -> vwap == px always -> vwap_dist == 0
        bars = _make_bars(n=50, price=4.0, volume=100.0)
        feats = build_features(bars)
        # Skip row 0 where cumvol starts accumulating; all should be ~0
        assert feats["vwap_dist"].dropna().abs().max() < 1e-10

    def test_vwap_dist_step_change(self):
        # First 10 bars at $4.00, next 10 at $6.00, volume=100 each
        n = 20
        price = np.array([4.0] * 10 + [6.0] * 10, dtype=float)
        bars = _make_bars(n=n, price=price, volume=100.0)
        feats = build_features(bars)

        # At bar 9 (0-indexed): px=4, session_vwap=4 => dist=0
        assert feats["vwap_dist"].iloc[9] == pytest.approx(0.0, abs=1e-10)

        # At bar 19: px=6, session_vwap=(10*4*100 + 10*6*100)/(20*100)=5.0
        # vwap_dist = (6 - 5) / 5 = 0.2
        assert feats["vwap_dist"].iloc[19] == pytest.approx(0.2, abs=1e-10)


class TestPullbackExact:
    """Hand-computed pullback values."""

    def test_pullback_at_new_high_is_zero(self):
        # Monotonically rising price: each bar is the rolling max -> pullback=0
        price = np.linspace(3.0, 6.0, 50)
        bars = _make_bars(n=50, price=price)
        feats = build_features(bars)
        assert (feats["pullback"].abs() < 1e-10).all()

    def test_pullback_dip_below_prior_high(self):
        # Rise to 6.00 then drop to 5.00 within 300-bar window
        price = np.array([3.0, 4.0, 5.0, 6.0, 5.0], dtype=float)
        bars = _make_bars(n=5, price=price)
        feats = build_features(bars)
        # At bar 4: px=5.0, rolling_max=6.0 -> pullback = 5/6 - 1 = -1/6
        assert feats["pullback"].iloc[4] == pytest.approx(5.0 / 6.0 - 1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# bars_features: vol_surge guards
# ---------------------------------------------------------------------------

class TestVolSurgeGuards:
    """vol_surge on a zero-volume tape must return NaN, not inf."""

    def test_zero_volume_returns_nan(self):
        bars = _make_bars(n=100, volume=0.0)
        feats = build_features(bars)
        # Median of zeros is zero -> division by zero -> NaN (not inf)
        # At minimum the first 300 rows (the long-median window) will eventually
        # produce a non-NaN when there is volume; here all zeros so always NaN.
        assert not np.isinf(feats["vol_surge"]).any()
        # All NaN is acceptable; inf is not
        non_nan = feats["vol_surge"].dropna()
        if len(non_nan):
            assert not np.isinf(non_nan).any()

    def test_positive_volume_after_zeros_not_inf(self):
        # Start with 50 zero-volume bars then normal
        vol = np.concatenate([np.zeros(50), np.full(50, 500.0)])
        bars = _make_bars(n=100, volume=vol)
        feats = build_features(bars)
        assert not np.isinf(feats["vol_surge"]).any()


# ---------------------------------------------------------------------------
# triple_barrier: constructed-path tests
# ---------------------------------------------------------------------------

_PRIOR_TRADE_WARMUP = 15   # bars before the entry bar so prior-10s check passes


def _make_labeler_bars(
    bid_future: np.ndarray,
    entry_ask: float,
    spread: float = 0.02,
    n_trades: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Build a second-bar tape with a valid entry row at index PRIOR_TRADE_WARMUP.

    Returns (bars, entry_idx) where entry_idx is the row index of the entry bar.
    All rows before entry_idx have n_trades > 0 so the prior-10s eligibility
    check passes.  bid_future[0] is the bid at the entry bar itself (overridden
    to entry_ask - spread); bid_future[1], [2], ... are bids at t+1, t+2, ...
    """
    warmup = _PRIOR_TRADE_WARMUP
    n_future = len(bid_future)
    n_total = warmup + n_future

    bid_arr = np.full(n_total, entry_ask - spread)
    ask_arr = np.full(n_total, entry_ask)
    # Warmup: stable price at entry level
    bid_arr[:warmup] = entry_ask - spread
    ask_arr[:warmup] = entry_ask
    # From entry_idx onward: follow bid_future
    bid_arr[warmup:] = bid_future
    ask_arr[warmup:] = bid_future + spread
    # Entry bar: ask must be entry_ask exactly
    bid_arr[warmup] = entry_ask - spread
    ask_arr[warmup] = entry_ask

    idx = pd.date_range("2024-06-03 13:30:00", periods=n_total, freq="1s", tz="UTC")
    nt = np.full(n_total, float(n_trades))

    bars = pd.DataFrame(
        {
            "open": ask_arr,
            "high": ask_arr,
            "low": bid_arr,
            "close": (bid_arr + ask_arr) / 2,
            "volume": np.full(n_total, 100.0),
            "vwap": (bid_arr + ask_arr) / 2,
            "n_trades": nt,
            "bid": bid_arr,
            "ask": ask_arr,
            "bid_size": np.full(n_total, 10.0),
            "ask_size": np.full(n_total, 10.0),
            "spread": np.full(n_total, float(spread)),
        },
        index=idx,
    )
    return bars, warmup


class TestTripleBarrierPaths:
    """Verify each of the four label outcomes on hand-crafted tapes."""

    def setup_method(self):
        self.fees = FeeModel()
        self.cfg = BarrierConfig(
            target_ps=0.05,
            stop_ps=0.03,
            timeout_s=10,
            fees=self.fees,
            clip_shares=1000,
        )

    def test_win_label_and_exit_s(self):
        """Bid rises above entry + target + sell_cost 5 steps after entry -> label 1."""
        entry_ask = 5.00
        sell_cost = self.fees.sell_cost_per_share(entry_ask, 1000)
        win_bid = entry_ask + self.cfg.target_ps + sell_cost + 0.001  # just above threshold

        # bid_future: index 0 = entry bar, index 5 = WIN bar
        n_future = 20
        bid_future = np.full(n_future, entry_ask - 0.01)
        bid_future[5] = win_bid   # WIN at 5 steps after entry

        bars, eidx = _make_labeler_bars(bid_future, entry_ask)
        labels = label_scalps(bars, self.cfg)

        row = labels.iloc[eidx]
        assert row["label"] == pytest.approx(1.0)
        assert row["exit_s"] == pytest.approx(5.0)
        assert row["entry_px"] == pytest.approx(entry_ask)
        assert np.isnan(row["timeout_edge"])

    def test_loss_label_and_exit_s(self):
        """Bid drops below entry - stop 3 steps after entry -> label -1, exit_s 3."""
        entry_ask = 5.00
        loss_bid = entry_ask - self.cfg.stop_ps - 0.001  # just below stop

        n_future = 20
        bid_future = np.full(n_future, entry_ask - 0.01)
        bid_future[3] = loss_bid   # LOSS at 3 steps after entry

        bars, eidx = _make_labeler_bars(bid_future, entry_ask)
        labels = label_scalps(bars, self.cfg)

        row = labels.iloc[eidx]
        assert row["label"] == pytest.approx(-1.0)
        assert row["exit_s"] == pytest.approx(3.0)
        assert np.isnan(row["timeout_edge"])

    def test_timeout_label_and_edge(self):
        """Neither barrier hit within timeout_s -> label 0, timeout_edge correct."""
        entry_ask = 5.00
        timeout = self.cfg.timeout_s

        # bid stays flat just inside both barriers; need timeout+1 future bars
        n_future = timeout + 5
        bid_future = np.full(n_future, entry_ask - 0.001)

        bars, eidx = _make_labeler_bars(bid_future, entry_ask)
        labels = label_scalps(bars, self.cfg)

        row = labels.iloc[eidx]
        assert row["label"] == pytest.approx(0.0)
        assert row["exit_s"] == pytest.approx(float(timeout))

        # timeout_edge = bid[t+timeout] - entry - sell_cost_at_exit
        exit_bid = bid_future[timeout]
        exit_sell_cost = self.fees.sell_cost_per_share(exit_bid, 1000)
        expected_edge = exit_bid - entry_ask - exit_sell_cost
        assert row["timeout_edge"] == pytest.approx(expected_edge)

    def test_nan_nbbo_at_entry_gives_nan_label(self):
        """NaN bid/ask at the entry bar -> label NaN (INVALID)."""
        n = 30
        bid_arr = np.full(n, 5.00)
        ask_arr = np.full(n, 5.02)
        # Put NaN NBBO at a row that otherwise has prior trades (row 15)
        entry_idx = 15
        bid_arr[entry_idx] = np.nan
        ask_arr[entry_idx] = np.nan

        idx = pd.date_range("2024-06-03 13:30:00", periods=n, freq="1s", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": 5.01, "high": 5.02, "low": 5.00, "close": 5.01,
                "volume": 100.0, "vwap": 5.01, "n_trades": 5.0,
                "bid": bid_arr, "ask": ask_arr,
                "bid_size": 10.0, "ask_size": 10.0, "spread": 0.02,
            },
            index=idx,
        )
        cfg = BarrierConfig(target_ps=0.05, stop_ps=0.03, timeout_s=10)
        labels = label_scalps(bars, cfg)

        assert np.isnan(labels.iloc[entry_idx]["label"])
        assert np.isnan(labels.iloc[entry_idx]["entry_px"])


# ---------------------------------------------------------------------------
# triple_barrier: perturbation / no-lookahead
# ---------------------------------------------------------------------------

class TestTripleBarrierNolookahead:
    """Changing bars beyond t+timeout_s must not change label at t."""

    def test_perturbation_beyond_timeout(self):
        target_ps = 0.05
        stop_ps = 0.03
        timeout_s = 10
        cfg = BarrierConfig(
            target_ps=target_ps, stop_ps=stop_ps, timeout_s=timeout_s
        )

        entry_ask = 5.00
        # Flat tape well inside both barriers -> TIMEOUT for entry row
        n_future = timeout_s + 20
        bid_future = np.full(n_future, entry_ask - 0.001)

        bars, eidx = _make_labeler_bars(bid_future, entry_ask)

        labels_before = label_scalps(bars, cfg)

        # Mutate bars strictly beyond (eidx + timeout_s)
        perturbed = bars.copy()
        beyond = eidx + timeout_s + 1
        perturbed.iloc[beyond:, perturbed.columns.get_loc("bid")] = 999.0
        perturbed.iloc[beyond:, perturbed.columns.get_loc("ask")] = 999.0

        labels_after = label_scalps(perturbed, cfg)

        # Entry row label must be identical before and after mutation
        assert labels_before.iloc[eidx]["label"] == pytest.approx(
            labels_after.iloc[eidx]["label"]
        )
        assert labels_before.iloc[eidx]["exit_s"] == pytest.approx(
            labels_after.iloc[eidx]["exit_s"]
        )
        # Full rows up to and including entry_idx must be unchanged
        pd.testing.assert_frame_equal(
            labels_before.iloc[: eidx + 1],
            labels_after.iloc[: eidx + 1],
        )


# ---------------------------------------------------------------------------
# triple_barrier: fee boundary — gross move = target but sell_cost eats it
# ---------------------------------------------------------------------------

class TestFeesBoundary:
    """A gross move exactly equal to target_ps is NOT a win after sell_cost."""

    def test_exact_gross_move_not_a_win(self):
        fees = FeeModel()
        target_ps = 0.05
        stop_ps = 0.10
        timeout_s = 20
        cfg = BarrierConfig(
            target_ps=target_ps, stop_ps=stop_ps, timeout_s=timeout_s, fees=fees
        )

        entry_ask = 5.00
        # Bid rises by exactly target_ps — NOT enough to cover sell_cost too
        gross_bid = entry_ask + target_ps  # below WIN threshold (sell_cost unpaid)

        n_future = timeout_s + 5
        bid_future = np.full(n_future, entry_ask - 0.001)
        bid_future[5] = gross_bid   # gross hit but NOT a WIN

        bars, eidx = _make_labeler_bars(bid_future, entry_ask)
        labels = label_scalps(bars, cfg)

        # Should NOT be a WIN (0 or -1, depending on further movement)
        assert labels.iloc[eidx]["label"] != pytest.approx(1.0)

    def test_gross_move_plus_epsilon_is_a_win(self):
        fees = FeeModel()
        target_ps = 0.05
        stop_ps = 0.10
        timeout_s = 20
        cfg = BarrierConfig(
            target_ps=target_ps, stop_ps=stop_ps, timeout_s=timeout_s, fees=fees
        )

        entry_ask = 5.00
        sell_cost = fees.sell_cost_per_share(entry_ask, cfg.clip_shares)
        # Bid rises by target_ps + sell_cost + epsilon -> exactly crosses WIN threshold
        win_bid = entry_ask + target_ps + sell_cost + 1e-6

        n_future = timeout_s + 5
        bid_future = np.full(n_future, entry_ask - 0.001)
        bid_future[5] = win_bid

        bars, eidx = _make_labeler_bars(bid_future, entry_ask)
        labels = label_scalps(bars, cfg)

        assert labels.iloc[eidx]["label"] == pytest.approx(1.0)
        assert labels.iloc[eidx]["exit_s"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# bars_features: additional feature sanity
# ---------------------------------------------------------------------------

class TestReturnFeatures:
    """Sanity checks on return features."""

    def test_ret_5s_known_value(self):
        # Price doubles at bar 5 (0-indexed), so ret_5s at bar 5 = (10-5)/5 = 1.0
        price = np.array([5.0] * 5 + [10.0] * 10, dtype=float)
        bars = _make_bars(n=15, price=price)
        feats = build_features(bars)
        assert feats["ret_5s"].iloc[5] == pytest.approx(1.0, abs=1e-10)

    def test_ret_features_nan_at_start(self):
        # ret_300s needs 300 bars; with n=50 all should be NaN
        bars = _make_bars(n=50)
        feats = build_features(bars)
        assert feats["ret_300s"].iloc[:50].isna().all()

    def test_mom_accel_zero_on_flat(self):
        # Flat price -> all rets zero -> mom_accel = 0
        bars = _make_bars(n=100, price=5.0)
        feats = build_features(bars)
        # Once warm-up is done (>= 30 bars for 15+15 shift), should be 0
        assert feats["mom_accel"].iloc[30:].dropna().abs().max() < 1e-10


class TestQuoteFeatures:
    """quote_ok and spread_bps."""

    def test_quote_ok_nan_bid(self):
        bars = _make_bars(n=10)
        bars.iloc[3, bars.columns.get_loc("bid")] = np.nan
        feats = build_features(bars)
        assert feats["quote_ok"].iloc[3] == 0.0
        assert feats["quote_ok"].iloc[4] == 1.0

    def test_spread_bps_exact(self):
        # spread=0.02, price=5.0, mid=5.0 -> 0.02/5.0 * 1e4 = 40 bps
        bars = _make_bars(n=5, price=5.0, spread=0.02)
        feats = build_features(bars)
        assert feats["spread_bps"].iloc[0] == pytest.approx(40.0, abs=1e-6)

    def test_spread_bps_narrow(self):
        # spread=0.01, price=10.0 -> 0.01/10.0 * 1e4 = 10 bps
        bars = _make_bars(n=5, price=10.0, spread=0.01)
        feats = build_features(bars)
        assert feats["spread_bps"].iloc[0] == pytest.approx(10.0, abs=1e-6)


class TestTodMin:
    """tod_min: minutes since 09:30 ET."""

    def test_tod_min_at_open(self):
        # Start at exactly 09:30 ET = 13:30 UTC
        bars = _make_bars(n=10, start="2024-06-03 13:30:00")
        feats = build_features(bars)
        # Bar 0 timestamp 13:30:00 UTC = 09:30:00 ET -> 0 min
        assert feats["tod_min"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        # Bar 1 = 1 second = 1/60 minutes
        assert feats["tod_min"].iloc[1] == pytest.approx(1.0 / 60.0, abs=1e-6)


def test_labeler_truncated_window_is_invalid():
    """Entries whose label window runs past end-of-day must be NaN, not 0."""
    timeout_s = 10
    cfg = BarrierConfig(target_ps=0.05, stop_ps=0.03, timeout_s=timeout_s)
    entry_ask = 5.00
    # only 5 future seconds < timeout_s: window is truncated at end of day
    bid_future = np.full(5, entry_ask - 0.001)
    bars, eidx = _make_labeler_bars(bid_future, entry_ask)
    labels = label_scalps(bars, cfg)
    assert np.isnan(labels.iloc[eidx]["label"])
    # with a full window the same flat tape resolves to TIMEOUT (0), not NaN
    bars_full, eidx_full = _make_labeler_bars(
        np.full(timeout_s + 5, entry_ask - 0.001), entry_ask)
    labels_full = label_scalps(bars_full, cfg)
    assert labels_full.iloc[eidx_full]["label"] == 0.0
