"""Targeted exact-value tests for the causal features added to
bars_features.py::build_features (order flow, LULD proximity, round-number
distance, range compression, momentum freshness, vwap slope).

Generic causality for ALL build_features columns (including these) is
already covered by test_no_lookahead.py; this file checks hand-computed
semantics on small constructed tapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scalp.bars_features import build_features


def _make_bars(
    n: int,
    price: float | np.ndarray = 5.0,
    volume: float | np.ndarray = 100.0,
    start: str = "2024-06-03 13:30:00",  # 09:30 ET
) -> pd.DataFrame:
    """Build a minimal contiguous-second bar DataFrame."""
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    px = np.full(n, float(price)) if np.isscalar(price) else np.asarray(price, float)
    vol = np.full(n, float(volume)) if np.isscalar(volume) else np.asarray(volume, float)
    return pd.DataFrame(
        {
            "open": px, "high": px, "low": px, "close": px,
            "volume": vol, "vwap": px,
            "n_trades": np.full(n, 1.0),
            "bid": px - 0.01, "ask": px + 0.01,
            "bid_size": np.full(n, 10.0), "ask_size": np.full(n, 10.0),
            "spread": np.full(n, 0.02),
        },
        index=idx,
    )


class TestFlowImb:
    def test_monotonic_up_is_fully_positive(self):
        price = np.linspace(3.0, 4.0, 150)
        bars = _make_bars(n=150, price=price, volume=100.0)
        feats = build_features(bars)
        # once the 60s window no longer straddles row 0 (whose diff is
        # undefined), every signed volume in-window is positive -> ratio 1.0
        vals = feats["flow_imb_60s"].iloc[60:].to_numpy()
        assert np.allclose(vals, 1.0, atol=1e-10)

    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(3)
        n = 500
        price = np.round(3.0 + rng.normal(0, 0.02, n).cumsum(), 2).clip(0.5)
        vol = rng.integers(1, 500, n).astype(float)
        bars = _make_bars(n=n, price=price, volume=vol)
        feats = build_features(bars)
        vals = feats["flow_imb_60s"].dropna()
        assert (vals >= -1.0 - 1e-9).all() and (vals <= 1.0 + 1e-9).all()

    def test_flow_imb_chg_is_flat_when_regime_unchanged(self):
        price = np.linspace(3.0, 4.0, 150)
        bars = _make_bars(n=150, price=price, volume=100.0)
        feats = build_features(bars)
        # ratio is pinned at 1.0 from row 60 on -> chg is exactly 0 once
        # both the current and 30s-ago values fall in that stable regime
        chg = feats["flow_imb_chg"].iloc[90:].to_numpy()
        assert np.allclose(chg, 0.0, atol=1e-10)

    def test_zero_volume_no_inf(self):
        bars = _make_bars(n=60, volume=0.0)
        feats = build_features(bars)
        assert not np.isinf(feats["flow_imb_60s"]).any()


class TestLuldUpDist:
    def test_flat_price_gives_exact_ten_percent(self):
        bars = _make_bars(n=60, price=5.0, volume=100.0)
        feats = build_features(bars)
        # reference == 5.0 everywhere once warmed up (min_periods=30)
        # band = 5.5, feature = (5.5 - 5.0) / 5.0 = 0.10
        vals = feats["luld_up_dist"].iloc[29:].to_numpy()
        assert np.allclose(vals, 0.10, atol=1e-10)

    def test_near_band_gives_small_value(self):
        # price ramps right up to the 10% band -> dist shrinks toward 0
        n = 60
        price = np.full(n, 5.0)
        price[-1] = 5.5  # equals the band computed off a ~5.0 reference
        bars = _make_bars(n=n, price=price, volume=100.0)
        feats = build_features(bars)
        assert feats["luld_up_dist"].iloc[-1] < feats["luld_up_dist"].iloc[-2]


class TestRoundDist:
    def test_exact_half_dollar_levels(self):
        price = np.array([5.00, 5.20, 5.24, 5.26, 5.50, 5.74])
        bars = _make_bars(n=len(price), price=price, volume=100.0)
        feats = build_features(bars)
        expected = np.array([
            (5.00 - 5.00) / 5.00,   # exactly on a level
            (5.20 - 5.00) / 5.20,   # nearer to 5.00 than 5.50
            (5.24 - 5.00) / 5.24,   # 0.24 to 5.00 vs 0.26 to 5.50 -> 5.00 wins
            (5.26 - 5.50) / 5.26,   # 0.26 to 5.00 vs 0.24 to 5.50 -> 5.50 wins
            (5.50 - 5.50) / 5.50,   # exactly on a level
            (5.74 - 5.50) / 5.74,   # nearer to 5.50 than 6.00
        ])
        np.testing.assert_allclose(feats["round_dist"].to_numpy(), expected, atol=1e-10)


class TestRangeCompress:
    def test_matches_hand_computed_ratio(self):
        # First 300 bars: wide swings (range 2.0). Last 60 bars: tight (range 0.1).
        wide = 3.0 + np.tile([0.0, 2.0], 150)
        tight = np.full(60, 5.0)
        tight[::2] += 0.1
        price = np.concatenate([wide, tight])
        bars = _make_bars(n=len(price), price=price, volume=100.0)
        bars["high"] = price
        bars["low"] = price
        feats = build_features(bars)
        last = feats["range_compress"].iloc[-1]
        range_60 = price[-60:].max() - price[-60:].min()
        range_300 = price[-300:].max() - price[-300:].min()
        assert last == pytest.approx(range_60 / range_300, abs=1e-10)
        assert last < 1.0  # compression: short range much smaller than long range


class TestMomFresh:
    def test_resets_at_new_high_then_grows_stale(self):
        price = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0])
        bars = _make_bars(n=len(price), price=price, volume=100.0)
        feats = build_features(bars)
        mf = feats["mom_fresh"].to_numpy()
        # rows 0..4 are all new running highs (monotonically increasing) -> 0
        np.testing.assert_allclose(mf[:5], 0.0, atol=1e-10)
        # after the peak, staleness grows by 1 each second, scaled by /600
        np.testing.assert_allclose(mf[5:9], np.array([1, 2, 3, 4]) / 600.0, atol=1e-10)

    def test_capped_at_one(self):
        price = np.concatenate([[10.0], np.full(700, 1.0)])
        bars = _make_bars(n=len(price), price=price, volume=100.0)
        feats = build_features(bars)
        assert feats["mom_fresh"].iloc[-1] == pytest.approx(1.0, abs=1e-10)


class TestVwapSlope:
    def test_equals_vwap_dist_change_over_60s(self):
        n = 150
        price = np.concatenate([np.full(75, 4.0), np.full(75, 5.0)])
        bars = _make_bars(n=n, price=price, volume=100.0)
        feats = build_features(bars)
        expected = feats["vwap_dist"] - feats["vwap_dist"].shift(60)
        pd.testing.assert_series_equal(
            feats["vwap_slope"], expected, check_names=False, check_exact=False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
