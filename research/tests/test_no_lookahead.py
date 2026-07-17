"""Rewrite-the-future battery: mutate the tape after a cut point and prove
features, barriers, and labels before the cut are bit-identical. Any failure
here is lookahead on the money path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.bars_features import build_features
from scalp.triple_barrier import BarrierConfig, label_scalps
from scalp.walkforward import TrainConfig, barrier_arrays, split_days

N = 600
CUT = 400
TIMEOUT = 60


def make_frame(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-06 15:00:00", periods=n, freq="1s", tz="UTC")
    close = np.round(3.0 + rng.normal(0, 0.01, n).cumsum(), 2).clip(0.5)
    vol = rng.integers(0, 400, n).astype(float)
    close_nan = close.copy()
    close_nan[vol == 0] = np.nan            # trade-empty seconds: NaN OHLC
    return pd.DataFrame({
        "open": close_nan, "high": close_nan, "low": close_nan,
        "close": close_nan, "volume": vol,
        "n_trades": (vol > 0).astype(float) * rng.integers(1, 9, n),
        "vwap": close_nan, "bid": close - 0.01, "ask": close + 0.01,
        "bid_size": 5.0, "ask_size": 5.0, "spread": 0.02,
    }, index=idx)


def rewrite_future(frame: pd.DataFrame, cut: int = CUT) -> pd.DataFrame:
    """A violently different future: prices x1.5, volume x3, busier tape."""
    f = frame.copy()
    price_cols = ["open", "high", "low", "close", "vwap", "bid", "ask"]
    f.iloc[cut:, [f.columns.get_loc(c) for c in price_cols]] *= 1.5
    f.iloc[cut:, f.columns.get_loc("volume")] *= 3
    f.iloc[cut:, f.columns.get_loc("n_trades")] += 5
    return f


def test_features_are_causal():
    a, b = make_frame(), rewrite_future(make_frame())
    pd.testing.assert_frame_equal(build_features(a).iloc[: CUT],
                                  build_features(b).iloc[: CUT])


def test_barrier_arrays_are_causal():
    cfg = TrainConfig(barrier_mode="vol", vol_window_s=300)
    ta, sa = barrier_arrays(make_frame(), cfg)
    tb, sb = barrier_arrays(rewrite_future(make_frame()), cfg)
    np.testing.assert_array_equal(ta[:CUT], tb[:CUT])
    np.testing.assert_array_equal(sa[:CUT], sb[:CUT])


def test_labels_depend_only_on_their_window():
    cfg = TrainConfig(barrier_mode="vol", vol_window_s=300, timeout_s=TIMEOUT)
    a, b = make_frame(), rewrite_future(make_frame())
    la = label_scalps(a, cfg.barrier(), *barrier_arrays(a, cfg))
    lb = label_scalps(b, cfg.barrier(), *barrier_arrays(b, cfg))
    # every row whose full label window (t, t+timeout] closes before the cut
    safe = CUT - TIMEOUT - 1
    pd.testing.assert_frame_equal(la.iloc[:safe], lb.iloc[:safe])
    assert la["label"].notna().sum() > 50        # battery isn't vacuous


def test_in_window_future_does_flip_labels():
    # flat tape -> TIMEOUT; a bid spike inside the window -> WIN. If this
    # fails, the labeler stopped reading the future it is SUPPOSED to read.
    idx = pd.date_range("2025-01-06 15:00:00", periods=200, freq="1s", tz="UTC")
    flat = pd.DataFrame({
        "open": 3.0, "high": 3.0, "low": 3.0, "close": 3.0, "volume": 100.0,
        "n_trades": 1.0, "vwap": 3.0, "bid": 2.99, "ask": 3.0,
        "bid_size": 5.0, "ask_size": 5.0, "spread": 0.01}, index=idx)
    cfg = BarrierConfig(target_ps=0.05, stop_ps=0.04, timeout_s=TIMEOUT)
    spiked = flat.copy()
    spiked.iloc[80, spiked.columns.get_loc("bid")] = 3.10
    assert label_scalps(flat, cfg)["label"].iloc[50] == 0.0
    assert label_scalps(spiked, cfg)["label"].iloc[50] == 1.0


def test_split_days_temporal_with_embargo():
    dates = [str(d.date()) for d in
             pd.date_range("2025-01-01", periods=40, freq="3D")]
    cfg = TrainConfig(embargo_days=2, test_frac=0.25)
    train, test = split_days(dates, cfg)
    assert train and test
    assert max(train) < min(test)
    gap = (pd.Timestamp(min(test)) - pd.Timestamp(max(train))).days
    assert gap > cfg.embargo_days


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
