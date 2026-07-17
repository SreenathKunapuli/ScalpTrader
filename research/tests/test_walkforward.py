"""Walk-forward harness: split integrity, dataset build, exact expectancy math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.walkforward import TrainConfig, _non_overlapping, build_dataset, \
    evaluate, fit_model, split_days

CFG = TrainConfig(target_ps=0.05, stop_ps=0.04, timeout_s=20,
                  max_rows_per_day=500, prob_threshold_grid=(0.4, 0.6))


def _synth_day(path, seed: int, n: int = 900, base: float = 3.0) -> None:
    """One synthetic stock-day parquet with drifting price + NBBO.
    Vol is set so all three labels (win/loss/timeout) occur at CFG barriers."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-06-03 14:00:00", periods=n, freq="1s", tz="UTC")
    mid = base + np.cumsum(rng.normal(0.0004, 0.012, n))
    spread = 0.02
    df = pd.DataFrame({
        "open": mid, "high": mid + 0.004, "low": mid - 0.004, "close": mid,
        "volume": rng.integers(0, 3000, n).astype(float), "vwap": mid,
        "n_trades": rng.integers(1, 12, n).astype(float),
        "bid": mid - spread / 2, "ask": mid + spread / 2,
        "bid_size": np.full(n, 8.0), "ask_size": np.full(n, 8.0),
        "spread": np.full(n, spread),
    }, index=idx)
    df.to_parquet(path)


def test_split_days_temporal_and_embargo():
    dates = [f"2024-06-{d:02d}" for d in range(3, 21)]
    cfg = TrainConfig(test_frac=0.25, embargo_days=2)
    train, test = split_days(dates, cfg)
    assert max(train) < min(test)
    gap = (pd.Timestamp(min(test)) - pd.Timestamp(max(train))).days
    assert gap > cfg.embargo_days
    # deterministic
    assert split_days(dates, cfg) == (train, test)


def test_build_dataset_caps_and_drops_invalid(tmp_path):
    files = []
    for i, d in enumerate(["2024-06-03", "2024-06-04", "2024-06-05"]):
        f = tmp_path / f"SYM{i}_{d}.parquet"
        _synth_day(f, seed=i)
        files.append(f)
    x, y, meta = build_dataset(files, CFG)
    assert len(x) == len(y) == len(meta)
    assert y.notna().all()
    per_day = meta.groupby("date").size()
    assert (per_day <= CFG.max_rows_per_day + 3).all()  # ceil rounding slack
    assert set(meta["date"]) == {"2024-06-03", "2024-06-04", "2024-06-05"}


class _StubModel:
    """predict_proba returns fixed P(win) per row; classes [-1, 0, 1]."""

    classes_ = np.array([-1.0, 0.0, 1.0])

    def __init__(self, p_win: np.ndarray) -> None:
        self._p = p_win

    def predict_proba(self, x) -> np.ndarray:  # noqa: ANN001
        rest = (1 - self._p) / 2
        return np.column_stack([rest, rest, self._p])


def test_evaluate_exact_expectancy_math():
    idx = pd.date_range("2024-06-03 14:00:00", periods=4, freq="120s", tz="UTC")
    x = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]}, index=idx)
    y = pd.Series([1.0, -1.0, 0.0, 1.0], index=idx)
    meta = pd.DataFrame({
        "symbol": "AAA", "date": "2024-06-03",
        "timeout_edge": [np.nan, np.nan, 0.01, np.nan],
        "exit_s": [10.0, 10.0, 20.0, 10.0],
        "target_ps": 0.05, "stop_ps": 0.04,
    }, index=idx)
    model = _StubModel(np.array([0.9, 0.9, 0.9, 0.3]))
    cfg = TrainConfig(target_ps=0.05, stop_ps=0.04, prob_threshold_grid=(0.5,))
    summary, per_thr = evaluate(model, x, y, meta, cfg)
    row = per_thr.iloc[0]
    # rows 0,1,2 selected (p=0.9), spaced 120s apart -> no overlap removal
    assert row["n_trades"] == 3
    assert row["hit_rate"] == pytest.approx(1 / 3)
    assert row["expectancy_ps"] == pytest.approx((0.05 - 0.04 + 0.01) / 3)
    assert row["sum_pnl_1000sh"] == pytest.approx((0.05 - 0.04 + 0.01) * 1000)


def test_non_overlap_enforced_within_symbol_day():
    idx = pd.date_range("2024-06-03 14:00:00", periods=5, freq="1s", tz="UTC")
    sel = pd.DataFrame({
        "symbol": "AAA", "date": "2024-06-03",
        "p_win": 0.9, "edge": 0.05, "label": 1.0,
        "exit_s": [3.0, 3.0, 3.0, 3.0, 3.0],
    }, index=idx)
    kept = _non_overlapping(sel)
    # entries at t=0 (busy to 3s), t=3 (busy to 6s): only 2 of 5 survive
    assert len(kept) == 2


def test_vol_barriers_causal_and_floored(tmp_path):
    from scalp.walkforward import barrier_arrays
    f = tmp_path / "AAA_2024-06-03.parquet"
    _synth_day(f, seed=3)
    bars = pd.read_parquet(f)
    cfg = TrainConfig(barrier_mode="vol", vol_window_s=60,
                      vol_target_mult=1.0, vol_stop_mult=0.5)
    tgt, stp = barrier_arrays(bars, cfg)
    assert np.isnan(tgt[0])                       # warmup rows undefined
    assert np.nanmin(tgt) >= cfg.min_target_ps    # floored
    assert np.nanmin(stp) >= cfg.min_stop_ps
    # causality: rewriting the future must not change barrier at t0
    t0 = 400
    pert = bars.copy()
    pert.iloc[t0 + 1:, pert.columns.get_loc("close")] *= 9.0
    tgt2, _ = barrier_arrays(pert, cfg)
    np.testing.assert_allclose(tgt[: t0 + 1], tgt2[: t0 + 1])
    # stop scales with target (half by config) where defined
    ratio = stp[~np.isnan(stp) & (tgt > cfg.min_target_ps)] \
        / tgt[~np.isnan(stp) & (tgt > cfg.min_target_ps)]
    assert (ratio <= 0.5 + 1e-9).all()


def test_end_to_end_smoke(tmp_path):
    files = []
    for i, d in enumerate(["2024-06-03", "2024-06-04", "2024-06-05",
                           "2024-06-06", "2024-06-07"]):
        f = tmp_path / f"SYM{i}_{d}.parquet"
        _synth_day(f, seed=10 + i)
        files.append(f)
    dates = [f.stem.rsplit("_", 1)[1] for f in files]
    train_d, test_d = split_days(dates, CFG)
    x_tr, y_tr, _ = build_dataset(
        [f for f, d in zip(files, dates, strict=True) if d in train_d], CFG)
    x_te, y_te, m_te = build_dataset(
        [f for f, d in zip(files, dates, strict=True) if d in test_d], CFG)
    model = fit_model(x_tr, y_tr, CFG.seed)
    summary, per_thr = evaluate(model, x_te, y_te, m_te, CFG)
    assert set(per_thr.columns) >= {"threshold", "n_trades", "hit_rate",
                                    "expectancy_ps", "sum_pnl_1000sh"}
    # higher threshold can never select more trades
    assert per_thr["n_trades"].is_monotonic_decreasing
    assert summary["n_test_days"] >= 1
