"""Unit tests for the hyperparameter-search additions to the train CLI:
fit_model kwarg forwarding, --drop-features column logic, and the
--val-frac inner temporal split. Function-level only — no full CLI runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.walkforward import TrainConfig, fit_model, split_val_days
from scripts.sim_eval import day_entries
from scripts.train_scalper import drop_columns, parse_drop_features


def _xy(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n))
    return x, y


# --------------------------------------------------------------------------
# fit_model hyperparameter forwarding
# --------------------------------------------------------------------------

def test_fit_model_default_matches_sklearn_defaults():
    """Omitting all hp kwargs must reproduce today's exact behavior."""
    x, y = _xy()
    model = fit_model(x, y, seed=7)
    assert model.learning_rate == 0.1
    assert model.max_iter == 100
    assert model.max_leaf_nodes == 31
    assert model.min_samples_leaf == 20
    assert model.l2_regularization == 0.0
    assert model.random_state == 7


def test_fit_model_forwards_hyperparameters():
    x, y = _xy()
    model = fit_model(x, y, seed=7, learning_rate=0.05, max_iter=17,
                      max_leaf_nodes=9, min_samples_leaf=3,
                      l2_regularization=0.25)
    assert model.learning_rate == 0.05
    assert model.max_iter == 17
    assert model.max_leaf_nodes == 9
    assert model.min_samples_leaf == 3
    assert model.l2_regularization == 0.25


def test_fit_model_partial_hyperparameters_leave_rest_default():
    x, y = _xy()
    model = fit_model(x, y, seed=7, max_iter=5)
    assert model.max_iter == 5
    # untouched knobs keep sklearn defaults
    assert model.learning_rate == 0.1
    assert model.max_leaf_nodes == 31
    assert model.min_samples_leaf == 20
    assert model.l2_regularization == 0.0


# --------------------------------------------------------------------------
# --drop-features
# --------------------------------------------------------------------------

def test_parse_drop_features_splits_and_strips():
    assert parse_drop_features("a, b,,c ") == ["a", "b", "c"]
    assert parse_drop_features(None) == []
    assert parse_drop_features("") == []


def test_drop_columns_removes_named_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    out = drop_columns(df, ["b"])
    assert list(out.columns) == ["a", "c"]
    assert len(out) == len(df)


def test_drop_columns_noop_when_empty():
    df = pd.DataFrame({"a": [1, 2]})
    out = drop_columns(df, [])
    assert out is df


def test_drop_columns_errors_on_unknown_name():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="unknown column"):
        drop_columns(df, ["nope"])


# --------------------------------------------------------------------------
# --val-frac inner temporal split
# --------------------------------------------------------------------------

def test_split_val_days_strictly_temporal():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]  # 20 unique days
    core, val = split_val_days(dates, 0.2)
    assert core and val
    assert max(core) < min(val)
    assert set(core) | set(val) == set(dates)
    assert set(core).isdisjoint(val)
    # deterministic
    assert split_val_days(dates, 0.2) == (core, val)


def test_split_val_days_zero_frac_is_noop():
    dates = [f"2024-06-{d:02d}" for d in range(1, 6)]
    core, val = split_val_days(dates, 0.0)
    assert core == sorted(dates)
    assert val == []


def test_split_val_days_keeps_at_least_one_core_day():
    dates = ["2024-06-01", "2024-06-02"]
    core, val = split_val_days(dates, 0.99)
    assert len(core) >= 1
    assert max(core) < min(val)


# --------------------------------------------------------------------------
# sim_eval.py / export_model.py: shared-helper reuse + HP/drop-features
# plumbing into evaluation and export
# --------------------------------------------------------------------------

def test_sim_eval_reuses_shared_drop_helpers():
    """sim_eval must import (not reimplement) train_scalper's parse/drop
    helpers, so the two scripts can never drift on --drop-features semantics."""
    from scripts import sim_eval, train_scalper
    assert sim_eval.drop_columns is train_scalper.drop_columns
    assert sim_eval.parse_drop_features is train_scalper.parse_drop_features


def test_export_model_reuses_shared_drop_helpers():
    from scripts import export_model, train_scalper
    assert export_model.drop_columns is train_scalper.drop_columns
    assert export_model.parse_drop_features is train_scalper.parse_drop_features


def test_day_entries_drops_features_before_predict():
    """--drop-features must be applied to the per-day feature frame BEFORE
    predict_proba, not just to the training fit matrix — a column mismatch
    here would crash predict_proba in the live-shaped OOS loop."""
    idx = pd.date_range("2024-06-03 14:00:00", periods=400, freq="1s", tz="UTC")
    rng = np.random.default_rng(0)
    mid = 3.0 + np.cumsum(rng.normal(0.0004, 0.012, 400))
    bars = pd.DataFrame({
        "open": mid, "high": mid + 0.004, "low": mid - 0.004, "close": mid,
        "volume": rng.integers(0, 3000, 400).astype(float), "vwap": mid,
        "n_trades": rng.integers(1, 12, 400).astype(float),
        "bid": mid - 0.01, "ask": mid + 0.01,
        "bid_size": np.full(400, 8.0), "ask_size": np.full(400, 8.0),
        "spread": np.full(400, 0.02),
    }, index=idx)

    class _RecordingModel:
        """predict_proba always signals WIN; records the columns it saw."""

        classes_ = np.array([-1.0, 0.0, 1.0])
        seen_columns: list[str] | None = None

        def predict_proba(self, x):  # noqa: ANN001
            type(self).seen_columns = list(x.columns)
            n = len(x)
            return np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])

    cfg = TrainConfig(target_ps=0.05, stop_ps=0.04, timeout_s=20,
                      barrier_mode="fixed")
    entries, lab = day_entries(bars, _RecordingModel(), cfg, threshold=0.5,
                               qty=100, drop_feats=["spread_bps"])
    assert _RecordingModel.seen_columns is not None
    assert "spread_bps" not in _RecordingModel.seen_columns
    assert not entries.empty
    assert not lab.empty
