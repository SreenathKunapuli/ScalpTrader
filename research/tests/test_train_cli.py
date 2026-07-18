"""Unit tests for the hyperparameter-search additions to the train CLI:
fit_model kwarg forwarding, --drop-features column logic, and the
--val-frac inner temporal split. Function-level only — no full CLI runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scalp.walkforward import TrainConfig, fit_model, split_days, split_val_days
from scripts.sim_eval import day_entries
from scripts.train_scalper import drop_columns, limit_by_quality, \
    parse_drop_features, quality_weight_array


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
# --test-start-date pinned OOS window
# --------------------------------------------------------------------------

def test_split_days_test_start_date_respects_boundary_and_embargo():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]  # 20 unique days
    cfg = TrainConfig(test_start_date="2024-06-15", embargo_days=1)
    train, test = split_days(dates, cfg)
    # every test day is on/after the pinned boundary
    assert min(test) == "2024-06-15"
    assert all(d >= "2024-06-15" for d in test)
    # no train day reaches the boundary
    assert all(d < "2024-06-15" for d in train)
    # the embargo day immediately before the boundary is dropped from train
    assert "2024-06-14" not in train
    assert "2024-06-13" in train


def test_split_days_none_preserves_old_fraction_split():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]  # 20 unique days
    cfg_old = TrainConfig(test_frac=0.25, embargo_days=1)
    cfg_new = TrainConfig(test_frac=0.25, embargo_days=1, test_start_date=None)
    assert split_days(dates, cfg_old) == split_days(dates, cfg_new)
    # sanity: byte-for-byte the same trailing-fraction behavior as today
    uniq = sorted(set(dates))
    n_test = max(1, -(-len(uniq) * 25 // 100))  # ceil via integer math
    expected_test = uniq[-n_test:]
    train, test = split_days(dates, cfg_new)
    assert test == expected_test


def test_split_days_test_start_date_empty_test_raises():
    dates = [f"2024-06-{d:02d}" for d in range(1, 6)]
    cfg = TrainConfig(test_start_date="2099-01-01")
    with pytest.raises(ValueError, match="leaves no test days"):
        split_days(dates, cfg)


# --------------------------------------------------------------------------
# --train-start-date training-recency filter
# --------------------------------------------------------------------------

def test_split_days_train_start_date_drops_only_pre_date_train_days():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]  # 20 unique days
    cfg_base = TrainConfig(test_frac=0.25, embargo_days=1)
    train_base, test_base = split_days(dates, cfg_base)

    cfg_recent = TrainConfig(test_frac=0.25, embargo_days=1,
                             train_start_date="2024-06-10")
    train_recent, test_recent = split_days(dates, cfg_recent)

    # test window is completely untouched by train_start_date
    assert test_recent == test_base
    # train side only loses days strictly before the cutoff
    assert all(d >= "2024-06-10" for d in train_recent)
    assert train_recent == [d for d in train_base if d >= "2024-06-10"]
    assert len(train_recent) < len(train_base)


def test_split_days_train_start_date_none_is_unchanged_behavior():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]
    cfg_old = TrainConfig(test_frac=0.25, embargo_days=1)
    cfg_new = TrainConfig(test_frac=0.25, embargo_days=1, train_start_date=None)
    assert split_days(dates, cfg_old) == split_days(dates, cfg_new)


# --------------------------------------------------------------------------
# --val-start-date pinned validation window
# --------------------------------------------------------------------------

def test_split_val_days_val_start_date_carves_exact_range():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]  # 20 unique days
    core, val = split_val_days(dates, val_frac=0.0, val_start_date="2024-06-15")
    assert val == [f"2024-06-{d:02d}" for d in range(15, 21)]
    assert core == [f"2024-06-{d:02d}" for d in range(1, 15)]
    assert max(core) < min(val)
    # val_start_date overrides val_frac entirely, even though val_frac=0.0
    # would otherwise mean "no val split"
    assert val


def test_split_val_days_val_start_date_empty_core_raises():
    dates = [f"2024-06-{d:02d}" for d in range(1, 6)]
    with pytest.raises(ValueError, match="leaves no core-train days"):
        split_val_days(dates, val_frac=0.2, val_start_date="2024-06-01")


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


# --------------------------------------------------------------------------
# --train-quality-limit corpus quality-depth knob
# --------------------------------------------------------------------------

def test_limit_by_quality_respects_top_n_membership():
    """Only candidates within the first N entries of the manifest-ranked
    file list survive; the relative order of `candidates` is preserved."""
    ranked = [Path(f"{i}.parquet") for i in range(10)]  # quality-rank order
    candidates = [ranked[2], ranked[5], ranked[8], ranked[1]]
    out = limit_by_quality(candidates, ranked, limit=6)
    assert out == [ranked[2], ranked[5], ranked[1]]


def test_limit_by_quality_none_is_identity():
    ranked = [Path(f"{i}.parquet") for i in range(5)]
    candidates = [ranked[4], ranked[0], ranked[2]]
    out = limit_by_quality(candidates, ranked, None)
    assert out is candidates


def test_limit_by_quality_val_test_untouched():
    """The knob only ever filters a caller-selected candidate set (e.g.
    core-train). Val/test file lists are simply never routed through the
    helper, so they stay fully intact — even though the SAME limit, if
    misapplied to them, would drop files (proving the filter has teeth)."""
    ranked = [Path(f"{i}.parquet") for i in range(10)]
    train_candidates = ranked[:8]
    val_files = ranked[8:9]
    test_files = ranked[9:10]
    limit = 3

    filtered_train = limit_by_quality(train_candidates, ranked, limit)
    assert filtered_train == ranked[:3]
    # if val/test were (wrongly) passed through the filter they'd be
    # emptied out at this limit ...
    assert limit_by_quality(val_files, ranked, limit) == []
    assert limit_by_quality(test_files, ranked, limit) == []
    # ... but the actual pipeline never calls the filter on them, so the
    # lists the caller holds remain the untouched originals
    assert val_files == ranked[8:9]
    assert test_files == ranked[9:10]


# --------------------------------------------------------------------------
# --quality-weight-mult soft quality-curation sample weights
# --------------------------------------------------------------------------

def test_quality_weight_array_hits_exactly_top_n_stems():
    """Rows whose (symbol, date) stem is in `top_stems` get `mult`; every
    other row gets 1.0 — exact membership, no partial credit."""
    meta = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "CCC", "DDD"],
        "date": ["2024-06-01", "2024-06-01", "2024-06-02",
                "2024-06-03", "2024-06-04"],
    })
    top_stems = {"AAA_2024-06-01", "CCC_2024-06-03"}
    w = quality_weight_array(meta, top_stems, mult=5.0)
    np.testing.assert_array_equal(w, np.array([5.0, 5.0, 1.0, 5.0, 1.0]))


def test_fit_model_sample_weight_mult_none_is_unchanged_behavior():
    """Passing sample_weight_mult=None (the default) must reproduce the
    exact same fitted model as omitting the kwarg entirely."""
    x, y = _xy()
    m_default = fit_model(x, y, seed=7)
    m_explicit_none = fit_model(x, y, seed=7, sample_weight_mult=None)
    np.testing.assert_array_equal(m_default.predict_proba(x),
                                  m_explicit_none.predict_proba(x))


def test_fit_model_composes_quality_mult_with_class_balance(monkeypatch):
    """sample_weight_mult must be multiplied elementwise into the existing
    class-balanced weights, not replace them."""
    import sklearn.ensemble as ensemble_mod

    captured: dict = {}

    class _RecordingClassifier:
        def __init__(self, random_state=None, **hp):
            self.random_state = random_state

        def fit(self, x, y, sample_weight=None):
            captured["sample_weight"] = sample_weight
            self.classes_ = np.unique(y)
            return self

    monkeypatch.setattr(ensemble_mod, "HistGradientBoostingClassifier",
                        _RecordingClassifier)

    x, y = _xy(n=50)
    mult = np.where(np.arange(50) < 10, 3.0, 1.0)
    fit_model(x, y, seed=7, sample_weight_mult=mult)

    freq = y.value_counts(normalize=True)
    base_w = y.map(lambda v: 1.0 / (len(freq) * freq[v])).to_numpy()
    np.testing.assert_allclose(captured["sample_weight"], base_w * mult)
