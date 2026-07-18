"""Unit tests for the hyperparameter-search additions to the train CLI:
fit_model kwarg forwarding, --drop-features column logic, and the
--val-frac inner temporal split. Function-level only — no full CLI runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalp.walkforward import fit_model, split_val_days
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
