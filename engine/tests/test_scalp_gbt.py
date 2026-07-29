"""ScalpGbtSignal: artifact loading, gating, and feature-order parity."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scalpengine.signals.base import SignalOutput
from scalpengine.signals.scalp_gbt import ScalpGbtSignal, build_features

FEATURES = ["ret_5s", "ret_15s", "ret_60s", "ret_300s", "mom_accel",
            "vwap_dist", "pullback", "vol_surge", "tape_speed",
            "spread_bps", "quote_ok", "tod_min"]


@pytest.fixture()
def artifact(tmp_path):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, len(FEATURES))), columns=FEATURES)
    y = pd.Series(rng.choice([-1.0, 0.0, 1.0], 300))
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(random_state=0, max_iter=10)
    model.fit(x, y)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "features.json").write_text(json.dumps(FEATURES))
    (tmp_path / "inference.json").write_text(json.dumps({
        "threshold": 0.6, "timeout_s": 120, "barrier_mode": "vol",
        "vol_window_s": 300, "vol_target_mult": 1.0, "vol_stop_mult": 0.5,
        "min_target_ps": 0.02, "min_stop_ps": 0.015,
        "target_ps": 0.05, "stop_ps": 0.04}))
    return tmp_path


def synth_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-06 15:00:00", periods=n, freq="1s", tz="UTC")
    close = np.round(3.0 + rng.normal(0, 0.01, n).cumsum(), 2).clip(0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close, "volume": rng.integers(0, 500, n).astype(float),
        "n_trades": rng.integers(0, 9, n).astype(float), "vwap": close,
        "bid": close - 0.01, "ask": close + 0.01,
        "bid_size": 5.0, "ask_size": 5.0, "spread": 0.02,
    }, index=idx)


def test_artifact_features_exist_in_research(artifact):
    # every artifact feature must exist in build_features output — a missing
    # one would silently reindex to NaN at inference. build_features may grow
    # NEW columns (newer artifacts pick them up); it must never lose one.
    assert set(FEATURES) <= set(build_features(synth_frame()).columns)


def test_decision_on_warm_frame(artifact):
    sig = ScalpGbtSignal(artifact)
    dec = sig.compute_second("XYZ", synth_frame())
    assert dec is not None
    assert 0.0 <= dec.p_win <= 1.0
    assert dec.target_ps >= 0.02 and dec.stop_ps >= 0.015
    assert dec.target_ps >= dec.stop_ps          # 1.0x vs 0.5x multipliers
    assert dec.timeout_s == 120 and sig.threshold == 0.6


def test_cold_window_gates_out(artifact):
    sig = ScalpGbtSignal(artifact)
    assert sig.compute_second("XYZ", synth_frame(50)) is None
    assert sig.compute_second("XYZ", pd.DataFrame()) is None


def test_invalid_nbbo_gates_out(artifact):
    sig = ScalpGbtSignal(artifact)
    frame = synth_frame()
    frame.iloc[-1, frame.columns.get_loc("bid")] = np.nan
    assert sig.compute_second("XYZ", frame) is None
    frame2 = synth_frame()
    frame2.iloc[-1, frame2.columns.get_loc("bid")] = 3.10  # crossed
    frame2.iloc[-1, frame2.columns.get_loc("ask")] = 3.00
    assert sig.compute_second("XYZ", frame2) is None


def test_five_min_path_is_inert(artifact):
    sig = ScalpGbtSignal(artifact)
    assert sig.compute("XYZ", []) == SignalOutput(0.0, 0.0)


def test_exec_stop_mult_decouples_bracket_from_label(artifact):
    # default artifact (no exec_stop_mult key): bracket stop == label stop
    sig = ScalpGbtSignal(artifact)
    dec = sig.compute_second("XYZ", synth_frame())
    assert dec.bracket_stop_ps == dec.stop_ps
    # rewrite inference.json with the validated timeout-only geometry
    inf = json.loads((artifact / "inference.json").read_text())
    inf["exec_stop_mult"] = 1000.0
    (artifact / "inference.json").write_text(json.dumps(inf))
    sig2 = ScalpGbtSignal(artifact)
    dec2 = sig2.compute_second("XYZ", synth_frame())
    assert dec2.exec_stop_ps == pytest.approx(dec2.stop_ps * 1000.0)
    assert dec2.bracket_stop_ps == dec2.exec_stop_ps
    assert dec2.stop_ps == dec.stop_ps        # sizing's loss leg unchanged


def test_no_win_class_never_signals(artifact, tmp_path):
    # retrain the artifact model without any WIN labels
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    y = pd.Series(rng.choice([-1.0, 0.0], 200))
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=0, max_iter=10).fit(x, y)
    joblib.dump(m, artifact / "model.joblib")
    sig = ScalpGbtSignal(artifact)
    dec = sig.compute_second("XYZ", synth_frame())
    assert dec is not None and dec.p_win == 0.0
