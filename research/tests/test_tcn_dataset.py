"""TCN deep rung, phase A: causality (rewrite-the-future + day-start
padding), scaler train-only correctness, and causal-conv invariance —
synthetic bars only, no corpus, no real training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scalp.bars_features import build_features
from scalp.deep.dataset import (apply_scaler, build_windows, fit_scaler,
                                subsample_negatives, windows_for_all_seconds)
from scalp.triple_barrier import label_scalps
from scalp.walkforward import TrainConfig, barrier_arrays

N = 600
CUT = 400
TIMEOUT = 60
WINDOW = 64


def make_frame(n: int = N, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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


def rewrite_future(frame: pd.DataFrame, cut: int) -> pd.DataFrame:
    """A violently different future: prices x1.5, volume x3, busier tape."""
    f = frame.copy()
    price_cols = ["open", "high", "low", "close", "vwap", "bid", "ask"]
    f.iloc[cut:, [f.columns.get_loc(c) for c in price_cols]] *= 1.5
    f.iloc[cut:, f.columns.get_loc("volume")] *= 3
    f.iloc[cut:, f.columns.get_loc("n_trades")] += 5
    return f


def _cfg(**kw) -> TrainConfig:
    base = dict(barrier_mode="fixed", target_ps=0.05, stop_ps=0.04,
               timeout_s=TIMEOUT)
    base.update(kw)
    return TrainConfig(**base)


# --------------------------------------------------------------------------- #
# (a) rewrite-the-future causality
# --------------------------------------------------------------------------- #
def test_build_windows_is_causal():
    cfg = _cfg()
    a = make_frame()
    # A label at t reads bars up to t+timeout_s, so the perturbation must
    # start strictly after CUT+TIMEOUT for the literal "idx<=CUT is
    # bit-identical" guarantee to hold (same convention as
    # test_no_lookahead.py's "safe" boundary).
    b = rewrite_future(make_frame(), CUT + TIMEOUT + 1)

    Xa, ya, idxa = build_windows(a, cfg, window_s=WINDOW)
    Xb, yb, idxb = build_windows(b, cfg, window_s=WINDOW)

    cut_ts = a.index[CUT]
    keep_a = idxa <= cut_ts
    keep_b = idxb <= cut_ts
    assert keep_a.sum() > 5   # battery isn't vacuous
    np.testing.assert_array_equal(idxa[keep_a], idxb[keep_b])
    np.testing.assert_array_equal(ya[keep_a], yb[keep_b])
    np.testing.assert_allclose(Xa[keep_a], Xb[keep_b], equal_nan=True)


# --------------------------------------------------------------------------- #
# (b) day-start padding / no future rows
# --------------------------------------------------------------------------- #
def test_windows_for_all_seconds_pads_and_never_reads_future_rows():
    n, n_feat, window_s = 10, 3, 6
    feat_arr = np.arange(n * n_feat, dtype=np.float64).reshape(n, n_feat)
    feats = pd.DataFrame(feat_arr, columns=[f"f{i}" for i in range(n_feat)])

    windows = windows_for_all_seconds(feats, window_s)
    assert windows.shape == (n, n_feat, window_s)

    for t in range(n):
        # every window's LAST column is exactly row t — never a future row
        np.testing.assert_array_equal(windows[t][:, -1], feat_arr[t])
        n_pad = max(0, window_s - 1 - t)
        if n_pad:
            assert np.isnan(windows[t][:, :n_pad]).all()
        real = windows[t][:, n_pad:]
        expected = feat_arr[max(0, t - window_s + 1): t + 1].T
        np.testing.assert_array_equal(real, expected)


def test_build_windows_matches_manual_reference_and_pads_at_day_start():
    cfg = _cfg()
    bars = make_frame(n=300)
    feats = build_features(bars)
    tgt, stp = barrier_arrays(bars, cfg)
    lab = label_scalps(bars, cfg.barrier(), target_ps_arr=tgt, stop_ps_arr=stp)
    valid = lab["label"].notna().to_numpy()
    assert valid.sum() > 5   # battery isn't vacuous

    X, y, idx = build_windows(bars, cfg, window_s=WINDOW)
    assert X.shape == (int(valid.sum()), feats.shape[1], WINDOW)

    all_windows = windows_for_all_seconds(feats, WINDOW)
    np.testing.assert_allclose(X, all_windows[valid], equal_nan=True)
    np.testing.assert_array_equal(
        y, (lab["label"].to_numpy()[valid] == 1.0).astype(np.int8))
    np.testing.assert_array_equal(idx, bars.index[valid])

    valid_pos = np.nonzero(valid)[0]
    assert valid_pos[0] < WINDOW - 1   # exercises the day-start pad path
    for i, t in enumerate(valid_pos):
        n_pad = max(0, WINDOW - 1 - t)
        if n_pad:
            assert np.isnan(X[i][:, :n_pad]).all()
        np.testing.assert_array_equal(
            X[i][:, -1], feats.iloc[t].to_numpy(dtype=np.float32))


# --------------------------------------------------------------------------- #
# (c) ScalpTCN causal-conv invariance
# --------------------------------------------------------------------------- #
def test_scalp_tcn_forward_sequence_is_causal():
    torch = pytest.importorskip("torch")
    from scalp.deep.model import ScalpTCN

    torch.manual_seed(0)
    model = ScalpTCN(n_features=5, window_s=32, channels=8, blocks=2,
                     dropout=0.0)
    model.eval()
    x = torch.randn(3, 5, 32)
    t = 12
    x2 = x.clone()
    x2[:, :, t + 1:] = torch.randn_like(x2[:, :, t + 1:])

    with torch.no_grad():
        y1 = model.forward_sequence(x)
        y2 = model.forward_sequence(x2)

    torch.testing.assert_close(y1[:, :, : t + 1], y2[:, :, : t + 1])
    # sanity: the perturbed region actually does change something,
    # otherwise this test would pass vacuously (e.g. a broken all-zero net)
    assert not torch.allclose(y1[:, :, t + 1:], y2[:, :, t + 1:])


def test_scalp_tcn_forward_returns_single_logit_at_last_timestep():
    torch = pytest.importorskip("torch")
    from scalp.deep.model import ScalpTCN

    model = ScalpTCN(n_features=4, window_s=16, channels=8, blocks=2,
                     dropout=0.0)
    model.eval()
    x = torch.randn(5, 4, 16)
    with torch.no_grad():
        logit = model(x)
        seq = model.forward_sequence(x)
        expected = model.head(seq[:, :, -1]).squeeze(-1)
    assert logit.shape == (5,)
    torch.testing.assert_close(logit, expected)


# --------------------------------------------------------------------------- #
# (d) scaler fit ignores non-train files by construction
# --------------------------------------------------------------------------- #
def _write_days(tmp_path, n_files=3, n=320, seed0=100):
    files = []
    for i in range(n_files):
        f = tmp_path / f"SYM{i}_2024-06-{i + 1:02d}.parquet"
        make_frame(n=n, seed=seed0 + i).to_parquet(f)
        files.append(f)
    return files


def test_fit_scaler_train_only_matches_hand_computed(tmp_path):
    train_files = _write_days(tmp_path, n_files=3)
    extra_file = tmp_path / "EXTRA_2099-01-01.parquet"
    make_frame(n=320, seed=999).to_parquet(extra_file)   # NOT in train_files

    cfg = _cfg()
    scaler = fit_scaler(train_files, cfg, window_s=WINDOW)

    all_feats = pd.concat(
        [build_features(pd.read_parquet(f)) for f in train_files])
    arr = all_feats.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)

    assert scaler["feature_names"] == list(all_feats.columns)
    np.testing.assert_allclose(scaler["median"], med)
    np.testing.assert_allclose(scaler["iqr"], iqr)

    # applied transform matches the same hand-computed stats
    X, _, _ = build_windows(pd.read_parquet(train_files[0]), cfg, WINDOW)
    scaled = apply_scaler(X, scaler)
    manual = np.nan_to_num(
        (X - med.reshape(1, -1, 1)) / iqr.reshape(1, -1, 1), nan=0.0)
    np.testing.assert_allclose(scaled, manual, atol=1e-5)

    # the extra file genuinely was never read: including it changes the fit
    scaler_with_extra = fit_scaler(train_files + [extra_file], cfg, WINDOW)
    assert scaler_with_extra["median"] != scaler["median"]


# --------------------------------------------------------------------------- #
# negative subsampling (spec item 3) — reproducibility + keep-all-positives
# --------------------------------------------------------------------------- #
def test_subsample_negatives_keeps_all_positives_and_is_reproducible():
    n = 2000
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.05).astype(np.int8)
    X = rng.normal(size=(n, 3, 4)).astype(np.float32)
    idx = pd.date_range("2024-01-01", periods=n, freq="1s", tz="UTC")
    cfg = TrainConfig(seed=42)

    Xs, ys, idxs = subsample_negatives(X, y, idx, cfg, "SYM_2024-01-01",
                                       neg_frac=0.15)
    n_pos = int(y.sum())
    assert int(ys.sum()) == n_pos           # every positive survives
    assert len(ys) < n                       # negatives were actually thinned
    frac_kept_neg = (len(ys) - n_pos) / (n - n_pos)
    assert 0.08 < frac_kept_neg < 0.22       # roughly neg_frac (loose bound)

    # reproducible: same cfg.seed + same file_key -> identical draw
    Xs2, ys2, idxs2 = subsample_negatives(X, y, idx, cfg, "SYM_2024-01-01",
                                          neg_frac=0.15)
    np.testing.assert_array_equal(ys, ys2)
    np.testing.assert_array_equal(idxs, idxs2)
    np.testing.assert_allclose(Xs, Xs2)

    # a different file_key draws a (near-certainly) different subsample
    _, _, idxs3 = subsample_negatives(X, y, idx, cfg, "SYM_2024-01-02",
                                      neg_frac=0.15)
    assert list(idxs) != list(idxs3)


# --------------------------------------------------------------------------- #
# TcnProbModel: serving interface sim_eval.day_entries expects
# --------------------------------------------------------------------------- #
def test_tcn_prob_model_predict_proba_shape_and_warmup_zero():
    torch = pytest.importorskip("torch")
    from scalp.deep.model import ScalpTCN, TcnProbModel

    n, n_feat, window_s = 50, 4, 10
    feats = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n, n_feat)),
        columns=[f"f{i}" for i in range(n_feat)],
    )
    feats.iloc[20] = np.nan   # an intrinsically undefined (NaN) second

    scaler = {
        "feature_names": list(feats.columns),
        "median": [0.0] * n_feat, "iqr": [1.0] * n_feat,
        "window_s": window_s, "n_train_files": 1,
    }
    model = ScalpTCN(n_features=n_feat, window_s=window_s, channels=4,
                     blocks=1, dropout=0.0)
    wrapped = TcnProbModel(model, scaler, window_s=window_s,
                           device=torch.device("cpu"))

    proba = wrapped.predict_proba(feats)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert list(wrapped.classes_) == [0.0, 1.0]
    # day-start warmup (t < window_s-1) and the injected NaN row -> p=0
    assert (proba[: window_s - 1, 1] == 0.0).all()
    assert proba[20, 1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
