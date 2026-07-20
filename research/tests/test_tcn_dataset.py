"""TCN deep rung, phase A: causality (rewrite-the-future + day-start
padding), scaler train-only correctness, and causal-conv invariance —
synthetic bars only, no corpus, no real training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scalp.bars_features import build_features
from scalp.deep.dataset import (apply_scaler, build_windows, fit_scaler,
                                save_scaler, subsample_negatives,
                                windows_for_all_seconds)
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


def test_build_windows_matches_manual_reference_and_excludes_day_start_pad():
    """build_windows' sample population is label-valid AND servable (see
    _servable_mask below, mirroring TcnProbModel.predict_proba's warmup
    mask) — this reproduces that population by hand and checks build_windows
    matches it exactly, INCLUDING excluding day-start pad seconds even
    though some of them are label-valid on their own."""
    cfg = _cfg()
    # NB: needs n well past 300 -- ret_300s (a feature) is NaN for every
    # row before row 300 regardless of window_s, so a short frame would
    # leave the servable population empty and make this test vacuous.
    bars = make_frame(n=N)
    feats = build_features(bars)
    n = len(feats)
    tgt, stp = barrier_arrays(bars, cfg)
    lab = label_scalps(bars, cfg.barrier(), target_ps_arr=tgt, stop_ps_arr=stp)
    label_valid = lab["label"].notna().to_numpy()
    servable = (
        (np.arange(n) >= WINDOW - 1) & feats.notna().all(axis=1).to_numpy()
    )
    valid = label_valid & servable
    assert valid.sum() > 5   # battery isn't vacuous
    # the bug this guards: label validity alone DOES include day-start
    # seconds (t < WINDOW-1) that a servable-only gate would exclude
    assert label_valid[: WINDOW - 1].any()

    X, y, idx = build_windows(bars, cfg, window_s=WINDOW)
    assert X.shape == (int(valid.sum()), feats.shape[1], WINDOW)

    all_windows = windows_for_all_seconds(feats, WINDOW)
    np.testing.assert_allclose(X, all_windows[valid], equal_nan=True)
    np.testing.assert_array_equal(
        y, (lab["label"].to_numpy()[valid] == 1.0).astype(np.int8))
    np.testing.assert_array_equal(idx, bars.index[valid])

    valid_pos = np.nonzero(valid)[0]
    # train/serve parity: build_windows must never admit a day-start pad
    # second, even though such seconds can be label-valid on their own
    assert valid_pos[0] >= WINDOW - 1
    for i, t in enumerate(valid_pos):
        np.testing.assert_array_equal(
            X[i][:, -1], feats.iloc[t].to_numpy(dtype=np.float32))


# --------------------------------------------------------------------------- #
# (b2) regression: build_windows' sample population must equal exactly what
# TcnProbModel.predict_proba serves a probability for (train/serve parity)
# --------------------------------------------------------------------------- #
def test_build_windows_never_admits_seconds_predict_proba_would_abstain_on():
    """Before the fix, build_windows gated on label validity alone, which
    can include day-start pad seconds (t < window_s-1) and seconds whose
    own feature row is NaN — exactly the seconds TcnProbModel.predict_proba
    (model.py) hard-zeroes at serve time. That trained/early-stopped on a
    population inference never scores. This asserts the two populations
    now agree."""
    cfg = _cfg()
    bars = make_frame(n=N)   # see note above: needs n past 300 (ret_300s)
    feats = build_features(bars)

    X, y, idx = build_windows(bars, cfg, window_s=WINDOW)
    assert len(idx) > 5   # battery isn't vacuous

    day_start_ts = bars.index[: WINDOW - 1]
    assert not idx.isin(day_start_ts).any()

    nan_ts = feats.index[feats.isna().any(axis=1)]
    assert not idx.isin(nan_ts).any()

    # sanity: the bug this guards against is real -- label validity alone
    # (the pre-fix gate) DOES include some of these serve-abstained seconds
    tgt, stp = barrier_arrays(bars, cfg)
    lab = label_scalps(bars, cfg.barrier(), target_ps_arr=tgt, stop_ps_arr=stp)
    label_valid_ts = bars.index[lab["label"].notna().to_numpy()]
    assert (label_valid_ts.isin(day_start_ts).any()
            or label_valid_ts.isin(nan_ts).any())


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
    idx = pd.date_range("2025-01-06 09:30:00", periods=n, freq="1s", tz="UTC")
    feats = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n, n_feat)),
        columns=[f"f{i}" for i in range(n_feat)], index=idx,
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


# --------------------------------------------------------------------------- #
# regression: predict_proba must reject a non-contiguous / multi-day index
# --------------------------------------------------------------------------- #
def test_tcn_prob_model_predict_proba_rejects_multi_day_concatenation():
    """windows_for_all_seconds builds causal windows over ROW POSITION and
    the warmup mask only zeroes the first window_s-1 ROWS of whatever frame
    it's given. Fed a multi-day/multi-symbol concatenation (the shape
    scalp.walkforward.build_dataset's x_test has), it would silently splice
    rows from OTHER stock-days into a window instead of erroring. This
    reproduces that shape (two single-day blocks with a multi-month gap
    between them, same as concatenating two different stock-days'
    valid-row frames) and checks predict_proba now fails loudly instead."""
    torch = pytest.importorskip("torch")
    from scalp.deep.model import ScalpTCN, TcnProbModel

    n_feat, window_s = 4, 10
    scaler = {
        "feature_names": [f"f{i}" for i in range(n_feat)],
        "median": [0.0] * n_feat, "iqr": [1.0] * n_feat,
        "window_s": window_s, "n_train_files": 1,
    }
    model = ScalpTCN(n_features=n_feat, window_s=window_s, channels=4,
                     blocks=1, dropout=0.0)
    wrapped = TcnProbModel(model, scaler, window_s=window_s,
                           device=torch.device("cpu"))

    day1 = pd.date_range("2024-01-02 09:30:00", periods=20, freq="1s", tz="UTC")
    day2 = pd.date_range("2024-06-03 09:30:00", periods=20, freq="1s", tz="UTC")
    idx = day1.append(day2)
    feats = pd.DataFrame(
        np.random.default_rng(0).normal(size=(len(idx), n_feat)),
        columns=scaler["feature_names"], index=idx,
    )

    with pytest.raises(ValueError, match="contiguous"):
        wrapped.predict_proba(feats)

    # sanity: the guard isn't rejecting everything -- a genuine single-day
    # contiguous slice is accepted
    proba = wrapped.predict_proba(feats.loc[day1])
    assert proba.shape == (len(day1), 2)


# --------------------------------------------------------------------------- #
# phase B: train-time augmentation (JitterDataset) — TRAIN SET ONLY
# --------------------------------------------------------------------------- #
def test_jitter_dataset_augments_train_batches_and_changes_between_epochs():
    """JitterDataset is the wrapper train_tcn.py puts ONLY around the train
    split (never val, see train_loop.train and its own test below). This
    checks the class itself:

    - noise is actually added per-sample in __getitem__ (no large tensor)
    - same (seed, epoch, index) -> identical noise (determinism contract)
    - different epoch -> different noise for same sample
    - sigma<=0 disables augmentation entirely (exact passthrough)
    - no attribute holds a tensor with the dataset's full shape (OOM guard)
    """
    torch = pytest.importorskip("torch")
    from scalp.deep.dataset import JitterDataset

    n, n_feat, window_s = 20, 3, 5
    rng = np.random.default_rng(1)
    X = torch.from_numpy(rng.normal(size=(n, n_feat, window_s)).astype(np.float32))
    y = torch.from_numpy((rng.random(n) < 0.5).astype(np.float32))

    ds = JitterDataset(X, y, sigma=0.5, seed=42)
    ds.set_epoch(0)
    batch0 = torch.stack([ds[i][0] for i in range(n)])
    assert not torch.allclose(batch0, X)          # noise was actually added

    # determinism: same (seed, epoch, index) -> identical noise
    ds2 = JitterDataset(X, y, sigma=0.5, seed=42)
    ds2.set_epoch(0)
    batch0_again = torch.stack([ds2[i][0] for i in range(n)])
    torch.testing.assert_close(batch0, batch0_again)

    # per-sample determinism: each individual index is reproducible
    for i in (0, 1, n // 2, n - 1):
        x_i_a = ds[i][0]
        x_i_b = ds2[i][0]
        torch.testing.assert_close(x_i_a, x_i_b,
                                   msg=f"sample {i} noise not deterministic")

    # different epoch -> different noise for the same sample
    ds.set_epoch(1)
    batch1 = torch.stack([ds[i][0] for i in range(n)])
    assert not torch.allclose(batch0, batch1)
    # verify at least one individual sample differs (not just aggregate)
    assert not torch.allclose(ds[0][0], ds2[0][0])  # epoch 1 vs epoch 0

    # sigma<=0 disables augmentation entirely -> exact passthrough
    ds_off = JitterDataset(X, y, sigma=0.0, seed=42)
    batch_off = torch.stack([ds_off[i][0] for i in range(n)])
    torch.testing.assert_close(batch_off, X)

    # OOM guard: no attribute OTHER than 'X' (the data tensor itself) should
    # hold a tensor with the full dataset shape (n, n_feat, window_s).
    # The old implementation stored self._noise with exactly this shape —
    # the new per-sample implementation must not materialise such a tensor.
    ds_check = JitterDataset(X, y, sigma=0.5, seed=42)
    ds_check.set_epoch(0)
    full_shape = tuple(X.shape)  # (n, n_feat, window_s)
    for attr_name, attr_val in vars(ds_check).items():
        if attr_name in ("X",):
            continue  # the data store itself is exempt
        if isinstance(attr_val, torch.Tensor):
            assert tuple(attr_val.shape) != full_shape, (
                f"attribute '{attr_name}' holds a tensor with the full dataset "
                f"shape {full_shape} — this would OOM on large datasets"
            )

    # labels are never touched by jitter
    ds.set_epoch(0)
    for i in range(n):
        torch.testing.assert_close(ds[i][1], y[i])


def test_train_loop_val_batches_bit_identical_with_train_jitter_on():
    """scalp.deep.train_loop.train calls set_epoch on train_loader.dataset
    only -- val_loader.dataset must never be wrapped in JitterDataset (this
    reproduces train_tcn.py's exact wiring: train wrapped, val a plain
    TensorDataset) and its batches must come out bit-identical to the raw
    val tensor, epoch after epoch, even with jitter cranked up on train."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from scalp.deep.dataset import JitterDataset
    from scalp.deep.model import ScalpTCN
    from scalp.deep.train_loop import train

    n_feat, window_s = 3, 6
    n_tr, n_val = 32, 10
    rng = np.random.default_rng(2)
    x_tr = torch.from_numpy(rng.normal(size=(n_tr, n_feat, window_s)).astype(np.float32))
    y_tr = torch.from_numpy((rng.random(n_tr) < 0.5).astype(np.float32))
    x_val = torch.from_numpy(rng.normal(size=(n_val, n_feat, window_s)).astype(np.float32))
    y_val = torch.from_numpy((rng.random(n_val) < 0.5).astype(np.float32))
    x_val_orig = x_val.clone()

    train_loader = DataLoader(JitterDataset(x_tr, y_tr, sigma=1.0, seed=7),
                              batch_size=8, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8,
                            shuffle=False)

    torch.manual_seed(0)
    model = ScalpTCN(n_features=n_feat, window_s=window_s, channels=4,
                     blocks=1, dropout=0.0)
    train(model, (train_loader, val_loader), epochs=3, lr=1e-3,
         pos_weight=1.0, device=torch.device("cpu"), patience=10,
         verbose=False)

    val_batches = torch.cat([xb for xb, _ in val_loader])
    torch.testing.assert_close(val_batches, x_val_orig)


# --------------------------------------------------------------------------- #
# phase B: sim_eval.py --tcn-run-dir adapter (TcnProbModel via day_entries)
# --------------------------------------------------------------------------- #
def test_sim_eval_tcn_run_dir_loads_and_scores_synthetic_day(tmp_path):
    """TcnProbModel.load(run_dir) must reconstruct a model.pt/scaler.json/
    config.json triple laid out exactly like train_tcn.py's artifacts, and
    the loaded model must be a drop-in for scripts.sim_eval.day_entries'
    `model` argument -- same predict_proba interface the GBT rung uses."""
    torch = pytest.importorskip("torch")
    import json

    from scalp.deep.model import ScalpTCN, TcnProbModel
    from scripts.sim_eval import day_entries

    bars = make_frame(n=N)
    day_file = tmp_path / "SYM_2025-01-06.parquet"
    bars.to_parquet(day_file)

    cfg = _cfg()
    scaler = fit_scaler([day_file], cfg, window_s=WINDOW)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_scaler(scaler, run_dir / "scaler.json")
    n_feat = len(scaler["feature_names"])
    tiny = ScalpTCN(n_features=n_feat, window_s=WINDOW, channels=4, blocks=1,
                    dropout=0.0)
    torch.save(tiny.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "window": WINDOW, "channels": 4, "blocks": 1, "dropout": 0.0,
    }))

    loaded = TcnProbModel.load(run_dir, device=torch.device("cpu"))
    assert list(loaded.classes_) == [0.0, 1.0]

    entries, lab = day_entries(bars, loaded, cfg, threshold=0.0, qty=100)
    assert not entries.empty                       # battery isn't vacuous
    assert set(entries.columns) == {"qty", "target_px", "stop_px", "deadline"}
    assert lab.index.equals(entries.index)


# --------------------------------------------------------------------------- #
# val subsample determinism (OOM-fix: val windows are now subsampled too)
# --------------------------------------------------------------------------- #
def _load_split_inline(
    files, cfg, window_s, scaler, neg_frac,
):
    """Local replica of train_tcn._load_split for testing — same logic,
    no script-level imports needed."""
    xs, ys = [], []
    for path in files:
        bars = pd.read_parquet(path)
        X, y, idx = build_windows(bars, cfg, window_s)
        del bars
        if len(y) == 0:
            continue
        if neg_frac is not None:
            file_key = path.stem
            X, y, idx = subsample_negatives(X, y, idx, cfg, file_key, neg_frac)
            if len(y) == 0:
                continue
        xs.append(apply_scaler(X, scaler))
        del X
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def test_val_subsample_is_deterministic_and_never_epoch_dependent(tmp_path):
    """Two calls to _load_split with the same val files and val_neg_frac must
    return bit-identical (X, y) arrays — same windows, same order. The seed
    must be derived purely from cfg.seed + per-file hash, never from anything
    epoch-dependent (that would break val AP comparability across runs)."""
    val_files = _write_days(tmp_path, n_files=4, n=N, seed0=200)
    cfg = _cfg(seed=13)

    scaler = fit_scaler(val_files, cfg, window_s=WINDOW)

    x1, y1 = _load_split_inline(val_files, cfg, WINDOW, scaler, neg_frac=0.15)
    x2, y2 = _load_split_inline(val_files, cfg, WINDOW, scaler, neg_frac=0.15)

    np.testing.assert_array_equal(y1, y2, err_msg="val labels differ between calls")
    np.testing.assert_array_equal(x1, x2, err_msg="val windows differ between calls")

    # confirm negatives were actually subsampled (not a no-op)
    total_samples = sum(
        len(build_windows(pd.read_parquet(f), cfg, WINDOW)[1]) for f in val_files
    )
    assert len(y1) < total_samples, (
        "negatives were not subsampled (val subset equals full val set)"
    )


# --------------------------------------------------------------------------- #
# TcnProbModel.load() map_location smoke: artifact saved on any device loads
# cleanly on CPU (and auto-device selection moves the model)
# --------------------------------------------------------------------------- #
def test_tcn_prob_model_load_map_location_smoke(tmp_path):
    """Save a tiny ScalpTCN state dict, then load it with TcnProbModel.load()
    which must use map_location='cpu' internally so artifacts trained on any
    device (cuda, mps) load on any other device without error."""
    torch = pytest.importorskip("torch")
    import json

    from scalp.deep.model import ScalpTCN, TcnProbModel

    n_feat, window_s = 4, 8
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    scaler = {
        "feature_names": [f"f{i}" for i in range(n_feat)],
        "median": [0.0] * n_feat,
        "iqr": [1.0] * n_feat,
        "window_s": window_s,
        "n_train_files": 1,
    }
    (run_dir / "scaler.json").write_text(json.dumps(scaler))
    (run_dir / "config.json").write_text(json.dumps(
        {"window": window_s, "channels": 4, "blocks": 1, "dropout": 0.0}
    ))

    tiny = ScalpTCN(n_features=n_feat, window_s=window_s, channels=4,
                    blocks=1, dropout=0.0)
    # Save with explicit map to CPU to simulate "trained anywhere" artifact.
    state = {k: v.cpu() for k, v in tiny.state_dict().items()}
    torch.save(state, run_dir / "model.pt")

    # load() without a device argument -> auto picks available device
    loaded = TcnProbModel.load(run_dir)
    assert loaded.model is not None
    # verify we can score — shape check is sufficient
    idx = pd.date_range("2025-01-06 09:30:00", periods=window_s + 5,
                        freq="1s", tz="UTC")
    feats = pd.DataFrame(
        np.random.default_rng(3).normal(size=(len(idx), n_feat)),
        columns=scaler["feature_names"], index=idx,
    )
    proba = loaded.predict_proba(feats)
    assert proba.shape == (len(idx), 2)

    # load() with explicit cpu device -> works regardless of what pick_device()
    # would choose (no cuda/mps on this machine, but this tests the kwarg path)
    loaded_cpu = TcnProbModel.load(run_dir, device=torch.device("cpu"))
    assert str(loaded_cpu.device) == "cpu"
    proba_cpu = loaded_cpu.predict_proba(feats)
    assert proba_cpu.shape == (len(idx), 2)


# --------------------------------------------------------------------------- #
# --device dml: clean ImportError when torch_directml is not installed
# --------------------------------------------------------------------------- #
def test_resolve_device_dml_raises_clean_error_when_torch_directml_missing(
    monkeypatch,
):
    """On a machine without torch_directml (e.g. a Mac or Linux CI runner),
    passing --device dml must raise an ImportError whose message tells the
    user exactly how to fix it ('pip install torch-directml'), rather than
    crashing with an opaque AttributeError or ModuleNotFoundError later."""
    import builtins

    real_import = builtins.__import__

    def _block_directml(name, *args, **kwargs):
        if name == "torch_directml":
            raise ImportError("No module named 'torch_directml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_directml)

    # import after patching so the lazy import inside _resolve_device is live
    import importlib
    import sys

    # force a fresh import of the script module (it may already be cached)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import scripts.train_tcn as _train_tcn  # noqa: PLC0415
    importlib.reload(_train_tcn)

    with pytest.raises(ImportError, match="pip install torch-directml"):
        _train_tcn._resolve_device("dml")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
