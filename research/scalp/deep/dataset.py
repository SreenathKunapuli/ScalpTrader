"""Windowed dataset for the TCN deep rung.

Every sample is a strictly-causal window of the SAME per-second features
the GBT rung and the live engine use (scalp.bars_features.build_features —
this module never reimplements a feature). A sample exists at second t iff
the triple-barrier label is warm there (reusing scalp.walkforward and
scalp.triple_barrier exactly as scalp.walkforward.build_dataset does) AND
t is in the population served at inference time — t >= window_s-1 and
feats.iloc[t] is NaN-free. That second condition matters: label validity
alone would admit day-start pad windows and NaN feature rows that
TcnProbModel.predict_proba (model.py) hard-zeroes at serve time, so
train/val would be fit and early-stopped on a population inference never
scores. build_windows enforces both so the sample population matches
serving exactly.

Causality: the window at t is features rows [t-window_s+1 .. t] — PAST rows
only. windows_for_all_seconds (serving) left-pads days that start with
fewer than window_s-1 seconds of history with NaN (never wrapped, never
filled from the future); build_windows (training) excludes those padded
seconds outright rather than training on them, per the parity note above.

Nothing in here normalizes: build_windows returns raw feature values so the
scaler fit (TRAIN DAYS ONLY) can never leak through a pre-normalized window.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..bars_features import build_features
from ..triple_barrier import label_scalps
from ..walkforward import TrainConfig, barrier_arrays

__all__ = [
    "build_windows",
    "windows_for_all_seconds",
    "fit_scaler",
    "apply_scaler",
    "save_scaler",
    "load_scaler",
    "subsample_negatives",
    "JitterDataset",
]


def _sliding_feature_windows(feat_arr: np.ndarray, window_s: int) -> np.ndarray:
    """[n, n_feat] -> [n, n_feat, window_s], window i = rows (i-window_s+1..i)
    of `feat_arr`, left-padded with NaN where that range runs before row 0.

    Pure numpy stride-trick view + one fancy-index copy at the caller's
    selection step — never materializes the full [n, n_feat, window_s]
    array unless the caller actually indexes every row.
    """
    n, n_feat = feat_arr.shape
    pad = np.full((window_s - 1, n_feat), np.nan, dtype=feat_arr.dtype)
    padded = np.concatenate([pad, feat_arr], axis=0)          # [n+window_s-1, n_feat]
    # sliding_window_view(axis=0) -> shape (n, n_feat, window_s); window i's
    # last column is padded[i+window_s-1] == feat_arr[i], i.e. row t=i itself.
    return np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=window_s, axis=0
    )


def windows_for_all_seconds(feats: pd.DataFrame, window_s: int = 240) -> np.ndarray:
    """Causal window for EVERY second in `feats` (no label-warmth filter) —
    what serving (TcnProbModel.predict_proba) needs, since it must emit a
    probability at every second, not just the ones that were trainable
    samples. Returns float32 [n, n_feat, window_s]."""
    if feats.empty:
        return np.empty((0, feats.shape[1], window_s), dtype=np.float32)
    feat_arr = feats.to_numpy(dtype=np.float32)
    return _sliding_feature_windows(feat_arr, window_s)


def build_windows(
    bars: pd.DataFrame, cfg: TrainConfig, window_s: int = 240,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, y, idx) for one stock-day.

    X: float32 [n_samples, n_features, window_s] — channels-first (feature,
       time), matching ScalpTCN's expected input layout directly (no
       permute needed downstream).
    y: int8 [n_samples] — 1 iff the triple-barrier label at that second is
       WIN (label == 1.0), else 0 (covers LOSS and TIMEOUT).
    idx: DatetimeIndex of the sample seconds (== bars.index at the warm
       rows), aligned 1:1 with X/y.

    A sample exists at second t iff the triple-barrier label is warm there
    (barrier_arrays + label_scalps, identical to walkforward.build_dataset's
    validity gate) AND t is in the population TcnProbModel.predict_proba
    actually scores at serve time: t >= window_s-1 (not a day-start pad
    window) and feats.iloc[t] has no NaN (not an intrinsically-undefined
    second). Label validity alone is NOT enough here — the NBBO/barrier
    gate it checks is independent of the two serve-time exclusions above,
    so without intersecting them the train/val sample population would
    include seconds (notably the first window_s-1 of every day) serving
    never emits a probability for. No normalization is applied here.
    """
    feats = build_features(bars)
    n = len(feats)
    n_feat = feats.shape[1]
    empty = (
        np.empty((0, n_feat, window_s), dtype=np.float32),
        np.empty((0,), dtype=np.int8),
        bars.index[:0],
    )
    if n == 0:
        return empty

    tgt, stp = barrier_arrays(bars, cfg)
    lab = label_scalps(bars, cfg.barrier(), target_ps_arr=tgt, stop_ps_arr=stp)
    label_valid = lab["label"].notna().to_numpy()
    # Same population TcnProbModel.predict_proba's warmup mask keeps
    # (model.py): not a day-start pad window, and this second's own
    # feature row is NaN-free.
    servable = (
        (np.arange(n) >= window_s - 1) & feats.notna().all(axis=1).to_numpy()
    )
    valid = label_valid & servable
    if not valid.any():
        return empty

    windows = windows_for_all_seconds(feats, window_s)       # [n, n_feat, window_s]
    valid_pos = np.nonzero(valid)[0]
    X = windows[valid_pos].copy()                              # materialize selection only
    label_vals = lab["label"].to_numpy()[valid_pos]
    y = (label_vals == 1.0).astype(np.int8)
    idx = bars.index[valid_pos]
    return X, y, idx


# --------------------------------------------------------------------------- #
# Scaler: median/IQR from TRAIN DAYS ONLY, streamed file-by-file
# --------------------------------------------------------------------------- #
def fit_scaler(
    train_files: list[Path], cfg: TrainConfig, window_s: int = 240,
) -> dict:
    """Per-feature median/IQR from TRAIN DAYS ONLY.

    Streams over `train_files` one at a time (only build_features' small
    [n_seconds, n_feat] per-second frame is ever held per file — never the
    ~window_s-times-larger windowed array build_windows would produce, and
    never more than one file's bars at once). The small per-second frames
    are concatenated once at the end to compute exact median/IQR; this is
    the "no giant concat" the caller cares about — the thing that would
    actually blow up memory (a corpus-wide windowed X) is never built here.
    """
    chunks: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for path in sorted(train_files):
        bars = pd.read_parquet(path)
        feats = build_features(bars)
        if feature_names is None:
            feature_names = list(feats.columns)
        elif list(feats.columns) != feature_names:
            raise ValueError(
                f"fit_scaler: feature column mismatch in {path} "
                f"(expected {feature_names}, got {list(feats.columns)})"
            )
        chunks.append(feats.to_numpy(dtype=np.float64))
    if not chunks:
        raise ValueError("fit_scaler: no train files given")

    arr = np.concatenate(chunks, axis=0)
    med = np.nanmedian(arr, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    iqr = q75 - q25
    # degenerate/near-constant features would divide by ~0 -> guard with 1.0
    iqr_safe = np.where(iqr > 1e-12, iqr, 1.0)
    return {
        "feature_names": feature_names,
        "median": med.tolist(),
        "iqr": iqr_safe.tolist(),
        "window_s": window_s,
        "n_train_files": len(train_files),
    }


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    """(X - median) / IQR per feature (axis=1, the channel axis), then
    nan_to_num(0). X: float32 [n, n_feat, window_s] (or [n_feat, window_s]
    for a single sample)."""
    med = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    if X.ndim == 3:
        med = med.reshape(1, -1, 1)
        iqr = iqr.reshape(1, -1, 1)
    elif X.ndim == 2:
        med = med.reshape(-1, 1)
        iqr = iqr.reshape(-1, 1)
    else:
        raise ValueError(f"apply_scaler: expected X.ndim in (2, 3), got {X.ndim}")
    out = (X.astype(np.float32) - med) / iqr
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def save_scaler(scaler: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(scaler, indent=2))


def load_scaler(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Negative subsampling: keep all positives, sample negatives at neg_frac
# --------------------------------------------------------------------------- #
def _stable_file_seed(cfg_seed: int, file_key: str) -> int:
    """Deterministic seed from cfg.seed + a stable hash of `file_key`
    (hashlib, NOT builtin hash() — Python's string hash is randomized
    per-process unless PYTHONHASHSEED is pinned, which would silently
    break reproducibility across runs)."""
    digest = hashlib.sha256(file_key.encode("utf-8")).hexdigest()
    file_hash = int(digest[:8], 16)
    return (int(cfg_seed) + file_hash) % (2**32 - 1)


def subsample_negatives(
    X: np.ndarray, y: np.ndarray, idx: pd.DatetimeIndex,
    cfg: TrainConfig, file_key: str, neg_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Keep every positive (y==1); keep each negative independently with
    probability `neg_frac`. RNG is a numpy Generator seeded from
    cfg.seed + a stable per-file hash, so re-running the same file with the
    same cfg.seed reproduces the identical subsample (different files never
    collide onto the same draw sequence)."""
    y = np.asarray(y)
    pos_mask = y == 1
    neg_mask = ~pos_mask
    rng = np.random.default_rng(_stable_file_seed(cfg.seed, file_key))
    neg_keep = rng.random(int(neg_mask.sum())) < neg_frac
    keep = pos_mask.copy()
    keep[neg_mask] = neg_keep
    return X[keep], y[keep], idx[keep]


# --------------------------------------------------------------------------- #
# Train-time augmentation: additive Gaussian jitter, re-drawn every epoch
# --------------------------------------------------------------------------- #
class JitterDataset(torch.utils.data.Dataset):
    """TRAIN-SET-ONLY augmentation wrapper: adds fresh N(0, sigma) Gaussian
    noise to the already-SCALED window tensor `X`, redrawn once per epoch.

    Val loaders must NEVER be wrapped in this — scalp.deep.train_loop.train
    calls set_epoch(epoch) on train_loader.dataset only (never
    val_loader.dataset), and train_tcn.py only ever wraps the train split.
    `sigma <= 0` disables augmentation entirely: __getitem__ then returns X
    unmodified and no RNG is ever drawn, so wrapping unconditionally at the
    default --jitter-sigma 0.0 is a no-op byte-for-byte.

    Reproducibility: set_epoch(epoch) reseeds a fresh torch.Generator from
    `seed + epoch` and materializes the noise tensor for that epoch in one
    shot (same shape as X) — the same (seed, epoch) pair always redraws
    identical noise, and every epoch draws a fresh one.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, sigma: float,
                seed: int):
        self.X = X
        self.y = y
        self.sigma = float(sigma)
        self.seed = int(seed)
        self._noise: torch.Tensor | None = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.sigma <= 0.0:
            self._noise = None
            return
        gen = torch.Generator().manual_seed(self.seed + int(epoch))
        self._noise = torch.randn(self.X.shape, generator=gen) * self.sigma

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[i]
        if self._noise is not None:
            x = x + self._noise[i]
        return x, self.y[i]
