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
from typing import Optional

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
    "MemmapWindows",
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
# Disk-backed window store: avoids accumulating all windows in RAM
# --------------------------------------------------------------------------- #
class MemmapWindows:
    """On-disk growable store for (X, y) window pairs.

    Files layout under `path` dir:
      X.f32.mmap  — float32 memmap, shape [n, n_features, window_s]
      y.i8.mmap   — int8 memmap, shape [n]
      meta.json   — {"n": int, "n_features": int, "window_s": int}

    Usage
    -----
    store = MemmapWindows.create(path, n_features=F, window_s=W)
    for X_chunk, y_chunk in day_chunks:
        store.append(X_chunk, y_chunk)    # X_chunk: float32 [k, F, W]
    ds = store.finalize()                 # returns a Dataset-like object

    The finalized object exposes __len__ and __getitem__(i) -> (Tensor, int8).
    Each __getitem__ copies the mmap row into a fresh numpy array before
    wrapping in a tensor so torch/DataLoader workers never hold an open
    file-descriptor reference across the full dataset.

    Deterministic: row i always returns the same data regardless of worker
    count or DataLoader shuffle — the mmap is read-only after finalize().
    """

    _X_FILE = "X.f32.mmap"
    _Y_FILE = "y.i8.mmap"
    _META_FILE = "meta.json"
    # Initial allocation size; grown by doubling when needed.
    _INIT_CAP = 4096

    def __init__(self, path: Path, n_features: int, window_s: int,
                 _cap: int = 0, _n: int = 0):
        self._path = Path(path)
        self._n_features = int(n_features)
        self._window_s = int(window_s)
        self._cap = int(_cap)
        self._n = int(_n)
        self._X: Optional[np.memmap] = None
        self._y: Optional[np.memmap] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, path: Path, n_features: int, window_s: int) -> "MemmapWindows":
        """Create a new empty store at `path` (directory must not already
        contain store files; `path` is created if it does not exist)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cap = cls._INIT_CAP
        n_features = int(n_features)
        window_s = int(window_s)
        # Open the backing files at initial capacity; mode='w+' creates/truncates.
        np.memmap(path / cls._X_FILE, dtype=np.float32, mode="w+",
                  shape=(cap, n_features, window_s))
        np.memmap(path / cls._Y_FILE, dtype=np.int8, mode="w+", shape=(cap,))
        inst = cls(path, n_features, window_s, _cap=cap, _n=0)
        inst._write_meta()
        return inst

    @classmethod
    def open(cls, path: Path) -> "_FinalizedMemmapWindows":
        """Re-open a finalized store for read-only indexing."""
        path = Path(path)
        meta = json.loads((path / cls._META_FILE).read_text())
        n = meta["n"]
        n_features = meta["n_features"]
        window_s = meta["window_s"]
        inst = cls(path, n_features, window_s, _cap=n, _n=n)
        return _FinalizedMemmapWindows(inst)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------
    def _open_mmaps(self, mode: str = "r+") -> tuple[np.memmap, np.memmap]:
        X = np.memmap(self._path / self._X_FILE, dtype=np.float32, mode=mode,
                      shape=(self._cap, self._n_features, self._window_s))
        y = np.memmap(self._path / self._Y_FILE, dtype=np.int8, mode=mode,
                      shape=(self._cap,))
        return X, y

    def _grow(self, needed: int) -> None:
        """Double capacity until `needed` additional rows fit."""
        new_cap = self._cap
        while new_cap < self._n + needed:
            new_cap = max(new_cap * 2, needed)
        # Resize by reading existing data, writing a larger file.
        old_X, old_y = self._open_mmaps(mode="r")
        snap_X = np.array(old_X[: self._n])
        snap_y = np.array(old_y[: self._n])
        del old_X, old_y  # close old mmaps before resizing files

        new_X = np.memmap(self._path / self._X_FILE, dtype=np.float32,
                          mode="w+",
                          shape=(new_cap, self._n_features, self._window_s))
        new_y = np.memmap(self._path / self._Y_FILE, dtype=np.int8,
                          mode="w+", shape=(new_cap,))
        new_X[: self._n] = snap_X
        new_y[: self._n] = snap_y
        new_X.flush()
        new_y.flush()
        del new_X, new_y
        self._cap = new_cap

    def append(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> None:
        """Append a chunk of windows to the store.

        X_chunk: float32 [k, n_features, window_s] (post-scale).
        y_chunk: int8    [k].
        """
        X_chunk = np.asarray(X_chunk, dtype=np.float32)
        y_chunk = np.asarray(y_chunk, dtype=np.int8)
        k = len(y_chunk)
        if k == 0:
            return
        if self._n + k > self._cap:
            self._grow(k)
        X_mm, y_mm = self._open_mmaps(mode="r+")
        X_mm[self._n: self._n + k] = X_chunk
        y_mm[self._n: self._n + k] = y_chunk
        X_mm.flush()
        y_mm.flush()
        del X_mm, y_mm
        self._n += k
        self._write_meta()

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def finalize(self) -> "_FinalizedMemmapWindows":
        """Truncate backing files to the actual count and return a
        read-only Dataset-compatible object."""
        if self._n == 0:
            # Write empty files at the correct (0,) shapes.
            np.memmap(self._path / self._X_FILE, dtype=np.float32, mode="w+",
                      shape=(0, self._n_features, self._window_s))
            np.memmap(self._path / self._Y_FILE, dtype=np.int8, mode="w+",
                      shape=(0,))
            self._cap = 0
            self._write_meta()
            return _FinalizedMemmapWindows(self)
        # Truncate to actual size by re-writing only valid rows.
        old_X, old_y = self._open_mmaps(mode="r")
        snap_X = np.array(old_X[: self._n])
        snap_y = np.array(old_y[: self._n])
        del old_X, old_y

        final_X = np.memmap(self._path / self._X_FILE, dtype=np.float32,
                            mode="w+",
                            shape=(self._n, self._n_features, self._window_s))
        final_y = np.memmap(self._path / self._Y_FILE, dtype=np.int8,
                            mode="w+", shape=(self._n,))
        final_X[:] = snap_X
        final_y[:] = snap_y
        final_X.flush()
        final_y.flush()
        del final_X, final_y
        self._cap = self._n
        self._write_meta()
        return _FinalizedMemmapWindows(self)

    def _write_meta(self) -> None:
        (self._path / self._META_FILE).write_text(json.dumps({
            "n": self._n,
            "n_features": self._n_features,
            "window_s": self._window_s,
        }, indent=2))


class _FinalizedMemmapWindows(torch.utils.data.Dataset):
    """Read-only Dataset view over a finalized MemmapWindows store.

    __getitem__(i) copies the i-th row from the mmap into a fresh numpy
    array so that the returned tensor is independent of the mmap file
    handle (safe across DataLoader workers; the mmap is never passed
    between processes).
    """

    def __init__(self, store: MemmapWindows):
        self._path = store._path
        self._n = store._n
        self._n_features = store._n_features
        self._window_s = store._window_s

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_mm = np.memmap(self._path / MemmapWindows._X_FILE, dtype=np.float32,
                         mode="r",
                         shape=(self._n, self._n_features, self._window_s))
        y_mm = np.memmap(self._path / MemmapWindows._Y_FILE, dtype=np.int8,
                         mode="r", shape=(self._n,))
        # Copy into a plain numpy array — severs the mmap reference so
        # torch never holds a mmap handle alive past this __getitem__ call.
        x_row = np.array(X_mm[i])
        y_val = int(y_mm[i])
        del X_mm, y_mm
        return torch.from_numpy(x_row), torch.tensor(y_val, dtype=torch.float32)

    # expose n_features / window_s for callers that query shape
    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def window_s(self) -> int:
        return self._window_s


# --------------------------------------------------------------------------- #
# Train-time augmentation: additive Gaussian jitter, re-drawn every epoch
# --------------------------------------------------------------------------- #
class JitterDataset(torch.utils.data.Dataset):
    """TRAIN-SET-ONLY augmentation wrapper: adds fresh N(0, sigma) Gaussian
    noise to the already-SCALED window tensor `X`, redrawn per sample.

    Val loaders must NEVER be wrapped in this — scalp.deep.train_loop.train
    calls set_epoch(epoch) on train_loader.dataset only (never
    val_loader.dataset), and train_tcn.py only ever wraps the train split.
    `sigma <= 0` disables augmentation entirely: __getitem__ then returns X
    unmodified and no RNG is ever drawn, so wrapping unconditionally at the
    default --jitter-sigma 0.0 is a no-op byte-for-byte.

    Reproducibility: noise is generated per-sample in __getitem__ using a
    torch.Generator seeded deterministically from (seed, current_epoch,
    index). This avoids materialising a noise tensor the size of the whole
    dataset (which OOM-killed machines at sigma > 0). The same (seed, epoch,
    index) triple always produces identical noise; a different epoch always
    produces different noise. set_epoch(epoch) stores the current epoch number
    and is the required epoch hook for train_loop.train.

    Accepts either:
    - (X: torch.Tensor, y: torch.Tensor) — in-RAM mode (original interface).
    - inner: _FinalizedMemmapWindows (or any Dataset returning (x_tensor,
      y_tensor) from __getitem__) — disk-backed mode. In this case X and y
      are NOT stored as full tensors; the inner dataset is indexed per sample
      so RAM usage is O(1) not O(n). Pass inner=<dataset> and omit X/y.
    """

    def __init__(
        self,
        X: "torch.Tensor | None" = None,
        y: "torch.Tensor | None" = None,
        sigma: float = 0.0,
        seed: int = 0,
        *,
        inner: "torch.utils.data.Dataset | None" = None,
    ):
        if inner is not None:
            # Disk-backed mode: do NOT store full tensors.
            self._inner = inner
            # X and y are intentionally not set as attributes to avoid the
            # full-dataset-sized in-RAM tensor that the OOM guard checks for.
            self._use_inner = True
        else:
            if X is None or y is None:
                raise ValueError("JitterDataset: supply either inner= or (X, y)")
            self.X = X
            self.y = y
            self._inner = None
            self._use_inner = False
        self.sigma = float(sigma)
        self.seed = int(seed)
        self._epoch: int = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self._use_inner:
            return len(self._inner)  # type: ignore[arg-type]
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_inner:
            x, y_i = self._inner[i]
        else:
            x = self.X[i]
            y_i = self.y[i]
        if self.sigma > 0.0:
            # Seed combines base_seed, epoch, and sample index so that:
            #   - same (epoch, i) always produces identical noise
            #   - different epochs produce different noise for each sample
            # All arithmetic stays in 64-bit to avoid wrap-around collisions.
            seed_val = (
                self.seed * 1_000_003
                + self._epoch * 1_000_000_007
                + int(i)
            ) % (2**63)
            gen = torch.Generator().manual_seed(seed_val)
            noise = torch.empty_like(x).normal_(generator=gen) * self.sigma
            x = x + noise
        return x, y_i
