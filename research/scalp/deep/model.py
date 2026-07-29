"""ScalpTCN: causal temporal-convolutional network for the scalp entry
signal, ported from ~/LOB research/lob/models.py's TCN block structure
(_CausalConv1d / _TCNBlock — left-pad-only dilated convs, weight-normed,
residual). LOB-specific I/O (raw40 column layout, 3-class down/flat/up
head, [B, T, F] input requiring a permute) is stripped: this rung is
feature-order agnostic (it consumes scalp.bars_features' 19-column output
as-is) and predicts a single binary WIN logit.

Input layout is channels-first, [B, n_features, window_s] — the SAME
layout scalp.deep.dataset.build_windows already produces, so no permute is
needed between dataset and model (unlike the LOB port, which took
[B, T, F] and permuted internally).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import apply_scaler, load_scaler, windows_for_all_seconds


def pick_device() -> torch.device:
    """auto-select: cuda (covers ROCm) > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _assert_single_day_contiguous_index(idx: pd.Index) -> None:
    """windows_for_all_seconds builds causal windows over ROW POSITION and
    the warmup mask only zeroes the first window_s-1 ROWS of whatever frame
    it's given — both are only correct if `idx` is a single stock-day's
    contiguous 1-second series (the day_entries-style full-day frame).

    Fed a multi-day/multi-symbol concatenation instead (e.g.
    scalp.walkforward.build_dataset's x_test — a valid-row subset, per-day
    stratified-subsampled, multi-stock-day concat sorted lexically by
    file), windows would silently splice rows from OTHER stock-days into
    each window and the warmup mask would only cover the very first window
    of the whole concatenation. Fail loudly instead of scoring garbage.
    """
    if len(idx) < 2:
        return
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(
            "TcnProbModel.predict_proba: feats_df.index must be a "
            f"DatetimeIndex, got {type(idx).__name__}")
    diffs = idx.to_series().diff().dropna()
    if not (diffs == pd.Timedelta(seconds=1)).all():
        raise ValueError(
            "TcnProbModel.predict_proba: feats_df.index must be a "
            "monotonic, contiguous 1-second DatetimeIndex spanning a "
            "SINGLE stock-day (the day_entries-style full-day feature "
            "frame) — got gaps or out-of-order timestamps, which usually "
            "means a multi-day/multi-symbol concatenation (e.g. "
            "scalp.walkforward.build_dataset's x_test) was passed in; see "
            "the TcnProbModel docstring for why that silently corrupts "
            "windows instead of erroring")


# --------------------------------------------------------------------------- #
# Causal TCN block (port of LOB's _CausalConv1d / _TCNBlock)
# --------------------------------------------------------------------------- #
class _CausalConv1d(nn.Module):
    """Left-padded conv so output at t sees only inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int,
                dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        return self.act(y + self.downsample(x))


class ScalpTCN(nn.Module):
    """Strictly causal TCN -> single WIN logit.

    forward_sequence(x) exposes the per-timestep trunk output [B, C, T] —
    this is what a causality test should probe (forward()'s single-logit
    output only ever corresponds to the LAST timestep of whatever window
    it's given, so "output at position t" only makes sense on the trunk).
    """

    def __init__(self, n_features: int = 19, window_s: int = 240,
                channels: int = 64, blocks: int = 4, dropout: float = 0.1,
                kernel: int = 3):
        super().__init__()
        self.n_features = n_features
        self.window_s = window_s
        layers = []
        in_ch = n_features
        for i in range(blocks):
            layers.append(_TCNBlock(in_ch, channels, kernel,
                                    dilation=2 ** i, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, 1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_features, T] -> [B, channels, T], causal at every t."""
        return self.tcn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_features, window_s] -> [B] single WIN logit (the
        window's last timestep, i.e. the decision second)."""
        y = self.forward_sequence(x)
        return self.head(y[:, :, -1]).squeeze(-1)


# --------------------------------------------------------------------------- #
# Serving wrapper: sklearn-shaped predict_proba over a FULL day feature frame
# --------------------------------------------------------------------------- #
class TcnProbModel:
    """Fit-nothing predict_proba(feats_df) -> np.ndarray [n, 2], matching
    the sklearn predict_proba SHAPE the GBT rung's scripts.sim_eval
    day_entries call site expects — but NOT a drop-in for
    scalp.walkforward.evaluate: that function's x_test is a valid-row
    subset, per-day stratified-subsampled, multi-stock-day concatenation
    (sorted lexically by file), and this class builds windows over ROW
    POSITION, not timestamp — see predict_proba's index guard. The TCN
    must be evaluated per stock-day through a day_entries-style FULL-day
    feature frame, one stock-day at a time, never through evaluate's
    concatenated x_test.

    `feats_df` must be the FULL day feature frame as build_features
    returns it (same columns, same index) — NOT a subset of warm rows.
    Builds a causal window ending at every second internally, applies the
    stored (train-days-only) scaler, batches through the net, and returns
    p=0 at "warmup" seconds: either fewer than window_s-1 seconds of prior
    history (heavily NaN-padded windows the net never trained on — see
    scalp.deep.dataset.build_windows, which excludes exactly this
    population from training so this guarantee actually holds) or a NaN
    in that second's own feature row (an intrinsically undefined second,
    e.g. before any quote/trade has established the rolling stats). This
    mirrors the conservative "abstain" behavior the barrier/NBBO gate in
    scripts.sim_eval.day_entries already applies to the GBT rung, just
    scoped to what the model itself needs to be well-defined.
    """

    def __init__(self, model: ScalpTCN, scaler: dict, window_s: int = 240,
                batch_size: int = 512, device: torch.device | None = None):
        self.model = model
        self.scaler = scaler
        self.window_s = window_s
        self.batch_size = batch_size
        self.device = device or pick_device()
        self.classes_ = np.array([0.0, 1.0])
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, run_dir: Path, device: torch.device | None = None,
            batch_size: int = 512) -> TcnProbModel:
        """Reconstruct from a train_tcn.py run dir: model.pt + scaler.json
        + config.json (architecture hyperparameters)."""
        run_dir = Path(run_dir)
        scaler = load_scaler(run_dir / "scaler.json")
        config = json.loads((run_dir / "config.json").read_text())
        model = ScalpTCN(
            n_features=len(scaler["feature_names"]),
            window_s=scaler.get("window_s", config.get("window", 240)),
            channels=config.get("channels", 64),
            blocks=config.get("blocks", 4),
            dropout=config.get("dropout", 0.1),
        )
        state = torch.load(run_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        return cls(model, scaler, window_s=scaler.get("window_s", 240),
                   batch_size=batch_size, device=device)

    @torch.no_grad()
    def predict_proba(self, feats_df: pd.DataFrame) -> np.ndarray:
        _assert_single_day_contiguous_index(feats_df.index)
        feature_names = self.scaler["feature_names"]
        missing = [c for c in feature_names if c not in feats_df.columns]
        if missing:
            raise ValueError(f"TcnProbModel.predict_proba: missing columns {missing}")
        feats = feats_df[feature_names]
        n = len(feats)
        if n == 0:
            return np.empty((0, 2), dtype=np.float32)

        raw = windows_for_all_seconds(feats, self.window_s)      # [n, F, T]
        scaled = apply_scaler(raw, self.scaler)

        F_arr = feats.to_numpy(dtype=np.float64)
        warmup = (np.arange(n) < self.window_s - 1) | np.isnan(F_arr).any(axis=1)

        probs = np.zeros(n, dtype=np.float32)
        keep_pos = np.nonzero(~warmup)[0]
        for start in range(0, len(keep_pos), self.batch_size):
            batch_idx = keep_pos[start:start + self.batch_size]
            xb = torch.from_numpy(scaled[batch_idx]).to(self.device)
            logits = self.model(xb)
            probs[batch_idx] = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1.0 - probs, probs]).astype(np.float32)
