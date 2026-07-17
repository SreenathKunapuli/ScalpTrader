"""ScalpGbtSignal — live inference for the walk-forward-validated GBT scalper.

Zero train/serve skew by construction: features come from the SAME
research function the model was trained on (research/scalp/bars_features
.build_features, imported directly from the monorepo), applied to the
SecondBarBuilder rolling frame whose bar semantics mirror the corpus.
Per-decision barriers replicate walkforward.barrier_arrays causally on
the trailing window, so the bracket the engine stages matches the labels
the model was trained against.

Artifact dir contract (written by research/scripts/export_model.py):
    model.joblib     fitted HistGradientBoostingClassifier
    features.json    ordered feature-column list
    inference.json   threshold + barrier/timeout parameters
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..data.bar_builder import Bar
from .base import Signal, SignalOutput

_RESEARCH = Path(__file__).resolve().parents[3] / "research"
if not (_RESEARCH / "scalp" / "bars_features.py").exists():
    raise ImportError(
        f"research package not found at {_RESEARCH} — ScalpGbtSignal requires "
        "the monorepo layout (engine and research side by side)")
if str(_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_RESEARCH))
from scalp.bars_features import build_features  # noqa: E402


@dataclass(frozen=True)
class ScalpDecision:
    p_win: float          # model probability of the WIN class
    target_ps: float      # vol-scaled label geometry, $/share off the entry
    stop_ps: float        # label stop — sizing's loss leg, NOT the bracket
    timeout_s: int
    # execution stop distance (label stop x exec_stop_mult). The sim study
    # (runs/sim_eval 2026-07-17) showed bid-triggered stops near the label
    # distance pay 16-65c/sh gap-through slippage and destroy the edge;
    # timeout-only exits with a far disaster stop are the validated shape.
    exec_stop_ps: float | None = None

    @property
    def bracket_stop_ps(self) -> float:
        return self.exec_stop_ps if self.exec_stop_ps is not None else self.stop_ps


class ScalpGbtSignal(Signal):
    name = "scalp_gbt"

    def __init__(self, artifact_dir: str | Path) -> None:
        import joblib

        d = Path(artifact_dir)
        self.model = joblib.load(d / "model.joblib")
        self.feature_order: list[str] = json.loads(
            (d / "features.json").read_text())
        inf = json.loads((d / "inference.json").read_text())
        self.threshold: float = float(inf["threshold"])
        self.timeout_s: int = int(inf["timeout_s"])
        self._barrier_mode: str = inf["barrier_mode"]
        self._vol_window_s: int = int(inf["vol_window_s"])
        self._vol_target_mult: float = float(inf["vol_target_mult"])
        self._vol_stop_mult: float = float(inf["vol_stop_mult"])
        self._min_target_ps: float = float(inf["min_target_ps"])
        self._min_stop_ps: float = float(inf["min_stop_ps"])
        self._exec_stop_mult: float = float(inf.get("exec_stop_mult", 1.0))
        self._fixed_target_ps: float = float(inf["target_ps"])
        self._fixed_stop_ps: float = float(inf["stop_ps"])
        classes = list(self.model.classes_)
        self._win_col = classes.index(1.0) if 1.0 in classes else None

    # The 5-min ensemble path is inert for this signal: scalp decisions run
    # on the second-cadence loop via compute_second().
    def compute(self, symbol: str, bars: list[Bar]) -> SignalOutput:
        return SignalOutput(0.0, 0.0)

    def _barriers(self, frame: pd.DataFrame) -> tuple[float, float] | None:
        """Trailing-window replica of walkforward.barrier_arrays' last row.
        None while the window is too cold to price a bracket."""
        if self._barrier_mode == "fixed":
            return self._fixed_target_ps, self._fixed_stop_ps
        w = self._vol_window_s
        px = frame["close"].ffill().iloc[-w:]
        if int(px.notna().sum()) < max(2, w // 3):
            return None
        rng = float(px.max() - px.min())
        return (max(self._vol_target_mult * rng, self._min_target_ps),
                max(self._vol_stop_mult * rng, self._min_stop_ps))

    def compute_second(self, symbol: str,
                       frame: pd.DataFrame) -> ScalpDecision | None:
        """Entry decision on the latest finalized 1s bar, or None when the
        gate fails (cold barriers / invalid NBBO) — the same gating the
        OOS sim evaluation applied, so live selectivity matches it."""
        if frame.empty:
            return None
        last = frame.iloc[-1]
        bid, ask = last["bid"], last["ask"]
        if not (pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid):
            return None
        barriers = self._barriers(frame)
        if barriers is None:
            return None
        exec_stop = barriers[1] * self._exec_stop_mult
        if self._win_col is None:      # trained without WIN labels: never enter
            return ScalpDecision(0.0, barriers[0], barriers[1], self.timeout_s,
                                 exec_stop)
        feats = build_features(frame).iloc[[-1]].reindex(
            columns=self.feature_order)
        p_win = float(self.model.predict_proba(feats)[0, self._win_col])
        return ScalpDecision(p_win, barriers[0], barriers[1], self.timeout_s,
                             exec_stop)
