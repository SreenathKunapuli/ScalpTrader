"""Triple-barrier labeler for scalp entries on second bars.

Label long-only taker entries with a triple-barrier scheme:
  +1 (WIN)     — first second where bid >= entry + target_ps + sell_cost
  -1 (LOSS)    — first second where bid <= entry - stop_ps
   0 (TIMEOUT) — neither barrier hit within timeout_s seconds
 NaN (INVALID) — no valid NBBO at entry, no recent trade activity, or the
                 label window would run past the end of the day (truncated
                 windows must not produce fake TIMEOUTs)

Entry is at ask[t]; exit side uses bid[t'] to compute live P&L.

Complexity note: the inner scan walks at most *timeout_s* rows for each
candidate entry, so the algorithm is O(n * timeout_s) worst case.  For
typical parameters (timeout_s=120, n~23400 RTH seconds) this is ~2.8 M
comparisons — fast enough with a simple Python forward loop plus numpy
array slicing to avoid per-element Python overhead on the inner window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .viability import FeeModel

_ACTIVE_WINDOW_S = 10   # require a trade in the prior N seconds for entry


@dataclass(frozen=True)
class BarrierConfig:
    target_ps: float           # profit target in dollars per share
    stop_ps: float             # stop-loss in dollars per share (positive value)
    timeout_s: int = 120       # max holding period in seconds
    fees: FeeModel = field(default_factory=FeeModel)
    clip_shares: int = 1000    # used to compute sell_cost_per_share


def label_scalps(
    bars: pd.DataFrame,
    cfg: BarrierConfig,
    target_ps_arr: np.ndarray | None = None,
    stop_ps_arr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assign triple-barrier labels to every second in *bars*.

    Parameters
    ----------
    bars:
        Second-bar DataFrame (corpus.second_bars output).  Must have a
        monotonically increasing DatetimeIndex and columns bid, ask, n_trades.
    cfg:
        Barrier parameters.

    Returns
    -------
    pd.DataFrame
        Same index as *bars*, columns:
        - label        float: 1.0 / -1.0 / 0.0 / NaN
        - entry_px     float: ask[t], NaN for INVALID rows
        - exit_s       float: seconds to resolution (NaN for INVALID)
        - timeout_edge float: bid[t+timeout_s] - entry - sell_cost (NaN unless TIMEOUT)
    """
    n = len(bars)
    ask = bars["ask"].to_numpy(dtype=float)
    bid = bars["bid"].to_numpy(dtype=float)
    n_trades = bars["n_trades"].to_numpy(dtype=float)
    # per-row barriers (volatility-scaled callers) or broadcast scalars
    tgt = (np.asarray(target_ps_arr, dtype=float) if target_ps_arr is not None
           else np.full(n, cfg.target_ps))
    stp = (np.asarray(stop_ps_arr, dtype=float) if stop_ps_arr is not None
           else np.full(n, cfg.stop_ps))
    if len(tgt) != n or len(stp) != n:
        raise ValueError("barrier arrays must match bars length")

    label = np.full(n, np.nan)
    entry_px = np.full(n, np.nan)
    exit_s = np.full(n, np.nan)
    timeout_edge = np.full(n, np.nan)

    # Pre-compute rolling sum of n_trades over the prior _ACTIVE_WINDOW_S rows
    # (causal: at index t, this is sum of rows [t-10, t-1]).
    # We use a simple cumsum trick for O(n).
    cum_trades = np.concatenate([[0.0], np.cumsum(n_trades)])
    # recent_trades[t] = trades in rows (t-window, t) exclusive of t
    w = _ACTIVE_WINDOW_S

    for t in range(n):
        # --- entry eligibility ---
        if np.isnan(ask[t]) or np.isnan(bid[t]):
            continue  # no valid NBBO -> INVALID (label stays NaN)

        # Trades in the prior _ACTIVE_WINDOW_S seconds (rows max(0,t-w)..t-1)
        start = max(0, t - w)
        recent = cum_trades[t] - cum_trades[start]
        if recent < 1:
            continue   # no recent trade activity -> INVALID

        if t + cfg.timeout_s >= n:
            continue  # truncated window at end of day -> INVALID, not a fake TIMEOUT

        if np.isnan(tgt[t]) or np.isnan(stp[t]):
            continue  # no barrier defined (e.g. vol warmup) -> INVALID
        entry = ask[t]
        sell_cost = cfg.fees.sell_cost_per_share(entry, cfg.clip_shares)
        win_threshold = entry + tgt[t] + sell_cost
        loss_threshold = entry - stp[t]

        # Mark as TIMEOUT by default (we know entry is valid)
        label[t] = 0.0
        entry_px[t] = entry

        # --- inner scan over (t, t+timeout_s] ---
        end = min(t + cfg.timeout_s + 1, n)   # exclusive upper bound
        resolved = False
        for k in range(t + 1, end):
            b = bid[k]
            if np.isnan(b):
                continue
            if b >= win_threshold:
                label[t] = 1.0
                exit_s[t] = float(k - t)
                resolved = True
                break
            if b <= loss_threshold:
                label[t] = -1.0
                exit_s[t] = float(k - t)
                resolved = True
                break

        if not resolved:
            # Timeout: record exit_s and timeout_edge
            exit_idx = t + cfg.timeout_s
            exit_s[t] = float(cfg.timeout_s)
            if exit_idx < n and not np.isnan(bid[exit_idx]):
                # sell_cost at exit uses bid at timeout
                exit_sell_cost = cfg.fees.sell_cost_per_share(
                    bid[exit_idx], cfg.clip_shares
                )
                timeout_edge[t] = bid[exit_idx] - entry - exit_sell_cost
            # else timeout_edge stays NaN (no valid quote at timeout)

    return pd.DataFrame(
        {
            "label": label,
            "entry_px": entry_px,
            "exit_s": exit_s,
            "timeout_edge": timeout_edge,
        },
        index=bars.index,
    )
