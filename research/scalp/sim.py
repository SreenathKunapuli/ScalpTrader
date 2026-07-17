"""Event-driven scalp fill simulator on second bars — the realism layer
between label-based expectancy (triple_barrier) and live trading.

Turns entry DECISIONS into attempted trades against the recorded tape,
charging the fills you would actually have gotten rather than the barrier
prices you wanted. Every rule errs conservative:

- One position at a time: entries are processed chronologically and any
  entry whose decision time falls while an earlier trade is still working
  or open (decision time through exit_t, inclusive) is skipped with
  filled=False and exit_reason 'unfilled'.
- Taker entry: cross at ask[t] when the NBBO is valid (bid/ask present,
  bid > 0). fill_qty = min(qty, ask_size[t] * lot_size, max_participation
  x rolling 60s volume ending at t). Zero displayed size or a zero
  participation budget means no fill — you cannot buy from an empty book.
- Maker entry: post at bid[t]; filled at bid[t] only if some later low
  within maker_wait_s prints STRICTLY below bid[t] (trade-through proxy,
  same convention as viability.maker_oracle: conservative on queue
  priority we lose, ignoring fills at exactly our price we might get).
  Otherwise unfilled. Maker fills take the full qty (a trade-through
  clears the level); capacity caps apply to taker entries only.
- Target exit: first second after the entry fill where bid >= target_px —
  exit at target_px. Resting-limit assumption, conservative: the BID must
  cross up to our offer; a high merely touching the level is not a fill.
- Stop exit: first second where bid <= stop_px. You do not get the stop
  price, you get the market: exit at the next valid bid at or after that
  second (the trigger bar itself, since triggering requires a valid bid)
  MINUS half the then-current spread. slippage_ps = stop_px - exit_px
  (positive = adverse; a tape that gaps through the stop shows up here).
- Stop and target on the SAME second: stop wins (conservative).
- Timeout: neither barrier hit by the entry's deadline — exit market-ish
  at the bid on the deadline bar (next valid bid if that one is stale).
- EOD: if the trade cannot resolve by the last bar (deadline past the end
  of the data, or no valid quote left to exit on), exit at the last valid
  bid with reason 'eod'.
- Fees: FeeModel.sell_cost_per_share on every exit; entries are free
  (Alpaca charges nothing on buys). pnl = (exit - entry - fee_ps) * qty.

No lookahead: each trade's outcome depends only on bars up to its exit_t
(the participation volume window is trailing), so mutating the tape after
a trade's exit cannot change that trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .viability import LOT_SIZE, FeeModel

VOLUME_WINDOW_S = 60   # trailing window for the participation cap

_COLUMNS = ["filled", "fill_qty", "entry_px", "exit_px", "exit_reason",
            "entry_t", "exit_t", "pnl", "slippage_ps"]


@dataclass(frozen=True)
class SimConfig:
    entry_mode: str = "taker"        # 'taker' | 'maker'
    maker_wait_s: int = 30
    fees: FeeModel = FeeModel()
    max_participation: float = 0.05  # of rolling 60s volume
    lot_size: int = LOT_SIZE


def _unfilled_row() -> dict:
    return {"filled": False, "fill_qty": 0, "entry_px": np.nan,
            "exit_px": np.nan, "exit_reason": "unfilled",
            "entry_t": pd.NaT, "exit_t": pd.NaT,
            "pnl": 0.0, "slippage_ps": np.nan}


def simulate(bars: pd.DataFrame, entries: pd.DataFrame,
             cfg: SimConfig) -> pd.DataFrame:
    """Simulate fills for entry decisions against one day of second bars.

    Parameters
    ----------
    bars:
        Second-bar DataFrame (corpus.second_bars output) with a
        monotonically increasing DatetimeIndex and columns bid, ask,
        ask_size, low, volume, spread.
    entries:
        One row per entry decision, indexed by decision time (must be a
        subset of bars.index), columns qty, target_px, stop_px, deadline
        (tz-aware timestamps).
    cfg:
        Fill-model parameters (see module docstring for the rules).

    Returns
    -------
    pd.DataFrame
        One row per ATTEMPTED entry, indexed by decision time, columns:
        - filled       bool: did the entry get any shares
        - fill_qty     int: shares filled (0 if unfilled)
        - entry_px     float: actual entry price (NaN if unfilled)
        - exit_px      float: actual exit price (NaN if unfilled)
        - exit_reason  'target' | 'stop' | 'timeout' | 'unfilled' | 'eod'
        - entry_t      entry fill time (NaT if unfilled)
        - exit_t       exit fill time (NaT if unfilled)
        - pnl          float: fee-adjusted dollars
        - slippage_ps  float: intended barrier px minus realized exit px
                       (0 on target, positive on adverse stop fills, NaN
                       when there was no intended barrier price)
    """
    if cfg.entry_mode not in ("taker", "maker"):
        raise ValueError(f"unknown entry_mode {cfg.entry_mode!r}")
    ent = entries.sort_index()
    if ent.empty:
        return pd.DataFrame(columns=_COLUMNS, index=ent.index)

    n = len(bars)
    bid = bars["bid"].to_numpy(dtype=float)
    ask = bars["ask"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    ask_size = bars["ask_size"].to_numpy(dtype=float)
    spread = bars["spread"].to_numpy(dtype=float)
    bid_ok = ~np.isnan(bid) & (bid > 0)
    nbbo_ok = bid_ok & ~np.isnan(ask)
    # trailing (causal) 60s volume ending at each bar, for the taker cap
    vol_w = bars["volume"].rolling(VOLUME_WINDOW_S,
                                   min_periods=1).sum().to_numpy(dtype=float)

    tpos_arr = bars.index.get_indexer(ent.index)
    if (tpos_arr < 0).any():
        raise ValueError("entry decision times must be a subset of bars.index")

    rows = []
    busy_until = None        # exit_t of the currently open trade, if any
    for (t_dec, e), tpos in zip(ent.iterrows(), tpos_arr):
        tpos = int(tpos)
        # one position at a time: skip while a prior trade occupies the tape
        if (busy_until is not None and t_dec <= busy_until) \
                or not nbbo_ok[tpos]:
            rows.append(_unfilled_row())
            continue
        qty = int(e["qty"])
        target, stop = float(e["target_px"]), float(e["stop_px"])
        deadline = e["deadline"]

        # --- entry fill ---
        if cfg.entry_mode == "taker":
            displayed = ask_size[tpos] * cfg.lot_size
            part = cfg.max_participation * vol_w[tpos]
            if not (np.isfinite(displayed) and displayed > 0
                    and np.isfinite(part)):
                rows.append(_unfilled_row())       # empty book: no fill
                continue
            fill_qty = int(min(qty, np.floor(displayed), np.floor(part)))
            if fill_qty <= 0:
                rows.append(_unfilled_row())       # participation budget zero
                continue
            e_idx, entry_px = tpos, ask[tpos]
        else:  # maker: needs a trade-through within the wait window
            entry_px = bid[tpos]
            win = low[tpos + 1: tpos + 1 + cfg.maker_wait_s]
            hit = np.nonzero(win < entry_px)[0]
            if hit.size == 0:
                rows.append(_unfilled_row())
                continue
            e_idx, fill_qty = tpos + 1 + int(hit[0]), qty

        # --- exit: barriers scanned over (entry, deadline], stop wins ties ---
        dd = int(bars.index.searchsorted(deadline, side="right")) - 1
        exit_idx, exit_px, reason, slip = -1, np.nan, None, np.nan
        for k in range(e_idx + 1, min(dd, n - 1) + 1):
            if not bid_ok[k]:
                continue
            b = bid[k]
            if b <= stop:      # checked first: same-second target loses
                # market exit at the next valid bid at/after the trigger
                # (== the trigger bar's bid) minus half the current spread
                half = spread[k] / 2 if np.isfinite(spread[k]) else 0.0
                exit_idx, exit_px, reason = k, b - half, "stop"
                slip = stop - exit_px
                break
            if b >= target:    # bid crossed our resting offer
                exit_idx, exit_px, reason, slip = k, target, "target", 0.0
                break

        if reason is None and deadline <= bars.index[-1]:
            # timeout at the deadline bar: first valid bid at/after it
            j = max(dd, e_idx)
            while j < n and not bid_ok[j]:
                j += 1
            if j < n:
                exit_idx, exit_px, reason = j, bid[j], "timeout"
        if reason is None:
            # end of data before resolution: last valid bid closes the book
            tail = np.nonzero(bid_ok[e_idx:])[0]
            j = e_idx + int(tail[-1]) if tail.size else tpos
            exit_idx, exit_px, reason = j, bid[j], "eod"

        fee_ps = cfg.fees.sell_cost_per_share(float(exit_px), fill_qty)
        pnl = (exit_px - entry_px - fee_ps) * fill_qty
        busy_until = bars.index[exit_idx]
        rows.append({"filled": True, "fill_qty": fill_qty,
                     "entry_px": float(entry_px), "exit_px": float(exit_px),
                     "exit_reason": reason, "entry_t": bars.index[e_idx],
                     "exit_t": bars.index[exit_idx], "pnl": float(pnl),
                     "slippage_ps": slip})

    return pd.DataFrame(rows, index=ent.index)[_COLUMNS]
