"""Cost-aware oracle PnL on second bars — the GO/NO-GO viability core.

Question answered (port of the LOB cost_aware_experiment, adapted from LOB
snapshots to runner-day second bars): with PERFECT foresight of the price
`horizon_s` ahead, does scalping this tape clear real transaction costs?
If the oracle can't make money, no model can — that regime is dead. Where
the oracle is rich, its trades define the scanner target and the label
cost-threshold for Phase 2.

Execution styles:
- taker: cross the spread both ways — buy at ask[t], sell at bid[t+h].
  This is the user's historical manual style (market-ish in/out).
- maker: post at bid[t]; filled only if a later trade prints strictly
  below the bid within `maker_wait_s` (trade-through proxy — conservative:
  price had to trade through our level, ignoring queue priority we lose and
  fills at exactly our price we might get). Exit posts at ask[t+h]; if not
  trade-through-filled within `maker_wait_s`, exit crosses at the bid then
  prevailing (pay spread on the way out).
- frictionless: mid-to-mid, zero costs — the "does price even move" bound.

All entries are long-only (the strategy scalps upward bursts). Trades are
greedy and non-overlapping per style: one open position at a time; this is
an upper bound on capture, not a strategy.

Quote sizes: SIP top-of-book sizes are in ROUND LOTS (100 shares); capacity
numbers use LOT_SIZE and are reported separately from per-share edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOT_SIZE = 100


@dataclass(frozen=True)
class FeeModel:
    """Regulatory costs on SELLS (Alpaca charges no commission).
    Rates drift year to year — configurable so the study states its inputs."""
    sec_rate: float = 27.80e-6       # x sell notional
    taf_per_share: float = 0.000166  # x shares sold, capped per trade
    taf_cap: float = 8.30

    def sell_cost_per_share(self, sell_px: float, shares: int) -> float:
        taf = min(self.taf_per_share * shares, self.taf_cap) / max(shares, 1)
        return self.sec_rate * sell_px + taf


@dataclass(frozen=True)
class OracleConfig:
    horizon_s: int = 60
    clip_shares: int = 1000          # the "buy huge" unit economics clip
    maker_wait_s: int = 30
    fees: FeeModel = FeeModel()


def _valid_nbbo(bars: pd.DataFrame) -> np.ndarray:
    return bars["bid"].notna().to_numpy() & bars["ask"].notna().to_numpy() \
        & (bars["bid"].to_numpy() > 0)


def taker_oracle(bars: pd.DataFrame, cfg: OracleConfig) -> pd.DataFrame:
    """Perfect-foresight long scalps paying the spread both ways."""
    h = cfg.horizon_s
    ask = bars["ask"].to_numpy()
    bid = bars["bid"].to_numpy()
    ask_sz = bars["ask_size"].to_numpy()
    bid_sz = bars["bid_size"].to_numpy()
    ok = _valid_nbbo(bars)
    n = len(bars)
    trades = []
    t = 0
    while t + h < n:
        if not (ok[t] and ok[t + h]):
            t += 1
            continue
        entry, exit_ = ask[t], bid[t + h]
        edge = exit_ - entry - cfg.fees.sell_cost_per_share(exit_, cfg.clip_shares)
        if edge > 0:
            displayed = min(ask_sz[t], bid_sz[t + h]) * LOT_SIZE
            fill = min(cfg.clip_shares, displayed if displayed > 0 else 0)
            trades.append({"t": bars.index[t], "entry": entry, "exit": exit_,
                           "edge_ps": edge, "clip_pnl": edge * cfg.clip_shares,
                           "displayed_shares": displayed,
                           "displayed_pnl": edge * fill})
            t += h  # position occupies the horizon: non-overlapping
        else:
            t += 1
    return pd.DataFrame(trades)


def maker_oracle(bars: pd.DataFrame, cfg: OracleConfig) -> pd.DataFrame:
    """Perfect-foresight long scalps earning the spread when fills allow."""
    h, w = cfg.horizon_s, cfg.maker_wait_s
    bid = bars["bid"].to_numpy()
    ask = bars["ask"].to_numpy()
    low = bars["low"].to_numpy()
    high = bars["high"].to_numpy()
    ok = _valid_nbbo(bars)
    n = len(bars)
    trades = []
    t = 0
    while t + h + w < n:
        if not ok[t]:
            t += 1
            continue
        entry = bid[t]
        # entry fill: some trade prints strictly below our bid within the wait
        entry_win = low[t + 1: t + 1 + w]
        filled_at = np.nonzero(entry_win < entry)[0]
        if filled_at.size == 0:
            t += 1
            continue
        te = t + 1 + int(filled_at[0])           # second the entry filled
        tx = te + h                               # exit decision time
        if tx + w >= n or not ok[tx]:
            t += 1
            continue
        target = ask[tx]
        exit_win = high[tx + 1: tx + 1 + w]
        hit = np.nonzero(exit_win > target)[0]
        if hit.size:
            exit_px, exit_t = target, tx + 1 + int(hit[0])
        else:
            # unfilled: cross out at the then-prevailing bid
            if not ok[tx + w]:
                t += 1
                continue
            exit_px, exit_t = bid[tx + w], tx + w
        edge = exit_px - entry - cfg.fees.sell_cost_per_share(exit_px, cfg.clip_shares)
        if edge > 0:
            trades.append({"t": bars.index[t], "entry": entry, "exit": exit_px,
                           "edge_ps": edge, "clip_pnl": edge * cfg.clip_shares,
                           "exit_crossed": not bool(hit.size)})
            t = exit_t + 1
        else:
            t += 1
    return pd.DataFrame(trades)


def frictionless_oracle(bars: pd.DataFrame, cfg: OracleConfig) -> float:
    """Sum of positive mid moves at the horizon — 'does price even move'."""
    mid = ((bars["bid"] + bars["ask"]) / 2).to_numpy()
    h = cfg.horizon_s
    if len(mid) <= h:
        return 0.0
    moves = mid[h:] - mid[:-h]
    moves = moves[~np.isnan(moves)]
    return float(moves[moves > 0].sum())


def day_summary(bars: pd.DataFrame, cfg: OracleConfig) -> dict:
    """All three oracles + tape stats for one stock-day at one horizon."""
    taker = taker_oracle(bars, cfg)
    maker = maker_oracle(bars, cfg)
    mid = (bars["bid"] + bars["ask"]) / 2
    spread_bps = (bars["spread"] / mid * 1e4)
    traded = bars[bars["n_trades"] > 0]
    out = {
        "horizon_s": cfg.horizon_s,
        "n_seconds": len(bars),
        "quote_coverage": float(_valid_nbbo(bars).mean()),
        "med_spread": float(bars["spread"].median()),
        "med_spread_bps": float(spread_bps.median()),
        "day_volume": float(bars["volume"].sum()),
        "med_second_dollar_vol": float(
            (traded["volume"] * traded["vwap"]).median()) if len(traded) else 0.0,
        "frictionless_ps": frictionless_oracle(bars, cfg),
    }
    for name, tr in [("taker", taker), ("maker", maker)]:
        out[f"{name}_n"] = len(tr)
        out[f"{name}_clip_pnl"] = float(tr["clip_pnl"].sum()) if len(tr) else 0.0
        out[f"{name}_med_edge_ps"] = float(tr["edge_ps"].median()) if len(tr) else 0.0
    out["taker_displayed_pnl"] = (
        float(taker["displayed_pnl"].sum()) if len(taker) else 0.0)
    return out
