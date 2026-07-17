"""Model-owned scalp sizing: capped-Kelly edge bet inside hard guardrails.

The model owns HOW MUCH within the box; the box itself (position pct,
participation, price band) comes from the engine's ScalpConfig guardrails
and is enforced here AND again by the RiskManager — sizing must never be
the only line of defense.

Outcome geometry follows the triple-barrier labels: a WIN banks target_ps
net of sell costs (the labeler requires bid >= entry + target + sell_cost),
a LOSS pays stop_ps plus the sell-side fee. Kelly is the binary
approximation over those two legs (timeouts, roughly edge-neutral by
construction, are ignored); the cap keeps the approximation honest.
"""

from __future__ import annotations

import math

from .viability import LOT_SIZE, FeeModel


def size_scalp(p_win: float, target_ps: float, stop_ps: float, price: float,
               equity: float, vol_60s: float, ask_lots: float, cfg,
               kelly_cap: float = 0.25, displayed_mult: float = 2.0,
               fees: FeeModel = FeeModel(), lot_size: int = LOT_SIZE) -> int:
    """Shares to buy for one scalp, 0 when the edge or the box says no.

    Parameters
    ----------
    p_win:        model probability of the WIN class for this decision
    target_ps / stop_ps:  bracket legs, $/share (vol-scaled, from the signal)
    price:        entry price (ask)
    equity:       account equity in dollars
    vol_60s:      trailing 60s traded SHARE volume (same window as the sim cap)
    ask_lots:     displayed ask size in round lots (SIP convention)
    cfg:          guardrails, duck-typed: max_position_pct, max_participation,
                  price_min, price_max (engine ScalpConfig satisfies this)
    """
    vals = (p_win, target_ps, stop_ps, price, equity, vol_60s, ask_lots)
    if any(v is None or not math.isfinite(v) for v in vals):
        return 0
    if not (0.0 <= p_win <= 1.0) or target_ps <= 0 or stop_ps <= 0 \
            or price <= 0 or equity <= 0:
        return 0
    if not (cfg.price_min <= price <= cfg.price_max):
        return 0

    # sell-side fee per share, TAF uncapped (cap only ever lowers it)
    fee_ps = fees.sec_rate * price + fees.taf_per_share
    win_ps = target_ps                  # net by label construction
    loss_ps = stop_ps + fee_ps
    q = 1.0 - p_win
    edge_ps = p_win * win_ps - q * loss_ps
    if edge_ps <= 0:
        return 0

    b = win_ps / loss_ps
    f = (p_win * b - q) / b             # binary Kelly fraction
    f = min(max(f, 0.0), kelly_cap)

    shares = min(
        f * equity / price,                       # Kelly bet
        cfg.max_position_pct * equity / price,    # HARD: single-scalp notional
        cfg.max_participation * max(vol_60s, 0.0),  # HARD: tape participation
        displayed_mult * max(ask_lots, 0.0) * lot_size,  # displayed depth
    )
    return max(int(shares), 0)
