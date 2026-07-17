"""Scalp risk configs — the hard bounds for the scalp execution path.

Why a separate frozen config (mirroring config/tiers.py): scalp risk limits
are code-reviewed constants, not env vars — changing a bound is a diff, not a
tweak. Two account-size presets are provided.

Field split (READ THIS before wiring):
  - target_ps / stop_ps / timeout_s are PRE-MODEL PLACEHOLDERS. The per-trade
    model (later) emits its own target/stop/timeout per opportunity, and those
    outputs OVERRIDE these values at arm-time. They exist here only so the
    engine can run before the model is wired.
  - Every other field is a HARD BOUND. The model never widens these; risk
    enforcement (RiskManager, participation caps) clamps to them regardless of
    what the model asks for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalpConfig:
    name: str
    max_position_pct: float      # HARD: max single-scalp notional as frac of equity
    max_gross_pct: float         # HARD: max total scalp gross as frac of equity
    max_open_scalps: int         # HARD: max concurrent open scalp positions
    daily_loss_limit_pct: float  # HARD: scalp-book day loss -> halt scalping for the day
    per_symbol_loss_cap_pct: float  # HARD: per-symbol realized loss today -> block that symbol
    target_ps: float             # PLACEHOLDER: take-profit as frac of price (model overrides)
    stop_ps: float               # PLACEHOLDER: stop distance as frac of price (model overrides)
    timeout_s: int               # PLACEHOLDER: max holding time before timeout exit (model overrides)
    max_participation: float     # HARD: max frac of available volume we take
    price_min: float             # HARD: reject scalps priced below this (avoids sub-dollar names)
    price_max: float             # HARD: reject scalps priced above this (share-granularity risk)


# ~$1k accounts: one scalp at a time, nearly full deployment into that single
# scalp (0.9 position == 0.9 gross since only one can be open). Tighter daily
# and per-symbol loss caps because a single bad name is the whole book.
SCALP_SMALL = ScalpConfig(
    "small", 0.9, 0.9, 1, 0.05, 0.03, 0.05, 0.04, 120, 0.05, 0.5, 10.0,
)

# ~$100k accounts: up to 5 concurrent scalps, each capped small (10% position),
# gross capped at 50%. Tighter daily/per-symbol loss caps than the small tier —
# a larger book can afford a lower relative pain threshold.
SCALP_LARGE = ScalpConfig(
    "large", 0.10, 0.50, 5, 0.02, 0.01, 0.05, 0.04, 120, 0.02, 0.5, 10.0,
)
