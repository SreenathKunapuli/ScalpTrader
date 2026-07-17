"""RiskManager — every order passes through approve(); no exceptions.

Structural enforcement: OrderManager.submit() requires an `Approval` token,
and Approval's constructor is module-private by convention (`_issue`), so
the only way to obtain one is this class. Exits (stop / EOD / kill-flatten)
use approve_exit(), which skips *entry-only* checks (windows, position
caps) but still refuses in nonsensical states.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ..config.scalp_tiers import ScalpConfig
from ..config.tiers import TierConfig
from ..data import calendar
from .state import PortfolioState

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    qty: int
    price_hint: float                 # latest mark, for exposure math
    reason: str = "signal"            # signal / stop / eod / kill
    book: str = "intraday"            # which book's limits apply

    def as_dict(self) -> dict[str, object]:
        return {"symbol": self.symbol, "side": self.side, "qty": self.qty,
                "price_hint": self.price_hint, "reason": self.reason,
                "book": self.book}


@dataclass(frozen=True)
class Approval:
    intent: OrderIntent
    token: str
    issued_at: datetime
    urgent: bool = False   # exit path: certainty of execution beats spread cost


@dataclass(frozen=True)
class Rejection:
    intent: OrderIntent
    reason: str


class RiskManager:
    def __init__(self, tier: TierConfig, state: PortfolioState,
                 scalp_cfg: ScalpConfig | None = None) -> None:
        self.tier = tier
        self.state = state
        self.scalp_cfg = scalp_cfg
        self._dynamic_universe: set[str] = set()

    def add_to_universe(self, symbols: list[str]) -> None:
        self._dynamic_universe.update(symbols)

    def reset_dynamic_universe(self) -> None:
        self._dynamic_universe.clear()

    def _issue(self, intent: OrderIntent, urgent: bool = False) -> Approval:
        return Approval(intent=intent, token=secrets.token_hex(8),
                        issued_at=datetime.now(UTC), urgent=urgent)

    def _post_trade_position_value(self, intent: OrderIntent) -> float:
        pos = self.state.positions.get(intent.symbol)
        cur_qty = pos.qty if pos else 0
        delta = intent.qty if intent.side == "buy" else -intent.qty
        return abs((cur_qty + delta) * intent.price_hint)

    def _post_trade_gross(self, intent: OrderIntent) -> float:
        """Post-trade gross of the INTENT'S book — each book has its own
        budget, so one book must not consume another's headroom."""
        pos = self.state.positions.get(intent.symbol)
        cur_val = abs(pos.market_value) if pos else 0.0
        return (self.state.book_gross(intent.book) - cur_val
                + self._post_trade_position_value(intent))

    def _is_entry(self, intent: OrderIntent) -> bool:
        """Entry = increases absolute exposure in the symbol."""
        pos = self.state.positions.get(intent.symbol)
        cur_qty = pos.qty if pos else 0
        delta = intent.qty if intent.side == "buy" else -intent.qty
        return abs(cur_qty + delta) > abs(cur_qty)

    def approve(self, intent: OrderIntent, now: datetime | None = None,
                urgent: bool = False) -> Approval | Rejection:
        """Full check sequence per §5.4, in order. Entry path.

        `urgent=True` marks the approval for urgent execution (limit near
        touch, market replace) — every risk check still runs; only the
        execution style changes.
        """
        now = now or datetime.now(UTC)
        s, t = self.state, self.tier
        if s.halted:
            return Rejection(intent, "engine HALTED")
        if intent.qty <= 0:
            return Rejection(intent, "non-positive quantity")
        pos = s.positions.get(intent.symbol)
        if pos and pos.qty != 0 and pos.book != intent.book:
            return Rejection(intent, f"symbol held by {pos.book} book")
        if (intent.book == "intraday"
                and intent.symbol not in t.universe
                and intent.symbol not in self._dynamic_universe):
            return Rejection(intent, f"symbol {intent.symbol} not in tier universe")
        is_entry = self._is_entry(intent)
        if is_entry:
            cur_qty = pos.qty if pos else 0
            delta = intent.qty if intent.side == "buy" else -intent.qty
            if s.intraday_halted:
                return Rejection(intent, "intraday book halted")
            if self.scalp_cfg is not None:
                sc = self.scalp_cfg
                if not sc.price_min <= intent.price_hint <= sc.price_max:
                    return Rejection(intent, "outside scalp price band")
                if (s.symbol_realized_today.get(intent.symbol, 0.0)
                        <= -sc.per_symbol_loss_cap_pct * s.equity):
                    return Rejection(intent, "per-symbol loss cap")
                new_scalp = intent.symbol not in s.book_positions("intraday")
                if new_scalp and len(s.book_positions("intraday")) >= sc.max_open_scalps:
                    return Rejection(intent, "max concurrent scalps")
            if (cur_qty + delta) < 0 and not t.allow_short:
                return Rejection(intent, "shorting not allowed in this tier")
            if self._post_trade_position_value(intent) > t.max_position_pct * s.equity + 1e-6:
                return Rejection(intent, "exceeds max position size")
            if self._post_trade_gross(intent) > t.max_gross_pct * s.equity + 1e-6:
                return Rejection(intent, "exceeds max gross exposure")
            new_symbol = intent.symbol not in s.book_positions("intraday")
            if new_symbol and len(s.book_positions("intraday")) >= t.max_open_positions:
                return Rejection(intent, "exceeds max open positions")
            if s.intraday_day_pnl_pct <= -t.daily_loss_limit_pct:
                return Rejection(intent, "daily loss limit reached")
            if s.drawdown_pct >= t.max_drawdown_pct:
                return Rejection(intent, "account drawdown floor reached")
            if not calendar.in_entry_window(now):
                return Rejection(intent, "outside entry window")
        return self._issue(intent, urgent=urgent)

    def approve_exit(self, intent: OrderIntent) -> Approval | Rejection:
        """Stops / EOD / kill flatten: risk-checked but window/cap-exempt."""
        if intent.qty <= 0:
            return Rejection(intent, "non-positive quantity")
        if not self._is_entry(intent):
            return self._issue(intent, urgent=True)
        return Rejection(intent, "exit path used for an exposure-increasing order")
