"""BracketBook — in-memory stop/timeout tracking for open scalps.

Why in-memory only: this is a derived view of open positions. On restart the
engine reconcile path rebuilds it from broker positions (re-arming each open
scalp), so persisting it would only risk drift against the source of truth.

The target (take-profit) leg is a resting broker limit order — see
OrderManager.submit_bracket_target — so the bracket only tracks the two legs
this process must fire itself: the stop (price trigger) and the timeout
(deadline trigger). When the target fills, call on_target_fill to disarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class BracketAction:
    symbol: str
    kind: Literal["stop", "timeout"]
    qty: int


@dataclass
class _Bracket:
    qty: int
    entry_px: float
    target_px: float
    stop_px: float
    deadline: datetime


class BracketBook:
    def __init__(self) -> None:
        self._armed: dict[str, _Bracket] = {}

    def arm(self, symbol: str, qty: int, entry_px: float, target_px: float,
            stop_px: float, deadline: datetime) -> None:
        """Arm (or re-arm) a symbol's stop+timeout bracket. Re-arming a symbol
        replaces its existing bracket entirely."""
        self._armed[symbol] = _Bracket(qty=qty, entry_px=entry_px,
                                        target_px=target_px, stop_px=stop_px,
                                        deadline=deadline)

    def disarm(self, symbol: str) -> None:
        """Remove a symbol's bracket. Idempotent — no error if not armed."""
        self._armed.pop(symbol, None)

    def on_target_fill(self, symbol: str) -> None:
        """Alias for disarm, kept for call-site clarity: the resting target
        limit filled, so this scalp's protective legs no longer apply."""
        self.disarm(symbol)

    def check(self, now: datetime,
              marks: dict[str, tuple[float, float]]) -> list[BracketAction]:
        """Return the bracket actions triggered as of `now`.

        marks maps symbol -> (bid, ask). For each armed symbol PRESENT in marks:
          - stop fires when bid <= stop_px;
          - timeout fires when now >= deadline;
          - stop takes precedence if both would fire;
          - a fired bracket is removed before returning (never double-fires).
        Symbols missing from marks are skipped: stale data must not fire stops.
        """
        actions: list[BracketAction] = []
        for symbol, br in list(self._armed.items()):
            quote = marks.get(symbol)
            if quote is None:
                continue  # no fresh mark -> do not fire on stale data
            bid, _ask = quote
            if bid <= br.stop_px:
                actions.append(BracketAction(symbol=symbol, kind="stop", qty=br.qty))
                del self._armed[symbol]
            elif now >= br.deadline:
                actions.append(BracketAction(symbol=symbol, kind="timeout", qty=br.qty))
                del self._armed[symbol]
        return actions

    @property
    def armed(self) -> dict[str, _Bracket]:
        """Copy of the armed brackets, for inspection (not mutation)."""
        return dict(self._armed)
