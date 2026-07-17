"""Quote-staleness instrumentation for the free IEX feed.

Phase 5's data-vendor decision must be made from evidence, not vibes: this
tracker measures, per focus symbol, how old the latest quote is and the
rolling distribution of inter-quote gaps. If p95 staleness on the names we
actually scalp stays inside the bracket-check tolerance, the free feed is
fine; if not, the numbers say exactly how much a paid feed would buy.

Pure logic, injected thresholds (defaults mirror Settings.staleness_pause_s
/ staleness_kill_s): the engine loop calls record() from on_quote and
classify() from its staleness monitor; the dashboard reads snapshot().
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

WINDOW_S = 300.0          # rolling window for gap percentiles


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class QuoteStalenessTracker:
    def __init__(self, pause_s: float = 60.0, kill_s: float = 180.0) -> None:
        assert kill_s >= pause_s > 0
        self.pause_s = pause_s
        self.kill_s = kill_s
        self._last: dict[str, datetime] = {}
        # (recv_ts, gap_s) samples per symbol, pruned to WINDOW_S on record
        self._gaps: dict[str, deque] = defaultdict(deque)

    def record(self, symbol: str, quote_ts: datetime, recv_ts: datetime) -> None:
        """One quote arrival. quote_ts = exchange stamp, recv_ts = local now;
        the gap series uses recv-to-recv spacing (what the engine actually
        experiences), while age() uses the exchange stamp."""
        prev = self._last.get(symbol)
        if prev is not None:
            gap = (quote_ts - prev).total_seconds()
            if gap >= 0:
                dq = self._gaps[symbol]
                dq.append(((recv_ts), gap))
                horizon = recv_ts.timestamp() - WINDOW_S
                while dq and dq[0][0].timestamp() < horizon:
                    dq.popleft()
        if prev is None or quote_ts >= prev:
            self._last[symbol] = quote_ts

    def age(self, symbol: str, now: datetime) -> float:
        """Seconds since `symbol`'s newest quote; +inf if never seen."""
        last = self._last.get(symbol)
        return (now - last).total_seconds() if last else float("inf")

    def gap_percentiles(self, symbol: str) -> tuple[float, float]:
        """(p50, p95) of inter-quote gaps in the rolling window."""
        vals = sorted(g for _, g in self._gaps.get(symbol, ()))
        return _pct(vals, 0.50), _pct(vals, 0.95)

    def classify(self, now: datetime, symbols: list[str] | None = None) -> str:
        """'ok' | 'pause' | 'kill' from the WORST tracked symbol's age.
        Symbols never seen are ignored (subscription may still be pending)."""
        pool = symbols if symbols is not None else list(self._last)
        worst = max((self.age(s, now) for s in pool if s in self._last),
                    default=0.0)
        if worst > self.kill_s:
            return "kill"
        if worst > self.pause_s:
            return "pause"
        return "ok"

    def snapshot(self, now: datetime) -> dict[str, dict]:
        """Dashboard payload: per-symbol age + gap percentiles."""
        out = {}
        for sym in self._last:
            p50, p95 = self.gap_percentiles(sym)
            out[sym] = {"age_s": round(self.age(sym, now), 3),
                        "gap_p50_s": round(p50, 3) if p50 == p50 else None,
                        "gap_p95_s": round(p95, 3) if p95 == p95 else None}
        return out
