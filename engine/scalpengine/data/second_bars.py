"""Streaming 1-second bar builder — the live twin of research corpus.second_bars.

Parity with the research pipeline is the design requirement (train/serve
skew on bar semantics would silently invalidate the model): a bar labeled
T covers (T-1s, T]; OHLC/vwap come from trades only (NaN when no trades,
volume/n_trades 0); NBBO is the last valid quote at or before T, forward
-filled at most QUOTE_STALE_LIMIT_S seconds and never from before the
grid start (corpus drops pre-grid quotes on reindex — mirrored here);
crossed/degenerate quotes are dropped at ingest; only bars whose close
label falls inside RTH are emitted, though premarket events still advance
quote state exactly like the corpus full-grid ffill does.

Streaming-only concession: a bar is finalized `grace_s` after its close
label; trades arriving later than that are dropped and counted in
`late_events` (the research pipeline has perfect hindsight, a live stream
does not — a finalized bar is never revised).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

NY = "America/New_York"
QUOTE_STALE_LIMIT_S = 60          # keep in lockstep with research corpus
RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)
ONE_S = pd.Timedelta(seconds=1)

COLUMNS = ["open", "high", "low", "close", "volume", "n_trades", "vwap",
           "bid", "ask", "bid_size", "ask_size", "spread"]


@dataclass
class _SymbolState:
    grid_start: pd.Timestamp | None = None
    next_emit: pd.Timestamp | None = None
    # pending per-second trade accumulators: label -> [o, h, l, c, vol, notional, n]
    acc: dict[pd.Timestamp, list[float]] = field(default_factory=dict)
    # last valid quote per second, oldest left: (label, bid, ask, bsz, asz)
    quotes: deque = field(default_factory=deque)
    rows: deque = field(default_factory=deque)          # (label, row list)
    late_events: int = 0


class SecondBarBuilder:
    def __init__(self, window_s: int = 900, grace_s: float = 0.25) -> None:
        self.window_s = window_s
        self.grace = pd.Timedelta(seconds=grace_s)
        self._sym: dict[str, _SymbolState] = {}

    def _state(self, symbol: str) -> _SymbolState:
        st = self._sym.get(symbol)
        if st is None:
            st = self._sym[symbol] = _SymbolState()
            st.rows = deque(maxlen=self.window_s)
        return st

    def add_trade(self, symbol: str, price: float, size: float,
                  ts: datetime) -> None:
        label = pd.Timestamp(ts).ceil("1s")
        st = self._state(symbol)
        if st.grid_start is None:
            st.grid_start = label
            st.next_emit = label
        if st.next_emit is not None and label < st.next_emit:
            st.late_events += 1        # bar already finalized: never revise
            return
        a = st.acc.get(label)
        if a is None:
            st.acc[label] = [price, price, price, price, size, price * size, 1]
        else:
            a[1] = max(a[1], price)
            a[2] = min(a[2], price)
            a[3] = price
            a[4] += size
            a[5] += price * size
            a[6] += 1

    def add_quote(self, symbol: str, bid: float, ask: float, bid_size: float,
                  ask_size: float, ts: datetime) -> None:
        if not (bid > 0 and ask > 0 and ask >= bid):
            return                     # crossed/degenerate: same drop as corpus
        label = pd.Timestamp(ts).ceil("1s")
        st = self._state(symbol)
        q = st.quotes
        if q and label < q[-1][0]:
            return                     # out-of-order older quote
        snap = (label, float(bid), float(ask), float(bid_size), float(ask_size))
        if q and label == q[-1][0]:
            q[-1] = snap                                    # last-in-second wins
        else:
            q.append(snap)

    def _nbbo(self, st: _SymbolState, label: pd.Timestamp,
              ) -> tuple[float, float, float, float]:
        oldest_ok = label - pd.Timedelta(seconds=QUOTE_STALE_LIMIT_S)
        for qlabel, bid, ask, bsz, asz in reversed(st.quotes):
            if qlabel > label:
                continue
            if qlabel < oldest_ok or (st.grid_start is not None
                                      and qlabel < st.grid_start):
                break                  # stale, or pre-grid (corpus drops these)
            return bid, ask, bsz, asz
        return np.nan, np.nan, np.nan, np.nan

    def poll(self, now: datetime) -> list[tuple[str, pd.Timestamp, dict]]:
        """Finalize and return every bar whose close label + grace <= now.

        Bars are appended to the rolling frame in label order; non-RTH bars
        advance state but are neither returned nor kept.
        """
        cutoff = pd.Timestamp(now) - self.grace
        out: list[tuple[str, pd.Timestamp, dict]] = []
        for symbol, st in self._sym.items():
            while st.next_emit is not None and st.next_emit <= cutoff:
                label = st.next_emit
                st.next_emit = label + ONE_S
                a = st.acc.pop(label, None)
                if a is None:
                    o = h = lo = c = vwap = np.nan
                    vol = n = 0.0
                else:
                    o, h, lo, c, vol, notional, n = a
                    vwap = notional / vol if vol > 0 else np.nan
                bid, ask, bsz, asz = self._nbbo(st, label)
                # prune quote history that can never be used again
                horizon = label - pd.Timedelta(seconds=QUOTE_STALE_LIMIT_S + 1)
                while st.quotes and st.quotes[0][0] < horizon:
                    st.quotes.popleft()
                t_local = label.tz_convert(NY).time()
                if not (RTH_START <= t_local < RTH_END):
                    continue
                row = [o, h, lo, c, float(vol), float(n), vwap,
                       bid, ask, bsz, asz, ask - bid]
                st.rows.append((label, row))
                out.append((symbol, label, dict(zip(COLUMNS, row, strict=True))))
        return out

    def get_frame(self, symbol: str) -> pd.DataFrame:
        """Rolling RTH frame for `symbol`, column-identical to the corpus."""
        st = self._sym.get(symbol)
        if st is None or not st.rows:
            return pd.DataFrame(columns=COLUMNS)
        idx = pd.DatetimeIndex([r[0] for r in st.rows])
        return pd.DataFrame([r[1] for r in st.rows], index=idx, columns=COLUMNS)

    def late_events(self, symbol: str) -> int:
        st = self._sym.get(symbol)
        return st.late_events if st else 0

    def reset(self, symbol: str | None = None) -> None:
        """Drop state for one symbol (or all) — day roll / slot reassignment."""
        if symbol is None:
            self._sym.clear()
        else:
            self._sym.pop(symbol, None)
