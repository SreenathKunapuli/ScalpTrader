"""The engine: asyncio loop wiring data -> signals -> risk -> execution.

Task graph (§5.6):
  stream consumer -> BarBuilder(1m) -> aggregate(5m) -> on tier cadence:
  signals -> ensemble -> target portfolio -> diff -> intents -> RiskManager
  -> OrderManager. Independent tasks: stop monitor, staleness monitor, EOD
  flattener, heartbeat persister, command poller (API control channel).

Testability: the engine takes its broker/stream via constructor injection;
the replay test drives `on_minute_bar` directly with a MockBroker.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog

from .config.scalp_tiers import ScalpConfig
from .config.settings import Settings
from .config.tiers import TIERS, Tier, TierConfig
from .data import calendar
from .data.bar_builder import Bar, BarBuilder, aggregate
from .data.second_bars import SecondBarBuilder
from .data.staleness import QuoteStalenessTracker
from .execution.brackets import BracketAction, BracketBook
from .execution.order_manager import OrderManager, is_target_coid
from .persistence.repo import Repo
from .pubsub import PubSub
from .risk.kill_switch import KillSwitch
from .risk.risk_manager import OrderIntent, Rejection, RiskManager
from .risk.sizing import size_position
from .risk.state import PortfolioState
from .signals.ensemble import Ensemble
from .signals.health import SignalHealthTracker

log = structlog.get_logger()

ATR_PERIOD = 14
MAX_5M_BARS = 2400  # ~1 month of 5-min bars kept in memory per symbol


def atr_from_bars(bars: list[Bar], period: int = ATR_PERIOD) -> float:
    if len(bars) < 2:
        return 0.0
    highs = np.array([b.high for b in bars[-(period + 1):]])
    lows = np.array([b.low for b in bars[-(period + 1):]])
    closes = np.array([b.close for b in bars[-(period + 1):]])
    prev = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))
    return float(tr.mean())


class Engine:
    def __init__(self, settings: Settings, tier: TierConfig, repo: Repo,
                 order_manager: OrderManager, ensemble: Ensemble,
                 state: PortfolioState, pubsub: PubSub,
                 scalp_cfg: ScalpConfig | None = None) -> None:
        self.settings = settings
        self.tier = tier
        self.repo = repo
        self.om = order_manager
        self.ensemble = ensemble
        self.state = state
        self.pubsub = pubsub
        self.scalp_cfg = scalp_cfg
        self.risk = RiskManager(tier, state, scalp_cfg=scalp_cfg)
        self.kill = KillSwitch(state, tier, repo, order_manager, pubsub_emit(pubsub),
                               staleness_kill_s=settings.staleness_kill_s,
                               broker_error_count=settings.broker_error_kill_count,
                               broker_error_window_s=settings.broker_error_kill_window_s,
                               flatten_intraday=self._flatten_intraday,
                               min_equity_usd=settings.min_equity_halt_usd or None)
        self.builder = BarBuilder(interval_s=settings.bar_interval_s)
        self.bars_1m: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=12000))
        self.bars_5m: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=MAX_5M_BARS))
        self._pending_1m: dict[str, list[Bar]] = defaultdict(list)
        # microstructure accumulators drained into each incoming stream bar
        self._spread_acc: dict[str, list[float]] = defaultdict(list)
        self._qimb_acc: dict[str, list[float]] = defaultdict(list)
        self._flow_acc: dict[str, float] = defaultdict(float)
        self._flow_tot: dict[str, float] = defaultdict(float)
        self._last_px: dict[str, float] = {}
        self._last_quote: dict[str, tuple[float, float]] = {}
        self._last_rebalance = datetime.min.replace(tzinfo=UTC)
        self.health = SignalHealthTracker(repo, list(ensemble.signals))
        self._paused_stale = False
        # starts as the frozen tier universe; day_scanner appends to this each session
        self._live_universe: list[str] = list(tier.universe)
        # stop prices staged at order-submit time, applied on fill (avoids async race)
        self._pending_stops: dict[str, float] = {}
        # scalp brackets: stop/timeout legs tracked in-process (target rests at broker)
        self.brackets = BracketBook()
        # bracket params staged at entry-submit time, armed on FILL (no bracket
        # before shares exist): symbol -> (qty, target_px, stop_px, deadline)
        self._pending_brackets: dict[str, tuple[int, float, float, datetime]] = {}
        # second-cadence scalp path: builder always exists; the model is wired
        # by cli when scalp_artifact_dir is configured (duck-typed: needs
        # .threshold and .compute_second(symbol, frame) -> ScalpDecision|None)
        # full-RTH window: features like vwap_dist/sess_hi_dist are
        # session-cumulative — truncating the frame would be train/serve skew
        self.second_bars = SecondBarBuilder(window_s=23400)
        self.scalp_signal: Any | None = None
        # SCALP_PROFILE=auto: cli sets this; day roll re-picks the guardrail
        # preset from actual equity so account growth upgrades the band
        self.scalp_auto = False
        # quote-staleness instrumentation (Phase 5)
        self.staleness = QuoteStalenessTracker(
            pause_s=float(settings.staleness_pause_s),
            kill_s=float(settings.staleness_kill_s),
        )

    # ---------------- data path ---------------- #
    def warmup(self, history: dict[str, list[Bar]]) -> None:
        """Seed bar caches from REST backfill (1-min bars)."""
        for sym, bars in history.items():
            self.bars_1m[sym].extend(bars)
            group: list[Bar] = []
            for b in bars:
                group.append(b)
                if len(group) == 5:
                    agg = aggregate(group, 300)
                    if agg:
                        self.bars_5m[sym].append(agg)
                    group = []
        log.info("warmup.done", symbols=len(history))

    async def on_trade(self, symbol: str, ts: datetime, price: float, size: int) -> None:
        self.state.last_data_ts = datetime.now(UTC)
        bid, ask = self._last_quote.get(symbol, (0.0, 0.0))
        last = self._last_px.get(symbol)
        sign = 0
        if ask > 0 and price >= ask:
            sign = 1
        elif bid > 0 and price <= bid:
            sign = -1
        elif last is not None and price != last:
            sign = 1 if price > last else -1
        self._flow_acc[symbol] += sign * size
        self._flow_tot[symbol] += size
        self._last_px[symbol] = price
        if self.scalp_signal is not None:
            self.second_bars.add_trade(symbol, price, size, ts)
        pos = self.state.positions.get(symbol)
        if pos:
            pos.mark = price

    async def on_quote(self, symbol: str, ts: datetime, bid: float, bid_sz: int,
                       ask: float, ask_sz: int) -> None:
        now = datetime.now(UTC)
        self.state.last_data_ts = now
        self.staleness.record(symbol, quote_ts=ts, recv_ts=now)
        self._last_quote[symbol] = (bid, ask)
        if bid > 0 and ask > bid:
            self._spread_acc[symbol].append(ask - bid)
        denom = bid_sz + ask_sz
        if denom > 0:
            self._qimb_acc[symbol].append((bid_sz - ask_sz) / denom)
        if self.scalp_signal is not None:
            self.second_bars.add_quote(symbol, bid, ask, bid_sz, ask_sz, ts)
        # scalp fast path: per-tick stop/timeout check. Only this symbol's
        # just-received quote is passed — never a stale mark — and an invalid
        # quote (zero/crossed) must not fire a stop.
        if symbol in self.brackets.armed and bid > 0 and ask >= bid:
            now = datetime.now(UTC)
            for action in self.brackets.check(now, {symbol: (bid, ask)}):
                await self._fire_bracket_exit(action, bid, ask, now)

    async def on_stream_bar(self, symbol: str, ts: datetime, o: float, h: float,
                            low: float, c: float, vol: int, vwap: float,
                            tcount: int) -> None:
        """Official 1-min bar from Alpaca, enriched with accumulated
        quote/trade microstructure (zeros where we hold no subscription)."""
        self.state.last_data_ts = datetime.now(UTC)
        spreads = self._spread_acc.pop(symbol, [])
        qimbs = self._qimb_acc.pop(symbol, [])
        flow = self._flow_acc.pop(symbol, 0.0)
        tot = self._flow_tot.pop(symbol, 0.0)
        bar = Bar(
            symbol=symbol, ts=ts, interval_s=60, open=o, high=h, low=low,
            close=c, volume=vol, vwap=vwap, trade_count=tcount,
            mean_spread=sum(spreads) / len(spreads) if spreads else 0.0,
            mean_quote_imbalance=sum(qimbs) / len(qimbs) if qimbs else 0.0,
            flow_imbalance=flow / tot if tot > 0 else 0.0,
        )
        pos = self.state.positions.get(symbol)
        if pos:
            pos.mark = c
        await self.on_minute_bar(bar)

    async def on_minute_bar(self, bar: Bar) -> None:
        """Finalized 1-min bar: cache, aggregate to 5-min, maybe act."""
        self.bars_1m[bar.symbol].append(bar)
        pend = self._pending_1m[bar.symbol]
        pend.append(bar)
        if len(pend) >= 5:
            five = aggregate(pend[:5], 300)
            del pend[:5]
            if five:
                self.bars_5m[bar.symbol].append(five)
                await self.on_five_min_bar(five)

    async def on_five_min_bar(self, bar: Bar) -> None:
        await self.check_stops(bar)
        trig = self.kill.check_triggers()
        if trig:
            scope, reason = trig
            if scope == "account" and not self.state.halted:
                await self.kill.fire(reason)
                return
            if scope == "intraday" and not self.state.intraday_halted:
                await self.kill.fire_intraday(reason)
        now = datetime.now(UTC)
        if (now - self._last_rebalance).total_seconds() >= self.tier.rebalance_seconds:
            self._last_rebalance = now
            await self.rebalance(now)

    # ---------------- decision path ---------------- #
    async def rebalance(self, now: datetime) -> None:
        if self.state.halted or self.state.intraday_halted or self._paused_stale:
            return
        if not calendar.in_entry_window(now):
            return

        # Advance trailing stops before computing new signals.
        # Intraday longs: only trail when a stop is already set (pending_stops
        # covers the gap between submit and fill for brand-new positions).
        mult = self.tier.stop_atr_multiple
        for sym, pos in list(self.state.positions.items()):
            if pos.qty <= 0:
                continue
            sym_bars = list(self.bars_5m.get(sym, ()))
            if len(sym_bars) < 30:
                continue
            atr = atr_from_bars(sym_bars)
            price = sym_bars[-1].close
            trailing = price - mult * atr
            if pos.book == "intraday" and pos.stop_price is not None:
                if trailing > pos.stop_price:
                    pos.stop_price = trailing
                    log.info("stop.trail", symbol=sym,
                             stop=round(trailing, 2), price=round(price, 2))

        for symbol in self._live_universe:
            bars = list(self.bars_5m.get(symbol, ()))
            if len(bars) < 30:
                continue
            res = self.ensemble.compute(symbol, bars, self.tier)
            self.repo.add_signal(now, symbol, res.per_signal, res.final_score)
            await self.pubsub.publish("signals", {
                "symbol": symbol, "ensemble": res.final_score,
                "per_signal": res.per_signal, "ts": now.isoformat()})
            if self.scalp_signal is not None:
                # Scalp-only book: the minute-bar ensemble is telemetry, never
                # a trader. Live 2026-07-20: it re-pegged into a runaway ADVB
                # spread twice (-$6.78, 85% of day loss) on a z-score signal
                # never validated for runners. Entries AND signal-exits are cut
                # (a signal-exit would fight the bracket that owns each scalp).
                continue
            if not res.is_candidate(self.tier):
                # candidate exit: existing position whose signal died
                await self._maybe_exit_on_signal(symbol, bars, res.final_score)
                continue
            await self._enter_or_adjust(symbol, bars, res, now)

    async def _maybe_exit_on_signal(self, symbol: str, bars: list[Bar],
                                    score: float) -> None:
        """Exit a position when the ensemble is no longer confident enough to hold it.

        Called only from the `not is_candidate` branch of rebalance(), so score is
        already below the confidence threshold. We exit on that alone — a weakening
        but still-positive signal (e.g. 0.3 with threshold 0.45) is not worth holding
        because the same gate that blocked entry now blocks continued holding.
        """
        pos = self.state.positions.get(symbol)
        if not pos:
            return
        threshold = self.tier.confidence_threshold
        if pos.book == "intraday":
            # Exit longs when not confidently bullish; shorts when not confidently bearish
            if (pos.qty > 0 and score < threshold) or (pos.qty < 0 and score > -threshold):
                await self._exit_position(symbol, "signal")

    async def _enter_or_adjust(self, symbol: str, bars: list[Bar],
                               res: Any, now: datetime) -> None:
        price = bars[-1].close
        atr = atr_from_bars(bars)
        target_qty = size_position(self.tier, self.state.equity, price, atr, res.vol_mult)
        if res.final_score < 0:
            target_qty = -target_qty if self.tier.allow_short else 0

        pos = self.state.positions.get(symbol)
        cur_qty = pos.qty if pos else 0

        delta = target_qty - cur_qty
        if delta == 0 or (target_qty == 0 and cur_qty == 0):
            return
        side: Literal["buy", "sell"] = "buy" if delta > 0 else "sell"

        intent = OrderIntent(symbol=symbol, side=side, qty=abs(delta),
                             price_hint=price, reason="signal")
        approval = self.risk.approve(intent, now)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            log.info("risk.rejected", symbol=symbol, reason=approval.reason)
            return
        spread = bars[-1].mean_spread
        try:
            order = await self.om.submit(approval, bars[-1].ts, mid=price, spread=spread)
        except Exception as exc:
            log.error("order.submit_failed", symbol=symbol, error=str(exc))
            if self.kill.record_broker_error():
                await self.kill.fire("5 consecutive broker errors in 60s")
            return
        if order:
            stop_mult = self.tier.stop_atr_multiple
            stop = price - stop_mult * atr if delta > 0 else price + stop_mult * atr
            p = self.state.positions.get(symbol)
            if p:
                p.stop_price = stop
            self._pending_stops[symbol] = stop
            if self.scalp_cfg is not None and side == "buy" and target_qty > 0:
                # Stage the bracket now; it is armed on FILL (cli fill handler)
                # so no bracket exists before shares do.
                sc = self.scalp_cfg
                self._pending_brackets[symbol] = (
                    abs(delta), price + sc.target_ps, price - sc.stop_ps,
                    now + timedelta(seconds=sc.timeout_s))
            await self.pubsub.publish("orders", {
                "symbol": symbol, "side": side, "qty": abs(delta),
                "ts": now.isoformat(), "reason": "signal"})

    async def _exit_position(self, symbol: str, reason: str, target_qty: int = 0) -> None:
        """Exit to `target_qty` (default 0 = full close). Positive target keeps
        that many shares — used when a partial close is required."""
        pos = self.state.positions.get(symbol)
        if not pos or pos.qty == 0:
            return
        close_qty = pos.qty - target_qty
        if close_qty == 0:
            return
        # A signal/eod/kill exit supersedes an armed bracket: disarm it and
        # cancel the resting take-profit leg, or that -tgt limit would sell
        # shares this exit is about to sell (double-sell -> short).
        if symbol in self.brackets.armed:
            self.brackets.disarm(symbol)
            await self._cancel_resting_target(symbol)
        side: Literal["buy", "sell"] = "sell" if close_qty > 0 else "buy"
        intent = OrderIntent(symbol=symbol, side=side, qty=abs(close_qty),
                             price_hint=pos.mark, reason=reason)
        approval = self.risk.approve_exit(intent)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            return
        try:
            await self.om.submit(approval, datetime.now(UTC),
                                 mid=pos.mark, spread=0.0)
        except Exception as exc:
            log.error("exit.submit_failed", symbol=symbol, error=str(exc))

    async def _fire_bracket_exit(self, action: BracketAction, bid: float,
                                 ask: float, now: datetime) -> None:
        """Fire a bracket stop/timeout exit: cancel the resting take-profit
        (-tgt) leg, then submit an urgent sell. BracketBook.check already
        disarmed the bracket, so this fires at most once per arm."""
        log.info("bracket.fired", symbol=action.symbol, kind=action.kind,
                 qty=action.qty, bid=bid)
        intent = OrderIntent(symbol=action.symbol, side="sell", qty=action.qty,
                             price_hint=bid, reason=action.kind)
        approval = self.risk.approve_exit(intent)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            log.info("bracket.exit_rejected", symbol=action.symbol,
                     reason=approval.reason)
            return
        # cancel the resting target first so both legs can't fill
        await self._cancel_resting_target(action.symbol)
        try:
            await self.om.submit(approval, now, mid=(bid + ask) / 2,
                                 spread=ask - bid)
        except Exception as exc:
            log.error("bracket.exit_submit_failed", symbol=action.symbol,
                      error=str(exc))

    async def _cancel_resting_target(self, symbol: str) -> None:
        """Best-effort cancel of a symbol's resting take-profit (-tgt) limit.
        Failure is logged, never raised: an unfilled sell limit above market
        is far less dangerous than skipping the exit that follows."""
        try:
            for o in await self.om._broker.get_open_orders():
                if o.symbol == symbol and is_target_coid(o.client_order_id):
                    await self.om._broker.cancel_order(o.id)
                    log.info("bracket.target_cancelled", symbol=symbol,
                             coid=o.client_order_id)
        except Exception as exc:
            log.error("bracket.target_cancel_failed", symbol=symbol,
                      error=str(exc))

    async def scalp_loop(self) -> None:
        """Second-cadence decision loop: poll finalized 1s bars and let the
        scalp model strike. Inert unless cli wired a model artifact."""
        if self.scalp_signal is None or self.scalp_cfg is None:
            return
        # per-symbol eval telemetry: [evals, max_p, n_ge_thr, n_qty0], flushed
        # every 60s — the decision path's silent exits (p<thr, qty=0) were
        # invisible on 2026-07-20 and cost a day of debugging
        self._scalp_stats: dict[str, list[float]] = {}
        last_flush = datetime.now(UTC)
        while True:
            await asyncio.sleep(0.25)
            now = datetime.now(UTC)
            if (now - last_flush).total_seconds() >= 60 and self._scalp_stats:
                log.info("scalp.telemetry", window_s=60, stats={
                    s: {"evals": int(v[0]), "max_p": round(v[1], 3),
                        "ge_thr": int(v[2]), "qty0": int(v[3])}
                    for s, v in sorted(self._scalp_stats.items())})
                self._scalp_stats = {}
                last_flush = now
            finalized = self.second_bars.poll(now)
            if not finalized:
                continue
            if self.state.halted or self.state.intraday_halted \
                    or self._paused_stale or not calendar.in_entry_window(now):
                continue
            for symbol in {s for s, _, _ in finalized}:
                try:
                    await self._maybe_scalp(symbol, now)
                except Exception as exc:  # decision errors must not kill the loop
                    log.error("scalp.decision_failed", symbol=symbol,
                              error=str(exc))

    async def _maybe_scalp(self, symbol: str, now: datetime) -> None:
        """One scalp decision on `symbol`'s latest finalized second bar."""
        pos = self.state.positions.get(symbol)
        if (pos and pos.qty != 0) or symbol in self._pending_brackets:
            return  # one scalp per symbol; entry already working
        frame = self.second_bars.get_frame(symbol)
        dec = self.scalp_signal.compute_second(symbol, frame)
        stats = getattr(self, "_scalp_stats", None)
        if stats is not None and dec is not None:
            st = stats.setdefault(symbol, [0, 0.0, 0, 0])
            st[0] += 1
            st[1] = max(st[1], dec.p_win)
            if dec.p_win >= self.scalp_signal.threshold:
                st[2] += 1
        if dec is None or dec.p_win < self.scalp_signal.threshold:
            return
        last = frame.iloc[-1]
        price = float(last["ask"])
        # research sizing head; scalp_gbt's import already put research on
        # sys.path (this method only runs when a model is wired)
        from scalp.sizing import size_scalp
        qty = size_scalp(dec.p_win, dec.target_ps, dec.stop_ps, price,
                         self.state.equity,
                         float(frame["volume"].iloc[-60:].sum()),
                         float(last["ask_size"]) if pd.notna(last["ask_size"])
                         else 0.0,
                         self.scalp_cfg)
        if qty <= 0:
            # above-threshold signal zeroed by the sizing box — rare and worth
            # a line each time (participation/depth caps starve on thin tape)
            if stats is not None:
                stats.setdefault(symbol, [0, 0.0, 0, 0])[3] += 1
            log.info("scalp.qty_zero", symbol=symbol, p_win=round(dec.p_win, 3),
                     px=price, vol_60s=float(frame["volume"].iloc[-60:].sum()),
                     ask_size=float(last["ask_size"])
                     if pd.notna(last["ask_size"]) else 0.0)
            return
        intent = OrderIntent(symbol=symbol, side="buy", qty=qty,
                             price_hint=price, reason="scalp")
        approval = self.risk.approve(intent, now)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            log.info("scalp.rejected", symbol=symbol, reason=approval.reason)
            return
        bid = float(last["bid"])
        try:
            order = await self.om.submit(approval, frame.index[-1],
                                         mid=(bid + price) / 2,
                                         spread=price - bid)
        except Exception as exc:
            log.error("scalp.submit_failed", symbol=symbol, error=str(exc))
            if self.kill.record_broker_error():
                await self.kill.fire("5 consecutive broker errors in 60s")
            return
        if order:
            # bracket stop uses the EXECUTION distance (disaster-only per the
            # sim study); the label stop above priced the Kelly loss leg
            stop_ps = getattr(dec, "bracket_stop_ps", dec.stop_ps)
            self._pending_brackets[symbol] = (
                qty, price + dec.target_ps, price - stop_ps,
                now + timedelta(seconds=dec.timeout_s))
            log.info("scalp.entry", symbol=symbol, qty=qty, px=price,
                     p_win=round(dec.p_win, 3),
                     tgt=round(dec.target_ps, 4), stp=round(dec.stop_ps, 4))
            await self.pubsub.publish("orders", {
                "symbol": symbol, "side": "buy", "qty": qty,
                "ts": now.isoformat(), "reason": "scalp"})

    async def _flatten_intraday(self, reason: str) -> None:
        """Graceful exit (urgent limit -> market) of every intraday-book
        position. Kill switch's intraday path."""
        for sym in list(self.state.positions):
            await self._exit_position(sym, "kill")

    async def check_stops(self, bar: Bar) -> None:
        if bar.symbol in self.brackets.armed:
            return  # an armed bracket owns the exit (fast path); no double-fire
        pos = self.state.positions.get(bar.symbol)
        if not pos or pos.stop_price is None:
            return
        hit = (pos.qty > 0 and bar.low <= pos.stop_price) or \
              (pos.qty < 0 and bar.high >= pos.stop_price)
        if hit:
            log.info("stop.hit", symbol=bar.symbol, stop=pos.stop_price)
            await self._exit_position(bar.symbol, "stop")

    # ---------------- background tasks ---------------- #
    async def staleness_monitor(self) -> None:
        while True:
            await asyncio.sleep(5)
            now = datetime.now(UTC)
            if not calendar.is_session_open(now) or self.state.halted:
                continue
            stale = (now - self.state.last_data_ts).total_seconds()
            if stale > self.settings.staleness_kill_s:
                await self.kill.fire_intraday(f"data staleness {stale:.0f}s")
            elif stale > self.settings.staleness_pause_s and not self._paused_stale:
                self._paused_stale = True
                self.repo.update_state(status="PAUSED")
                await self.pubsub.publish("engine_status",
                                          {"status": "PAUSED", "reason": "stale data"})
            elif stale <= self.settings.staleness_pause_s and self._paused_stale:
                self._paused_stale = False
                self.repo.update_state(status="RUNNING")
                await self.pubsub.publish("engine_status", {"status": "RUNNING"})

    async def eod_flattener(self) -> None:
        while True:
            await asyncio.sleep(20)
            now = datetime.now(UTC)
            if self.state.halted:
                continue
            if not calendar.in_eod_flatten_window(now):
                continue
            open_positions = [(s, p) for s, p in self.state.positions.items()
                              if p.qty != 0]
            if not open_positions:
                continue
            log.info("eod.flatten", n=len(open_positions))
            for sym, _pos in open_positions:
                await self._exit_position(sym, "eod")

    async def heartbeat(self) -> None:
        last_beat = datetime.now(UTC)
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval_s)
            now = datetime.now(UTC)
            gap = (now - last_beat).total_seconds()
            last_beat = now
            if gap > 120:  # machine slept: local mirror is suspect
                log.warning("wake.detected", slept_s=int(gap))
                try:
                    await self.om.reconcile_state()
                except Exception as exc:
                    log.error("wake.reconcile_failed", error=str(exc))
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
            def _pos_dict(p, now=now):
                return {
                    "symbol": p.symbol,
                    "side": "long" if p.qty > 0 else "short",
                    "qty": p.qty,
                    "book": p.book,
                    "entry": p.entry_price,
                    "mark": p.mark,
                    "market_value": round(p.market_value, 2),
                    "upnl": round(p.unrealized_pnl, 2),
                    "stop": p.stop_price,
                    "age_s": (now - p.entry_ts).total_seconds() if p.entry_ts else None,
                    "entry_signals": p.entry_signals,
                }
            pos_payload = [_pos_dict(p) for p in self.state.positions.values()]
            staleness_payload = self.staleness.snapshot(now)
            self.repo.update_state(heartbeat_ts=now, last_data_ts=self.state.last_data_ts,
                                   peak_equity=self.state.peak_equity,
                                   positions_json=pos_payload,
                                   staleness_json=staleness_payload)
            self.repo.add_equity_snapshot(now, self.state.equity, self.state.cash,
                                          self.state.gross_exposure)
            await self.pubsub.publish("equity", {
                "ts": now.isoformat(), "equity": self.state.equity,
                "cash": self.state.cash, "gross": self.state.gross_exposure})
            await self.pubsub.publish("positions", {"positions": pos_payload})
            await self.pubsub.publish("staleness", staleness_payload)

    async def command_poller(self) -> None:
        """API -> engine control channel (kill / reset / set_tier)."""
        while True:
            await asyncio.sleep(2)
            for cmd in self.repo.pending_commands():
                log.info("command.received", command=cmd.command)
                if cmd.command == "kill":
                    await self.kill.fire("manual kill via API")
                elif cmd.command == "reset":
                    self.state.halted = False
                    self.state.halted_reason = ""
                    self.repo.update_state(status="RUNNING", halted_reason="")
                    await self.pubsub.publish("engine_status", {"status": "RUNNING"})
                elif cmd.command == "reset_intraday":
                    self.state.intraday_halted = False
                    self.state.last_data_ts = datetime.now(UTC)
                    self.repo.update_state(status="RUNNING", halted_reason="")
                    await self.pubsub.publish("engine_status", {"status": "RUNNING"})
                elif cmd.command == "set_tier":
                    tier_name = str(cmd.payload_json.get("tier", "")).lower()
                    if tier_name in [t.value for t in Tier] and not self.state.halted:
                        self.tier = TIERS[Tier(tier_name)]
                        self.risk.tier = self.tier
                        self.kill.tier = self.tier
                        self.repo.update_state(tier=tier_name)
                self.repo.mark_command_done(cmd.id)

    def expand_universe(self, symbols: list[str]) -> None:
        """Add screener-sourced symbols to the live universe for this session.

        Caller (cli.day_scanner) must have already warmed up bar history via
        engine.warmup() before calling this so rebalance() has enough bars to
        score them immediately.
        """
        new = [s for s in symbols if s not in self._live_universe]
        if not new:
            return
        self._live_universe.extend(new)
        self.risk.add_to_universe(new)
        log.info("universe.expanded", added=new, total=len(self._live_universe))

    def rotate_universe(self, add: list[str], evict: list[str]) -> None:
        """Swap stale idle candidates out and fresh ones in.

        Only evicts symbols that have no active position and are not in the
        hardcoded tier universe — held positions are never touched (we need
        to keep monitoring them for exits).
        """
        protected = set(self.tier.universe) | set(self.state.positions)
        actual_evict = [s for s in evict if s not in protected]
        if actual_evict:
            self._live_universe = [s for s in self._live_universe
                                   if s not in actual_evict]
            for s in actual_evict:
                self.risk._dynamic_universe.discard(s)
            log.info("universe.evicted", removed=actual_evict,
                     total=len(self._live_universe))
        new = [s for s in add if s not in self._live_universe]
        if new:
            self._live_universe.extend(new)
            self.risk.add_to_universe(new)
            log.info("universe.rotated_in", added=new, total=len(self._live_universe))

    async def day_roll(self) -> None:
        """Reset day-start equity at each session open."""
        last_day = None
        while True:
            await asyncio.sleep(30)
            now = datetime.now(UTC)
            if calendar.is_session_open(now) and last_day != now.date():
                last_day = now.date()
                self._roll_day(now)

    def _roll_day(self, now: datetime) -> None:
        """Once-per-session-day resets (extracted from day_roll's loop for
        testability)."""
        self.state.day_start_equity = self.state.equity
        self.state.intraday_realized_today = 0.0
        self.state.symbol_realized_today.clear()  # per-symbol scalp loss caps
        if self.state.intraday_halted:  # day-scoped halt: new-day amnesty
            self.state.intraday_halted = False
            self.repo.update_state(status="RUNNING", halted_reason="")
            log.info("intraday_halt.cleared")
        self.repo.update_state(day_start_equity=self.state.equity)
        if self.scalp_auto and self.scalp_cfg is not None:
            from .config.scalp_tiers import profile_for_equity

            new_cfg = profile_for_equity(self.state.equity)
            if new_cfg.name != self.scalp_cfg.name:
                log.info("scalp.auto_profile_changed", old=self.scalp_cfg.name,
                         new=new_cfg.name, equity=self.state.equity)
                self.scalp_cfg = new_cfg
                self.risk.scalp_cfg = new_cfg
        # reset dynamic universe — yesterday's movers don't carry over
        self._live_universe = list(self.tier.universe)
        self.risk.reset_dynamic_universe()
        # shadow evaluation: refresh per-signal health multipliers
        self.ensemble.health_multipliers = self.health.evaluate(now)
        log.info("day.roll", equity=self.state.equity,
                 health=self.ensemble.health_multipliers)


def pubsub_emit(ps: PubSub) -> Any:
    async def emit(channel: str, data: dict[str, Any]) -> None:
        await ps.publish(channel, data)
    return emit
