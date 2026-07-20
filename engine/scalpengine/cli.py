"""scalpctl — engine entrypoint: run / halt / reset / status / flatten.

`run` wires real Alpaca clients; everything else talks through the DB
(commands table / engine_state), so it works whether or not the engine
process is up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

import structlog

from .config.settings import get_settings

if TYPE_CHECKING:
    from .persistence.repo import Repo
from .config.tiers import TIERS, Tier

structlog.configure(processors=[
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.add_log_level,
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()


def _repo() -> Repo:
    from .persistence.repo import Repo

    s = get_settings()
    return Repo(s.resolved_database_url())


async def _day_scanner(engine: "Engine", stream: "MarketStream",
                       settings: "Settings", tier: "TierConfig") -> None:
    """Continuously refresh the trading universe throughout the day.

    Phase 1 — morning scan (once per calendar day, fires at engine startup):
        Scores ~3 000 stocks by short-term daily momentum (universe3000.csv →
        sp500_constituents.csv fallback), warms up minute-bar history for the top
        candidates, and seeds the live universe before the first bar arrives.
        Reserves 8 stream slots for intraday screener picks.

    Phase 2 — intraday rescan (every 30 min during market hours):
        Calls the Alpaca screener for today's top gainers and most-active names.
        If new candidates appear that aren't already subscribed, idle (no-position)
        members of the universe are evicted in FIFO order to free slots, then new
        names are warmed up and added. This catches underdogs that break out mid-day
        on news, earnings, or sudden volume surges — not just pre-market leaders.

    Stream budget: SUBSCRIPTION_LIMIT = 30.
    Protected from eviction: tier base universe, any open position symbols.
    """
    import datetime as dt

    from .data import calendar
    from .data.alpaca_stream import SUBSCRIPTION_LIMIT
    from .data.history import fetch_minute_bars
    from .data.morning_scan import build_daily_candidates
    from .data.screener import scan_candidates

    INTRADAY_RESCAN_S = 1800          # rescan screener every 30 min
    INTRADAY_SLOT_RESERVE = 8         # always keep 8 slots open for intraday picks

    morning_done: set = set()         # dates where morning scan completed
    ranker_done: set = set()          # dates where the 10:01 ranked scan ran
    last_intraday_ts = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    dynamic_added: list[str] = []     # FIFO queue of dynamically-added symbols

    while True:
        await asyncio.sleep(30)
        now = dt.datetime.now(dt.timezone.utc)
        today = now.date()

        # ── Phase 1.5: model-ranked morning scan (once per day, ~10:01 ET —
        # the 09:30-09:45 window + the free tier's 15-min SIP embargo).
        # Rescores the current universe with the trained scanner ranker and
        # reorders stream priority; falls back silently to the momentum
        # universe when no artifact is trained.
        if today not in ranker_done and calendar.is_session_open(now):
            import zoneinfo

            et = now.astimezone(zoneinfo.ZoneInfo("America/New_York"))
            if et.time() >= dt.time(10, 1):
                ranker_done.add(today)
                try:
                    import time as _time

                    from alpaca.data.historical import StockHistoricalDataClient

                    from .scanner.live_scan import run_morning_scan

                    hist = StockHistoricalDataClient(
                        settings.alpaca_api_key, settings.alpaca_secret_key)
                    result = await asyncio.to_thread(
                        run_morning_scan,
                        symbols=list(engine._live_universe),
                        session_date=today, client=hist, repo=engine.repo,
                        rate_limiter=lambda: _time.sleep(0.35))
                    if result.plan_focus:
                        log.info("day_scanner.ranked_scan",
                                 top5=result.plan_focus[:5],
                                 dropped=result.dropped_budget)
                except Exception as exc:
                    log.warning("day_scanner.ranked_scan_failed", error=str(exc))

        # ── Phase 1: morning scan ────────────────────────────────────────────
        if today not in morning_done:
            morning_done.add(today)
            dynamic_added = []   # reset FIFO for the new session

            exclude = set(engine._live_universe)
            slots = max(0, SUBSCRIPTION_LIMIT - len(engine._live_universe)
                        - INTRADAY_SLOT_RESERVE)
            if slots > 0:
                try:
                    candidates = await asyncio.to_thread(
                        build_daily_candidates,
                        settings.alpaca_api_key, settings.alpaca_secret_key,
                        top_n=slots, exclude=exclude,
                    )
                except Exception as exc:
                    log.warning("day_scanner.morning_failed", error=str(exc))
                    candidates = []
                if candidates:
                    try:
                        history = await asyncio.to_thread(
                            fetch_minute_bars,
                            settings.alpaca_api_key, settings.alpaca_secret_key,
                            candidates, settings.history_warmup_days,
                        )
                        engine.warmup(history)
                    except Exception as exc:
                        log.warning("day_scanner.morning_warmup_failed", error=str(exc))
                    engine.expand_universe(candidates)
                    dynamic_added.extend(candidates)
                    engine.state.last_data_ts = dt.datetime.now(dt.timezone.utc)
                    stream.update_symbols(engine._live_universe,
                                         context=set(tier.universe))
                    log.info("day_scanner.morning_done", added=len(candidates),
                             top5=candidates[:5],
                             universe=len(engine._live_universe))

        # ── Phase 2: intraday rescan ─────────────────────────────────────────
        if not calendar.is_session_open(now):
            continue
        if (now - last_intraday_ts).total_seconds() < INTRADAY_RESCAN_S:
            continue
        last_intraday_ts = now

        exclude = set(engine._live_universe)
        try:
            new_candidates = await asyncio.to_thread(
                scan_candidates,
                settings.alpaca_api_key, settings.alpaca_secret_key,
                exclude, 10,
            )
        except Exception as exc:
            log.warning("day_scanner.intraday_failed", error=str(exc))
            continue

        if not new_candidates:
            log.info("day_scanner.intraday_no_new")
            continue

        # How many slots do we need to free?
        slots_free = SUBSCRIPTION_LIMIT - len(engine._live_universe)
        slots_needed = max(0, len(new_candidates) - slots_free)

        # Evict oldest idle (no-position) dynamic names to make room
        held = set(engine.state.positions)
        evict: list[str] = []
        if slots_needed > 0:
            for sym in list(dynamic_added):
                if sym not in held and sym not in tier.universe:
                    evict.append(sym)
                    if len(evict) >= slots_needed:
                        break

        engine.rotate_universe(new_candidates, evict)

        for sym in evict:
            if sym in dynamic_added:
                dynamic_added.remove(sym)
        dynamic_added.extend(new_candidates)

        # Warm up history only for symbols that were actually added and lack bars
        actually_added = [s for s in new_candidates
                          if s in engine._live_universe
                          and len(engine.bars_5m.get(s, [])) < 30]
        if actually_added:
            try:
                history = await asyncio.to_thread(
                    fetch_minute_bars,
                    settings.alpaca_api_key, settings.alpaca_secret_key,
                    actually_added, settings.history_warmup_days,
                )
                engine.warmup(history)
            except Exception as exc:
                log.warning("day_scanner.intraday_warmup_failed", error=str(exc))

        # Reset staleness clock before reconnect so the monitor doesn't fire
        # during the few seconds the stream is tearing down and rebuilding.
        engine.state.last_data_ts = dt.datetime.now(dt.timezone.utc)
        stream.update_symbols(engine._live_universe, context=set(tier.universe))
        log.info("day_scanner.intraday_done",
                 added=new_candidates, evicted=evict,
                 universe=len(engine._live_universe))


def make_fill_handler(engine: "Engine", om: "OrderManager"):  # type: ignore[no-untyped-def]
    """Build the trade-updates fill callback (module-level so tests can wire
    it against a MockBroker-backed engine).

    Routing: target (-tgt) fills disarm the bracket and stop; ordinary fills
    apply the staged ATR stop, and a BUY fill with a staged bracket arms it
    and rests the take-profit leg at the broker (armed on FILL, not submit —
    no bracket before shares exist)."""
    from .execution.order_manager import is_target_coid

    async def _on_fill_event(symbol: str, side: str, qty: int, price: float,
                             coid: str) -> None:
        engine.repo.record_order_fill(coid, qty, price)
        if is_target_coid(coid):
            om.on_fill(symbol, side, qty, price, reason="target", book="intraday")
            engine.brackets.on_target_fill(symbol)
            engine._pending_brackets.pop(symbol, None)
            return
        om.on_fill(symbol, side, qty, price, reason="stream", book="intraday")
        # Apply ATR stop staged at submit time.
        stop = engine._pending_stops.pop(symbol, None)
        if stop is not None:
            pos = engine.state.positions.get(symbol)
            if pos and pos.book == "intraday":
                pos.stop_price = stop
        pending = engine._pending_brackets.get(symbol)
        if pending is not None and side == "buy":
            engine._pending_brackets.pop(symbol)
            b_qty, target_px, stop_px, deadline = pending
            engine.brackets.arm(symbol, b_qty, entry_px=price, target_px=target_px,
                                stop_px=stop_px, deadline=deadline)
            asyncio.create_task(
                om.submit_bracket_target(symbol, b_qty, target_px, entry_coid=coid))
            log.info("bracket.armed", symbol=symbol, qty=b_qty,
                     target=target_px, stop=stop_px)

    return _on_fill_event


def resolve_scalp_profile(profile: str,
                          equity: float | None = None) -> "ScalpConfig | None":
    """Map settings.scalp_profile to its frozen ScalpConfig.

    "off" -> None: the engine runs with every scalp/bracket path dormant
    (legacy behavior). "auto" -> preset chosen from the account's ACTUAL
    equity (change equity at the broker, restart, and the guardrails
    follow; re-checked at each day roll). Anything else unrecognized is a
    config error on the money path — fail loudly rather than silently
    trading without brackets."""
    from .config.scalp_tiers import SCALP_LARGE, SCALP_MID, SCALP_SMALL, \
        profile_for_equity

    if profile == "off":
        return None
    if profile == "auto":
        if equity is None:
            raise ValueError("scalp_profile=auto needs the account equity")
        return profile_for_equity(equity)
    if profile == "small":
        return SCALP_SMALL
    if profile == "mid":
        return SCALP_MID
    if profile == "large":
        return SCALP_LARGE
    raise ValueError(
        f"unknown scalp_profile {profile!r} (expected off|auto|small|mid|large)")


def rearm_open_scalps(engine: "Engine", scalp_cfg: "ScalpConfig") -> None:
    """Re-arm brackets for open intraday longs found at startup reconcile.

    Conservative restart behavior: the original arm-time target/stop/deadline
    are not persisted (BracketBook is a derived in-memory view), so each
    position gets ScalpConfig-derived legs off its entry price and a fresh
    deadline of now + timeout_s — bounding the extra holding time of any
    scalp that outlived its original bracket to one timeout window."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    for sym, pos in engine.state.positions.items():
        if pos.book != "intraday" or pos.qty <= 0:
            continue  # brackets exit long scalps only
        engine.brackets.arm(sym, pos.qty, entry_px=pos.entry_price,
                            target_px=pos.entry_price + scalp_cfg.target_ps,
                            stop_px=pos.entry_price - scalp_cfg.stop_ps,
                            deadline=now + timedelta(seconds=scalp_cfg.timeout_s))
        log.info("bracket.rearmed_on_restart", symbol=sym, qty=pos.qty)


async def _run(tier_name: str) -> None:
    import atexit
    import os
    from pathlib import Path

    from .data.alpaca_stream import MarketStream, TradeUpdateStream
    from .data.history import fetch_minute_bars
    from .engine import Engine
    from .execution.broker import AlpacaBroker
    from .execution.order_manager import OrderManager
    from .execution.reconcile import reconcile
    from .persistence.repo import Repo
    from .pubsub import PubSub
    from .risk.state import PortfolioState
    from .signals.ensemble import Ensemble
    from .signals.mean_reversion import MeanReversionSignal
    from .signals.momentum import MomentumSignal

    # Alpaca allows ONE data websocket per account: two engines silently kick
    # each other off the stream (observed live 2026-07-07). Refuse dual launch.
    lock = Path("scalpengine.pid")
    if lock.exists():
        try:
            old_pid = int(lock.read_text().strip())
            os.kill(old_pid, 0)  # raises if not running
            print(f"FATAL: another engine is already running (pid {old_pid}). "
                  "Alpaca allows one data connection — two engines starve each other. "
                  "Stop it first (Ctrl-C or `kill`).")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # stale lockfile
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))

    s = get_settings()
    if s.trading_mode != "paper":
        print("FATAL: only TRADING_MODE=paper is supported in this build.")
        sys.exit(2)
    if not s.alpaca_api_key or not s.alpaca_secret_key:
        print("FATAL: ALPACA_API_KEY / ALPACA_SECRET_KEY missing in .env")
        sys.exit(2)

    tier = TIERS[Tier(tier_name)]
    repo = Repo(s.resolved_database_url())
    state = PortfolioState()
    db_state = repo.get_state()
    if db_state.status == "HALTED":
        print(f"Engine is HALTED ({db_state.halted_reason}). Run `scalpctl reset` first.")
        sys.exit(1)
    state.peak_equity = db_state.peak_equity
    pubsub = PubSub(s.redis_url)
    broker = AlpacaBroker(s.alpaca_api_key, s.alpaca_secret_key, s.alpaca_paper_base_url)
    om = OrderManager(broker, repo, state)
    ensemble = Ensemble([MomentumSignal(), MeanReversionSignal()])
    account = await broker.get_account()
    scalp_cfg = resolve_scalp_profile(s.scalp_profile, equity=account.equity)
    if s.scalp_profile == "auto":
        from .config.scalp_tiers import AUTO_FLOOR_BAND_EQUITY, AUTO_FLOOR_USD

        # small accounts get the day-trading equity floor by default; an
        # explicit MIN_EQUITY_HALT_USD always wins
        if not s.min_equity_halt_usd and account.equity < AUTO_FLOOR_BAND_EQUITY:
            s.min_equity_halt_usd = AUTO_FLOOR_USD
        log.info("scalp.auto_profile", equity=account.equity,
                 profile=scalp_cfg.name if scalp_cfg else "off",
                 equity_floor=s.min_equity_halt_usd or None)
    engine = Engine(s, tier, repo, om, ensemble, state, pubsub, scalp_cfg=scalp_cfg)
    engine.scalp_auto = s.scalp_profile == "auto"
    if scalp_cfg is not None and s.scalp_artifact_dir:
        from .signals.scalp_gbt import ScalpGbtSignal

        engine.scalp_signal = ScalpGbtSignal(s.scalp_artifact_dir)
        log.info("scalp.model_loaded", artifact=s.scalp_artifact_dir,
                 threshold=engine.scalp_signal.threshold)

    await reconcile(broker, repo, state)
    if scalp_cfg is not None:
        rearm_open_scalps(engine, scalp_cfg)
    state.day_start_equity = state.equity
    state.peak_equity = max(state.peak_equity, state.equity)
    repo.update_state(status="RUNNING", tier=tier_name,
                      day_start_equity=state.equity, peak_equity=state.peak_equity)

    log.info("warmup.backfill", days=s.history_warmup_days)
    history = fetch_minute_bars(s.alpaca_api_key, s.alpaca_secret_key,
                                tier.universe, s.history_warmup_days)
    engine.warmup(history)

    stream = MarketStream(s.alpaca_api_key, s.alpaca_secret_key, tier.universe,
                          engine.on_trade, engine.on_quote, engine.on_stream_bar,
                          context=set(tier.universe))

    trade_stream = TradeUpdateStream(s.alpaca_api_key, s.alpaca_secret_key,
                                     paper=True, on_fill=make_fill_handler(engine, om))
    tasks = [
        stream.run_forever(),
        trade_stream.run_forever(),
        engine.staleness_monitor(),
        engine.eod_flattener(),
        engine.heartbeat(),
        engine.command_poller(),
        engine.day_roll(),
        engine.scalp_loop(),
        _day_scanner(engine, stream, s, tier),
    ]
    log.info("engine.start", tier=tier_name, universe=len(tier.universe),
             scalp_profile=s.scalp_profile)
    await asyncio.gather(*tasks)


def main() -> None:
    p = argparse.ArgumentParser(prog="scalpctl")
    sub = p.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run the engine (paper only)")
    runp.add_argument("--tier", default=get_settings().risk_tier,
                      choices=["low", "medium", "high"])
    sub.add_parser("status")
    sub.add_parser("halt", help="fire the kill switch")
    sub.add_parser("reset", help="clear HALTED state")
    sub.add_parser("flatten", help="alias for halt (cancel+flatten)")
    args = p.parse_args()

    if args.cmd == "run":
        asyncio.run(_run(args.tier))
    elif args.cmd == "status":
        st = _repo().get_state()
        print(f"status={st.status} tier={st.tier} reason={st.halted_reason!r} "
              f"heartbeat={st.heartbeat_ts} peak={st.peak_equity}")
    elif args.cmd in ("halt", "flatten"):
        _repo().enqueue_command("kill")
        print("kill command enqueued (engine executes within 2s if running)")
    elif args.cmd == "reset":
        repo = _repo()
        repo.enqueue_command("reset")
        repo.update_state(status="STOPPED", halted_reason="")
        print("reset enqueued + state cleared")


if __name__ == "__main__":
    main()
