"""Staleness wiring through the live engine:
  - QuoteStalenessTracker instantiated from settings
  - record() called on every on_quote
  - heartbeat persists snapshot to repo (staleness_json) and publishes on pubsub
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from scalpengine.config.settings import Settings
from scalpengine.config.tiers import TIERS, Tier
from scalpengine.data.staleness import QuoteStalenessTracker
from scalpengine.engine import Engine
from scalpengine.execution.order_manager import OrderManager
from scalpengine.pubsub import PubSub
from scalpengine.signals.ensemble import Ensemble
from scalpengine.signals.mean_reversion import MeanReversionSignal
from scalpengine.signals.momentum import MomentumSignal

from .conftest import MockBroker

MED = TIERS[Tier.MEDIUM]
T0 = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)  # Monday 11:00 ET, in session


def make_engine(state, repo, mock_broker):  # type: ignore[no-untyped-def]
    s = Settings(_env_file=None, staleness_pause_s=60, staleness_kill_s=180)  # type: ignore[call-arg]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ens = Ensemble([MomentumSignal(), MeanReversionSignal()])
    return Engine(s, MED, repo, om, ens, state, PubSub())


# ---- instantiation ----

def test_staleness_tracker_instantiated(state, repo, mock_broker):
    engine = make_engine(state, repo, mock_broker)
    assert isinstance(engine.staleness, QuoteStalenessTracker)
    assert engine.staleness.pause_s == 60.0
    assert engine.staleness.kill_s == 180.0


# ---- record() called from on_quote ----

async def test_on_quote_records_staleness(state, repo, mock_broker):
    engine = make_engine(state, repo, mock_broker)

    # before any quote: age is +inf
    assert engine.staleness.age("SPY", T0) == float("inf")

    await engine.on_quote("SPY", T0, bid=100.0, bid_sz=100, ask=100.02, ask_sz=100)
    # age should be ~0 immediately after the record (quote_ts = T0)
    assert engine.staleness.age("SPY", T0) == pytest.approx(0.0)

    # second quote 5s later
    t1 = T0 + timedelta(seconds=5)
    await engine.on_quote("SPY", t1, bid=100.01, bid_sz=100, ask=100.03, ask_sz=100)
    assert engine.staleness.age("SPY", t1) == pytest.approx(0.0)
    # age from T0 perspective is 5s
    assert engine.staleness.age("SPY", T0 + timedelta(seconds=6)) == pytest.approx(1.0)


async def test_on_quote_records_gap_percentiles(state, repo, mock_broker):
    engine = make_engine(state, repo, mock_broker)

    # feed 6 quotes 2s apart (5 gaps)
    for i in range(6):
        ts = T0 + timedelta(seconds=2 * i)
        await engine.on_quote("SPY", ts, bid=100.0, bid_sz=50, ask=100.02, ask_sz=50)

    p50, p95 = engine.staleness.gap_percentiles("SPY")
    assert p50 == pytest.approx(2.0)
    assert p95 == pytest.approx(2.0)


# ---- heartbeat persists snapshot and publishes ----

async def test_heartbeat_persists_staleness(state, repo, mock_broker):
    engine = make_engine(state, repo, mock_broker)

    # Deliver one quote so snapshot has content
    await engine.on_quote("AAPL", T0, bid=150.0, bid_sz=200, ask=150.05, ask_sz=200)

    # Patch heartbeat_interval_s to 0 so the first sleep finishes immediately
    engine.settings.heartbeat_interval_s = 0

    # Run heartbeat for one iteration (cancel after it writes once)
    async def run_once():
        task = asyncio.create_task(engine.heartbeat())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await run_once()

    st = repo.get_state()
    assert isinstance(st.staleness_json, dict)
    assert "AAPL" in st.staleness_json
    sym_data = st.staleness_json["AAPL"]
    assert "age_s" in sym_data
    assert "gap_p50_s" in sym_data
    assert "gap_p95_s" in sym_data


async def test_heartbeat_publishes_staleness_channel(state, repo, mock_broker):
    engine = make_engine(state, repo, mock_broker)
    published: list[tuple[str, object]] = []

    async def capture(channel: str, data: object) -> None:  # type: ignore[override]
        published.append((channel, data))

    engine.pubsub.publish = capture  # type: ignore[method-assign]

    await engine.on_quote("QQQ", T0, bid=350.0, bid_sz=100, ask=350.05, ask_sz=100)

    engine.settings.heartbeat_interval_s = 0

    async def run_once():
        task = asyncio.create_task(engine.heartbeat())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await run_once()

    channels = [ch for ch, _ in published]
    assert "staleness" in channels
    staleness_payloads = [d for ch, d in published if ch == "staleness"]
    assert len(staleness_payloads) >= 1
    payload = staleness_payloads[0]
    assert isinstance(payload, dict)
    assert "QQQ" in payload


# ---- multiple symbols each get their own slot ----

async def test_multiple_symbols_tracked_independently(state, repo, mock_broker):
    engine = make_engine(state, repo, mock_broker)

    await engine.on_quote("SPY", T0, bid=400.0, bid_sz=100, ask=400.02, ask_sz=100)
    await engine.on_quote("QQQ", T0 + timedelta(seconds=3), bid=300.0,
                          bid_sz=100, ask=300.02, ask_sz=100)

    now = T0 + timedelta(seconds=10)
    snap = engine.staleness.snapshot(now)
    assert "SPY" in snap and "QQQ" in snap
    # SPY is 10s old, QQQ is 7s old
    assert snap["SPY"]["age_s"] == pytest.approx(10.0)
    assert snap["QQQ"]["age_s"] == pytest.approx(7.0)
