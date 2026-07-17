"""Bracket-scalp wiring through the live engine: entry-fill arming, the
per-tick stop/timeout fast path, coid-routed target fills, day-roll resets,
and the off-profile leaving every legacy path untouched."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from scalpengine.cli import make_fill_handler, rearm_open_scalps, resolve_scalp_profile
from scalpengine.config.scalp_tiers import SCALP_LARGE, SCALP_SMALL, ScalpConfig
from scalpengine.config.settings import Settings
from scalpengine.config.tiers import TIERS, Tier
from scalpengine.data.bar_builder import Bar
from scalpengine.engine import Engine
from scalpengine.execution.broker import BrokerOrder
from scalpengine.execution.order_manager import OrderManager, is_target_coid
from scalpengine.pubsub import PubSub
from scalpengine.risk.state import Position
from scalpengine.signals.ensemble import Ensemble
from scalpengine.signals.mean_reversion import MeanReversionSignal
from scalpengine.signals.momentum import MomentumSignal

MED = TIERS[Tier.MEDIUM]
T0 = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)  # Monday 11:00 ET, in session

# Test profile: SCALP_LARGE hard bounds but a price band that admits the $100
# synthetic tape (the shipped profiles cap at $10) and round bracket legs.
SCALP = ScalpConfig(
    name="test", max_position_pct=0.10, max_gross_pct=0.50, max_open_scalps=5,
    daily_loss_limit_pct=0.02, per_symbol_loss_cap_pct=0.01,
    target_ps=0.50, stop_ps=0.40, timeout_s=120,
    max_participation=0.02, price_min=0.5, price_max=500.0,
)


def make_flat_bars(symbol: str, n: int, start: datetime, price: float) -> list[Bar]:
    """n identical bars at a fixed price — enough range for a nonzero ATR."""
    return [Bar(symbol=symbol, ts=start + timedelta(minutes=5 * i), interval_s=300,
                open=price, high=price * 1.001, low=price * 0.999, close=price,
                volume=1000, vwap=price, trade_count=50, mean_spread=0.02,
                mean_quote_imbalance=0.0, flow_imbalance=0.0)
            for i in range(n)]


def make_scalp_engine(state, repo, mock_broker,  # type: ignore[no-untyped-def]
                      scalp_cfg: ScalpConfig | None = SCALP):
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ens = Ensemble([MomentumSignal(), MeanReversionSignal()])
    return Engine(s, MED, repo, om, ens, state, PubSub(), scalp_cfg=scalp_cfg), om


async def submit_entry(engine: Engine, mock_broker, symbol: str = "SPY"):  # type: ignore[no-untyped-def]
    """Drive a real BUY entry through _enter_or_adjust at $100."""
    bars = make_flat_bars(symbol, 40, T0 - timedelta(hours=4), price=100.0)
    mock_broker.price[symbol] = 100.0
    res = SimpleNamespace(final_score=1.0, vol_mult=1.0)
    await engine._enter_or_adjust(symbol, bars, res, T0)
    entries = [o for o in mock_broker.submitted if o.side == "buy"]
    assert entries, "entry order must have been submitted"
    return entries[-1]


def _order_reason(repo, coid: str) -> str | None:  # type: ignore[no-untyped-def]
    from scalpengine.persistence.models import Order
    from sqlalchemy import select

    with repo.session() as s:
        row = s.scalar(select(Order).where(Order.client_order_id == coid))
        return row.reason if row is not None else None


# ---------- (a) entry fill -> armed bracket + resting target ---------- #
async def test_entry_fill_arms_bracket_and_rests_target(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_scalp_engine(state, repo, mock_broker)
    entry = await submit_entry(engine, mock_broker)
    # staged at submit, armed only on fill — no bracket before shares exist
    assert "SPY" in engine._pending_brackets
    assert "SPY" not in engine.brackets.armed

    on_fill = make_fill_handler(engine, om)
    await on_fill("SPY", "buy", entry.qty, 100.0, entry.client_order_id)
    for _ in range(3):
        await asyncio.sleep(0)  # let the created task rest the target leg

    br = engine.brackets.armed.get("SPY")
    assert br is not None and br.qty == entry.qty
    assert br.target_px == pytest.approx(100.0 + SCALP.target_ps)
    assert br.stop_px == pytest.approx(100.0 - SCALP.stop_ps)
    assert "SPY" not in engine._pending_brackets
    tgts = [o for o in mock_broker.submitted if is_target_coid(o.client_order_id)]
    assert len(tgts) == 1 and tgts[0].side == "sell" and tgts[0].qty == entry.qty
    assert tgts[0].client_order_id == f"{entry.client_order_id[:24]}-tgt"


# ---------- (b) stop tick -> urgent exit + target cancelled ---------- #
async def test_stop_tick_exits_and_cancels_target(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_scalp_engine(state, repo, mock_broker)
    state.positions["SPY"] = Position(symbol="SPY", qty=10, entry_price=100.0, mark=100.0)
    engine.brackets.arm("SPY", 10, entry_px=100.0, target_px=100.5, stop_px=99.6,
                        deadline=datetime.now(UTC) + timedelta(seconds=120))
    tgt = BrokerOrder(id="tgt1", client_order_id="cafebabecafebabecafebab-tgt",
                      symbol="SPY", side="sell", qty=10, status="new")
    mock_broker.open_orders.append(tgt)

    await engine.on_quote("SPY", T0, bid=99.60, bid_sz=100, ask=99.62, ask_sz=100)

    assert "SPY" not in engine.brackets.armed
    assert "tgt1" in mock_broker.cancelled  # resting target pulled first
    sells = [o for o in mock_broker.submitted if o.side == "sell" and o.symbol == "SPY"]
    assert len(sells) == 1 and sells[0].qty == 10
    assert _order_reason(repo, sells[0].client_order_id) == "stop"


# ---------- (c) target fill routed via coid -> disarm, no more exits ---------- #
async def test_target_fill_routed_by_coid_disarms(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_scalp_engine(state, repo, mock_broker)
    now = datetime.now(UTC)
    state.positions["SPY"] = Position(symbol="SPY", qty=10, entry_price=100.0, mark=100.0)
    engine.brackets.arm("SPY", 10, entry_px=100.0, target_px=100.5, stop_px=99.6,
                        deadline=now + timedelta(seconds=120))
    engine._pending_brackets["SPY"] = (10, 100.5, 99.6, now + timedelta(seconds=120))

    on_fill = make_fill_handler(engine, om)
    await on_fill("SPY", "sell", 10, 100.5, "deadbeefdeadbeefdeadbeef-tgt")

    assert "SPY" not in engine.brackets.armed
    assert "SPY" not in engine._pending_brackets
    assert "SPY" not in state.positions  # closed via om.on_fill(reason="target")
    assert state.symbol_realized_today["SPY"] == pytest.approx(5.0)
    n = len(mock_broker.submitted)
    # a later tick through the old stop must not fire anything
    await engine.on_quote("SPY", T0, bid=99.0, bid_sz=100, ask=99.02, ask_sz=100)
    assert len(mock_broker.submitted) == n


# ---------- (d) timeout deadline -> exit fires exactly once ---------- #
async def test_timeout_exit_fires_once(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_scalp_engine(state, repo, mock_broker)
    state.positions["SPY"] = Position(symbol="SPY", qty=10, entry_price=100.0, mark=100.0)
    engine.brackets.arm("SPY", 10, entry_px=100.0, target_px=100.5, stop_px=90.0,
                        deadline=datetime.now(UTC) - timedelta(seconds=1))

    await engine.on_quote("SPY", T0, bid=100.0, bid_sz=100, ask=100.02, ask_sz=100)
    sells = [o for o in mock_broker.submitted if o.side == "sell"]
    assert len(sells) == 1 and sells[0].qty == 10
    assert _order_reason(repo, sells[0].client_order_id) == "timeout"

    # fired bracket is disarmed: an identical second tick submits nothing
    await engine.on_quote("SPY", T0, bid=100.0, bid_sz=100, ask=100.02, ask_sz=100)
    assert len([o for o in mock_broker.submitted if o.side == "sell"]) == 1


# ---------- (e) day roll clears the per-symbol scalp loss ledger ---------- #
def test_day_roll_clears_symbol_realized(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    state.symbol_realized_today["SPY"] = -123.0
    state.intraday_realized_today = -55.0
    engine._roll_day(T0)
    assert state.symbol_realized_today == {}
    assert state.intraday_realized_today == 0.0


# ---------- (f) scalp_profile=off leaves the legacy path untouched ---------- #
async def test_scalp_off_leaves_legacy_path_untouched(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_scalp_engine(state, repo, mock_broker, scalp_cfg=None)
    entry = await submit_entry(engine, mock_broker)
    assert engine._pending_brackets == {}  # nothing staged when scalping is off

    on_fill = make_fill_handler(engine, om)
    await on_fill("SPY", "buy", entry.qty, 100.0, entry.client_order_id)
    for _ in range(3):
        await asyncio.sleep(0)

    assert engine.brackets.armed == {}
    assert not any(is_target_coid(o.client_order_id) for o in mock_broker.submitted)
    # legacy ATR stop staged at submit still lands on the position
    assert state.positions["SPY"].stop_price is not None


def test_scalp_profile_resolution() -> None:
    assert Settings(_env_file=None).scalp_profile == "off"  # type: ignore[call-arg]
    assert resolve_scalp_profile("off") is None
    assert resolve_scalp_profile("small") is SCALP_SMALL
    assert resolve_scalp_profile("large") is SCALP_LARGE
    with pytest.raises(ValueError, match="scalp_profile"):
        resolve_scalp_profile("bogus")


# ---------- restart reconcile re-arm (conservative fresh deadline) ---------- #
def test_restart_rearms_open_intraday_longs(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    state.positions["SPY"] = Position(symbol="SPY", qty=10, entry_price=100.0, mark=100.0)
    state.positions["QQQ"] = Position(symbol="QQQ", qty=-5, entry_price=50.0, mark=50.0)
    rearm_open_scalps(engine, SCALP)
    br = engine.brackets.armed.get("SPY")
    assert br is not None and br.qty == 10
    assert br.target_px == pytest.approx(100.0 + SCALP.target_ps)
    assert br.stop_px == pytest.approx(100.0 - SCALP.stop_ps)
    assert "QQQ" not in engine.brackets.armed  # brackets exit long scalps only
