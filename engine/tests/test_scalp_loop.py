"""Second-cadence scalp loop: model decision -> sizing -> risk -> maker entry
-> staged vol-scaled bracket. Uses a duck-typed fake model so the loop's
plumbing is tested independently of sklearn."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from scalpengine.signals.scalp_gbt import ScalpDecision

from .test_bracket_wiring import SCALP, T0, make_scalp_engine


class FakeModel:
    """compute_second returns a canned decision; records calls."""

    def __init__(self, decision: ScalpDecision | None, threshold: float = 0.6):
        self.decision = decision
        self.threshold = threshold
        self.calls: list[str] = []

    def compute_second(self, symbol, frame):
        self.calls.append(symbol)
        return self.decision


def feed_tape(engine, symbol: str = "SPY", price: float = 100.0,
              seconds: int = 90, start: datetime = T0) -> None:
    """90s of a busy $100 tape into the second-bar builder."""
    for i in range(seconds):
        ts = start + timedelta(seconds=i, milliseconds=300)
        engine.second_bars.add_trade(symbol, price, 500, ts)
        engine.second_bars.add_quote(symbol, price - 0.02, price, 50.0, 50.0, ts)
    engine.second_bars.poll(start + timedelta(seconds=seconds + 1))


DEC = ScalpDecision(p_win=0.9, target_ps=0.50, stop_ps=0.40, timeout_s=120)


async def test_scalp_entry_submits_and_stages_bracket(state, repo, mock_broker):
    engine, om = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(DEC)
    mock_broker.price["SPY"] = 100.0
    feed_tape(engine)

    now = T0 + timedelta(seconds=95)
    await engine._maybe_scalp("SPY", now)

    buys = [o for o in mock_broker.submitted if o.side == "buy"]
    assert len(buys) == 1 and buys[0].qty > 0
    qty, tgt, stp, deadline = engine._pending_brackets["SPY"]
    assert qty == buys[0].qty
    assert tgt == pytest.approx(100.0 + DEC.target_ps)   # model's vol-scaled legs
    assert stp == pytest.approx(100.0 - DEC.stop_ps)
    assert deadline == now + timedelta(seconds=DEC.timeout_s)
    # hard caps respected: notional within tier box, participation within tape
    assert buys[0].qty * 100.0 <= SCALP.max_position_pct * state.equity + 100.0
    assert buys[0].qty <= SCALP.max_participation * 500 * 60 + 1


async def test_below_threshold_no_entry(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(
        ScalpDecision(p_win=0.4, target_ps=0.5, stop_ps=0.4, timeout_s=120))
    feed_tape(engine)
    await engine._maybe_scalp("SPY", T0 + timedelta(seconds=95))
    assert not [o for o in mock_broker.submitted if o.side == "buy"]
    assert "SPY" not in engine._pending_brackets


async def test_gate_none_no_entry(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(None)   # cold window / bad NBBO
    feed_tape(engine)
    await engine._maybe_scalp("SPY", T0 + timedelta(seconds=95))
    assert not mock_broker.submitted


async def test_existing_position_or_pending_skips(state, repo, mock_broker):
    from scalpengine.risk.state import Position

    engine, _ = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(DEC)
    feed_tape(engine)
    state.positions["SPY"] = Position(symbol="SPY", qty=10,
                                      entry_price=100.0, mark=100.0)
    await engine._maybe_scalp("SPY", T0 + timedelta(seconds=95))
    assert not mock_broker.submitted
    del state.positions["SPY"]
    engine._pending_brackets["SPY"] = (10, 100.5, 99.6,
                                       T0 + timedelta(seconds=200))
    await engine._maybe_scalp("SPY", T0 + timedelta(seconds=95))
    assert not mock_broker.submitted
    assert engine.scalp_signal.calls == []   # skipped before model ran


async def test_zero_size_no_entry(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(DEC)
    # empty book: displayed ask size 0 -> sizing refuses
    for i in range(90):
        ts = T0 + timedelta(seconds=i, milliseconds=300)
        engine.second_bars.add_trade("SPY", 100.0, 500, ts)
        engine.second_bars.add_quote("SPY", 99.98, 100.0, 50.0, 0.0, ts)
    engine.second_bars.poll(T0 + timedelta(seconds=91))
    await engine._maybe_scalp("SPY", T0 + timedelta(seconds=95))
    assert not mock_broker.submitted


async def test_risk_rejection_records_no_bracket(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(DEC)
    feed_tape(engine, price=600.0)          # outside SCALP price band (max 500)
    await engine._maybe_scalp("SPY", T0 + timedelta(seconds=95))
    assert not mock_broker.submitted
    assert "SPY" not in engine._pending_brackets


async def test_loop_inert_without_model(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    assert engine.scalp_signal is None
    await engine.scalp_loop()               # returns immediately, no hang
    # and the data path stays cheap: builder untouched by on_trade
    await engine.on_trade("SPY", T0, 100.0, 100)
    assert engine.second_bars.get_frame("SPY").empty


async def test_on_trade_and_quote_feed_builder_when_wired(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker)
    engine.scalp_signal = FakeModel(None)
    t = T0 + timedelta(milliseconds=100)
    await engine.on_trade("SPY", t, 100.0, 100)
    await engine.on_quote("SPY", t, bid=99.98, bid_sz=5, ask=100.0, ask_sz=5)
    engine.second_bars.poll(T0 + timedelta(seconds=3))
    frame = engine.second_bars.get_frame("SPY")
    # first bar carries the trade; later quote-only seconds carry NBBO ffill
    assert frame["close"].iloc[0] == 100.0
    assert (frame["bid"] == 99.98).all()
    assert frame["close"].iloc[1:].isna().all()
