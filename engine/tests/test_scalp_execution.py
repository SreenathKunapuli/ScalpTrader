"""BracketBook trigger logic + OrderManager.submit_bracket_target behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from scalpengine.execution.brackets import BracketAction, BracketBook
from scalpengine.execution.order_manager import (
    OrderManager,
    is_target_coid,
    make_client_order_id,
)

T0 = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)


def _arm(book: BracketBook, symbol: str = "AAPL", *, qty: int = 10,
         entry_px: float = 100.0, target_px: float = 104.0, stop_px: float = 96.0,
         deadline: datetime | None = None) -> None:
    book.arm(symbol, qty=qty, entry_px=entry_px, target_px=target_px,
             stop_px=stop_px, deadline=deadline or T0 + timedelta(seconds=120))


# ---------- BracketBook ---------- #
def test_stop_fires_at_boundary_not_one_tick_above() -> None:
    book = BracketBook()
    _arm(book, stop_px=96.0)
    # bid exactly at stop_px fires
    actions = book.check(T0, {"AAPL": (96.0, 96.02)})
    assert actions == [BracketAction("AAPL", "stop", 10)]
    assert "AAPL" not in book.armed  # removed after firing

    # bid one tick above stop does NOT fire
    book2 = BracketBook()
    _arm(book2, stop_px=96.0)
    assert book2.check(T0, {"AAPL": (96.01, 96.03)}) == []
    assert "AAPL" in book2.armed


def test_timeout_fires_at_exactly_deadline() -> None:
    book = BracketBook()
    deadline = T0 + timedelta(seconds=120)
    _arm(book, stop_px=90.0, deadline=deadline)
    # just before deadline: nothing (bid well above stop)
    assert book.check(deadline - timedelta(seconds=1), {"AAPL": (100.0, 100.02)}) == []
    # exactly at deadline: timeout fires
    actions = book.check(deadline, {"AAPL": (100.0, 100.02)})
    assert actions == [BracketAction("AAPL", "timeout", 10)]
    assert "AAPL" not in book.armed


def test_stop_takes_precedence_over_timeout() -> None:
    book = BracketBook()
    deadline = T0 + timedelta(seconds=120)
    _arm(book, stop_px=96.0, deadline=deadline)
    # both would fire: bid at stop AND now past deadline -> stop wins
    actions = book.check(deadline + timedelta(seconds=5), {"AAPL": (96.0, 96.02)})
    assert actions == [BracketAction("AAPL", "stop", 10)]


def test_no_double_fire() -> None:
    book = BracketBook()
    _arm(book, stop_px=96.0)
    first = book.check(T0, {"AAPL": (95.0, 95.02)})
    assert len(first) == 1
    # a second check with the same triggering marks yields nothing
    assert book.check(T0, {"AAPL": (95.0, 95.02)}) == []


def test_missing_marks_skipped_stale_data_never_fires() -> None:
    book = BracketBook()
    deadline = T0 - timedelta(seconds=1)  # already past -> would timeout if seen
    _arm(book, stop_px=200.0, deadline=deadline)  # stop also above any bid
    # symbol absent from marks -> skipped entirely, stays armed
    assert book.check(T0, {}) == []
    assert "AAPL" in book.armed
    assert book.check(T0, {"MSFT": (50.0, 50.02)}) == []
    assert "AAPL" in book.armed


def test_rearm_replaces_bracket() -> None:
    book = BracketBook()
    _arm(book, qty=10, stop_px=96.0)
    _arm(book, qty=20, stop_px=90.0)  # re-arm same symbol
    assert len(book.armed) == 1
    br = book.armed["AAPL"]
    assert br.qty == 20 and br.stop_px == 90.0
    # old stop (96) no longer fires; new stop (90) governs
    assert book.check(T0, {"AAPL": (95.0, 95.02)}) == []
    actions = book.check(T0, {"AAPL": (90.0, 90.02)})
    assert actions == [BracketAction("AAPL", "stop", 20)]


def test_multi_symbol_independence() -> None:
    book = BracketBook()
    _arm(book, "AAPL", qty=10, stop_px=96.0)
    _arm(book, "MSFT", qty=5, stop_px=300.0)
    # only AAPL's stop triggers; MSFT stays armed and untouched
    actions = book.check(T0, {"AAPL": (96.0, 96.02), "MSFT": (400.0, 400.02)})
    assert actions == [BracketAction("AAPL", "stop", 10)]
    assert "AAPL" not in book.armed and "MSFT" in book.armed


def test_disarm_and_on_target_fill_idempotent() -> None:
    book = BracketBook()
    _arm(book)
    book.disarm("AAPL")
    assert "AAPL" not in book.armed
    book.disarm("AAPL")           # idempotent, no error
    book.on_target_fill("AAPL")   # alias, also safe when not armed
    _arm(book)
    book.on_target_fill("AAPL")
    assert "AAPL" not in book.armed


def test_armed_property_is_a_copy() -> None:
    book = BracketBook()
    _arm(book)
    snapshot = book.armed
    snapshot.clear()
    assert "AAPL" in book.armed  # mutating the copy does not touch the book


# ---------- submit_bracket_target ---------- #
async def test_bracket_target_coid_format_and_recorded(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    entry_coid = make_client_order_id("scalp", "AAPL", "buy", T0)
    order = await om.submit_bracket_target("AAPL", qty=10, limit_px=104.0,
                                           entry_coid=entry_coid)
    assert order is not None
    expected_coid = f"{entry_coid[:24]}-tgt"
    assert order.client_order_id == expected_coid
    assert is_target_coid(order.client_order_id)
    assert not is_target_coid(entry_coid)
    # resting SELL limit at the target price
    assert order.side == "sell"
    # recorded in the repo with reason="target"
    with repo.session() as s:
        from sqlalchemy import select
        from scalpengine.persistence.models import Order

        row = s.scalar(select(Order).where(Order.client_order_id == expected_coid))
        assert row is not None
        assert row.reason == "target" and row.side == "sell"
        assert row.order_type == "limit" and row.limit_price == pytest.approx(104.0)


async def test_bracket_target_duplicate_suppressed(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    entry_coid = make_client_order_id("scalp", "AAPL", "buy", T0)
    o1 = await om.submit_bracket_target("AAPL", 10, 104.0, entry_coid)
    # MockBroker.submit_order raises "client_order_id must be unique" on a repeat
    o2 = await om.submit_bracket_target("AAPL", 10, 104.0, entry_coid)
    assert o1 is not None and o2 is None  # duplicate swallowed


async def test_bracket_target_reraises_other_errors(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]

    async def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("broker exploded")

    mock_broker.submit_order = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="broker exploded"):
        await om.submit_bracket_target("AAPL", 10, 104.0, "x")
