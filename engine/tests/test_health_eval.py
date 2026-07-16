"""Signal health tracker tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scalpengine.signals.health import SignalHealthTracker


def _add_trades(repo, pnls: list[float], signal: str = "lob_flow") -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    for i, pnl in enumerate(pnls):
        repo.add_trade(symbol="SPY", side="long", qty=1,
                       entry_ts=now - timedelta(days=1), exit_ts=now - timedelta(hours=i),
                       entry_price=100.0, exit_price=100.0 + pnl, pnl=pnl,
                       signal_scores_json={signal: {"contribution": 0.5}})


def test_health_halves_weight_on_losses(repo) -> None:  # type: ignore[no-untyped-def]
    tracker = SignalHealthTracker(repo, ["lob_flow", "momentum"])
    _add_trades(repo, [-1.0] * 10)
    mults = tracker.evaluate()
    assert mults["lob_flow"] == 0.5           # degraded -> halved
    assert mults["momentum"] == 1.0           # no evidence -> untouched
    tracker.restore("lob_flow")
    assert tracker.multipliers["lob_flow"] == 1.0


def test_health_keeps_weight_on_wins(repo) -> None:  # type: ignore[no-untyped-def]
    tracker = SignalHealthTracker(repo, ["lob_flow"])
    _add_trades(repo, [1.0] * 10)
    assert tracker.evaluate()["lob_flow"] == 1.0
