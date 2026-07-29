"""Subscription budget planning + stream-bar enrichment tests."""

from __future__ import annotations

from datetime import UTC, datetime

from scalpengine.data.alpaca_stream import plan_subscriptions


def test_plan_20_runners_2_context() -> None:
    """20 scalpable + SPY/QQQ context, limit=30.

    K = 30 // 3 = 10 full packages; budget used = 30; leftover = 0.
    SPY/QQQ are dropped (correct: depth beats breadth for scalping).
    """
    runners = [f"S{i}" for i in range(20)]
    ctx = {"SPY", "QQQ"}
    symbols = runners + ["SPY", "QQQ"]
    bars, quotes, trades = plan_subscriptions(symbols, context=ctx)

    expected_full = runners[:10]
    assert bars == expected_full
    assert quotes == expected_full
    assert trades == expected_full
    assert "SPY" not in bars and "QQQ" not in bars


def test_plan_6_runners_2_context() -> None:
    """6 scalpable + SPY/QQQ context, limit=30.

    K = min(6, 30//3) = 6 full packages; budget used = 18; leftover = 12.
    Leftover bars: SPY, QQQ (context first), no more scalpable to add.
    """
    runners = [f"S{i}" for i in range(6)]
    ctx = {"SPY", "QQQ"}
    symbols = runners + ["SPY", "QQQ"]
    bars, quotes, trades = plan_subscriptions(symbols, context=ctx)

    assert quotes == runners
    assert trades == runners
    # bars = 6 full-pkg + up to 12 leftover; context symbols get leftover bars
    assert set(bars) == set(runners) | {"SPY", "QQQ"}
    assert len(bars) == 8  # 6 runners + 2 context


def test_plan_no_context_all_scalpable() -> None:
    """No context set: all symbols are scalpable, get full packages up to K."""
    runners = [f"S{i}" for i in range(10)]
    bars, quotes, trades = plan_subscriptions(runners)

    # K = min(10, 30//3) = 10 full packages (budget = 30, no leftover)
    assert bars == runners
    assert quotes == runners
    assert trades == runners


async def test_on_stream_bar_enriches_and_feeds(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    from .test_replay import make_engine

    engine, _ = make_engine(state, repo, mock_broker)
    ts = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)
    # accumulate microstructure, then deliver the official bar
    await engine.on_quote("SPY", ts, 99.99, 300, 100.01, 100)
    await engine.on_trade("SPY", ts, 100.01, 100)   # at ask -> +100
    await engine.on_trade("SPY", ts, 99.99, 50)     # at bid -> -50
    await engine.on_stream_bar("SPY", ts, 100.0, 100.1, 99.9, 100.05, 150, 100.02, 2)

    bars = list(engine.bars_1m["SPY"])
    assert len(bars) == 1
    b = bars[0]
    assert b.close == 100.05 and b.volume == 150
    assert abs(b.mean_spread - 0.02) < 1e-9
    assert abs(b.mean_quote_imbalance - 0.5) < 1e-9
    assert abs(b.flow_imbalance - (100 - 50) / 150) < 1e-9
    # accumulators drained
    await engine.on_stream_bar("SPY", ts, 100.0, 100.1, 99.9, 100.0, 10, 100.0, 1)
    assert list(engine.bars_1m["SPY"])[-1].flow_imbalance == 0.0
