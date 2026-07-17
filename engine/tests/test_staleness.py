"""QuoteStalenessTracker: ages, rolling gap percentiles, classify ladder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scalpengine.data.staleness import QuoteStalenessTracker

T0 = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def test_age_and_never_seen():
    tr = QuoteStalenessTracker()
    assert tr.age("X", T0) == float("inf")
    tr.record("X", at(0), at(0.1))
    assert tr.age("X", at(5)) == 5.0


def test_gap_percentiles_rolling():
    tr = QuoteStalenessTracker()
    for i in range(11):                       # 10 gaps of exactly 2s
        tr.record("X", at(2 * i), at(2 * i))
    p50, p95 = tr.gap_percentiles("X")
    assert p50 == 2.0 and p95 == 2.0
    # a 40s outage shows up in the tail
    tr.record("X", at(60), at(60))
    _, p95b = tr.gap_percentiles("X")
    assert p95b > 2.0


def test_window_prunes_old_gaps():
    tr = QuoteStalenessTracker()
    tr.record("X", at(0), at(0))
    tr.record("X", at(50), at(50))            # 50s gap sample
    # 400s later: only fresh 1s gaps remain inside the 300s window
    for i in range(5):
        tr.record("X", at(450 + i), at(450 + i))
    p50, p95 = tr.gap_percentiles("X")
    assert p95 <= 400.0 - 50.0                # the 50s outage sample aged out
    assert p50 == 1.0


def test_classify_ladder_uses_worst_symbol():
    tr = QuoteStalenessTracker(pause_s=60, kill_s=180)
    tr.record("FAST", at(0), at(0))
    tr.record("SLOW", at(0), at(0))
    tr.record("FAST", at(100), at(100))
    assert tr.classify(at(100)) == "pause"    # SLOW is 100s old
    assert tr.classify(at(100), ["FAST"]) == "ok"
    assert tr.classify(at(200)) == "kill"     # SLOW is 200s old
    assert tr.classify(at(200), ["NEVER_SEEN"]) == "ok"


def test_out_of_order_quote_ignored_for_last():
    tr = QuoteStalenessTracker()
    tr.record("X", at(10), at(10))
    tr.record("X", at(5), at(11))             # late, older stamp
    assert tr.age("X", at(12)) == 2.0         # newest stamp still t=10


def test_snapshot_shape():
    tr = QuoteStalenessTracker()
    tr.record("X", at(0), at(0))
    tr.record("X", at(1), at(1))
    snap = tr.snapshot(at(3))
    assert snap["X"]["age_s"] == 2.0
    assert snap["X"]["gap_p50_s"] == 1.0
