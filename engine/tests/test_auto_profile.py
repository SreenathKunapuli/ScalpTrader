"""SCALP_PROFILE=auto: equity-band preset selection + day-roll re-banding."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from scalpengine.cli import resolve_scalp_profile
from scalpengine.config.scalp_tiers import (SCALP_LARGE, SCALP_MID,
                                            SCALP_SMALL, profile_for_equity)

from .test_bracket_wiring import make_scalp_engine

T0 = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)


def test_equity_bands():
    assert profile_for_equity(2_500) is SCALP_SMALL
    assert profile_for_equity(5_000) is SCALP_SMALL
    assert profile_for_equity(9_999) is SCALP_SMALL
    assert profile_for_equity(10_000) is SCALP_MID
    assert profile_for_equity(49_999) is SCALP_MID
    assert profile_for_equity(50_000) is SCALP_LARGE
    assert profile_for_equity(100_000) is SCALP_LARGE


def test_resolve_auto_and_manual():
    assert resolve_scalp_profile("auto", equity=5_000) is SCALP_SMALL
    assert resolve_scalp_profile("auto", equity=100_000) is SCALP_LARGE
    assert resolve_scalp_profile("mid") is SCALP_MID
    with pytest.raises(ValueError, match="auto needs"):
        resolve_scalp_profile("auto")


def test_day_roll_rebands_in_auto_mode(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker,
                                  scalp_cfg=SCALP_SMALL)
    engine.scalp_auto = True
    state.equity = 60_000.0          # grew out of the small band
    engine._roll_day(T0)
    assert engine.scalp_cfg is SCALP_LARGE
    assert engine.risk.scalp_cfg is SCALP_LARGE


def test_day_roll_keeps_band_without_auto(state, repo, mock_broker):
    engine, _ = make_scalp_engine(state, repo, mock_broker,
                                  scalp_cfg=SCALP_SMALL)
    state.equity = 60_000.0
    engine._roll_day(T0)
    assert engine.scalp_cfg is SCALP_SMALL   # manual profile never swapped
