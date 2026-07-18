"""Absolute equity floor: halt before the account sinks under the
day-trading minimum (small-account mode)."""

from __future__ import annotations

from scalpengine.config.settings import Settings
from scalpengine.config.tiers import TIERS, Tier
from scalpengine.risk.kill_switch import KillSwitch


def make_kill(state, repo, floor):
    return KillSwitch(state, TIERS[Tier.MEDIUM], repo, object(),
                      emit=None, min_equity_usd=floor)


def test_floor_fires_account_scope(state, repo, mock_broker):
    state.equity = 2_400.0
    scope, reason = make_kill(state, repo, 2_600.0).check_triggers()
    assert scope == "account" and "equity floor" in reason


def test_above_floor_no_fire(state, repo, mock_broker):
    state.equity = 2_601.0
    state.peak_equity = 2_601.0
    state.day_start_equity = 2_601.0
    assert make_kill(state, repo, 2_600.0).check_triggers() is None


def test_disabled_and_zero_equity_guard(state, repo, mock_broker):
    # None = disabled; equity 0 (pre-reconcile) must not false-fire either
    state.equity = 0.0
    state.peak_equity = 0.0
    assert make_kill(state, repo, None).check_triggers() is None
    assert make_kill(state, repo, 2_600.0).check_triggers() is None


def test_settings_default_off():
    s = Settings(_env_file=None)
    assert s.min_equity_halt_usd == 0.0
