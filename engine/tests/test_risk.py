"""Sizing property tests + full RiskManager.approve() branch coverage."""

from __future__ import annotations

from datetime import UTC

from hypothesis import given
from hypothesis import strategies as st
from scalpengine.config.scalp_tiers import SCALP_LARGE
from scalpengine.config.tiers import TIERS, Tier
from scalpengine.risk.risk_manager import Approval, OrderIntent, Rejection, RiskManager
from scalpengine.risk.sizing import size_position
from scalpengine.risk.state import Position

from .conftest import IN_SESSION, HIGH_WIDE as HIGH, MED_WIDE as MED


# ---------- sizing ---------- #
@given(equity=st.floats(1_000, 10_000_000), price=st.floats(1, 5_000),
       atr=st.floats(0.01, 500), vol=st.floats(0.3, 1.0))
def test_sizing_never_exceeds_caps(equity: float, price: float, atr: float, vol: float) -> None:
    for tier in TIERS.values():
        shares = size_position(tier, equity, price, atr, vol)
        assert shares >= 0
        assert shares * price <= tier.max_position_pct * equity + price  # floor slack
        risk = shares * tier.stop_atr_multiple * atr
        assert risk <= tier.risk_per_trade_pct * equity + tier.stop_atr_multiple * atr


def test_sizing_zero_on_degenerate() -> None:
    assert size_position(MED, 0, 100, 1) == 0
    assert size_position(MED, 100_000, 0, 1) == 0
    assert size_position(MED, 100_000, 100, 0) == 0
    assert size_position(MED, 100, 5000, 1) == 0  # can't afford one share


# ---------- approve() branches ---------- #
def _intent(symbol: str = "AAPL", side: str = "buy", qty: int = 10,
            price: float = 100.0, reason: str = "signal") -> OrderIntent:
    return OrderIntent(symbol=symbol, side=side, qty=qty, price_hint=price, reason=reason)  # type: ignore[arg-type]


def test_rejects_when_halted(state) -> None:  # type: ignore[no-untyped-def]
    state.halted = True
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "HALTED" in r.reason


def test_rejects_unknown_symbol(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(symbol="GME"), IN_SESSION)
    assert isinstance(r, Rejection) and "universe" in r.reason


def test_rejects_nonpositive_qty(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(qty=0), IN_SESSION)
    assert isinstance(r, Rejection)


def test_rejects_short_in_medium(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(side="sell"), IN_SESSION)
    assert isinstance(r, Rejection) and "short" in r.reason


def test_allows_short_in_high(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(HIGH, state).approve(_intent(side="sell"), IN_SESSION)
    assert isinstance(r, Approval)


def test_rejects_position_size(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(qty=200, price=100.0), IN_SESSION)
    # 200*100 = 20k > 10% of 100k
    assert isinstance(r, Rejection) and "position size" in r.reason


def test_rejects_gross_exposure(state) -> None:  # type: ignore[no-untyped-def]
    for _i, sym in enumerate(["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"]):
        state.positions[sym] = Position(symbol=sym, qty=95, entry_price=100.0, mark=100.0)
    # gross = 8*9500 = 76k; +9.5k > 80% of 100k
    r = RiskManager(MED, state).approve(_intent(symbol="AAPL", qty=95), IN_SESSION)
    assert isinstance(r, Rejection) and "gross" in r.reason


def test_rejects_max_positions(state) -> None:  # type: ignore[no-untyped-def]
    for sym in MED.universe[:10]:
        state.positions[sym] = Position(symbol=sym, qty=1, entry_price=10.0, mark=10.0)
    r = RiskManager(MED, state).approve(_intent(symbol="UNH", qty=1, price=10.0), IN_SESSION)
    assert isinstance(r, Rejection) and "open positions" in r.reason


def test_rejects_daily_loss(state) -> None:  # type: ignore[no-untyped-def]
    state.intraday_realized_today = -2_500.0  # intraday book -2.5% on the day
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "daily loss" in r.reason


def test_rejects_drawdown(state) -> None:  # type: ignore[no-untyped-def]
    state.peak_equity = 160_000.0  # 100k now -> 37.5% DD > 35% MEDIUM floor
    state.day_start_equity = state.equity  # keep day pnl at 0
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "drawdown" in r.reason


def test_intraday_halt_blocks_entries(state) -> None:  # type: ignore[no-untyped-def]
    state.intraday_halted = True
    rm = RiskManager(MED, state)
    r = rm.approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "intraday" in r.reason


def test_rejects_outside_entry_window(state) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime

    weekend = datetime(2026, 6, 13, 15, 0, tzinfo=UTC)  # Saturday
    r = RiskManager(MED, state).approve(_intent(), weekend)
    assert isinstance(r, Rejection) and "window" in r.reason


def test_approves_valid_entry(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Approval) and r.token


def test_exit_bypasses_windows_but_not_direction(state) -> None:  # type: ignore[no-untyped-def]
    state.positions["AAPL"] = Position(symbol="AAPL", qty=10, entry_price=100.0, mark=100.0)
    rm = RiskManager(MED, state)
    ok = rm.approve_exit(_intent(side="sell", qty=10, reason="stop"))
    assert isinstance(ok, Approval)
    bad = rm.approve_exit(_intent(side="buy", qty=10))  # would increase exposure
    assert isinstance(bad, Rejection)


# ---------- scalp_cfg checks (only active when RiskManager has scalp_cfg) ---------- #
def _scalp_rm(state) -> RiskManager:  # type: ignore[no-untyped-def]
    return RiskManager(MED, state, scalp_cfg=SCALP_LARGE)


def test_scalp_price_band_rejects_out_of_band(state) -> None:  # type: ignore[no-untyped-def]
    rm = _scalp_rm(state)
    # SCALP_LARGE band is [0.5, 10.0]; 100.0 is above price_max
    r = rm.approve(_intent(price=100.0, qty=1), IN_SESSION)
    assert isinstance(r, Rejection) and "price band" in r.reason
    # below price_min
    r2 = rm.approve(_intent(price=0.25, qty=1), IN_SESSION)
    assert isinstance(r2, Rejection) and "price band" in r2.reason


def test_scalp_price_band_allows_in_band(state) -> None:  # type: ignore[no-untyped-def]
    rm = _scalp_rm(state)
    r = rm.approve(_intent(price=5.0, qty=1), IN_SESSION)
    assert isinstance(r, Approval)


def test_scalp_per_symbol_loss_cap_blocks_that_symbol_only(state) -> None:  # type: ignore[no-untyped-def]
    rm = _scalp_rm(state)
    # per_symbol_loss_cap_pct = 0.01 of 100k equity = -1000 threshold
    state.symbol_realized_today["AAPL"] = -1_000.0
    r = rm.approve(_intent(symbol="AAPL", price=5.0, qty=1), IN_SESSION)
    assert isinstance(r, Rejection) and "per-symbol loss cap" in r.reason
    # a DIFFERENT symbol (no realized loss) is unaffected
    ok = rm.approve(_intent(symbol="MSFT", price=5.0, qty=1), IN_SESSION)
    assert isinstance(ok, Approval)


def test_scalp_per_symbol_cap_still_allows_exit(state) -> None:  # type: ignore[no-untyped-def]
    rm = _scalp_rm(state)
    state.symbol_realized_today["AAPL"] = -2_000.0  # well past the cap
    state.positions["AAPL"] = Position(symbol="AAPL", qty=100, entry_price=5.0, mark=5.0)
    # exit path bypasses the entry-only scalp checks
    ok = rm.approve_exit(_intent(symbol="AAPL", side="sell", qty=100,
                                 price=5.0, reason="stop"))
    assert isinstance(ok, Approval)


def test_scalp_max_open_scalps_rejects_new_symbol(state) -> None:  # type: ignore[no-untyped-def]
    rm = _scalp_rm(state)
    # SCALP_LARGE.max_open_scalps = 5; fill the intraday book with 5 names
    # (universe[5:10] — the AAPL intent below must be a NEW sixth symbol)
    for sym in MED.universe[5:10]:
        state.positions[sym] = Position(symbol=sym, qty=1, entry_price=5.0, mark=5.0)
    r = rm.approve(_intent(symbol="AAPL", price=5.0, qty=1), IN_SESSION)
    assert isinstance(r, Rejection) and "max concurrent scalps" in r.reason
    # adding to an EXISTING scalp symbol is not a new scalp -> allowed
    ok = rm.approve(_intent(symbol=MED.universe[5], price=5.0, qty=1), IN_SESSION)
    assert isinstance(ok, Approval)


def test_scalp_checks_inert_without_cfg(state) -> None:  # type: ignore[no-untyped-def]
    # default RiskManager (scalp_cfg=None) ignores band/symbol-cap entirely
    rm = RiskManager(MED, state)
    state.symbol_realized_today["AAPL"] = -50_000.0
    r = rm.approve(_intent(price=100.0), IN_SESSION)
    assert isinstance(r, Approval)
