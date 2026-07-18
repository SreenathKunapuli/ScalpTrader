"""Causal per-second features from 1-second bar data.

All features at time t use ONLY bars up to and including t — no lookahead.
The bar grid is assumed to be contiguous seconds within RTH, so row-based
rolling windows are equivalent to time-based windows; this avoids the
ambiguity of pandas time-based rolling on gappy grids.

Input shape: DatetimeIndex (UTC), columns
    open, high, low, close, volume, vwap, n_trades, bid, ask,
    bid_size, ask_size, spread
Seconds with no trades have NaN OHLC but volume=0. NBBO can be NaN.

Output: feature DataFrame with same index as input, no dropna applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
_RTH_OPEN_H, _RTH_OPEN_M = 9, 30  # 09:30 ET


@dataclass(frozen=True)
class FeatureConfig:
    ret_windows: tuple[int, ...] = (5, 15, 60, 300)   # seconds for ret_Xs
    mom_accel_window: int = 15                          # for ret-diff acceleration
    vwap_dist: bool = True
    pullback_window: int = 300
    vol_surge_short: int = 5
    vol_surge_long: int = 300
    tape_speed_short: int = 10
    tape_speed_long: int = 300


def build_features(
    bars: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """Compute causal per-second features from 1-second bars.

    Parameters
    ----------
    bars:
        Second-bar DataFrame produced by corpus.second_bars().  Index must be
        a monotonically increasing UTC DatetimeIndex with contiguous 1-second
        steps within RTH.
    cfg:
        Tuning knobs; defaults match the spec.

    Returns
    -------
    pd.DataFrame
        Same index as *bars*, columns as described in the module docstring.
        No rows are dropped; caller decides what to do with NaN-heavy rows.
    """
    if bars.empty:
        return pd.DataFrame(index=bars.index)

    out = pd.DataFrame(index=bars.index)

    # ------------------------------------------------------------------
    # Price series: last-traded price, forward-filled from close
    # ------------------------------------------------------------------
    px = bars["close"].ffill()

    # ------------------------------------------------------------------
    # Returns: pct-change over fixed row windows
    # ------------------------------------------------------------------
    for w in cfg.ret_windows:
        out[f"ret_{w}s"] = px.pct_change(periods=w)

    # ------------------------------------------------------------------
    # Momentum acceleration: burst proxy
    # ------------------------------------------------------------------
    w_mom = cfg.mom_accel_window
    ret_mom = px.pct_change(periods=w_mom)
    out["mom_accel"] = ret_mom - ret_mom.shift(w_mom)

    # ------------------------------------------------------------------
    # VWAP distance from running session VWAP (causal cumsum)
    # ------------------------------------------------------------------
    # Use vwap*volume for notional; bars with volume=0 contribute 0 notional.
    notional = (bars["vwap"].fillna(0.0) * bars["volume"])
    cum_notional = notional.cumsum()
    cum_volume = bars["volume"].cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        session_vwap = cum_notional / cum_volume.replace(0, np.nan)
    out["vwap_dist"] = (px - session_vwap) / session_vwap

    # ------------------------------------------------------------------
    # Pullback: depth below rolling max (row-window, causal)
    # ------------------------------------------------------------------
    pw = cfg.pullback_window
    rolling_max = px.rolling(pw, min_periods=1).max()
    out["pullback"] = px / rolling_max - 1.0

    # ------------------------------------------------------------------
    # Volume surge: short rolling sum vs long rolling median of that sum
    # ------------------------------------------------------------------
    vs = cfg.vol_surge_short
    vl = cfg.vol_surge_long
    vol_short = bars["volume"].rolling(vs, min_periods=1).sum()
    vol_long_med = vol_short.rolling(vl, min_periods=1).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["vol_surge"] = vol_short / vol_long_med.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Tape speed: n_trades rolling sum vs long rolling median
    # ------------------------------------------------------------------
    ts = cfg.tape_speed_short
    tl = cfg.tape_speed_long
    trade_short = bars["n_trades"].rolling(ts, min_periods=1).sum()
    trade_long_med = trade_short.rolling(tl, min_periods=1).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["tape_speed"] = trade_short / trade_long_med.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Spread in basis points
    # ------------------------------------------------------------------
    mid = (bars["bid"] + bars["ask"]) / 2.0
    out["spread_bps"] = bars["spread"] / mid * 1e4

    # ------------------------------------------------------------------
    # Book-pressure microstructure (all causal: rolling/shift/cummax only)
    # ------------------------------------------------------------------
    # displayed-size imbalance: bid-heavy book -> buyers stacking up
    denom = bars["bid_size"] + bars["ask_size"]
    with np.errstate(invalid="ignore", divide="ignore"):
        qimb = (bars["bid_size"] - bars["ask_size"]) / denom.replace(0, np.nan)
    out["qimb"] = qimb
    out["qimb_chg_30s"] = qimb - qimb.shift(30)
    # spread regime: current spread vs its trailing norm (narrowing = urgency)
    spread_med = out["spread_bps"].rolling(300, min_periods=30).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["spread_rel"] = out["spread_bps"] / spread_med.replace(0, np.nan) - 1.0
    # bid-side momentum: the BID moving up is real buyer pressure, not prints
    out["bid_ret_15s"] = bars["bid"].pct_change(periods=15)
    # breakout proximity: distance below the running session high
    out["sess_hi_dist"] = px / px.cummax() - 1.0
    # momentum persistence: consecutive up-seconds (capped at 30)
    up = (px.diff() > 0).astype(float)
    grp = (up == 0).cumsum()
    out["up_streak"] = up.groupby(grp).cumsum().clip(upper=30)
    # large-print detector: recent mean trade size vs trailing norm
    with np.errstate(invalid="ignore", divide="ignore"):
        tsize = (bars["volume"].rolling(60, min_periods=1).sum()
                 / bars["n_trades"].rolling(60, min_periods=1).sum()
                 .replace(0, np.nan))
        out["tsize_surge"] = tsize / tsize.rolling(300, min_periods=30) \
            .median().replace(0, np.nan)

    # ------------------------------------------------------------------
    # Quote validity flag
    # ------------------------------------------------------------------
    out["quote_ok"] = np.where(
        bars["bid"].notna() & bars["ask"].notna(), 1.0, 0.0
    )

    # ------------------------------------------------------------------
    # Time of day: minutes since 09:30 ET
    # ------------------------------------------------------------------
    local = bars.index.tz_convert(NY)
    rth_open = local.normalize() + pd.Timedelta(
        hours=_RTH_OPEN_H, minutes=_RTH_OPEN_M
    )
    out["tod_min"] = (local - rth_open).total_seconds() / 60.0

    return out
