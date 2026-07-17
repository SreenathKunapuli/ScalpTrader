"""Morning-observable scanner ranker: which runner candidates deserve a
websocket slot, decided WITHOUT hindsight.

The scanner runs at ~09:45 ET on a firehose of gappers and must commit a
handful of names to the (finite) live tick pipeline before the day plays
out. Every feature here is computable from what is KNOWN by 09:45: yesterday's
close/dollar-volume (fully settled at the open) and the first fifteen minutes
of today's 1s tape. The day's CLOSE-based gain — the RunnerCriteria trigger in
scalp.corpus — is HINDSIGHT and is deliberately absent; using it would leak the
answer into the question.

The label the ranker chases is realized SCALPABILITY, not direction: the
perfect-foresight oracle's taker_clip_pnl at horizon 60 from the viability
study. That is the dollar edge a flawless scalper could have extracted, i.e.
exactly what a websocket slot is worth. Cross-sectional rank-IC (predicted vs
realized) is the honest score — the ranker only has to order candidates, not
price them.

Everything here is pure pandas so the morning-only guarantee is
perturbation-testable: rewriting the afternoon must not move a single feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
EARLY_START = "09:30"        # RTH open (ET)
EARLY_END = "09:45"          # first fifteen minutes; scanner decision time
LABEL_HORIZON_S = 60         # viability horizon whose taker_clip_pnl is the label

# Morning-observable feature order (also the model's column order).
FEATURES = ["gap_pct", "prev_close", "prev_day_dollar_vol", "relvol_at_open",
            "first15_range_pct", "first15_vwap_dist", "first15_spread_bps_med",
            "first15_dollar_vol"]


def _assert_morning_only(early_bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the ET-localized index after asserting it lies in [09:30, 09:45).

    The window is closed-open on the right: a bar labeled 09:45:00 covers
    (09:44:59, 09:45:00] and is still within the first fifteen minutes, so the
    guard rejects only bars STRICTLY after 09:45:00. Raising here (rather than
    silently slicing) is the whole point — a caller that hands us the afternoon
    has a lookahead bug, and we want it to fail loudly.
    """
    if early_bars.empty:
        raise ValueError("early_bars is empty; need first-15-min 1s bars")
    idx = early_bars.index
    if getattr(idx, "tz", None) is None:
        raise ValueError("early_bars index must be tz-aware (UTC ok)")
    local = idx.tz_convert(NY)
    lo = pd.Timestamp(EARLY_START).time()
    hi = pd.Timestamp(EARLY_END).time()
    t = local.time
    if (t < lo).any() or (t > hi).any():
        raise ValueError(
            f"early_bars must be within [{EARLY_START}, {EARLY_END}] ET only; "
            f"got {local.min()} .. {local.max()} — pass ONLY the first 15 min")
    return local


def build_scanner_features(daily_row: pd.Series | dict,
                           early_bars: pd.DataFrame) -> dict:
    """Morning-observable features for one runner candidate at ~09:45 ET.

    `daily_row`: settled-by-open fields for the stock-day — `open` and
    `prev_close` (required, for the gap), plus optional `prev_day_dollar_vol`
    and `prev_day_volume` (yesterday is fully known at the open, so these are
    fair game; today's full-day volume/dollar_vol are NOT and are ignored even
    if present). `early_bars`: 1s bars from 09:30:00-09:45:00 ET ONLY — the
    function asserts this window and raises if handed anything later.

    relvol_at_open uses the first-15-min volume against yesterday's TOTAL volume
    as a same-time-of-day-free proxy for how unusually active the name is; a big
    fraction of a full prior session traded in fifteen minutes screams runner.
    """
    row = daily_row if isinstance(daily_row, dict) else daily_row.to_dict()
    self_local = _assert_morning_only(early_bars)  # noqa: F841 (guard side effect)

    open_px = float(row.get("open", np.nan))
    prev_close = float(row.get("prev_close", np.nan))
    gap_pct = open_px / prev_close - 1.0 if prev_close and prev_close == prev_close \
        else np.nan

    prev_day_dollar_vol = float(row.get("prev_day_dollar_vol", np.nan))
    prev_day_volume = float(row.get("prev_day_volume", np.nan))

    px = early_bars["close"].astype(float)
    vol = early_bars["volume"].astype(float).fillna(0.0)
    traded = px.dropna()

    first15_dollar_vol = float((px.fillna(0.0) * vol).sum())
    first15_volume = float(vol.sum())
    relvol_at_open = (first15_volume / prev_day_volume
                      if prev_day_volume and prev_day_volume == prev_day_volume
                      else np.nan)

    if len(traded):
        hi, lo = float(traded.max()), float(traded.min())
        ref = open_px if open_px == open_px and open_px > 0 else float(traded.iloc[0])
        first15_range_pct = (hi - lo) / ref if ref else np.nan
        # VWAP of the window, and how far last print sits above/below it.
        w_vwap = (float((traded * vol.reindex(traded.index).fillna(0.0)).sum())
                  / first15_volume) if first15_volume > 0 else float(traded.mean())
        last_px = float(traded.iloc[-1])
        first15_vwap_dist = last_px / w_vwap - 1.0 if w_vwap else np.nan
    else:
        first15_range_pct = np.nan
        first15_vwap_dist = np.nan

    if "spread" in early_bars.columns and "bid" in early_bars.columns:
        mid = (early_bars["bid"].astype(float) + early_bars["ask"].astype(float)) / 2.0
        spread_bps = (early_bars["spread"].astype(float) / mid) * 1e4
        med = spread_bps.replace([np.inf, -np.inf], np.nan).dropna()
        first15_spread_bps_med = float(med.median()) if len(med) else np.nan
    else:
        first15_spread_bps_med = np.nan

    return {
        "gap_pct": gap_pct,
        "prev_close": prev_close,
        "prev_day_dollar_vol": prev_day_dollar_vol,
        "relvol_at_open": relvol_at_open,
        "first15_range_pct": first15_range_pct,
        "first15_vwap_dist": first15_vwap_dist,
        "first15_spread_bps_med": first15_spread_bps_med,
        "first15_dollar_vol": first15_dollar_vol,
    }


def slice_early_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Take a full runner-day 1s panel down to the 09:30:00-09:45:00 ET window.

    Convenience for callers holding the whole session; `build_scanner_features`
    itself never widens the window — it only ever narrows/validates.
    """
    if bars.empty:
        return bars
    local = bars.index.tz_convert(NY)
    lo = pd.Timestamp(EARLY_START).time()
    hi = pd.Timestamp(EARLY_END).time()
    mask = (local.time >= lo) & (local.time <= hi)
    return bars[mask]


def label_scalpability(viability: pd.DataFrame, symbol: str, date: str,
                       horizon_s: int = LABEL_HORIZON_S) -> float:
    """Realized scalpability = taker_clip_pnl at `horizon_s` for the stock-day.

    Returns NaN when the (symbol, date, horizon) row is absent from the
    viability results (e.g. corpus day the study skipped).
    """
    m = ((viability["symbol"] == symbol) & (viability["date"] == date)
         & (viability["horizon_s"] == horizon_s))
    hit = viability.loc[m, "taker_clip_pnl"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def rank_ic(predicted: pd.Series | np.ndarray,
            realized: pd.Series | np.ndarray) -> float:
    """Spearman rank-IC between predicted and realized scalpability.

    NaN pairs are dropped; fewer than two comparable pairs -> NaN. A perfect
    monotone ranking returns 1.0 (the training script's headline OOS number).
    """
    p = pd.Series(np.asarray(predicted, dtype=float)).reset_index(drop=True)
    r = pd.Series(np.asarray(realized, dtype=float)).reset_index(drop=True)
    both = pd.concat([p, r], axis=1).dropna()
    if len(both) < 2:
        return float("nan")
    return float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))


def decile_table(predicted: pd.Series | np.ndarray,
                 realized: pd.Series | np.ndarray,
                 n_deciles: int = 10) -> pd.DataFrame:
    """Mean realized scalpability per predicted decile (0 = lowest predicted).

    The go/no-go table for the ranker: a useful scanner puts the fat realized
    PnL in the top predicted deciles. Columns: decile, n, mean_realized_pnl,
    median_realized_pnl. Bins are quantile cuts on the prediction; when ties or
    scarcity collapse bins, the surviving deciles are reported as-is.
    """
    p = pd.Series(np.asarray(predicted, dtype=float)).reset_index(drop=True)
    r = pd.Series(np.asarray(realized, dtype=float)).reset_index(drop=True)
    both = pd.concat([p.rename("pred"), r.rename("real")], axis=1).dropna()
    if both.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_realized_pnl",
                                     "median_realized_pnl"])
    q = min(n_deciles, both["pred"].nunique())
    both["decile"] = pd.qcut(both["pred"].rank(method="first"), q,
                             labels=False, duplicates="drop")
    g = both.groupby("decile")["real"]
    out = pd.DataFrame({
        "decile": g.mean().index.astype(int),
        "n": g.size().to_numpy(),
        "mean_realized_pnl": g.mean().to_numpy(),
        "median_realized_pnl": g.median().to_numpy(),
    }).reset_index(drop=True)
    return out
