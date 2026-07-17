"""Morning-only guarantee + join/IC correctness for the scanner ranker.

The bug class that fabricates scanner alpha is the AFTERNOON leaking into a
09:45 decision. Every feature test perturbs data STRICTLY after 09:45 and
asserts the features are byte-for-byte unchanged; the rest check hand-computed
joins and a hand-built perfect ranking (IC == 1.0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scanner import rank  # noqa: E402
from train_scanner import build_scanner_dataset  # noqa: E402


def _session_bars(date: str = "2024-06-03", n_seconds: int = 23400,
                  base: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """A full RTH runner day of 1s bars (UTC index), 09:30->16:00 ET."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{date} 09:30:01", tz="America/New_York").tz_convert("UTC")
    idx = pd.date_range(start, periods=n_seconds, freq="1s")
    close = base * np.exp(np.cumsum(rng.normal(0, 0.0005, n_seconds)))
    vol = rng.integers(50, 5000, n_seconds).astype(float)
    bid = close - 0.01
    ask = close + 0.01
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": vol, "n_trades": rng.integers(1, 40, n_seconds),
        "vwap": close, "bid": bid, "ask": ask,
        "bid_size": 100.0, "ask_size": 100.0, "spread": ask - bid,
    }, index=idx)


def _daily_row(open_px: float = 3.30, prev_close: float = 3.00,
               prev_day_volume: float = 1e6, prev_day_dollar_vol: float = 3e6):
    return {"open": open_px, "prev_close": prev_close,
            "prev_day_volume": prev_day_volume,
            "prev_day_dollar_vol": prev_day_dollar_vol}


# --------------------------------------------------------------------------- #
# 1. features are morning-only: the function rejects bars past 09:45 ET.
# --------------------------------------------------------------------------- #

def test_build_features_rejects_bars_past_0945():
    bars = _session_bars()               # full session, runs to 16:00 ET
    with pytest.raises(ValueError, match="09:45"):
        rank.build_scanner_features(_daily_row(), bars)


def test_build_features_rejects_naive_index():
    early = rank.slice_early_bars(_session_bars())
    naive = early.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        rank.build_scanner_features(_daily_row(), naive)


def test_first_15_min_window_is_accepted_and_values_sane():
    bars = _session_bars()
    early = rank.slice_early_bars(bars)
    # window is 09:30..09:45 inclusive of the closing 09:45:00 bar
    local = early.index.tz_convert("America/New_York")
    assert local.min().time() >= pd.Timestamp("09:30").time()
    assert local.max().time() <= pd.Timestamp("09:45").time()
    f = rank.build_scanner_features(_daily_row(), early)
    assert set(f) == set(rank.FEATURES)
    assert f["gap_pct"] == pytest.approx(3.30 / 3.00 - 1.0)
    assert f["prev_close"] == 3.00
    # relvol_at_open = first15 volume / prev-day total volume, in (0, 1)-ish
    assert 0.0 < f["relvol_at_open"] < 5.0
    assert f["first15_dollar_vol"] > 0
    assert f["first15_spread_bps_med"] > 0
    assert np.isfinite(f["first15_range_pct"])


# --------------------------------------------------------------------------- #
# 2. rewrite-the-afternoon perturbation: features unchanged.
# --------------------------------------------------------------------------- #

def test_afternoon_rewrite_does_not_move_features():
    bars = _session_bars(seed=1)
    early = rank.slice_early_bars(bars)
    before = rank.build_scanner_features(_daily_row(), early)

    mutated = bars.copy()
    ny = mutated.index.tz_convert("America/New_York")
    after_mask = ny.time > pd.Timestamp("09:45").time()
    assert after_mask.any()
    for col in ["open", "high", "low", "close", "vwap", "bid", "ask", "spread"]:
        mutated.loc[after_mask, col] *= 9.0
    mutated.loc[after_mask, "volume"] *= 1000.0

    early2 = rank.slice_early_bars(mutated)
    after = rank.build_scanner_features(_daily_row(), early2)
    assert before == after  # identical dict -> no afternoon leakage


def test_slice_early_bars_drops_everything_after_0945():
    early = rank.slice_early_bars(_session_bars())
    ny = early.index.tz_convert("America/New_York")
    assert (ny.time <= pd.Timestamp("09:45").time()).all()
    # 09:30:01 .. 09:45:00 inclusive -> 900 one-second bars
    assert len(early) == 900


# --------------------------------------------------------------------------- #
# 3. dataset join correctness on 3 synthetic stock-days.
# --------------------------------------------------------------------------- #

def test_dataset_join_three_days(tmp_path):
    specs = [("AAA", "2024-06-03", 3.0, 111.0),
             ("BBB", "2024-06-04", 4.0, 222.0),
             ("CCC", "2024-06-05", 5.0, 333.0)]
    corpus = tmp_path / "1s"
    corpus.mkdir()
    for sym, date, base, _pnl in specs:
        _session_bars(date=date, base=base).to_parquet(corpus / f"{sym}_{date}.parquet")

    # viability: label rows at the target horizon plus a decoy horizon that
    # must NOT be picked up.
    vrows = []
    for sym, date, _base, pnl in specs:
        vrows.append({"symbol": sym, "date": date, "horizon_s": 60,
                      "taker_clip_pnl": pnl})
        vrows.append({"symbol": sym, "date": date, "horizon_s": 300,
                      "taker_clip_pnl": pnl * 10})  # decoy
    viability = pd.DataFrame(vrows)

    # runner_index keyed by (symbol, ts-at-session-midnight ET->UTC).
    irows, ikeys = [], []
    for sym, date, base, _pnl in specs:
        ts = pd.Timestamp(f"{date} 00:00:00", tz="America/New_York").tz_convert("UTC")
        ikeys.append((sym, ts))
        irows.append({"open": base * 1.10, "prev_close": base, "volume": 1e6,
                      "dollar_vol": base * 1e6})
    index = pd.DataFrame(irows, index=pd.MultiIndex.from_tuples(
        ikeys, names=["symbol", "ts"]))

    files = [corpus / f"{s}_{d}.parquet" for s, d, _b, _p in specs]
    ds = build_scanner_dataset(viability, index, files, horizon_s=60)

    assert len(ds) == 3
    assert set(ds["symbol"]) == {"AAA", "BBB", "CCC"}
    # labels come from horizon 60, not the x10 decoy
    got = dict(zip(ds["symbol"], ds["label"]))
    assert got == {"AAA": 111.0, "BBB": 222.0, "CCC": 333.0}
    # gap_pct = open/prev_close - 1 = 1.10 - 1 = 0.10 for every row
    for gp in ds["gap_pct"]:
        assert gp == pytest.approx(0.10)
    # every morning feature present, no label overlap columns leaked
    assert set(rank.FEATURES).issubset(ds.columns)


def test_dataset_drops_day_without_label(tmp_path):
    corpus = tmp_path / "1s"
    corpus.mkdir()
    _session_bars(date="2024-06-03", base=3.0).to_parquet(
        corpus / "AAA_2024-06-03.parquet")
    _session_bars(date="2024-06-04", base=4.0).to_parquet(
        corpus / "BBB_2024-06-04.parquet")
    # only AAA has a horizon-60 label
    viability = pd.DataFrame([{"symbol": "AAA", "date": "2024-06-03",
                               "horizon_s": 60, "taker_clip_pnl": 50.0}])
    ts = pd.Timestamp("2024-06-03 00:00:00", tz="America/New_York").tz_convert("UTC")
    index = pd.DataFrame([{"open": 3.3, "prev_close": 3.0, "volume": 1e6,
                           "dollar_vol": 3e6}],
                         index=pd.MultiIndex.from_tuples(
                             [("AAA", ts)], names=["symbol", "ts"]))
    files = [corpus / "AAA_2024-06-03.parquet", corpus / "BBB_2024-06-04.parquet"]
    ds = build_scanner_dataset(viability, index, files, horizon_s=60)
    assert list(ds["symbol"]) == ["AAA"]


def test_prev_day_fields_derive_from_prior_session(tmp_path):
    """A recurring symbol's prev-day dollar-volume = its PRIOR runner row."""
    corpus = tmp_path / "1s"
    corpus.mkdir()
    for date, base in [("2024-06-03", 3.0), ("2024-07-01", 3.5)]:
        _session_bars(date=date, base=base).to_parquet(
            corpus / f"AAA_{date}.parquet")
    viability = pd.DataFrame([
        {"symbol": "AAA", "date": "2024-06-03", "horizon_s": 60, "taker_clip_pnl": 10.0},
        {"symbol": "AAA", "date": "2024-07-01", "horizon_s": 60, "taker_clip_pnl": 20.0},
    ])
    keys, rows = [], []
    for date, base, dv in [("2024-06-03", 3.0, 3e6), ("2024-07-01", 3.5, 7e6)]:
        keys.append(("AAA", pd.Timestamp(f"{date} 00:00:00",
                     tz="America/New_York").tz_convert("UTC")))
        rows.append({"open": base * 1.1, "prev_close": base, "volume": dv / base,
                     "dollar_vol": dv})
    index = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        keys, names=["symbol", "ts"]))
    files = [corpus / "AAA_2024-06-03.parquet", corpus / "AAA_2024-07-01.parquet"]
    ds = build_scanner_dataset(viability, index, files, horizon_s=60)
    ds = ds.set_index("date")
    # first session has no prior -> NaN; second inherits the first's dollar_vol
    assert np.isnan(ds.loc["2024-06-03", "prev_day_dollar_vol"])
    assert ds.loc["2024-07-01", "prev_day_dollar_vol"] == pytest.approx(3e6)


# --------------------------------------------------------------------------- #
# 4. IC computation exact on a hand-built case (perfect ranking -> 1.0).
# --------------------------------------------------------------------------- #

def test_rank_ic_perfect_and_inverted():
    realized = np.array([10.0, 50.0, 5.0, 100.0, 30.0])
    # a monotone transform of realized -> perfect Spearman ordering
    predicted = np.log1p(realized)
    assert rank.rank_ic(predicted, realized) == pytest.approx(1.0)
    assert rank.rank_ic(-predicted, realized) == pytest.approx(-1.0)


def test_rank_ic_drops_nan_pairs_and_needs_two():
    pred = np.array([1.0, 2.0, np.nan, 4.0])
    real = np.array([10.0, 20.0, 30.0, 40.0])
    assert rank.rank_ic(pred, real) == pytest.approx(1.0)  # 3 clean pairs, monotone
    assert np.isnan(rank.rank_ic([np.nan, 1.0], [2.0, np.nan]))  # <2 pairs


def test_decile_table_orders_realized_by_prediction():
    rng = np.random.default_rng(3)
    realized = rng.uniform(0, 1000, 200)
    predicted = realized + rng.normal(0, 10, 200)  # strong but noisy signal
    dec = rank.decile_table(predicted, realized, n_deciles=10)
    assert len(dec) == 10
    assert list(dec["decile"]) == list(range(10))
    # top predicted decile realizes more than the bottom
    assert dec["mean_realized_pnl"].iloc[-1] > dec["mean_realized_pnl"].iloc[0]
    assert dec["n"].sum() == 200


def test_label_scalpability_selects_horizon():
    viability = pd.DataFrame([
        {"symbol": "AAA", "date": "2024-06-03", "horizon_s": 60, "taker_clip_pnl": 42.0},
        {"symbol": "AAA", "date": "2024-06-03", "horizon_s": 300, "taker_clip_pnl": 99.0},
    ])
    assert rank.label_scalpability(viability, "AAA", "2024-06-03", 60) == 42.0
    assert np.isnan(rank.label_scalpability(viability, "ZZZ", "2024-06-03", 60))
