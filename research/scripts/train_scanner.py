"""Train + walk-forward-evaluate the morning-observable scanner ranker.

The scanner decides — at ~09:45 ET, without hindsight — which runner candidates
earn a websocket slot. This script assembles one row per corpus stock-day from
three sources joined on (symbol, date):
  * viability results.parquet -> the LABEL (taker_clip_pnl at horizon 60, the
    realized scalpability a perfect scalper could have extracted),
  * runner_index.parquet     -> settled-by-open daily fields (open, prev_close,
    and — where the symbol recurs — yesterday's dollar-volume / volume),
  * corpus 1s parquets        -> the first-15-min tape (09:30-09:45 ET only).

Days are split strictly temporally (scalp.walkforward.split_days), a HistGBT
REGRESSOR is fit on log1p(label clipped at 0), and every reported number is
out-of-sample: the Spearman rank-IC between predicted and realized scalpability
plus a decile table (mean realized PnL per predicted decile). Artifacts land in
<out>/<ts>/ (config incl. git head, metrics.json, decile_table.csv), mirroring
train_scalper.py.

Usage:
  .venv/bin/python research/scripts/train_scanner.py [--limit 0]
      [--horizon 60] [--out runs/scanner]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp.walkforward import TrainConfig, split_days  # noqa: E402
from scanner.rank import (  # noqa: E402
    FEATURES, LABEL_HORIZON_S, build_scanner_features, decile_table, rank_ic,
    slice_early_bars,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"
INDEX_PATH = ROOT / "data" / "runner_index.parquet"
VIABILITY_PATH = ROOT / "data" / "viability" / "results.parquet"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _daily_lookup(index: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """(symbol, date) -> settled-by-open daily fields for build_scanner_features.

    Yesterday's dollar-volume/volume are attached ONLY when the same symbol has
    an earlier runner-index session (its prior row); otherwise NaN (the model
    handles missing values natively). Same-day dollar_vol/volume are hindsight
    and are never forwarded as prev-day fields.
    """
    idx = index.sort_index()
    lut: dict[tuple[str, str], dict] = {}
    for sym, sub in idx.groupby(level="symbol", sort=False):
        prev_dv = sub["dollar_vol"].shift(1)
        prev_vol = sub["volume"].shift(1)
        for i, (_, ts) in enumerate(sub.index):
            date_str = pd.Timestamp(ts).date().isoformat()
            row = sub.iloc[i]
            lut[(sym, date_str)] = {
                "open": float(row.get("open", np.nan)),
                "prev_close": float(row.get("prev_close", np.nan)),
                "prev_day_dollar_vol": float(prev_dv.iloc[i]),
                "prev_day_volume": float(prev_vol.iloc[i]),
            }
    return lut


def build_scanner_dataset(viability: pd.DataFrame, index: pd.DataFrame,
                          files: list[Path], horizon_s: int = LABEL_HORIZON_S,
                          ) -> pd.DataFrame:
    """One row per stock-day: morning features + realized-scalpability label.

    Joins the three sources on (symbol, date). A day is dropped only when it has
    no first-15-min bars or no viability label at the horizon — never for a
    missing prev-day field (that stays NaN).
    """
    daily = _daily_lookup(index)
    lab = viability[viability["horizon_s"] == horizon_s]
    label_lut = {(r.symbol, r.date): float(r.taker_clip_pnl)
                 for r in lab.itertuples()}

    rows = []
    for path in sorted(files):
        sym, date = path.stem.rsplit("_", 1)
        label = label_lut.get((sym, date))
        if label is None or label != label:  # missing / NaN label
            continue
        bars = pd.read_parquet(path)
        early = slice_early_bars(bars)
        if early.empty:
            continue
        drow = daily.get((sym, date), {"open": np.nan, "prev_close": np.nan})
        feats = build_scanner_features(drow, early)
        feats.update({"symbol": sym, "date": date, "label": label})
        rows.append(feats)
    if not rows:
        raise ValueError("no scanner rows assembled (check corpus / viability)")
    return pd.DataFrame(rows)


def fit_regressor(x: pd.DataFrame, y: pd.Series, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(x, y)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=LABEL_HORIZON_S,
                   help="viability horizon (s) whose taker_clip_pnl is the label")
    p.add_argument("--limit", type=int, default=0, help="cap #stock-days")
    p.add_argument("--out", default="runs/scanner")
    args = p.parse_args()

    if not VIABILITY_PATH.exists():
        print(f"missing {VIABILITY_PATH}; run the viability study first")
        return

    viability = pd.read_parquet(VIABILITY_PATH)
    index = pd.read_parquet(INDEX_PATH)
    man = pd.read_csv(MANIFEST)
    ok = man[man["status"] == "ok"]
    files = [CORPUS_DIR / f"{r.symbol}_{r.date}.parquet"
             for r in ok.itertuples()
             if (CORPUS_DIR / f"{r.symbol}_{r.date}.parquet").exists()]
    if args.limit:
        files = files[: args.limit]

    print(f"assembling scanner dataset from {len(files)} stock-days "
          f"(label = taker_clip_pnl @ {args.horizon}s) ...", flush=True)
    ds = build_scanner_dataset(viability, index, files, args.horizon)
    print(f"  {len(ds)} rows with a label", flush=True)

    cfg = TrainConfig()
    dates = ds["date"].tolist()
    train_dates, test_dates = split_days(dates, cfg)
    tr = ds[ds["date"].isin(train_dates)]
    te = ds[ds["date"].isin(test_dates)]
    print(f"  train {len(tr)} (≤{max(train_dates) if train_dates else '-'}) | "
          f"test {len(te)} (≥{min(test_dates)})", flush=True)

    # log1p of the label clipped at 0 — the scalper can decline any day, so only
    # positive realized edge is worth ordering by; log tames the fat right tail.
    y_tr = np.log1p(tr["label"].clip(lower=0.0))
    model = fit_regressor(tr[FEATURES], y_tr, cfg.seed)

    pred_te = model.predict(te[FEATURES])
    ic = rank_ic(pred_te, te["label"].to_numpy())
    dec = decile_table(pred_te, te["label"].to_numpy())

    out = ROOT / args.out / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "features": FEATURES,
        "label": f"taker_clip_pnl@{args.horizon}s (log1p, clip>=0)",
        "seed": cfg.seed,
        "git_head": _git_head(),
        "n_train_days": len(train_dates), "n_test_days": len(test_dates),
        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
    }, indent=2))
    metrics = {
        "oos_rank_ic": ic,
        "n_test_rows": int(len(te)),
        "test_range": [min(test_dates), max(test_dates)],
        "decile_table": dec.to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    dec.to_csv(out / "decile_table.csv", index=False)

    print(f"\nOOS REPORT (test days {min(test_dates)} .. {max(test_dates)})")
    print(f"  rank-IC (predicted vs realized scalpability): {ic:.4f}")
    print("  decile table (mean realized taker_clip_pnl per predicted decile):")
    print(dec.round(2).to_string(index=False))
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
