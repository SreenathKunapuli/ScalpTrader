"""Rollup viability report across the corpus of runner-day second bars.

Question answered: across real market microstructure (spreads, sizes, fees),
does a perfect-foresight oracle make money scalping upward bursts on runner
days? And where does it work — by price bucket, by horizon, by fill style?
This report surfaces the go/no-go signal and the levers that most explain the
variance, so Phase 2 modelling effort is focused on viable regimes only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp.viability import FeeModel, OracleConfig, day_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus" / "1s"
MANIFEST = DATA / "corpus" / "manifest.csv"
INDEX_PATH = DATA / "runner_index.parquet"

PRICE_BUCKETS = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]
BUCKET_LABELS = ["0.5-2", "2-5", "5-10"]


def _price_bucket(price: float) -> str:
    for (lo, hi), label in zip(PRICE_BUCKETS, BUCKET_LABELS):
        if lo <= price < hi:
            return label
    return "other"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        return pd.DataFrame(columns=["symbol", "date", "status", "rows", "n_trades"])
    return pd.read_csv(MANIFEST)


def _load_runner_index() -> pd.DataFrame | None:
    if not INDEX_PATH.exists():
        return None
    return pd.read_parquet(INDEX_PATH)


def _parse_horizons(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _run_study(horizons: list[int], clip: int, limit: int | None) -> list[dict]:
    """Core loop: for each ok stock-day x horizon, run day_summary."""
    manifest = _load_manifest()
    ok_rows = manifest[manifest["status"] == "ok"].copy()

    runner_idx = _load_runner_index()

    # Build a lookup: (symbol, date_str) -> {gain, relvol, dollar_vol, open}
    meta: dict[tuple[str, str], dict] = {}
    if runner_idx is not None:
        for (sym, ts), row in runner_idx.iterrows():
            date_str = pd.Timestamp(ts).date().isoformat()
            meta[(sym, date_str)] = {
                "gain": float(row.get("gain", float("nan"))),
                "relvol": float(row.get("relvol", float("nan"))),
                "dollar_vol": float(row.get("dollar_vol", float("nan"))),
                "open": float(row.get("open", float("nan"))),
            }

    # Cap stock-days if --limit supplied
    if limit is not None:
        ok_rows = ok_rows.head(limit)

    records: list[dict] = []
    fee_model = FeeModel()

    for _, mrow in ok_rows.iterrows():
        sym, date = str(mrow["symbol"]), str(mrow["date"])
        parquet = CORPUS_DIR / f"{sym}_{date}.parquet"
        if not parquet.exists():
            continue

        try:
            bars = pd.read_parquet(parquet)
        except Exception:  # noqa: BLE001
            continue

        m = meta.get((sym, date), {})
        open_px = m.get("open", float("nan"))
        bucket = _price_bucket(open_px) if not (open_px != open_px) else "other"

        for h in horizons:
            cfg = OracleConfig(horizon_s=h, clip_shares=clip, fees=fee_model)
            try:
                summary = day_summary(bars, cfg)
            except Exception:  # noqa: BLE001
                continue
            records.append({
                "symbol": sym,
                "date": date,
                "price_bucket": bucket,
                "gain": m.get("gain"),
                "relvol": m.get("relvol"),
                "dollar_vol": m.get("dollar_vol"),
                **summary,
            })

    return records


def _pivot_median(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot_table(index="price_bucket", columns="horizon_s",
                          values=value_col, aggfunc="median")


def _pivot_pct_positive(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pos = (df[value_col] > 0).astype(float)
    tmp = df.copy()
    tmp["_pos"] = pos
    return tmp.pivot_table(index="price_bucket", columns="horizon_s",
                           values="_pos", aggfunc="mean") * 100


def _print_report(df: pd.DataFrame) -> None:
    n_stock_days = df[["symbol", "date"]].drop_duplicates().shape[0]
    h60 = df[df["horizon_s"] == 60] if 60 in df["horizon_s"].values else pd.DataFrame()

    dates = pd.to_datetime(df["date"])
    date_range = f"{dates.min().date()} – {dates.max().date()}"
    bucket_counts = df.groupby("price_bucket")[["symbol", "date"]].apply(
        lambda g: g.drop_duplicates().shape[0]
    )
    med_qc = df.groupby(["symbol", "date"])["quote_coverage"].first().median()

    print("=" * 70)
    print("VIABILITY STUDY REPORT")
    print("=" * 70)
    print(f"Corpus: {n_stock_days} stock-days  |  {date_range}")
    print(f"Bucket stock-days: {dict(bucket_counts)}")
    print(f"Median quote_coverage: {med_qc:.3f}")
    print()

    print("Table A — Median taker_clip_pnl/day ($ at clip_shares) by bucket x horizon")
    ta = _pivot_median(df, "taker_clip_pnl")
    print(ta.to_string(float_format=lambda x: f"{x:,.2f}"))
    print()

    print("Table B — Median maker_clip_pnl/day ($ at clip_shares) by bucket x horizon")
    tb = _pivot_median(df, "maker_clip_pnl")
    print(tb.to_string(float_format=lambda x: f"{x:,.2f}"))
    print()

    print("Table C — % stock-days with positive clip_pnl (taker | maker) by bucket x horizon")
    tc_t = _pivot_pct_positive(df, "taker_clip_pnl")
    tc_m = _pivot_pct_positive(df, "maker_clip_pnl")
    tc_t.columns = pd.MultiIndex.from_tuples(
        [("taker%", c) for c in tc_t.columns])
    tc_m.columns = pd.MultiIndex.from_tuples(
        [("maker%", c) for c in tc_m.columns])
    tc = pd.concat([tc_t, tc_m], axis=1)
    print(tc.to_string(float_format=lambda x: f"{x:.1f}"))
    print()

    if not h60.empty:
        print("Table D — Median med_spread_bps and taker_n (horizon=60 only) by bucket")
        td = h60.groupby("price_bucket").agg(
            med_spread_bps=("med_spread_bps", "median"),
            med_taker_n=("taker_n", "median"),
        )
        print(td.to_string(float_format=lambda x: f"{x:.2f}"))
        print()

    print("=" * 70)
    print("VERDICT INPUTS")
    print("=" * 70)
    for h in sorted(df["horizon_s"].unique()):
        sub = df[df["horizon_s"] == h]
        tp = (sub["taker_clip_pnl"] > 0).mean() * 100
        mp = (sub["maker_clip_pnl"] > 0).mean() * 100
        print(f"  horizon={h:>4}s  taker-positive={tp:5.1f}%  "
              f"maker-positive={mp:5.1f}%  (n={len(sub)})")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizons", default="30,60,120,300,900",
                   help="Comma-separated horizon seconds (default: 30,60,120,300,900)")
    p.add_argument("--clip", type=int, default=1000,
                   help="Clip shares for unit-economics calculation (default: 1000)")
    p.add_argument("--out", default="data/viability",
                   help="Output directory (default: data/viability)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of stock-days processed (for quick runs)")
    args = p.parse_args()

    horizons = _parse_horizons(args.horizons)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running viability study: horizons={horizons} clip={args.clip} "
          f"limit={args.limit}", flush=True)

    records = _run_study(horizons, args.clip, args.limit)

    if not records:
        print("No stock-days processed — corpus may still be downloading or "
              "no status=='ok' rows with existing parquets found.")
        return

    df = pd.DataFrame(records)
    results_path = out_dir / "results.parquet"
    df.to_parquet(results_path)
    print(f"Saved {len(df)} rows -> {results_path}", flush=True)

    fee = FeeModel()
    n_stock_days = df[["symbol", "date"]].drop_duplicates().shape[0]
    config = {
        "horizons": horizons,
        "clip": args.clip,
        "fees": {
            "sec_rate": fee.sec_rate,
            "taf_per_share": fee.taf_per_share,
            "taf_cap": fee.taf_cap,
        },
        "n_stock_days": n_stock_days,
        "git_head": _git_head(),
    }
    config_path = out_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved config -> {config_path}", flush=True)

    _print_report(df)


if __name__ == "__main__":
    main()
