"""Train + walk-forward-evaluate the GBT scalp-entry model on the corpus.

Every number printed here is out-of-sample (later days than training).
Artifacts land in <out>/<ts>/ so results are reproducible from cached data:
config.json (incl. git head), metrics.json, per_threshold.csv,
feature_importances.csv.

Usage:
  .venv/bin/python research/scripts/train_scalper.py [--limit 60]
      [--target-ps 0.05] [--stop-ps 0.04] [--timeout 120] [--out runs/scalper]
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
from scalp.walkforward import (
    TrainConfig,
    build_dataset,
    evaluate,
    fit_model,
    split_days,  # noqa: E402
    split_val_days,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"


def parse_drop_features(arg: str | None) -> list[str]:
    """"a, b,,c" -> ["a", "b", "c"]; None/"" -> []."""
    if not arg:
        return []
    return [c.strip() for c in arg.split(",") if c.strip()]


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop `cols` from `df`; raise on any name not present."""
    if not cols:
        return df
    unknown = [c for c in cols if c not in df.columns]
    if unknown:
        raise ValueError(f"--drop-features: unknown column(s) {unknown}; "
                         f"available: {sorted(df.columns)}")
    return df.drop(columns=cols)


def limit_by_quality(candidates: list[Path], ranked_files: list[Path],
                     limit: int | None) -> list[Path]:
    """Corpus quality-depth knob: restrict `candidates` to the files that
    also appear among the first `limit` entries of `ranked_files` (manifest
    fetch-priority / quality-rank order — index 0 is the highest-quality
    row). `limit=None` is the identity (no filter, today's behavior).

    This is applied ONLY to the training-FIT file set (core-train / the fit
    set), never to val or test — those must be judged on identical days
    regardless of this knob. Order and any duplicates in `candidates` are
    preserved; only membership is tested."""
    if limit is None:
        return candidates
    allowed = set(ranked_files[:limit])
    return [f for f in candidates if f in allowed]


def quality_weight_array(meta: pd.DataFrame, top_stems: set[str],
                         mult: float) -> np.ndarray:
    """Per-row sample-weight multiplier: `mult` for rows whose source file
    stem ("{symbol}_{date}") is in `top_stems`, 1.0 for everything else.
    `meta` is build_dataset's third return (needs "symbol" and "date")."""
    stems = meta["symbol"].astype(str) + "_" + meta["date"].astype(str)
    return np.where(stems.isin(top_stems), mult, 1.0)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-ps", type=float, default=0.05)
    p.add_argument("--stop-ps", type=float, default=0.04)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--barrier-mode", choices=["fixed", "vol"], default="fixed")
    p.add_argument("--vol-target-mult", type=float, default=1.0)
    p.add_argument("--vol-stop-mult", type=float, default=0.5)
    p.add_argument("--vol-window", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="cap #stock-days")
    p.add_argument("--out", default="runs/scalper")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--max-leaf-nodes", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=None)
    p.add_argument("--l2-regularization", type=float, default=None)
    p.add_argument("--drop-features", default=None,
                   help="comma-separated feature columns to drop, e.g. 'a,b,c'")
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="carve this fraction of the LAST train days into an "
                        "inner validation set, evaluated before the OOS test")
    p.add_argument("--test-start-date", default=None,
                   help="pin the OOS test window to all days >= this ISO "
                        "date (yyyy-mm-dd) instead of the trailing "
                        "test_frac fraction, so a growing corpus keeps a "
                        "comparable test set")
    p.add_argument("--train-start-date", default=None,
                   help="drop train days older than this ISO date "
                        "(yyyy-mm-dd), applied after the test/embargo "
                        "split so it never touches the test window — a "
                        "training-recency knob")
    p.add_argument("--val-start-date", default=None,
                   help="pin the inner validation window to all TRAIN "
                        "days >= this ISO date (yyyy-mm-dd) instead of "
                        "the trailing --val-frac fraction; passing this "
                        "alone (without --val-frac) still activates the "
                        "val split")
    p.add_argument("--train-quality-limit", type=int, default=None,
                   help="corpus quality-depth knob: after the day splits "
                        "are computed on the full file list, restrict the "
                        "TRAINING-FIT file set (core-train) to files among "
                        "the first N rows of the manifest (status=='ok' "
                        "rows, in fetch-priority / quality-rank order). "
                        "Val and test file sets are NEVER filtered. "
                        "None (default) applies no filter.")
    p.add_argument("--quality-weight-mult", type=float, default=None,
                   help="soft quality-curation knob: rows whose source file "
                        "is among the first --quality-weight-top status=='ok' "
                        "manifest rows get this sample-weight multiplier "
                        "(composed with the existing class-balanced "
                        "weights); all other rows get 1.0. None (default) "
                        "applies no reweighting.")
    p.add_argument("--quality-weight-top", type=int, default=451,
                   help="number of leading status=='ok' manifest rows "
                        "(fetch-priority / quality-rank order) treated as "
                        "'top quality' for --quality-weight-mult")
    args = p.parse_args()

    cfg = TrainConfig(target_ps=args.target_ps, stop_ps=args.stop_ps,
                      timeout_s=args.timeout, barrier_mode=args.barrier_mode,
                      vol_target_mult=args.vol_target_mult,
                      vol_stop_mult=args.vol_stop_mult,
                      vol_window_s=args.vol_window,
                      test_start_date=args.test_start_date,
                      train_start_date=args.train_start_date)
    drop_feats = parse_drop_features(args.drop_features)
    hp = dict(learning_rate=args.learning_rate, max_iter=args.max_iter,
             max_leaf_nodes=args.max_leaf_nodes,
             min_samples_leaf=args.min_samples_leaf,
             l2_regularization=args.l2_regularization)
    man = pd.read_csv(MANIFEST)
    ok = man[man["status"] == "ok"]
    files = [CORPUS_DIR / f"{r.symbol}_{r.date}.parquet"
             for r in ok.itertuples() if (CORPUS_DIR / f"{r.symbol}_{r.date}.parquet").exists()]
    if args.limit:
        files = files[: args.limit]
    dates = [f.stem.rsplit("_", 1)[1] for f in files]
    train_dates, test_dates = split_days(dates, cfg)
    train_files = [f for f, d in zip(files, dates, strict=True) if d in train_dates]
    test_files = [f for f, d in zip(files, dates, strict=True) if d in test_dates]
    print(f"stock-days: {len(files)} total -> train {len(train_files)} "
          f"(≤{max(train_dates) if train_dates else '-'}) | "
          f"test {len(test_files)} (≥{min(test_dates)})")

    if args.val_frac > 0 or args.val_start_date is not None:
        core_train_dates, val_dates = split_val_days(
            train_dates, args.val_frac, args.val_start_date)
        core_train_files = [f for f, d in zip(files, dates, strict=True)
                            if d in core_train_dates]
        val_files = [f for f, d in zip(files, dates, strict=True) if d in val_dates]
        print(f"  val-split: core-train {len(core_train_files)} "
              f"(≤{max(core_train_dates) if core_train_dates else '-'}) | "
              f"val {len(val_files)} (≥{min(val_dates) if val_dates else '-'})")
    else:
        core_train_dates, val_dates = train_dates, []
        core_train_files, val_files = train_files, []

    if args.train_quality_limit is not None:
        core_train_files = limit_by_quality(core_train_files, files,
                                            args.train_quality_limit)
        print(f"  quality-limit: core-train restricted to top "
              f"{args.train_quality_limit} manifest rows -> "
              f"{len(core_train_files)} files")

    out = ROOT / args.out / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print("building train dataset ...", flush=True)
    x_tr, y_tr, m_tr = build_dataset(core_train_files, cfg)
    x_tr = drop_columns(x_tr, drop_feats)
    print(f"  {len(x_tr):,} rows; label rates "
          f"{y_tr.value_counts(normalize=True).round(3).to_dict()}", flush=True)

    weight_mult = None
    if args.quality_weight_mult is not None:
        top_stems = {f.stem for f in files[:args.quality_weight_top]}
        weight_mult = quality_weight_array(m_tr, top_stems,
                                           args.quality_weight_mult)
        print(f"  quality-weight: top {args.quality_weight_top} manifest "
              f"rows -> x{args.quality_weight_mult} "
              f"({int((weight_mult != 1.0).sum())} of {len(weight_mult)} "
              f"rows)", flush=True)

    model = fit_model(x_tr, y_tr, cfg.seed, sample_weight_mult=weight_mult,
                      **hp)

    if val_files:
        print("building val dataset ...", flush=True)
        x_val, y_val, m_val = build_dataset(val_files, cfg)
        x_val = drop_columns(x_val, drop_feats)
        print(f"  {len(x_val):,} rows", flush=True)
        val_summary, val_per_thr = evaluate(model, x_val, y_val, m_val, cfg)
        (out / "val_metrics.json").write_text(
            json.dumps(val_summary, indent=2, default=str))
        val_per_thr.to_csv(out / "val_per_threshold.csv", index=False)
        print(f"\nVAL REPORT (val days {min(val_dates)} .. {max(val_dates)})")
        print(val_per_thr.round(4).to_string(index=False))

    print("building test dataset ...", flush=True)
    x_te, y_te, m_te = build_dataset(test_files, cfg)
    x_te = drop_columns(x_te, drop_feats)
    print(f"  {len(x_te):,} rows", flush=True)

    summary, per_thr = evaluate(model, x_te, y_te, m_te, cfg)

    (out / "config.json").write_text(json.dumps({
        "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items() if k != "fees"},
        "fees": cfg.fees.__dict__, "git_head": _git_head(),
        "n_train_days": len(train_files), "n_test_days": len(test_files),
        "n_core_train_days": len(core_train_files), "n_val_days": len(val_files),
        "learning_rate": args.learning_rate, "max_iter": args.max_iter,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "l2_regularization": args.l2_regularization,
        "drop_features": drop_feats, "val_frac": args.val_frac,
        "val_start_date": args.val_start_date,
        "train_quality_limit": args.train_quality_limit,
        "quality_weight_mult": args.quality_weight_mult,
        "quality_weight_top": args.quality_weight_top,
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    per_thr.to_csv(out / "per_threshold.csv", index=False)
    try:
        imp = pd.Series(getattr(model, "feature_importances_", None)
                        if getattr(model, "feature_importances_", None) is not None
                        else [], dtype=float)
    except Exception:  # noqa: BLE001
        imp = pd.Series(dtype=float)
    if imp.empty:  # HistGBT has no impurity importances; use permutation on a sample
        from sklearn.inspection import permutation_importance
        # positional sampling — index labels repeat across stock-days
        rng = np.random.default_rng(0)
        pos = rng.choice(len(x_te), min(5000, len(x_te)), replace=False)
        r = permutation_importance(model, x_te.iloc[pos], y_te.iloc[pos],
                                   n_repeats=3, random_state=0)
        imp = pd.Series(r.importances_mean, index=x_te.columns)
    imp.sort_values(ascending=False).to_csv(out / "feature_importances.csv")

    print(f"\nOOS REPORT (test days {min(test_dates)} .. {max(test_dates)})")
    print(per_thr.round(4).to_string(index=False))
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
