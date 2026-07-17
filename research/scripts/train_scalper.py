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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp.walkforward import TrainConfig, build_dataset, evaluate, fit_model, \
    split_days  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"


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
    args = p.parse_args()

    cfg = TrainConfig(target_ps=args.target_ps, stop_ps=args.stop_ps,
                      timeout_s=args.timeout, barrier_mode=args.barrier_mode,
                      vol_target_mult=args.vol_target_mult,
                      vol_stop_mult=args.vol_stop_mult,
                      vol_window_s=args.vol_window)
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

    print("building train dataset ...", flush=True)
    x_tr, y_tr, _ = build_dataset(train_files, cfg)
    print(f"  {len(x_tr):,} rows; label rates "
          f"{y_tr.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    print("building test dataset ...", flush=True)
    x_te, y_te, m_te = build_dataset(test_files, cfg)
    print(f"  {len(x_te):,} rows", flush=True)

    model = fit_model(x_tr, y_tr, cfg.seed)
    summary, per_thr = evaluate(model, x_te, y_te, m_te, cfg)

    out = ROOT / args.out / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items() if k != "fees"},
        "fees": cfg.fees.__dict__, "git_head": _git_head(),
        "n_train_days": len(train_files), "n_test_days": len(test_files),
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
