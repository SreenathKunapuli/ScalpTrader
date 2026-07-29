"""Export a deployment artifact from a validated train_scalper run.

Walk-forward already produced the OOS evidence for this config; the
deployment fit uses ALL corpus stock-days (train+test) with the exact
same TrainConfig, then dumps everything ScalpGbtSignal needs:
model.joblib + features.json + inference.json, into the run dir.

Usage:
  .venv/bin/python research/scripts/export_model.py \
      --run-dir runs/scalper/<ts> --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp.viability import FeeModel  # noqa: E402
from scalp.walkforward import TrainConfig, build_dataset, fit_model  # noqa: E402
from scripts.train_scalper import (
    drop_columns,
    limit_by_quality,
    parse_drop_features,  # noqa: E402
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="runs/scalper/<ts> dir holding config.json")
    p.add_argument("--threshold", type=float, required=True,
                   help="entry threshold chosen from that run's OOS report")
    p.add_argument("--exec-stop-mult", type=float, default=1000.0,
                   help="execution stop = label stop x this. Default 1000 "
                        "(timeout-only exits; sim study showed nearby stops "
                        "pay ruinous gap-through slippage)")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--max-leaf-nodes", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=None)
    p.add_argument("--l2-regularization", type=float, default=None)
    p.add_argument("--drop-features", default=None,
                   help="comma-separated feature columns to drop, e.g. 'a,b,c'")
    args = p.parse_args()
    drop_feats = parse_drop_features(args.drop_features)
    hp = dict(learning_rate=args.learning_rate, max_iter=args.max_iter,
             max_leaf_nodes=args.max_leaf_nodes,
             min_samples_leaf=args.min_samples_leaf,
             l2_regularization=args.l2_regularization)

    run_dir = (ROOT / args.run_dir).resolve()
    saved = json.loads((run_dir / "config.json").read_text())
    cfg_d = dict(saved["cfg"])
    cfg_d["prob_threshold_grid"] = tuple(cfg_d["prob_threshold_grid"])
    cfg = TrainConfig(**cfg_d, fees=FeeModel(**saved["fees"]))

    man = pd.read_csv(MANIFEST)
    ok = man[man["status"] == "ok"]
    files = [CORPUS_DIR / f"{r.symbol}_{r.date}.parquet"
             for r in ok.itertuples()
             if (CORPUS_DIR / f"{r.symbol}_{r.date}.parquet").exists()]
    ranked_files = files  # manifest fetch-priority / quality-rank order,
    # captured before any date filters below so --train-quality-limit's
    # "first N rows" always means first N of the FULL manifest.
    if cfg.train_start_date is not None:
        # deployment fit = train+test window days, minus pre-cutoff vintage:
        # a run trained with --train-start-date shouldn't silently dilute
        # its deployment fit back in with the dropped older days.
        files = [f for f in files
                if f.stem.rsplit("_", 1)[1] >= cfg.train_start_date]
    train_quality_limit = saved.get("train_quality_limit")
    if train_quality_limit is not None:
        # mirror the corpus quality-depth knob the run was trained with —
        # the deployment fit should match the evidence that validated it.
        files = limit_by_quality(files, ranked_files, train_quality_limit)
        print(f"  quality-limit: deployment fit restricted to top "
              f"{train_quality_limit} manifest rows -> {len(files)} files")
    print(f"deployment fit on ALL {len(files)} stock-days "
          f"(config from {run_dir.name}) ...", flush=True)
    x, y, _ = build_dataset(files, cfg)
    x = drop_columns(x, drop_feats)
    model = fit_model(x, y, cfg.seed, **hp)

    import joblib
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "features.json").write_text(json.dumps(list(x.columns)))
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = "unknown"
    (run_dir / "inference.json").write_text(json.dumps({
        "threshold": args.threshold,
        "exec_stop_mult": args.exec_stop_mult,
        "timeout_s": cfg.timeout_s,
        "barrier_mode": cfg.barrier_mode,
        "vol_window_s": cfg.vol_window_s,
        "vol_target_mult": cfg.vol_target_mult,
        "vol_stop_mult": cfg.vol_stop_mult,
        "min_target_ps": cfg.min_target_ps,
        "min_stop_ps": cfg.min_stop_ps,
        "target_ps": cfg.target_ps,
        "stop_ps": cfg.stop_ps,
        "n_stock_days": len(files),
        "n_rows": int(len(x)),
        "git_head": head,
        "hp": hp,
        "dropped_features": drop_feats,
        "train_quality_limit": train_quality_limit,
    }, indent=2))
    print(f"artifact -> {run_dir}/{{model.joblib, features.json, inference.json}}")


if __name__ == "__main__":
    main()
