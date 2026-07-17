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

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="runs/scalper/<ts> dir holding config.json")
    p.add_argument("--threshold", type=float, required=True,
                   help="entry threshold chosen from that run's OOS report")
    args = p.parse_args()

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
    print(f"deployment fit on ALL {len(files)} stock-days "
          f"(config from {run_dir.name}) ...", flush=True)
    x, y, _ = build_dataset(files, cfg)
    model = fit_model(x, y, cfg.seed)

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
    }, indent=2))
    print(f"artifact -> {run_dir}/{{model.joblib, features.json, inference.json}}")


if __name__ == "__main__":
    main()
