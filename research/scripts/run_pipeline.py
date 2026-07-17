"""One-command reproduction of every quoted model metric from cached data.

Chains the existing committed scripts (never reimplements their logic):
corpus manifest check -> viability study (if missing) -> train_scalper for
each swept config -> sim_eval realism pass for the chosen config -> one
report at runs/pipeline_report.json.

Usage:
  .venv/bin/python research/scripts/run_pipeline.py [--smoke] [--skip-sim]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
SCRIPTS = ROOT / "research" / "scripts"

# (barrier_mode, target_mult, stop_mult, timeout_s) — the swept configs
CONFIGS = [("vol", 1.0, 0.5, 120), ("vol", 1.5, 0.75, 60), ("vol", 2.0, 1.0, 120)]
CHOSEN = CONFIGS[0]          # winner by OOS sum PnL on the 451-day corpus
CHOSEN_THRESHOLD = 0.6


def run(cmd: list) -> None:
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="--limit 40 everywhere")
    p.add_argument("--skip-sim", action="store_true")
    args = p.parse_args()
    limit = ["--limit", "40"] if args.smoke else []

    if not (ROOT / "data" / "corpus" / "manifest.csv").exists():
        sys.exit("no corpus manifest — run build_runner_corpus.py --scan/--fetch first")
    if not (ROOT / "data" / "viability" / "results.parquet").exists():
        run([PY, SCRIPTS / "viability_study.py"])

    report: dict = {"train_runs": [], "smoke": args.smoke}
    for mode, tm, sm, to in CONFIGS:
        scalper_dir = ROOT / "runs" / "scalper"
        before = set(scalper_dir.glob("*")) if scalper_dir.exists() else set()
        run([PY, SCRIPTS / "train_scalper.py", "--barrier-mode", mode,
             "--vol-target-mult", tm, "--vol-stop-mult", sm, "--timeout", to,
             *limit])
        for d in sorted(set(scalper_dir.glob("*")) - before):
            report["train_runs"].append({
                "config": f"{mode} {tm}x/{sm}x @{to}s", "dir": str(d),
                "metrics": json.loads((d / "metrics.json").read_text())})

    if not args.skip_sim:
        mode, tm, sm, to = CHOSEN
        sim_dir = ROOT / "runs" / "sim_eval"
        before = set(sim_dir.glob("*")) if sim_dir.exists() else set()
        run([PY, SCRIPTS / "sim_eval.py", "--barrier-mode", mode,
             "--vol-target-mult", tm, "--vol-stop-mult", sm, "--timeout", to,
             "--threshold", CHOSEN_THRESHOLD, *limit])
        for d in sorted(set(sim_dir.glob("*")) - before):
            report["sim_eval"] = {"dir": str(d), "report": json.loads(
                (d / "report.json").read_text())}

    out = ROOT / "runs" / "pipeline_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
