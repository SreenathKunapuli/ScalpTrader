"""Kaggle kernel entrypoint: rebuild the repo layout, train the TCN on CUDA,
leave the artifact in /kaggle/working for `kaggle kernels output` to pull.

Attach the scalptrader-corpus dataset to the kernel. Torch/pandas/sklearn are
preinstalled on Kaggle GPU images. Hyperparameters are edited here per run
(each push is one experiment; config.json records everything).
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = Path("/kaggle/input/scalptrader-corpus")
WORK = Path("/kaggle/working/ScalpTrader")

TRAIN_ARGS = [
    "--barrier-mode", "vol", "--vol-target-mult", "1.0",
    "--vol-stop-mult", "0.5", "--timeout", "120",
    "--test-start-date", "2025-06-27",
    "--train-quality-limit", "451",
    "--val-start-date", "2025-01-01",
    "--device", "auto",            # resolves to cuda on Kaggle
    "--window", "240", "--channels", "64", "--blocks", "4",
    "--epochs", "40", "--patience", "6", "--lr", "1e-3",
    "--batch", "512", "--neg-frac", "0.12",
    "--jitter-sigma", "0.0",
]

WORK.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(DATASET / "code.zip") as z:
    z.extractall(WORK)
(WORK / "data").mkdir(exist_ok=True)
if not (WORK / "data/corpus").exists():
    # symlink keeps the read-only dataset in place; scripts only read it
    (WORK / "data/corpus").symlink_to(DATASET / "corpus")

r = subprocess.run(
    [sys.executable, str(WORK / "research/scripts/train_tcn.py"), *TRAIN_ARGS],
    cwd=WORK)
print("train exit:", r.returncode)

runs = sorted((WORK / "runs/tcn").glob("*"))
if runs:
    dest = Path("/kaggle/working/tcn_artifact")
    shutil.copytree(runs[-1], dest, dirs_exist_ok=True)
    print("artifact ->", dest, list(p.name for p in dest.iterdir()))
sys.exit(r.returncode)
