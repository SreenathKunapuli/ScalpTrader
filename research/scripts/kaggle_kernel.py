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

BASE_ARGS = [
    "--barrier-mode", "vol", "--vol-target-mult", "1.0",
    "--vol-stop-mult", "0.5", "--timeout", "120",
    "--test-start-date", "2025-06-27",
    "--val-start-date", "2025-01-01",
    "--device", "auto",            # resolves to cuda on Kaggle
    "--window", "240", "--channels", "64", "--blocks", "4",
    "--epochs", "40", "--patience", "6", "--lr", "1e-3",
    "--lr-schedule", "cosine", "--jitter-sigma", "0.05",
    "--batch", "512", "--neg-frac", "0.06", "--val-neg-frac", "0.06",
    "--window-store", "disk",
]
# wave 2: v8 overfit from epoch 1 -> regularize (cosine+jitter) and test
# whether 4x data beats quality curation for the deep rung
ARMS = {
    "top451_reg": ["--train-quality-limit", "451"],
    "full1250_reg": [],
}

import psutil
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
import torch
print(f"cuda available: {torch.cuda.is_available()}")
print("inputs:", [str(p) for p in Path("/kaggle/input").rglob("*")][:8])
assert DATASET.exists(), (
    "dataset not attached — run must be launched via `kaggle kernels push` "
    "(UI re-runs can drop the dataset attachment)")

WORK.mkdir(parents=True, exist_ok=True)
# Kaggle auto-extracts uploaded zips — handle both layouts
if (DATASET / "code.zip").exists():
    with zipfile.ZipFile(DATASET / "code.zip") as z:
        z.extractall(WORK)
elif (DATASET / "research").exists():
    shutil.copytree(DATASET / "research", WORK / "research", dirs_exist_ok=True)
else:  # code.zip extracted into a code/ folder
    shutil.copytree(DATASET / "code" / "research", WORK / "research",
                    dirs_exist_ok=True)
(WORK / "data").mkdir(exist_ok=True)
corpus_src = next(p for p in [DATASET / "corpus", DATASET / "corpus.zip"]
                  if p.exists())
if corpus_src.suffix == ".zip":
    with zipfile.ZipFile(corpus_src) as z:
        z.extractall(WORK / "data")
elif not (WORK / "data/corpus").exists():
    # symlink keeps the read-only dataset in place; scripts only read it
    (WORK / "data/corpus").symlink_to(corpus_src)

worst = 0
for name, extra in ARMS.items():
    print(f"===== ARM {name} =====", flush=True)
    r = subprocess.run(
        [sys.executable, str(WORK / "research/scripts/train_tcn.py"),
         *BASE_ARGS, *extra], cwd=WORK)
    print(f"arm {name} exit: {r.returncode}", flush=True)
    worst = max(worst, r.returncode)
    runs = sorted((WORK / "runs/tcn").glob("*"))
    if runs:
        dest = Path(f"/kaggle/working/tcn_{name}")
        shutil.copytree(runs[-1], dest, dirs_exist_ok=True)
        print("artifact ->", dest, [p.name for p in dest.iterdir()])
sys.exit(worst)
