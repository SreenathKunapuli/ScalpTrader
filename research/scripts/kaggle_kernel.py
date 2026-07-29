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
    "--batch", "512", "--val-neg-frac", "0.06",
]
# wave 2 attempt 2: v9 died on the /kaggle/working ~20GB output quota — the
# disk window store lived under the run dir. Storage is now per-arm:
#   top451   -> RAM (v8 proved it fits the T4 session)
#   full1250 -> memmap on /kaggle/tmp scratch (~57GB), neg-frac trimmed to
#               0.03 so train+val stores fit; positives are all kept, so the
#               4x-data question is still answered at matched positive count
ARMS = {
    "top451_reg": ["--train-quality-limit", "451", "--window-store", "ram",
                   "--neg-frac", "0.06"],
    "full1250_reg": ["--window-store", "disk", "--neg-frac", "0.03",
                     "--window-cache-dir", "/kaggle/tmp/wcache"],
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

def _disk(label: str) -> None:
    for mnt in ("/kaggle/working", "/kaggle/tmp"):
        u = shutil.disk_usage(mnt)
        print(f"[disk {label}] {mnt}: free {u.free / 1e9:.1f} GB "
              f"of {u.total / 1e9:.1f} GB", flush=True)


worst = 0
for name, extra in ARMS.items():
    print(f"===== ARM {name} =====", flush=True)
    _disk(f"before {name}")
    r = subprocess.run(
        [sys.executable, str(WORK / "research/scripts/train_tcn.py"),
         *BASE_ARGS, *extra], cwd=WORK)
    print(f"arm {name} exit: {r.returncode}", flush=True)
    _disk(f"after {name}")
    worst = max(worst, r.returncode)
    runs = sorted((WORK / "runs/tcn").glob("*"))
    if runs:
        dest = Path(f"/kaggle/working/tcn_{name}")
        shutil.copytree(runs[-1], dest, dirs_exist_ok=True)
        print("artifact ->", dest, [p.name for p in dest.iterdir()])
    # scratch + run-dir hygiene so arm 2 starts with full quota
    shutil.rmtree("/kaggle/tmp/wcache", ignore_errors=True)
    for d in runs:
        shutil.rmtree(d, ignore_errors=True)
sys.exit(worst)
