"""Package code + corpus into a Kaggle dataset folder, ready to upload.

Produces build/kaggle_dataset/ containing:
  code.zip                    research/ tree (scalp pkg + scripts, no tests)
  corpus/manifest.csv         quality-ranked manifest
  corpus/1s/<N parquet files> top-N stock-days (default: all on disk)

Upload (once ~/.kaggle/kaggle.json exists):
  kaggle datasets create -p build/kaggle_dataset        # first time
  kaggle datasets version -p build/kaggle_dataset -m "refresh"

Usage:
  .venv/bin/python research/scripts/kaggle_bundle.py [--top-n 451]
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "kaggle_dataset"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-n", type=int, default=0,
                   help="bundle only the first N manifest rows (0 = all)")
    p.add_argument("--slug", default="scalptrader-corpus")
    args = p.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "corpus" / "1s").mkdir(parents=True)

    zpath = OUT / "code.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ["research/scalp", "research/scripts"]:
            for f in (ROOT / sub).rglob("*.py"):
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f.relative_to(ROOT))
    print(f"code.zip: {zpath.stat().st_size / 1e6:.1f} MB")

    man = pd.read_csv(ROOT / "data/corpus/manifest.csv")
    ok = man[man["status"] == "ok"]
    if args.top_n:
        ok = ok.head(args.top_n)
    kept, missing = 0, 0
    for r in ok.itertuples():
        src = ROOT / f"data/corpus/1s/{r.symbol}_{r.date}.parquet"
        if src.exists():
            shutil.copy2(src, OUT / "corpus" / "1s" / src.name)
            kept += 1
        else:
            missing += 1
    man.to_csv(OUT / "corpus" / "manifest.csv", index=False)
    total_mb = sum(f.stat().st_size for f in (OUT / "corpus" / "1s").iterdir()) / 1e6
    print(f"corpus: {kept} days ({total_mb:,.0f} MB), {missing} missing")

    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "ScalpTrader corpus + research code",
        "id": f"KAGGLE_USERNAME/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    print(f"ready -> {OUT}  (fill KAGGLE_USERNAME in dataset-metadata.json)")


if __name__ == "__main__":
    main()
