"""Train the TCN deep rung (scalp.deep) on the corpus.

Walk-forward day splits identical to train_scalper.py (scalp.walkforward
.split_days), but a causal sequence model instead of the GBT. Train days
are quality-limited pre-window days; val days (>= --val-start-date) are for
EARLY STOPPING only. Test days are NEVER loaded by this script — split_days
only reads date STRINGS to compute the boundary, so no test parquet is ever
opened here; a separate eval script reads the OOS test window later.

Artifacts land in <out>/<ts>/: model.pt (best-val-AP state dict),
scaler.json (train-days-only per-feature median/IQR — serving MUST load
this to reproduce training-time normalization), config.json (architecture
+ data hyperparameters, git head), history.csv (per-epoch loss/AP).

Usage:
  .venv/bin/python research/scripts/train_tcn.py --limit 60 \
      --barrier-mode vol --vol-target-mult 1.0 --vol-stop-mult 0.5 \
      --val-start-date 2025-01-01 --train-quality-limit 450 \
      --window 240 --channels 64 --blocks 4 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp.deep.dataset import (
    JitterDataset,
    MemmapWindows,
    apply_scaler,
    build_windows,  # noqa: E402
    fit_scaler,
    save_scaler,
    subsample_negatives,
)
from scalp.deep.model import ScalpTCN, pick_device  # noqa: E402
from scalp.deep.train_loop import train  # noqa: E402
from scalp.walkforward import TrainConfig, split_days  # noqa: E402
from scripts.train_scalper import limit_by_quality  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"
# "top quality" convention shared with train_scalper.py / sim_eval.py's
# --quality-weight-top default: the first N status=='ok' manifest rows in
# fetch-priority / quality-rank order.
QUALITY_TOP_N = 451


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _file_key(path: Path) -> str:
    return path.stem  # "SYM_yyyy-mm-dd" — stable across machines/paths


def _resolve_device(device_arg: str):
    """Resolve --device argument to a usable device object.

    'auto' -> cuda > mps > cpu via pick_device() (dml is never auto-selected).
    'dml'  -> torch_directml.device(); lazy import so the Windows-only package
              is never imported on non-Windows machines.  Raises ImportError
              with an actionable pip hint if torch_directml is not installed.
    Other  -> torch.device(device_arg).
    """
    if device_arg == "auto":
        return pick_device()
    if device_arg == "dml":
        try:
            import torch_directml  # noqa: PLC0415 — Windows-only, lazy import
        except ImportError as exc:
            raise ImportError(
                "DirectML device requested (--device dml) but torch_directml "
                "is not installed. Install it with: "
                "pip install torch-directml"
            ) from exc
        return torch_directml.device()
    return torch.device(device_arg)


def _load_split(
    files: list[Path], cfg: TrainConfig, window_s: int, scaler: dict,
    neg_frac: float | None, quality_stems: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build causal windows for every file, apply the (train-days-only)
    scaler, and optionally subsample negatives.
    `neg_frac=None` disables subsampling (e.g. want the full class balance).
    `neg_frac` is a float -> subsample each file's negatives independently
    at that probability, seeded deterministically by cfg.seed + file hash.

    `quality_stems`, if given, additionally builds a bool array (aligned
    1:1 with the returned samples) marking which samples came from a file
    whose stem ("{symbol}_{date}") is in `quality_stems` — feeds
    --quality-oversample's WeightedRandomSampler weights. None (default)
    skips this bookkeeping and the third return is None."""
    xs, ys, quals = [], [], []
    for path in files:
        bars = pd.read_parquet(path)
        X, y, idx = build_windows(bars, cfg, window_s)
        del bars  # full-day frame no longer needed
        if len(y) == 0:
            continue
        if neg_frac is not None:
            X, y, idx = subsample_negatives(X, y, idx, cfg, _file_key(path),
                                            neg_frac)
            if len(y) == 0:
                continue
        xs.append(apply_scaler(X, scaler))
        del X  # raw (unscaled) windows freed once scaled copy is appended
        ys.append(y)
        if quality_stems is not None:
            quals.append(np.full(len(y), path.stem in quality_stems))
    if not xs:
        raise ValueError("no samples in this file set")
    q_all = np.concatenate(quals, axis=0) if quality_stems is not None else None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), q_all


def _build_split_disk(
    files: list[Path], cfg: TrainConfig, window_s: int, scaler: dict,
    neg_frac: float | None, store_path: Path,
    quality_stems: set[str] | None = None,
) -> tuple[MemmapWindows._FinalizedMemmapWindows, np.ndarray | None]:
    """Disk-backed variant of _load_split.

    Appends each day's (post-scale) windows to a MemmapWindows store under
    `store_path` one file at a time — never accumulates the full dataset in
    RAM.  Returns the finalized store (a Dataset-compatible object) and,
    optionally, a bool quality array aligned 1:1 with the stored samples
    (for WeightedRandomSampler).

    Scaling order (IMPORTANT): scaler must already be fit before calling
    this function; raw windows are built, scaler is applied, THEN the
    scaled chunk is appended to disk.  The scaler is never fit here.
    """
    from scalp.deep.dataset import MemmapWindows  # noqa: PLC0415 (local re-import ok)
    n_features = len(scaler["feature_names"])
    store = MemmapWindows.create(store_path, n_features=n_features,
                                 window_s=window_s)
    quals: list[np.ndarray] = []
    any_samples = False
    for path in files:
        bars = pd.read_parquet(path)
        X, y, idx = build_windows(bars, cfg, window_s)
        del bars
        if len(y) == 0:
            continue
        if neg_frac is not None:
            X, y, idx = subsample_negatives(X, y, idx, cfg, _file_key(path),
                                            neg_frac)
            if len(y) == 0:
                continue
        X_scaled = apply_scaler(X, scaler)
        del X
        store.append(X_scaled, y)
        del X_scaled
        if quality_stems is not None:
            quals.append(np.full(len(y), path.stem in quality_stems,
                                 dtype=bool))
        any_samples = True
    if not any_samples:
        raise ValueError("no samples in this file set")
    ds = store.finalize()
    q_all = np.concatenate(quals, axis=0) if quality_stems is not None else None
    return ds, q_all


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # data flags, mirroring train_scalper.py
    p.add_argument("--target-ps", type=float, default=0.05)
    p.add_argument("--stop-ps", type=float, default=0.04)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--barrier-mode", choices=["fixed", "vol"], default="fixed")
    p.add_argument("--vol-target-mult", type=float, default=1.0)
    p.add_argument("--vol-stop-mult", type=float, default=0.5)
    p.add_argument("--vol-window", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="cap #stock-days")
    p.add_argument("--test-start-date", default=None,
                   help="pin the OOS test window to all days >= this ISO "
                        "date (yyyy-mm-dd); test days are NEVER loaded by "
                        "this script — only used to compute the train/test "
                        "boundary from date strings")
    p.add_argument("--val-start-date", default=None,
                   help="pin the inner validation window (used ONLY for "
                        "early stopping) to all TRAIN days >= this ISO "
                        "date; if omitted, no val split and no early "
                        "stopping (trains the full --epochs)")
    p.add_argument("--train-quality-limit", type=int, default=None,
                   help="corpus quality-depth knob: restrict the core-train "
                        "file set to files among the first N manifest rows "
                        "(status=='ok', fetch-priority order); val is "
                        "NEVER filtered")
    p.add_argument("--seed", type=int, default=7)
    # model / training flags
    p.add_argument("--window", type=int, default=240)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--neg-frac", type=float, default=0.15)
    p.add_argument("--val-neg-frac", type=float, default=None,
                   help="negative subsampling fraction applied to the val "
                        "windows (same deterministic seeding as train: "
                        "cfg.seed + per-file hash, NEVER epoch-dependent). "
                        "Default None uses the same value as --neg-frac, so "
                        "every candidate model sees the identical val subset "
                        "and val AP is comparable across runs.")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--jitter-sigma", type=float, default=0.0,
                   help="train-only augmentation: each epoch, add fresh "
                        "N(0, sigma) Gaussian noise to the SCALED training "
                        "windows (torch.Generator seeded from --seed + "
                        "epoch, so runs reproduce and every epoch draws "
                        "fresh noise); 0.0 (default) disables augmentation "
                        "entirely. The val loader is NEVER augmented, "
                        "regardless of this flag.")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"],
                   default="constant",
                   help="LR schedule for training. 'constant' (default) keeps "
                        "AdamW's fixed lr. 'cosine' chains a 2-epoch linear "
                        "warmup with CosineAnnealingLR for the remaining "
                        "epochs via SequentialLR.")
    p.add_argument("--quality-oversample", type=int, default=1,
                   help="oversampling knob: training samples whose source "
                        f"day is among the first {QUALITY_TOP_N} "
                        "status=='ok' manifest rows (fetch-priority / "
                        "quality-rank order) get this x sampling weight "
                        "via WeightedRandomSampler; 1 (default) is a no-op "
                        "(keeps today's shuffle=True DataLoader). Only "
                        "meaningful when training beyond the top-"
                        f"{QUALITY_TOP_N} files (e.g. --train-quality-limit "
                        "unset or > that).")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu", "dml"],
                   default="auto",
                   help="compute device for training. 'auto' (default) "
                        "selects cuda if available, else mps, else cpu. "
                        "ROCm/AMD torch builds report as cuda. "
                        "'dml' selects Windows DirectML (requires "
                        "torch-directml; never auto-selected).")
    p.add_argument("--window-store", choices=["ram", "disk"], default="ram",
                   help="where to hold the windowed dataset during training. "
                        "'ram' (default) accumulates all windows in memory "
                        "(original behavior). 'disk' writes each day's "
                        "post-scale windows to memmap files under "
                        "<out>/wcache/ so RAM usage is O(window) not "
                        "O(dataset); wcache is deleted after a successful "
                        "training run.")
    p.add_argument("--window-cache-dir", default=None,
                   help="override the disk window-store location (default "
                        "<out>/wcache). On Kaggle this MUST point at scratch "
                        "(/kaggle/tmp): the run dir lives under "
                        "/kaggle/working whose ~20GB output quota a full "
                        "window store exceeds — observed kill 2026-07-27.")
    p.add_argument("--out", default="runs/tcn")
    args = p.parse_args()

    # --val-neg-frac defaults to the same value as --neg-frac so the val
    # subset is subsampled at the same rate as train, producing a fixed,
    # deterministic val set that is comparable across runs.
    val_neg_frac: float = (
        args.neg_frac if args.val_neg_frac is None else args.val_neg_frac
    )

    device = _resolve_device(args.device)

    cfg = TrainConfig(target_ps=args.target_ps, stop_ps=args.stop_ps,
                      timeout_s=args.timeout, barrier_mode=args.barrier_mode,
                      vol_target_mult=args.vol_target_mult,
                      vol_stop_mult=args.vol_stop_mult,
                      vol_window_s=args.vol_window,
                      test_start_date=args.test_start_date,
                      seed=args.seed)

    man = pd.read_csv(MANIFEST)
    ok = man[man["status"] == "ok"]
    files = [CORPUS_DIR / f"{r.symbol}_{r.date}.parquet"
             for r in ok.itertuples()
             if (CORPUS_DIR / f"{r.symbol}_{r.date}.parquet").exists()]
    if args.limit:
        files = files[: args.limit]
    dates = [f.stem.rsplit("_", 1)[1] for f in files]

    # test days are NEVER loaded: split_days only reads date STRINGS, no IO.
    train_dates, test_dates = split_days(dates, cfg)
    train_files = [f for f, d in zip(files, dates, strict=True) if d in train_dates]
    print(f"stock-days: {len(files)} total -> train {len(train_files)} "
          f"(<={max(train_dates) if train_dates else '-'}) | "
          f"test {len(test_dates)} days NEVER loaded "
          f"(>={min(test_dates) if test_dates else '-'})")

    if args.val_start_date is not None:
        core_train_dates = [d for d in train_dates if d < args.val_start_date]
        val_dates = [d for d in train_dates if d >= args.val_start_date]
        if not core_train_dates:
            raise ValueError(
                f"--val-start-date {args.val_start_date!r} leaves no "
                f"core-train days")
    else:
        core_train_dates, val_dates = train_dates, []
    core_train_files = [f for f, d in zip(files, dates, strict=True)
                        if d in core_train_dates]
    val_files = [f for f, d in zip(files, dates, strict=True) if d in val_dates]
    print(f"  val-split: core-train {len(core_train_files)} "
          f"(<={max(core_train_dates) if core_train_dates else '-'}) | "
          f"val {len(val_files)} (>={min(val_dates) if val_dates else '-'})")
    if not val_files:
        print("  WARNING: no val split (--val-start-date not set, or it "
              "leaves no val days) -> no early stopping, trains the full "
              "--epochs")

    if args.train_quality_limit is not None:
        core_train_files = limit_by_quality(core_train_files, files,
                                            args.train_quality_limit)
        print(f"  quality-limit: core-train restricted to top "
              f"{args.train_quality_limit} manifest rows -> "
              f"{len(core_train_files)} files")

    out = ROOT / args.out / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print("fitting scaler on core-train days ...", flush=True)
    scaler = fit_scaler(core_train_files, cfg, args.window)
    save_scaler(scaler, out / "scaler.json")
    n_features = len(scaler["feature_names"])
    print(f"  {n_features} features, {scaler['n_train_files']} files")

    quality_stems = None
    if args.quality_oversample != 1:
        quality_stems = {f.stem for f in files[:QUALITY_TOP_N]}

    wcache_dir = (Path(args.window_cache_dir) if args.window_cache_dir
                  else out / "wcache")
    use_disk = (args.window_store == "disk")

    if use_disk:
        print("building train windows (disk store) ...", flush=True)
        tr_store, q_tr = _build_split_disk(
            core_train_files, cfg, args.window, scaler,
            args.neg_frac, wcache_dir / "train", quality_stems,
        )
        n_tr_samples = len(tr_store)
        # Collect y values for pos_weight computation without loading X.
        # Re-open y mmap directly (O(n) ints, tiny).
        _y_mm = np.memmap(wcache_dir / "train" / MemmapWindows._Y_FILE,
                          dtype=np.int8, mode="r", shape=(n_tr_samples,))
        y_tr_arr = np.array(_y_mm)
        del _y_mm
        n_pos = int(y_tr_arr.sum())
        n_tr = n_tr_samples
        print(f"  {n_tr:,} samples; positive rate "
              f"{float(y_tr_arr.mean()):.4f}", flush=True)
        if q_tr is not None:
            print(f"  quality-oversample: {int(q_tr.sum()):,}/{n_tr:,} "
                  f"train samples from top {QUALITY_TOP_N} manifest rows -> "
                  f"x{args.quality_oversample} sampling weight", flush=True)
        # JitterDataset wraps the disk store (no full-tensor copy).
        torch.manual_seed(args.seed)
        train_dataset = JitterDataset(
            sigma=args.jitter_sigma, seed=args.seed, inner=tr_store,
        )
        if val_files:
            print(f"building val windows (disk store, val_neg_frac={val_neg_frac}) ...",
                  flush=True)
            val_store, _ = _build_split_disk(
                val_files, cfg, args.window, scaler,
                val_neg_frac, wcache_dir / "val",
            )
            n_val = len(val_store)
            _yv_mm = np.memmap(wcache_dir / "val" / MemmapWindows._Y_FILE,
                               dtype=np.int8, mode="r", shape=(n_val,))
            y_val_arr = np.array(_yv_mm)
            del _yv_mm
            print(f"  {n_val:,} val samples; positive rate "
                  f"{float(y_val_arr.mean()):.4f}", flush=True)
            val_loader = DataLoader(
                val_store, batch_size=args.batch * 2, shuffle=False,
            )
        else:
            n_val = 0
            y_val_arr = np.empty((0,), dtype=np.int8)
            val_loader = DataLoader(
                TensorDataset(
                    torch.empty((0, n_features, args.window), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.float32),
                ),
                batch_size=args.batch * 2, shuffle=False,
            )
        # pos_weight uses train labels only (already loaded above).
        n_neg = int(n_tr - n_pos)
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        y_tr = y_tr_arr   # alias for config.json reporting
        y_val = y_val_arr
    else:
        # RAM mode (original behavior).
        print("building train windows ...", flush=True)
        x_tr, y_tr, q_tr = _load_split(core_train_files, cfg, args.window,
                                        scaler, args.neg_frac, quality_stems)
        print(f"  {len(y_tr):,} samples; positive rate "
              f"{float(y_tr.mean()):.4f}", flush=True)
        if q_tr is not None:
            print(f"  quality-oversample: {int(q_tr.sum()):,}/{len(y_tr):,} "
                  f"train samples from top {QUALITY_TOP_N} manifest rows -> "
                  f"x{args.quality_oversample} sampling weight", flush=True)
        if val_files:
            print(f"building val windows (val_neg_frac={val_neg_frac}) ...",
                  flush=True)
            x_val, y_val, _ = _load_split(val_files, cfg, args.window, scaler,
                                          neg_frac=val_neg_frac)
            print(f"  {len(y_val):,} samples; positive rate "
                  f"{float(y_val.mean()):.4f}", flush=True)
        else:
            x_val = np.empty((0, n_features, args.window), dtype=np.float32)
            y_val = np.empty((0,), dtype=np.int8)
        torch.manual_seed(args.seed)
        train_dataset = JitterDataset(
            torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32)),
            sigma=args.jitter_sigma, seed=args.seed,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_val),
                         torch.from_numpy(y_val.astype(np.float32))),
            batch_size=args.batch * 2, shuffle=False,
        )
        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    sampler = None
    if q_tr is not None:
        weights = np.where(q_tr, float(args.quality_oversample), 1.0)
        sampler = WeightedRandomSampler(torch.from_numpy(weights),
                                        num_samples=len(weights),
                                        replacement=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, sampler=sampler,
        shuffle=(sampler is None), drop_last=True,
    )

    model = ScalpTCN(n_features=n_features, window_s=args.window,
                     channels=args.channels, blocks=args.blocks,
                     dropout=args.dropout)
    print(f"training on {device} (pos_weight={pos_weight:.3f}, "
          f"lr_schedule={args.lr_schedule}) ...", flush=True)
    history, best_state = train(model, (train_loader, val_loader), args.epochs,
                                args.lr, pos_weight, device, args.patience,
                                schedule=args.lr_schedule)

    state_to_save = best_state if best_state is not None else model.state_dict()
    torch.save(state_to_save, out / "model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "window": args.window, "channels": args.channels,
        "blocks": args.blocks, "dropout": args.dropout,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "neg_frac": args.neg_frac, "val_neg_frac": val_neg_frac,
        "patience": args.patience,
        "jitter_sigma": args.jitter_sigma,
        "quality_oversample": args.quality_oversample,
        "quality_top_n": QUALITY_TOP_N,
        "seed": args.seed, "n_features": n_features,
        "n_core_train_days": len(core_train_files),
        "n_val_days": len(val_files), "n_train_samples": int(len(y_tr)),
        "n_val_samples": int(len(y_val)), "pos_weight": pos_weight,
        "barrier_mode": args.barrier_mode, "target_ps": args.target_ps,
        "stop_ps": args.stop_ps, "timeout_s": args.timeout,
        "vol_target_mult": args.vol_target_mult,
        "vol_stop_mult": args.vol_stop_mult, "vol_window_s": args.vol_window,
        "test_start_date": args.test_start_date,
        "val_start_date": args.val_start_date,
        "train_quality_limit": args.train_quality_limit,
        "device": str(device),
        "device_arg": args.device,
        "device_resolved_date": time.strftime("%Y-%m-%d"),
        "lr_schedule": args.lr_schedule,
        "window_store": args.window_store,
        "git_head": _git_head(),
    }, indent=2))

    # Delete disk cache after a successful training run (model.pt is saved).
    if use_disk and wcache_dir.exists():
        shutil.rmtree(wcache_dir)
        print(f"  wcache deleted: {wcache_dir}")

    if history:
        best = max(history, key=lambda r: r["val_ap"])
        print(f"\nbest epoch {best['epoch']}: val_loss {best['val_loss']:.4f} "
              f"val_AP {best['val_ap']:.4f}")
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
