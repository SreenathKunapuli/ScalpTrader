"""Push the trained scalp model's OOS entry decisions through the fill
simulator — the realism haircut between barrier-price expectancy and
what the tape would actually have paid.

Reproduces the exact walk-forward model (same TrainConfig/seed/split as
train_scalper.py), then for each OOS test day: entry decisions are the
seconds where barriers are warm, the NBBO is valid, and p_win >= the
threshold — the same gating the live engine applies — and sim.simulate
charges real fills (capacity caps, stop slippage, one position at a time).

Reported side by side: barrier-assumption PnL on the SAME attempted
entries vs simulated PnL, taker and maker entry modes.

Usage:
  .venv/bin/python research/scripts/sim_eval.py --barrier-mode vol \
      --vol-target-mult 1.0 --vol-stop-mult 0.5 --timeout 120 \
      --threshold 0.6 [--qty 1000] [--out runs/sim_eval]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalp.bars_features import build_features  # noqa: E402
from scalp.sim import SimConfig, simulate  # noqa: E402
from scalp.triple_barrier import label_scalps  # noqa: E402
from scalp.walkforward import TrainConfig, barrier_arrays, build_dataset, \
    fit_model, split_days  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"


def day_entries(bars: pd.DataFrame, model, cfg: TrainConfig,
                threshold: float, qty: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry decisions for one stock-day + their barrier-assumption edges.

    Gating mirrors the live engine: barriers warm (finite), NBBO valid at
    the decision second, p_win >= threshold. Returns (entries for
    sim.simulate, label-edge frame aligned to the same index).
    """
    feats = build_features(bars)
    tgt, stp = barrier_arrays(bars, cfg)
    ask = bars["ask"].to_numpy(dtype=float)
    bid = bars["bid"].to_numpy(dtype=float)
    nbbo_ok = np.isfinite(bid) & (bid > 0) & np.isfinite(ask) & (ask >= bid)
    gate = np.isfinite(tgt) & np.isfinite(stp) & nbbo_ok
    if not gate.any():
        empty = pd.DataFrame()
        return empty, empty
    classes = list(model.classes_)
    if 1.0 in classes:
        p_win = model.predict_proba(feats)[:, classes.index(1.0)]
    else:
        p_win = np.zeros(len(feats))
    sel = gate & (p_win >= threshold)
    if not sel.any():
        empty = pd.DataFrame()
        return empty, empty
    idx = bars.index[sel]
    entries = pd.DataFrame({
        "qty": qty,
        "target_px": ask[sel] + tgt[sel],
        "stop_px": ask[sel] - stp[sel],
        "deadline": idx + pd.Timedelta(seconds=cfg.timeout_s),
    }, index=idx)
    # barrier-assumption outcome for the same decisions, from the labeler
    lab = label_scalps(bars, cfg.barrier(), target_ps_arr=tgt, stop_ps_arr=stp)
    lab_sel = lab.loc[idx]
    edge = np.where(lab_sel["label"] == 1.0, tgt[sel],
                    np.where(lab_sel["label"] == -1.0, -stp[sel],
                             lab_sel["timeout_edge"].fillna(0.0)))
    label_frame = pd.DataFrame({"label": lab_sel["label"], "edge_ps": edge,
                                "exit_s": lab_sel["exit_s"]}, index=idx)
    return entries, label_frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--barrier-mode", choices=["fixed", "vol"], default="vol")
    p.add_argument("--target-ps", type=float, default=0.05)
    p.add_argument("--stop-ps", type=float, default=0.04)
    p.add_argument("--vol-target-mult", type=float, default=1.0)
    p.add_argument("--vol-stop-mult", type=float, default=0.5)
    p.add_argument("--vol-window", type=int, default=300)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--qty", type=int, default=1000)
    p.add_argument("--max-participation", type=float, default=0.05)
    p.add_argument("--maker-wait", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="cap #stock-days")
    p.add_argument("--out", default="runs/sim_eval")
    args = p.parse_args()

    cfg = TrainConfig(target_ps=args.target_ps, stop_ps=args.stop_ps,
                      timeout_s=args.timeout, barrier_mode=args.barrier_mode,
                      vol_target_mult=args.vol_target_mult,
                      vol_stop_mult=args.vol_stop_mult,
                      vol_window_s=args.vol_window)
    man = pd.read_csv(MANIFEST)
    ok = man[man["status"] == "ok"]
    files = [CORPUS_DIR / f"{r.symbol}_{r.date}.parquet"
             for r in ok.itertuples()
             if (CORPUS_DIR / f"{r.symbol}_{r.date}.parquet").exists()]
    if args.limit:
        files = files[: args.limit]
    dates = [f.stem.rsplit("_", 1)[1] for f in files]
    train_dates, test_dates = split_days(dates, cfg)
    train_files = [f for f, d in zip(files, dates, strict=True) if d in train_dates]
    test_files = [f for f, d in zip(files, dates, strict=True) if d in test_dates]
    print(f"stock-days: {len(files)} -> train {len(train_files)} | "
          f"test {len(test_files)} | threshold {args.threshold}")

    print("fitting model on train days ...", flush=True)
    x_tr, y_tr, _ = build_dataset(train_files, cfg)
    model = fit_model(x_tr, y_tr, cfg.seed)

    modes = {
        "taker": SimConfig(entry_mode="taker", fees=cfg.fees,
                           max_participation=args.max_participation),
        "maker": SimConfig(entry_mode="maker", maker_wait_s=args.maker_wait,
                           fees=cfg.fees,
                           max_participation=args.max_participation),
    }
    trades: dict[str, list[pd.DataFrame]] = {m: [] for m in modes}
    label_frames: list[pd.DataFrame] = []
    n_attempted = 0
    for i, f in enumerate(sorted(test_files)):
        bars = pd.read_parquet(f)
        entries, lab = day_entries(bars, model, cfg, args.threshold, args.qty)
        if entries.empty:
            continue
        n_attempted += len(entries)
        label_frames.append(lab.assign(day=f.stem))
        for mode, scfg in modes.items():
            res = simulate(bars, entries, scfg)
            # label edge rides along per-row: same-index within the day, so
            # no cross-day alignment (timestamps repeat across symbols)
            res["label_edge_ps"] = lab["edge_ps"]
            trades[mode].append(res.assign(day=f.stem))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(test_files)} days simmed", flush=True)

    if not label_frames:
        print("no entries selected at this threshold — nothing to report")
        return
    labels = pd.concat(label_frames)
    out = ROOT / args.out / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "config": {**{k: v for k, v in vars(args).items()},
                   "n_test_days": len(test_files),
                   "test_span": [min(test_dates), max(test_dates)]},
        "n_attempted": int(n_attempted),
    }
    print(f"\nSIM REPORT — OOS {min(test_dates)}..{max(test_dates)}, "
          f"{n_attempted} attempted entries @thr {args.threshold}")
    for mode in modes:
        allt = pd.concat(trades[mode])
        filled = allt[allt["filled"]]
        n_f = len(filled)
        sh = filled["fill_qty"].sum()
        pnl = filled["pnl"].sum()
        # barrier-assumption PnL over the same FILLED decisions at fill_qty:
        # what the sweep's expectancy math would have credited these trades
        label_pnl = float((filled["label_edge_ps"] * filled["fill_qty"]).sum())
        row = {
            "n_filled": int(n_f),
            "fill_rate": round(n_f / n_attempted, 4),
            "mean_fill_qty": round(float(filled["fill_qty"].mean()), 1) if n_f else 0,
            "sim_pnl_usd": round(float(pnl), 2),
            "sim_expectancy_ps": round(float(pnl / sh), 5) if sh else float("nan"),
            "label_pnl_same_fills_usd": round(label_pnl, 2),
            "haircut_pct": round(100 * (1 - pnl / label_pnl), 1)
            if label_pnl > 0 else float("nan"),
            "exit_reasons": filled["exit_reason"].value_counts().to_dict(),
            "stop_slippage_ps_mean": round(float(
                filled.loc[filled["exit_reason"] == "stop", "slippage_ps"]
                .mean()), 5) if (filled["exit_reason"] == "stop").any() else 0.0,
        }
        report[mode] = row
        allt.to_parquet(out / f"trades_{mode}.parquet")
        print(f"\n[{mode}] filled {n_f}/{n_attempted} "
              f"(mean qty {row['mean_fill_qty']})")
        print(f"  sim PnL          ${row['sim_pnl_usd']:>12,.2f}  "
              f"({row['sim_expectancy_ps'] * 100:.2f} c/sh)"
              if sh else "  sim PnL: no fills")
        print(f"  label-assumption ${row['label_pnl_same_fills_usd']:>12,.2f}  "
              f"-> haircut {row['haircut_pct']}%")
        print(f"  exits: {row['exit_reasons']}  "
              f"stop slip {row['stop_slippage_ps_mean'] * 100:.2f} c/sh")

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
