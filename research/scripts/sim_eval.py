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
from scalp.deep.model import TcnProbModel  # noqa: E402
from scalp.sim import SimConfig, simulate  # noqa: E402
from scalp.triple_barrier import label_scalps  # noqa: E402
from scalp.walkforward import TrainConfig, barrier_arrays, build_dataset, \
    fit_model, split_days  # noqa: E402
from scripts.train_scalper import drop_columns, limit_by_quality, \
    parse_drop_features, quality_weight_array  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
CORPUS_DIR = ROOT / "data" / "corpus" / "1s"


def day_entries(bars: pd.DataFrame, model, cfg: TrainConfig,
                threshold: float, qty: int,
                exec_stop_mult: float = 1.0,
                drop_feats: list[str] | None = None,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry decisions for one stock-day + their barrier-assumption edges.

    Gating mirrors the live engine: barriers warm (finite), NBBO valid at
    the decision second, p_win >= threshold. Returns (entries for
    sim.simulate, label-edge frame aligned to the same index).
    """
    feats = build_features(bars)
    feats = drop_columns(feats, drop_feats or [])
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
        "stop_px": ask[sel] - stp[sel] * exec_stop_mult,
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
    p.add_argument("--exec-stop-mult", type=float, default=1.0,
                   help="execution stop distance as a multiple of the LABEL "
                        "stop (decouples the bracket from the label geometry; "
                        "large value ~ timeout-only exits with the timeout "
                        "bounding risk)")
    p.add_argument("--qty", type=int, default=1000)
    p.add_argument("--max-participation", type=float, default=0.05)
    p.add_argument("--maker-wait", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="cap #stock-days")
    p.add_argument("--out", default="runs/sim_eval")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--max-leaf-nodes", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=None)
    p.add_argument("--l2-regularization", type=float, default=None)
    p.add_argument("--drop-features", default=None,
                   help="comma-separated feature columns to drop, e.g. 'a,b,c'")
    p.add_argument("--test-start-date", default=None,
                   help="pin the OOS test window to all days >= this ISO "
                        "date (yyyy-mm-dd) instead of the trailing "
                        "test_frac fraction, so a growing corpus keeps a "
                        "comparable test set")
    p.add_argument("--train-start-date", default=None,
                   help="drop train days older than this ISO date "
                        "(yyyy-mm-dd), applied after the test/embargo "
                        "split so it never touches the test window — a "
                        "training-recency knob")
    p.add_argument("--train-quality-limit", type=int, default=None,
                   help="corpus quality-depth knob: after the day splits "
                        "are computed on the full file list, restrict the "
                        "TRAINING-FIT file set to files among the first N "
                        "rows of the manifest (status=='ok' rows, in "
                        "fetch-priority / quality-rank order). The test "
                        "file set is NEVER filtered. None (default) "
                        "applies no filter.")
    p.add_argument("--quality-weight-mult", type=float, default=None,
                   help="soft quality-curation knob: rows whose source file "
                        "is among the first --quality-weight-top status=='ok' "
                        "manifest rows get this sample-weight multiplier "
                        "(composed with the existing class-balanced "
                        "weights); all other rows get 1.0. None (default) "
                        "applies no reweighting. Applied to this script's "
                        "own train fit so a future gate run trains "
                        "identically to train_scalper.py.")
    p.add_argument("--quality-weight-top", type=int, default=451,
                   help="number of leading status=='ok' manifest rows "
                        "(fetch-priority / quality-rank order) treated as "
                        "'top quality' for --quality-weight-mult")
    p.add_argument("--tcn-run-dir", default=None,
                   help="load the TCN deep-rung model from this "
                        "train_tcn.py run dir (model.pt + scaler.json + "
                        "config.json) via scalp.deep.model.TcnProbModel."
                        "load, instead of fitting a GBT — SKIPS the GBT "
                        "dataset build + fit entirely. Everything "
                        "downstream (day_entries gating, simulate, report) "
                        "is unchanged. HP flags and --drop-features are "
                        "ignored in this mode and must be left unset.")
    args = p.parse_args()

    cfg = TrainConfig(target_ps=args.target_ps, stop_ps=args.stop_ps,
                      timeout_s=args.timeout, barrier_mode=args.barrier_mode,
                      vol_target_mult=args.vol_target_mult,
                      vol_stop_mult=args.vol_stop_mult,
                      vol_window_s=args.vol_window,
                      test_start_date=args.test_start_date,
                      train_start_date=args.train_start_date)
    drop_feats = parse_drop_features(args.drop_features)
    hp = dict(learning_rate=args.learning_rate, max_iter=args.max_iter,
             max_leaf_nodes=args.max_leaf_nodes,
             min_samples_leaf=args.min_samples_leaf,
             l2_regularization=args.l2_regularization)
    if args.tcn_run_dir is not None:
        assert not drop_feats, \
            "--drop-features is ignored with --tcn-run-dir; leave it unset"
        assert all(v is None for v in hp.values()), \
            "HP flags are ignored with --tcn-run-dir; leave them unset"
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

    if args.tcn_run_dir is not None:
        run_dir = Path(args.tcn_run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        print(f"loading TCN model from {run_dir} ...", flush=True)
        model = TcnProbModel.load(run_dir)
    else:
        if args.train_quality_limit is not None:
            train_files = limit_by_quality(train_files, files,
                                           args.train_quality_limit)
            print(f"  quality-limit: train fit restricted to top "
                  f"{args.train_quality_limit} manifest rows -> "
                  f"{len(train_files)} files")

        print("fitting model on train days ...", flush=True)
        x_tr, y_tr, m_tr = build_dataset(train_files, cfg)
        x_tr = drop_columns(x_tr, drop_feats)

        weight_mult = None
        if args.quality_weight_mult is not None:
            top_stems = {f.stem for f in files[:args.quality_weight_top]}
            weight_mult = quality_weight_array(m_tr, top_stems,
                                               args.quality_weight_mult)
            print(f"  quality-weight: top {args.quality_weight_top} manifest "
                  f"rows -> x{args.quality_weight_mult} "
                  f"({int((weight_mult != 1.0).sum())} of {len(weight_mult)} "
                  f"rows)", flush=True)

        model = fit_model(x_tr, y_tr, cfg.seed, sample_weight_mult=weight_mult,
                          **hp)

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
        entries, lab = day_entries(bars, model, cfg, args.threshold, args.qty,
                                   args.exec_stop_mult, drop_feats)
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
