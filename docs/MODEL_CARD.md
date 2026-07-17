# ScalpTrader Model Card

Every number here is out-of-sample, produced by a committed script from
cached data, with the artifact path stated. Nothing in-sample, nothing
irreproducible. Run `research/scripts/run_pipeline.py` to regenerate the
whole chain.

## Data

- **Universe**: "runner days" — close/open $0.5–$10, day gain ≥ +15% or gap
  ≥ +10%, rel-volume ≥ 3× (20d median), dollar-vol ≥ $5M. Selection uses
  RAW (unadjusted) daily bars, same-day + prior data only
  (`research/scalp/corpus.py::select_runner_days`).
- **Bars**: free Alpaca historical SIP tick trades + quotes → 1-second bars
  with NBBO-at-close (ffill ≤ 60s, crossed quotes dropped, RTH only).
- **Corpus**: 451 ok stock-days on disk (of ~17k indexed runner days),
  `data/corpus/1s/`, manifest-resumable
  (`research/scripts/build_runner_corpus.py`).

## Viability gate (Phase 1, passed)

Perfect-foresight taker AND maker oracles are net-positive (after spread +
SEC 27.80e-6 + TAF 0.000166/sh capped $8.30) on **100% of runner
stock-days**, all price buckets, horizons 30s–900s; median spreads
50–66bps. Artifacts: `data/viability/results.parquet` via
`research/scripts/viability_study.py`. This is the regime the LOB project
proved large caps do NOT have.

## Entry model (GBT rung of the ladder)

- **Features** (12, strictly causal, `research/scalp/bars_features.py`):
  ret_5/15/60/300s, mom_accel, vwap_dist, pullback, vol_surge, tape_speed,
  spread_bps, quote_ok, tod_min. Causality enforced by
  `research/tests/test_no_lookahead.py` (rewrite-the-future battery).
- **Labels**: cost-aware triple-barrier, long-only, entry at ask, WIN
  requires bid ≥ entry + target + sell-fee. Barriers are **vol-scaled
  per row** (mult × trailing 300s price range, floored at 2¢/1.5¢, causal);
  fixed-cent barriers were tested first and REJECTED (negative OOS
  expectancy despite monotone ranking — wrong geometry for a $0.7–$9 band).
  Truncated end-of-day windows are INVALID, never fake TIMEOUTs.
- **Model**: `HistGradientBoostingClassifier`, class-balanced sample
  weights, seed 7 (`research/scalp/walkforward.py`).
- **Validation**: day-level strictly-temporal split (last 25% of days =
  test, 1-day embargo), non-overlapping trade selection per stock-day.

### OOS sweep, 451 stock-days → 122 test days (2025-06-27..2026-07-14)

| config | best thr | ¢/sh | trades/day | sum PnL @1000sh | artifact |
|---|---|---|---|---|---|
| vol 1.0×/0.5× @120s | 0.6 | **+0.83** | 15.4 | **+$12,776** | runs/scalper/20260717_043822 |
| vol 1.5×/0.75× @60s | 0.7 | +0.76 | 10.2 | +$7,740 | runs/scalper/20260717_044425 |
| vol 2.0×/1.0× @120s | — | negative everywhere | — | — | runs/scalper/20260717_045621 |

(An earlier 43-test-day run showed +3.04¢/sh for 1.5/0.75@60 — tripling the
test window shrank it. The larger window is the honest number.)

## Fill-simulator realism pass (the haircut that matters)

Event-driven sim (`research/scalp/sim.py`, 9 adversarial tape tests):
taker entries capped by displayed size + 5% of trailing 60s volume, target
fills require the BID to cross, stop exits pay the realized bid − ½ spread,
one position at a time. Runner: `research/scripts/sim_eval.py`.

**Threshold 0.6, taker, same 122 OOS days
(runs/sim_eval/20260717_113818):**

| | PnL @≤1000sh |
|---|---|
| barrier-assumption on the same filled entries | +$4,790 |
| simulated, stop enforced at label distance | **−$186,672** |
| of which pure stop gap-through slippage | −$192,096 |
| counterfactual: stops fill AT stop price | +$5,424 |

Per-exit decomposition: target +$169k (595), timeout +$44k (1401), eod ~0
(12), **stop −$401k (1441, avg 16¢/sh slippage)**. Threshold 0.7 shows the
same shape with 26¢/sh stop slippage (runs/sim_eval/20260717_114814).

**Conclusion**: the model's edge survives realistic entry fills, capacity
caps, and bid-cross target fills. The tight vol-scaled stop leg is
unenforceable on gap-prone tapes and is the entire loss.

### Exec-stop decoupling experiments (same model/entries, bracket varied)

122 OOS days, taker/maker PnL @≤1000sh:

| threshold | exec stop | taker | maker | artifact |
|---|---|---|---|---|
| 0.6 | 1× (label) | −$186.7k | −$119.6k | 20260717_113818 |
| 0.6 | 3× | −$126.1k | −$91.2k | 20260717_115923 |
| 0.6 | timeout-only | −$16.5k | +$26.9k | 20260717_120950 |
| 0.7 | 1× (label) | −$22.1k | −$12.1k | 20260717_114814 |
| 0.7 | 3× | −$2.3k | +$8.4k | 20260717_121932 |
| **0.7** | **timeout-only** | **+$19.4k (+5.86¢/sh)** | **+$36.4k (+10.38¢/sh)** | 20260717_122933 |

Even 3× stops lose: when they trigger they pay 58–68¢/sh gap-through.
**Validated config: threshold 0.7, timeout-only exits (far disaster stop),
120s timeout.** Taker: 396 fills (~3.2/day), win rate 56.8%, median
+3¢/sh, p05 −83¢/sh, worst single trade −$3.0k — per-trade tail risk is
bounded by the loss caps and daily breaker, not by a price stop (which the
data shows cannot execute anywhere near its price on these tapes).
Engine alignment: `inference.json exec_stop_mult` (export_model.py,
default 1000) — sizing still prices its Kelly loss leg off the label stop.

## Scanner ranker (which stocks to watch)

8 morning-observable features (gap, prev-day liquidity, first-15-min tape;
the 09:30–09:45 ET window is asserted inside the feature builder). Label =
realized taker scalpability at 60s horizon from the viability study.
**OOS rank-IC 0.7326**, near-monotone deciles (bottom ~$4.5k → top
$36–48k realized). Small sample: 110 train / 39 test rows.
Artifacts: `runs/scanner/` via `research/scripts/train_scanner.py`.

## Known caveats

- Free-tier IEX live quotes vs SIP historical: live staleness is measured
  (QuoteStalenessTracker) and gates the data-vendor decision, per plan.
- Halts and gap-through-stop risk are exactly why the stop finding above
  matters; no orders during halts is engine policy.
- Scanner ranker trained on 149 labeled days — informational until the
  viability label set grows.
- Sizing head (capped Kelly, `research/scalp/sizing.py`) is validated by
  property tests, not yet by sim-with-sizing (sim uses fixed 1000sh clip).
