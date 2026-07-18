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
- **Microstructure features added 2026-07-18** (7, bringing the total to
  19): qimb, qimb_chg_30s, spread_rel, bid_ret_15s, sess_hi_dist, up_streak,
  tsize_surge — same causality battery. On the identical 451-day corpus /
  122 OOS test days, best-threshold expectancy improves at both grid
  thresholds: thr 0.6 +0.83¢/sh → +1.61¢/sh (sum PnL @1000sh $12,775 →
  $27,127), thr 0.7 +4.08¢/sh → +5.00¢/sh ($7,107 → $13,090). Run
  `runs/scalper/20260718_015432` vs baseline `runs/scalper/20260717_043822`.
  Permutation importance on that run (`feature_importances.csv`) is
  **negative for 3 of the 7 new features — qimb_chg_30s, bid_ret_15s,
  sess_hi_dist** — prune candidates at the next big-corpus retrain.
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
| 0.7 | timeout-only | +$19.4k (+5.86¢/sh) | +$36.4k (+10.38¢/sh) | 20260717_122933 |
| 0.7 | timeout-only, **19-feature model** (`--limit 451`) | +$18,511 | +$29,783 | runs/sim_eval/20260718_020553 |
| **0.6** | **timeout-only, 19-feature model** (`--limit 451`) | **+$12,330** | **+$65,582** | runs/sim_eval/20260718_021932 |

Even 3× stops lose: when they trigger they pay 58–68¢/sh gap-through.
The original 12-feature validated config was threshold 0.7, timeout-only
exits (far disaster stop), 120s timeout (taker: 396 fills ~3.2/day, win
rate 56.8%, median +3¢/sh, p05 −83¢/sh, worst single trade −$3.0k).

**NEW deployed config (2026-07-18): 19-feature model, threshold 0.6,
timeout-only exits.** Chosen over the thr-0.7 19-feature row because it is
sim-positive in BOTH entry modes (taker +$12,330, maker +$65,582) and the
live engine enters maker-first — thr 0.7 taker is also positive here, but
0.6 wins on the maker leg that the engine actually uses first. Per-trade
tail risk is bounded by the loss caps and daily breaker, not by a price
stop (the data shows a tight stop cannot execute anywhere near its price
on these tapes). **Caveat: this selection was made by comparing across
multiple sim_eval runs (two thresholds × two feature sets × two entry
modes) — a selection-bias risk inherent to picking the best cell after the
fact. Paper validation per `docs/LIVE_PROMOTION.md` must reproduce this
config's edge before it is trusted live.**
Engine alignment: `inference.json exec_stop_mult` (export_model.py,
default 1000) — sizing still prices its Kelly loss leg off the label stop.

## Hyperparameter search — round 1 (2026-07-18): no deploy

**Design.** Staged HP search on the pinned 451-day corpus. Selection used
ONLY inner temporal validation (`--val-frac 0.2`: 256 core-train days
≤2024-11-25, 72 val days 2024-12-02..2025-06-20); OOS test was untouched
during search.

| config | val thr-0.6 PnL @1000sh | note |
|---|---|---|
| default HP (baseline) | +$10,417 | reference |
| `--max-leaf-nodes 31` | — | worse |
| `--max-leaf-nodes 63` | **+$12,674 (+22%)** | **selected** |
| `--max-leaf-nodes 127` | +$1,665 | overfits |
| learning-rate / max-iter variants | all lower | rejected |
| permutation-guided feature drops | all lower | rejected |

Artifacts: `runs/hpsearch/{baseline,s1_leaf31,s1_leaf63,s1_leaf127,s2_lr05_it300,s2_lr1_it150,s3_bestdrop,s3_basedrop}/`.

**One-time OOS test read — leaf-63 winner** (full 328-day train fit, same
122 OOS days 2025-06-27..2026-07-14, `runs/scalper/20260718_062534`):

| | thr 0.6 |
|---|---|
| n_trades | 1797 (18.0/day) |
| hit_rate | 18.1% |
| expectancy | +1.04¢/sh |
| sum PnL @1000sh | +$18,669 |
| deployed default-HP model (same days) | **+$27,127** |

Leaf-63 trails the deployed model by $8,458 on the test set.

**Sim gate** (thr 0.6, exec-stop-mult 1000, `--limit 451`):

| model | taker @1000sh | maker @1000sh | artifact |
|---|---|---|---|
| leaf-63 | −$25,491 | +$19,718 | runs/sim_eval/20260718_064444 |
| deployed (default HP) | +$12,330 | +$65,582 | runs/sim_eval/20260718_021932 |

Gate requires beating the deployed model on maker with a non-negative taker
→ **FAILED**. Deployed artifact `runs/scalper/20260718_015432` unchanged.

**Diagnostic.** Leaf-63 attempted 42,327 sim entries at thr 0.6 vs the
deployed model's 38,193 — HP changes shift probability calibration so a
fixed threshold selects very different entry sets across configs; this
motivates adding a calibration lever before the next HP round.

**Conclusion.** Inner-val gains did not transfer to test or sim — exactly
the failure mode the round protocol (val-only selection, one test read,
mandatory sim gate) exists to catch.

## Big-corpus retrain — round 2 (2026-07-18): no deploy

**Setup.** Corpus grew 451 → 1,250 ok stock-days (`build_runner_corpus
--fetch 800`; new days are lower fetch-priority and skew 2020–2021). New
`--test-start-date` flag (commit bd99881) pins the OOS window at 2025-06-27
so it stays comparable as the corpus grows. Retrain of the deployed recipe
(default HP, 19 features) on all 887 pre-window days:
`runs/scalper/20260718_065308`. One-time test read on the expanded window
(361 test stock-days, 2025-06-27..2026-07-16): thr 0.6 hit 14.3%,
expectancy −0.65¢/sh, sum PnL @1000sh −$29,463; thr 0.7 +$9,344.

**Sim** (thr 0.6, exec-stop-mult 1000, full manifest):
`runs/sim_eval/20260718_075345` — full window taker −$68,799 (−0.96¢/sh,
8,716 fills), maker +$34,510 (+0.45¢/sh, 7,638 fills), 106,229 attempted.

**Matched-day comparison** (same simulator, identical 122 test days = test
days of the original 451-day corpus; baseline run `runs/sim_eval/20260718_021932`):

| model (training set) | subset | taker PnL | taker ¢/sh | maker PnL | maker ¢/sh |
|---|---|---|---|---|---|
| deployed (451-corpus) | matched 122 days | +$12,330.17 | +0.50 | +$65,581.64 | +2.37 |
| big-corpus (887 days) | matched 122 days | −$20,556.43 | −0.72 | +$30,843.04 | +0.96 |
| big-corpus (887 days) | 238 new test days | −$48,242.48 | −1.11 | +$3,667.00 | +0.08 |

Only the training data differs on the matched rows → the extra 2020–2021-vintage,
lower-priority training days actively dilute the edge. Verdict: gate FAILED,
deployed artifact unchanged.

**Implications.**
(a) Training-data vintage/regime match matters more than volume — motivates
a `--train-start-date` recency filter (round 3).
(b) The 238 lower-fetch-priority test days carry near-zero maker edge —
corpus quality tier matters, and the original high-priority days better
proxy what the live scanner selects; headline PnL/day figures should be
read against day quality.

## Scanner ranker (which stocks to watch)

8 morning-observable features (gap, prev-day liquidity, first-15-min tape;
the 09:30–09:45 ET window is asserted inside the feature builder). Label =
realized taker scalpability at 60s horizon from the viability study.
An adversarial review caught a train/serve provenance skew in the prev-day
volume features (training used the prior RUNNER-INDEX row, median 42 days
stale; serving uses the true prior calendar session) — fixed by deriving
training fields from the daily-raw cache, matching serving exactly.
**OOS rank-IC 0.7022 on clean provenance** (was 0.7326 on the skewed
features — the honest direction), near-monotone deciles (bottom ~$4.5k →
top ~$42–43k realized). Small sample: 110 train / 39 test rows.
Deployed artifact with model.joblib: `runs/scanner/20260717_171507`;
live path: `engine/scalpengine/scanner/live_scan.py`, invoked at 10:01 ET
(post-SIP-embargo) by the day scanner.

## Known caveats

- Free-tier IEX live quotes vs SIP historical: live staleness is measured
  (QuoteStalenessTracker) and gates the data-vendor decision, per plan.
- Halts and gap-through-stop risk are exactly why the stop finding above
  matters; no orders during halts is engine policy.
- Scanner ranker trained on 149 labeled days — informational until the
  viability label set grows.
- Sizing head (capped Kelly, `research/scalp/sizing.py`) is validated by
  property tests, not yet by sim-with-sizing (sim uses fixed 1000sh clip).
