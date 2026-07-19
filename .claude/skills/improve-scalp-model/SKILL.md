---
name: improve-scalp-model
description: Run one protocol-clean improvement round on the ScalpTrader scalp-entry model — candidate levers, exact commands, validation discipline, sim gate, matched-day comparison, deploy/rollback. Follow mechanically when asked to "make the model better".
---

# Improve the scalp model (one round)

All paths relative to /Users/sreenathkunapuli/ScalpTrader; python is `.venv/bin/python`.
Delegation — match the subagent model to the task, full ladder allowed:
claude-sonnet-5 (pass the id EXPLICITLY — the 'sonnet' alias is an older
model) for command runs, doc edits, and tightly-specced plumbing; opus for
mid-complexity logic where sonnet risks subtle errors; fable (tight scope,
low effort) only where judgment genuinely moves trading results — adversarial
causality/parity reviews, quant-critical design. Be mindful of tokens: strong
models only where they make a real difference. Haiku is too weak for this
project. You design, review, and gate. Long commands (20-60 min): agents must
nohup-background them and poll a log; never let a foreground timeout kill a
run.

## Non-negotiable ground rules

1. NEVER deploy on barrier/label numbers — only the fill simulator counts
   (history: a "+$4.8k" config lost −$187k simulated).
2. One OOS test read per round, winner only. Selection happens on validation.
   KNOWN HAZARD: the 2025-H1 val window rank-inverted vs test three times —
   demand a val margin beating the best-known val cell before burning the test
   read, and treat small val edges as noise.
3. Matched-day comparison is the deploy criterion. Deployed since
   2026-07-19 (round 7): 26-feature GBT @ thr 0.7
   (runs/scalper/20260719_112357) — matched-122-day sim taker +$40,930
   (+7.40¢/sh) / maker +$39,652 (+6.80¢/sh), runs/sim_eval/20260719_120559.
   A future candidate deploys only if, on the SAME matched days, it beats
   BOTH legs of that cell, or beats the historical 0.6-maker high-water
   mark (+$65,582, runs/sim_eval/20260718_021932) with taker positive.
   Rollback artifact: runs/scalper/20260718_015432 (19-feat @0.6).
4. Settled facts — do not re-test: vol-scaled barriers (fixed-cent dead);
   timeout-only execution (--exec-stop-mult 1000; nearby stops pay 16–68¢/sh
   gap-through); default HistGradientBoosting HPs (round 1); hard quality
   cutoffs other than top-451 (rounds 2/4: matched-day maker is monotone in
   quality depth — 451:$65.6k, 800:$41.5k, 887:$30.8k); date-recency filters
   (round 3). Full history: docs/MODEL_CARD.md.
5. Every quoted number needs a runs/ artifact path. No attribution of any kind
   in git commits (no Co-Authored-By, no "Generated with", no model names).
6. Agents never touch .env. Deploy flips SCALP_ARTIFACT_DIR in .env BY HAND.

## Round procedure

Standard data flags for every command below (call this BASE):
`--barrier-mode vol --vol-target-mult 1.0 --vol-stop-mult 0.5 --timeout 120 --test-start-date 2025-06-27`

1. **Baseline** = deployed artifact dir from `.env SCALP_ARTIFACT_DIR`
   (currently runs/scalper/20260719_112357: 26 features, thr 0.7,
   exec_stop_mult 1000, trained on the top-451-quality corpus).
2. **Candidates** (pick ONE lever per round, 2-4 cells):
   `research/scripts/train_scalper.py BASE --val-start-date 2025-01-01 <lever flags>`
   Lever flags available: `--train-quality-limit N` (top-N manifest rows),
   `--quality-weight-mult M [--quality-weight-top N]`, `--train-start-date D`,
   `--max-leaf-nodes/--learning-rate/--max-iter/--min-samples-leaf/
   --l2-regularization`, `--drop-features a,b,c`. New features go into
   `research/scalp/bars_features.py::build_features` (strictly causal —
   rolling/shift/cummax over PAST rows only; the test_no_lookahead battery
   auto-covers new columns; session-cumulative features already have a
   full-session live frame). Run candidates SEQUENTIALLY (18 GB RAM).
3. **Select** on `val_per_threshold.csv` thr-0.6 sum_pnl_1000sh. The bar to
   earn a test read: beat the best-known val cell (+$3,739, top-800) — update
   that number here when it moves.
4. **One test read**: same command, no val flags. Read per_threshold.csv.
5. **Sim gate**: `research/scripts/sim_eval.py BASE --threshold 0.6
   --exec-stop-mult 1000 <same lever flags>`.
6. **Matched-day breakdown** (pandas, FILLED rows only, on the sim run's
   trades_taker.parquet / trades_maker.parquet): matched set = stems of the
   first 451 status=="ok" rows of data/corpus/manifest.csv whose date part
   >= "2025-06-27" (122 stems). Per mode: sum(pnl), sum(fill_qty),
   expectancy_ps on matched days and on the complement. Compare to the
   baseline numbers in ground-rule 3.
7. **Deploy** (only if gate passes): `research/scripts/export_model.py
   --run-dir <winner> --threshold 0.6 --exec-stop-mult 1000 <lever flags if
   the export script takes them — quality/recency limits are read from the
   run's config.json automatically>`; user flips .env; old artifact dir stays
   as rollback.
8. **Record**: append a round section to docs/MODEL_CARD.md (design, val
   table, test read, sim + matched-day table, verdict) and commit; update the
   playbook memory's learned-facts and, if the earn-a-test-read bar moved,
   step 3 of this skill.

## Data augmentation doctrine (synthetic data)

Synthetic data adds NO new market information — it is a regularizer only.
Permitted forms: (a) quality-weighted sampling (`--quality-weight-mult`) —
soft oversampling of proven-good days; (b) train-time perturbation for the
deep rung (Gaussian jitter on input windows, oversampling top-quality days
per epoch) — never applied to val/test data. FORBIDDEN: generative tape
synthesis (GAN/diffusion) and any augmentation of evaluation data — these
manufacture fake edge that the sim gate exposes only after wasted compute.
Augmented candidates go through the exact same round procedure.

## Deep rung (TCN)

Code: research/scalp/deep/ + research/scripts/train_tcn.py (240s × 19-feature
causal windows, scaler fit on train only, TcnProbModel exposes predict_proba
so sim_eval works unchanged). Train on the top-451-quality set; val is for
early stopping only. The TCN promotes to the engine ONLY if it beats the GBT
baseline in the matched-day sim comparison (same bars as ground-rule 3).

## When a subagent dies mid-round ("monthly spend limit")

The error is often transient. Check disk for the dead agent's partial work
first (runs/ dirs, git status) — they usually finish the command before dying.
Probe with a one-line sonnet-5 agent; resume the workflow (completed stages
are cached) or finish locally with background Bash. Never redo finished work.
