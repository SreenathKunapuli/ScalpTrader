# Promotion-to-Live Criteria

Live trading is enabled by the owner flipping `TRADING_MODE` + adding live
keys — never by the engine. Before that flip, every criterion below must
hold. Numbers are defaults chosen for the $1k starting tier; veto/adjust
before the first live session.

## Hard criteria (all required)

1. **≥ 20 paper sessions** with the scalp loop live on real-time IEX data,
   spanning at least 4 distinct calendar weeks (no cherry-picked regimes).
2. **Net-positive paper PnL after all fees** over the window, AND the worst
   single session ≥ −2× the average winning session (no blow-up shape).
3. **Fill-quality gap bound**: realized entry slippage vs sim assumptions
   (taker at ask, target at bid-cross) within 25% on average. Measured from
   the trade tape the dashboard records; compared against
   `research/scripts/sim_eval.py` on the same dates' corpus days.
4. **Stop discipline**: realized loss on stop/timeout exits within 1.5× of
   the sim's per-trade loss distribution p95. If gap-through exceeds this,
   the data feed (or the stop design) is not live-ready.
5. **Zero guardrail breaches caused by bugs**: kill switch, daily-loss
   breaker, per-symbol cap, PDT-margin checks each either never fired or
   fired correctly (post-mortem on every firing, filed in docs/).
6. **Staleness bound**: p95 quote age on focus-list symbols < 5s during
   RTH (from `QuoteStalenessTracker` snapshots persisted daily). If not,
   the free-feed upgrade decision (plan §data-vendor) happens BEFORE live.
7. **Broker readiness**: verify Alpaca's FINRA-4210 margin phase-in status
   for intraday round-trips (rule amended eff. 2026-06-04; brokers may
   phase in until Oct 2027). Written confirmation in .env comments or
   docs/ before the flip.
8. **Reconcile drill**: kill the engine mid-session on paper with open
   scalps, restart, confirm brackets re-arm and positions match broker.

## Capital ramp

- Weeks 1–2 live: $1k tier (`SCALP_SMALL`), max one concurrent scalp.
- Only after 10 net-positive live sessions: raise to 25% of target
  capital; after 20: full tier. Any guardrail breach resets the ramp.

## Standing constraints

- `max_live_equity_usd` set in .env from day one.
- Kill switch reachable from the dashboard at all times (owner login).
- Weekly walk-forward retrain (`scalpctl train` / train_scalper.py) with
  the OOS report reviewed before the next session uses a new artifact.
