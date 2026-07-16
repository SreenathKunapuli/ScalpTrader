# ScalpTrader

ML scalping platform for cheap (<$10), high-relative-volume stocks: scanner → second-resolution signal models → risk-guarded execution on Alpaca (paper-trading only), with a live web dashboard (owner login + guest spectate).

Forked from the owner's LOB platform; the engine (execution, risk, kill switch, persistence) is signal-agnostic and battle-tested there. Strategy viability is gated by a cost-aware oracle study (`research/scripts/viability_study.py`) before any model is trained — every quoted metric must be reproducible from committed scripts.

- Engine: `scalpctl run --tier <low|medium|high>` (paper only, structurally enforced)
- Tests: `pytest`
