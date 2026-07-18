"""TCN deep rung: causal sequence model on top of the same per-second
features the GBT rung and the live engine use (scalp.bars_features).

Modules:
    dataset    — window construction, per-feature scaler, negative
                 subsampling (all pure / no I/O side effects beyond
                 reading the input parquet files the caller hands in).
    model      — ScalpTCN (causal TCN, ported from ~/LOB research/lob/
                 models.py) and TcnProbModel, the sklearn-shaped
                 predict_proba wrapper sim_eval.day_entries expects.
    train_loop — BCEWithLogits + AdamW + early-stopping-on-val-AP loop.
"""
