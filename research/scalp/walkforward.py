"""Walk-forward training/evaluation for the scalp entry model (GBT rung).

Discipline (non-negotiable, see plan):
- Splits are DAY-level and strictly temporal: every test day is later than
  every train day, with an embargo gap dropped from the TRAIN side.
- No cross-day features; each stock-day is featurized independently.
- Every reported number is out-of-sample; nothing in-sample leaves this
  module.
- Evaluation enforces NON-OVERLAPPING trades per stock-day: the labeler
  labels every second, and adjacent seconds flag the same move — counting
  them all would inflate trade counts and sum-PnL. Expectancy uses the
  label outcome (+target / -stop / timeout_edge), i.e. it assumes fills at
  the barrier prices; the event-driven simulator refines this later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bars_features import build_features
from .triple_barrier import BarrierConfig, label_scalps
from .viability import FeeModel


@dataclass(frozen=True)
class TrainConfig:
    target_ps: float = 0.05
    stop_ps: float = 0.04
    timeout_s: int = 120
    prob_threshold_grid: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7)
    embargo_days: int = 1
    test_frac: float = 0.25
    max_rows_per_day: int = 2000
    clip_shares: int = 1000
    seed: int = 7
    fees: FeeModel = field(default_factory=FeeModel)

    def barrier(self) -> BarrierConfig:
        return BarrierConfig(target_ps=self.target_ps, stop_ps=self.stop_ps,
                             timeout_s=self.timeout_s, fees=self.fees,
                             clip_shares=self.clip_shares)


def _day_key(path: Path) -> tuple[str, str]:
    sym, date = path.stem.rsplit("_", 1)
    return sym, date


def build_dataset(day_files: list[Path], cfg: TrainConfig,
                  ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Features/labels/meta across stock-days, per-day capped and stratified."""
    rng = np.random.default_rng(cfg.seed)
    xs, ys, ms = [], [], []
    for path in sorted(day_files):
        sym, date = _day_key(path)
        bars = pd.read_parquet(path)
        feats = build_features(bars)
        lab = label_scalps(bars, cfg.barrier())
        valid = lab["label"].notna()
        if not valid.any():
            continue
        f, y = feats[valid], lab.loc[valid, "label"]
        meta = pd.DataFrame({
            "symbol": sym, "date": date,
            "timeout_edge": lab.loc[valid, "timeout_edge"],
            "exit_s": lab.loc[valid, "exit_s"],
        }, index=f.index)
        if len(f) > cfg.max_rows_per_day:
            # stratified subsample: keep label proportions, seeded
            frac = cfg.max_rows_per_day / len(f)
            keep_idx = (
                pd.Series(np.arange(len(f)), index=f.index)
                .groupby(y.to_numpy())
                .apply(lambda s: s.sample(max(1, int(math.ceil(len(s) * frac))),
                                          random_state=rng.integers(2**31)))
                .droplevel(0).sort_values()
            )
            sel = f.index[keep_idx.to_numpy()]
            f, y, meta = f.loc[sel], y.loc[sel], meta.loc[sel]
        xs.append(f)
        ys.append(y)
        ms.append(meta)
    if not xs:
        raise ValueError("no labeled rows in any day file")
    return pd.concat(xs), pd.concat(ys), pd.concat(ms)


def split_days(dates: list[str], cfg: TrainConfig) -> tuple[list[str], list[str]]:
    """Strictly temporal day split with an embargo dropped from the train side."""
    uniq = sorted(set(dates))
    n_test = max(1, math.ceil(len(uniq) * cfg.test_frac))
    test = uniq[-n_test:]
    train = uniq[:-n_test]
    first_test = pd.Timestamp(test[0])
    train = [d for d in train
             if (first_test - pd.Timestamp(d)).days > cfg.embargo_days]
    return train, test


def fit_model(x: pd.DataFrame, y: pd.Series, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    freq = y.value_counts(normalize=True)
    w = y.map(lambda v: 1.0 / (len(freq) * freq[v])).to_numpy()
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(x, y, sample_weight=w)
    return model


def _non_overlapping(sel: pd.DataFrame) -> pd.DataFrame:
    """Greedy earliest-first non-overlap per (symbol, date), using exit_s."""
    kept = []
    for _, g in sel.groupby(["symbol", "date"], sort=False):
        g = g.sort_index()
        busy_until: pd.Timestamp | None = None
        for ts, row in g.iterrows():
            if busy_until is not None and ts < busy_until:
                continue
            kept.append(row)
            hold = row["exit_s"] if np.isfinite(row["exit_s"]) else 0.0
            busy_until = ts + pd.Timedelta(seconds=float(hold))
    return pd.DataFrame(kept)


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series,
             meta_test: pd.DataFrame, cfg: TrainConfig,
             ) -> tuple[dict, pd.DataFrame]:
    """OOS per-threshold economics with non-overlapping trade selection."""
    classes = list(model.classes_)
    if 1.0 in classes:
        p_win = model.predict_proba(x_test)[:, classes.index(1.0)]
    else:  # training data had no WIN labels — model can never signal an entry
        p_win = np.zeros(len(x_test))
    edge = np.where(y_test.to_numpy() == 1.0, cfg.target_ps,
                    np.where(y_test.to_numpy() == -1.0, -cfg.stop_ps,
                             meta_test["timeout_edge"].fillna(0.0).to_numpy()))
    base = meta_test.assign(p_win=p_win, edge=edge, label=y_test.to_numpy())
    n_days = meta_test["date"].nunique()
    rows = []
    for thr in cfg.prob_threshold_grid:
        sel = base[base["p_win"] >= thr]
        sel = _non_overlapping(sel) if len(sel) else sel
        n = len(sel)
        wins = int((sel["label"] == 1.0).sum()) if n else 0
        exp_ps = float(sel["edge"].mean()) if n else float("nan")
        rows.append({
            "threshold": thr, "n_trades": n,
            "trades_per_day": n / n_days if n_days else 0.0,
            "hit_rate": wins / n if n else float("nan"),
            "expectancy_ps": exp_ps,
            "expectancy_usd_1000sh": exp_ps * 1000 if n else float("nan"),
            "sum_pnl_1000sh": float(sel["edge"].sum() * 1000) if n else 0.0,
        })
    per_thr = pd.DataFrame(rows)
    summary = {
        "n_test_days": int(n_days),
        "n_test_rows": int(len(x_test)),
        "base_rates": y_test.value_counts(normalize=True).round(4).to_dict(),
        "per_threshold": per_thr.to_dict(orient="records"),
    }
    return summary, per_thr
