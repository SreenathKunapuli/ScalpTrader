"""Sizing head: Kelly monotonicity, hard caps, zero-edge refusal."""

from __future__ import annotations

import numpy as np
import pytest
from scalp.sizing import size_scalp
from scalp.viability import LOT_SIZE, FeeModel

from scalpengine.config.scalp_tiers import SCALP_LARGE, SCALP_SMALL

# generous tape so only the account-side caps bind unless a test says otherwise
DEEP = dict(vol_60s=1e9, ask_lots=1e6)
BAR = dict(target_ps=0.05, stop_ps=0.03, price=3.00)


def test_zero_when_edge_nonpositive():
    # p=0.4 on ~5/3 odds is negative edge
    assert size_scalp(0.40, 0.03, 0.05, 3.0, 100_000, cfg=SCALP_LARGE, **DEEP) == 0
    assert size_scalp(0.0, **BAR, equity=100_000, cfg=SCALP_LARGE, **DEEP) == 0


def test_monotone_nondecreasing_in_p_win():
    for cfg, equity in ((SCALP_SMALL, 1_000), (SCALP_LARGE, 100_000)):
        sizes = [size_scalp(p, **BAR, equity=equity, cfg=cfg, **DEEP)
                 for p in np.linspace(0.0, 1.0, 41)]
        assert sizes == sorted(sizes)
        assert sizes[-1] > 0


def test_never_exceeds_any_cap():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = rng.uniform(0, 1)
        tgt, stp = rng.uniform(0.01, 0.30), rng.uniform(0.01, 0.20)
        px = rng.uniform(0.3, 12.0)
        eq = rng.choice([1_000.0, 100_000.0])
        vol = rng.uniform(0, 2e6)
        lots = rng.uniform(0, 500)
        cfg = SCALP_SMALL if eq < 10_000 else SCALP_LARGE
        s = size_scalp(p, tgt, stp, px, eq, vol, lots, cfg)
        assert s >= 0
        assert s * px <= cfg.max_position_pct * eq + px       # int floor slack
        assert s <= cfg.max_participation * vol + 1
        assert s <= 2.0 * lots * LOT_SIZE + 1
        if not (cfg.price_min <= px <= cfg.price_max):
            assert s == 0


def test_price_band_guardrail():
    assert size_scalp(0.9, **BAR | {"price": 0.30}, equity=100_000,
                      cfg=SCALP_LARGE, **DEEP) == 0
    assert size_scalp(0.9, **BAR | {"price": 15.0}, equity=100_000,
                      cfg=SCALP_LARGE, **DEEP) == 0


def test_tier_economics_sane():
    # confident scalp on a deep tape: small tier bets most of the account,
    # large tier is bounded by its 10% single-scalp cap
    s_small = size_scalp(0.9, **BAR, equity=1_000, cfg=SCALP_SMALL, **DEEP)
    s_large = size_scalp(0.9, **BAR, equity=100_000, cfg=SCALP_LARGE, **DEEP)
    assert 0 < s_small * BAR["price"] <= 0.9 * 1_000 + BAR["price"]
    assert 0 < s_large * BAR["price"] <= 0.10 * 100_000 + BAR["price"]
    assert s_large > s_small


def test_thin_tape_binds():
    # 5% participation of 4000 shares = 200; 2x of 1 lot = 200
    s = size_scalp(0.9, **BAR, equity=100_000, vol_60s=4_000, ask_lots=1.0,
                   cfg=SCALP_LARGE)
    assert 0 < s <= 200


def test_invalid_inputs_refuse():
    for bad in (dict(p_win=float("nan")), dict(price=-1.0),
                dict(equity=0.0), dict(target_ps=0.0)):
        kw = dict(p_win=0.9, **BAR, equity=100_000) | bad
        assert size_scalp(cfg=SCALP_LARGE, **kw, **DEEP) == 0


def test_fees_reduce_edge():
    # with fees zeroed the same marginal setup sizes >= the fee-charged one
    p, kw = 0.45, dict(**BAR, equity=100_000, cfg=SCALP_LARGE) | DEEP
    with_fees = size_scalp(p, **kw)
    no_fees = size_scalp(p, **kw, fees=FeeModel(sec_rate=0.0, taf_per_share=0.0,
                                                taf_cap=0.0))
    assert no_fees >= with_fees


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
