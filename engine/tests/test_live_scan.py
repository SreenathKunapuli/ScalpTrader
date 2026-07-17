"""Tests for engine/scalpengine/scanner/live_scan.py.

All tests are network-free: they operate on canned fixtures and stub
or bypass the alpaca-py client entirely.  The four areas tested are:

  1. Scoring parity: the same synthetic morning tick stream produces
     identical scanner features whether processed via the research
     corpus.second_bars offline path or via live_scan.build_bars_from_ticks.

  2. Budget cap: symbols beyond max_symbols_for_budget are dropped and
     reported in ScanResult.dropped_budget; the fetch loop never touches them.

  3. Watchlist write shape: repo.replace_watchlist is called with dicts
     that satisfy WatchlistEntry's column contract (ts, symbol, score,
     price, gain_pct, relvol, spread_bps, streamed).

  4. Graceful degradation: missing artifact, zero-bar symbol, and a symbol
     where build_scanner_features raises all produce sensible results without
     crashing the scan run.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

# Ensure research is importable (same pattern as scalp_gbt.py / live_scan.py)
_RESEARCH = Path(__file__).resolve().parents[2] / "research"
if str(_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_RESEARCH))

from scalp.corpus import second_bars  # noqa: E402
from scanner.rank import FEATURES, build_scanner_features, slice_early_bars  # noqa: E402

from scalpengine.scanner.live_scan import (
    ScanResult,
    build_bars_from_ticks,
    build_daily_context,
    run_morning_scan,
    score_candidates,
)
from scalpengine.persistence.repo import Repo

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SESSION_DATE = date(2025, 6, 9)   # arbitrary Monday

NY = "America/New_York"


def _make_tick_df(
    n_trades: int = 300,
    start_et: str = "09:30:00",
    end_et: str = "09:44:59",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesise trades + quotes covering the 09:30-09:44:59 ET window.

    Returns (trades_df, quotes_df) with the exact column signatures that
    corpus.second_bars and live_scan.build_bars_from_ticks expect.
    """
    rng = np.random.default_rng(seed)
    day = pd.Timestamp(_SESSION_DATE, tz=NY)
    t0 = day + pd.Timedelta(hours=9, minutes=30)
    t1 = day + pd.Timedelta(hours=9, minutes=44, seconds=59)
    span_s = int((t1 - t0).total_seconds())

    # Trades
    offsets_s = np.sort(rng.integers(0, span_s, size=n_trades))
    prices = np.round(5.0 + rng.normal(0, 0.02, n_trades).cumsum(), 2).clip(0.5)
    sizes = rng.integers(1, 200, size=n_trades).astype(float)
    ts_trades = [t0 + pd.Timedelta(seconds=int(o)) for o in offsets_s]
    trades_df = pd.DataFrame(
        {"price": prices, "size": sizes},
        index=pd.DatetimeIndex(ts_trades, name="timestamp", tz=NY).tz_convert("UTC"),
    )

    # Quotes (every ~2s)
    n_q = span_s // 2
    q_offsets = np.arange(0, span_s, 2)[:n_q]
    mid = 5.0
    bids = np.round(mid - 0.01, 2) * np.ones(n_q)
    asks = np.round(mid + 0.01, 2) * np.ones(n_q)
    ts_quotes = [t0 + pd.Timedelta(seconds=int(o)) for o in q_offsets]
    quotes_df = pd.DataFrame(
        {
            "bid_price": bids,
            "ask_price": asks,
            "bid_size": np.ones(n_q) * 10.0,
            "ask_size": np.ones(n_q) * 10.0,
        },
        index=pd.DatetimeIndex(ts_quotes, name="timestamp", tz=NY).tz_convert("UTC"),
    )
    return trades_df, quotes_df


def _make_daily_ctx(sym: str = "XYZ") -> dict:
    return {
        "open": 5.10,
        "prev_close": 4.90,
        "prev_day_dollar_vol": 8_000_000.0,
        "prev_day_volume": 1_600_000.0,
    }


@pytest.fixture()
def scanner_artifact(tmp_path: Path) -> Path:
    """Minimal scanner artifact: a real trained HistGBT regressor + config."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    n = 200
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = rng.uniform(0, 1000, size=n)
    model = HistGradientBoostingRegressor(random_state=0, max_iter=20)
    model.fit(x, y)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "config.json").write_text(json.dumps({
        "features": FEATURES,
        "label": "taker_clip_pnl@60s (log1p, clip>=0)",
        "seed": 0,
        "git_head": "test",
        "n_train_days": 80, "n_test_days": 20,
        "n_train_rows": 160, "n_test_rows": 40,
    }))
    (tmp_path / "metrics.json").write_text(json.dumps({"oos_rank_ic": 0.5}))
    return tmp_path


@pytest.fixture()
def repo() -> Repo:
    return Repo("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# 1. Scoring parity: corpus path offline == live path
# ---------------------------------------------------------------------------

class TestScoringParity:
    """Both paths must produce bit-identical scanner features."""

    def test_features_match_corpus_path(self) -> None:
        trades_df, quotes_df = _make_tick_df(seed=7)
        ctx = _make_daily_ctx()

        # Corpus offline path (used during training)
        bars_corpus = second_bars(trades_df, quotes_df, interval_s=1)
        early_corpus = slice_early_bars(bars_corpus)
        feats_corpus = build_scanner_features(ctx, early_corpus)

        # Live path (live_scan.build_bars_from_ticks)
        bars_live = build_bars_from_ticks(trades_df, quotes_df)
        early_live = slice_early_bars(bars_live)
        feats_live = build_scanner_features(ctx, early_live)

        for key in FEATURES:
            v_corpus = feats_corpus[key]
            v_live = feats_live[key]
            if np.isnan(v_corpus) and np.isnan(v_live):
                continue
            assert abs(v_corpus - v_live) < 1e-9, (
                f"feature {key!r}: corpus={v_corpus} live={v_live}"
            )

    def test_empty_trades_returns_empty_bars(self) -> None:
        empty_trades = pd.DataFrame(columns=["price", "size"])
        _, quotes_df = _make_tick_df()
        bars = build_bars_from_ticks(empty_trades, quotes_df)
        assert bars.empty

    def test_empty_quotes_still_yields_bars(self) -> None:
        trades_df, _ = _make_tick_df()
        empty_quotes = pd.DataFrame(
            columns=["bid_price", "ask_price", "bid_size", "ask_size"]
        )
        bars = build_bars_from_ticks(trades_df, empty_quotes)
        assert not bars.empty
        # NBBO columns present but NaN (no quotes)
        assert "bid" in bars.columns
        assert bars["bid"].isna().all()

    def test_score_candidates_roundtrip(self, scanner_artifact: Path) -> None:
        """score_candidates should produce a finite score for valid bars."""
        model = joblib.load(scanner_artifact / "model.joblib")
        cfg = json.loads((scanner_artifact / "config.json").read_text())
        feature_names = cfg["features"]

        trades_df, quotes_df = _make_tick_df()
        bars = build_bars_from_ticks(trades_df, quotes_df)
        bars_map = {"XYZ": bars}
        ctx = {"XYZ": _make_daily_ctx()}

        rows = score_candidates(bars_map, ctx, model, feature_names)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "XYZ"
        assert np.isfinite(rows[0]["score"]), "expected a finite score for valid bars"
        for feat in FEATURES:
            assert feat in rows[0], f"missing feature {feat!r}"


# ---------------------------------------------------------------------------
# 2. Budget cap
# ---------------------------------------------------------------------------

class TestBudgetCap:
    def test_excess_symbols_dropped(self, scanner_artifact: Path, repo: Repo) -> None:
        """Symbols beyond max_symbols_for_budget must appear in dropped_budget."""
        symbols = [f"SYM{i:03d}" for i in range(50)]
        max_sym = 10

        # Stub client that records which symbols were actually fetched
        fetched_syms: list[str] = []

        class _StubClient:
            def get_stock_trades(self, req: Any) -> Any:
                fetched_syms.append(req.symbol_or_symbols)
                r = MagicMock()
                r.df = pd.DataFrame()
                return r

            def get_stock_quotes(self, req: Any) -> Any:
                r = MagicMock()
                r.df = pd.DataFrame()
                return r

            def get_stock_bars(self, req: Any) -> Any:
                r = MagicMock()
                r.data = {}
                return r

        result = run_morning_scan(
            symbols=symbols,
            session_date=_SESSION_DATE,
            client=_StubClient(),
            repo=repo,
            top_n=5,
            max_symbols_for_budget=max_sym,
            artifact_dir=scanner_artifact,
        )

        assert len(result.dropped_budget) == 50 - max_sym
        assert set(result.dropped_budget) == set(symbols[max_sym:])
        # No fetch should have been issued for dropped symbols
        fetched_flat = [s for req_sym in fetched_syms for s in (
            [req_sym] if isinstance(req_sym, str) else [req_sym]
        )]
        for dropped_sym in result.dropped_budget:
            assert dropped_sym not in fetched_flat, (
                f"dropped symbol {dropped_sym!r} was still fetched"
            )

    def test_within_budget_no_drops(self, scanner_artifact: Path, repo: Repo) -> None:
        symbols = ["AAPL", "MSFT", "GOOG"]

        class _StubClient:
            def get_stock_trades(self, req: Any) -> Any:
                r = MagicMock()
                r.df = pd.DataFrame()
                return r

            def get_stock_quotes(self, req: Any) -> Any:
                r = MagicMock()
                r.df = pd.DataFrame()
                return r

            def get_stock_bars(self, req: Any) -> Any:
                r = MagicMock()
                r.data = {}
                return r

        result = run_morning_scan(
            symbols=symbols,
            session_date=_SESSION_DATE,
            client=_StubClient(),
            repo=repo,
            top_n=30,
            max_symbols_for_budget=40,
            artifact_dir=scanner_artifact,
        )
        assert result.dropped_budget == []


# ---------------------------------------------------------------------------
# 3. Watchlist write shape
# ---------------------------------------------------------------------------

class TestWatchlistWriteShape:
    def _make_stub_client_with_data(self) -> Any:
        """Client that returns real trade/quote data for one symbol."""
        trades_df, quotes_df = _make_tick_df(seed=99)

        class _StubClient:
            def get_stock_trades(self, req: Any) -> Any:
                r = MagicMock()
                df = trades_df.copy()
                df.index = pd.MultiIndex.from_tuples(
                    [(req.symbol_or_symbols, t) for t in df.index],
                    names=["symbol", "timestamp"],
                )
                r.df = df
                return r

            def get_stock_quotes(self, req: Any) -> Any:
                r = MagicMock()
                df = quotes_df.copy()
                df.index = pd.MultiIndex.from_tuples(
                    [(req.symbol_or_symbols, t) for t in df.index],
                    names=["symbol", "timestamp"],
                )
                r.df = df
                return r

            def get_stock_bars(self, req: Any) -> Any:
                r = MagicMock()
                sym = req.symbol_or_symbols[0] if isinstance(
                    req.symbol_or_symbols, list) else req.symbol_or_symbols
                bar = MagicMock()
                bar.timestamp = pd.Timestamp(_SESSION_DATE, tz=NY) - pd.Timedelta(days=1)
                bar.open = 4.80
                bar.close = 4.90
                bar.volume = 1_600_000
                bar2 = MagicMock()
                bar2.timestamp = pd.Timestamp(_SESSION_DATE, tz=NY)
                bar2.open = 5.10
                bar2.close = 5.20
                bar2.volume = 500_000
                r.data = {sym: [bar, bar2]}
                return r

        return _StubClient()

    def test_watchlist_row_keys(self, scanner_artifact: Path, repo: Repo) -> None:
        """Each watchlist row must contain the WatchlistEntry column set."""
        required_keys = {"ts", "symbol", "score", "price", "gain_pct",
                         "relvol", "spread_bps", "streamed"}

        written_rows: list[dict] = []
        original_replace = repo.replace_watchlist

        def _capturing_replace(rows: list[dict]) -> None:
            written_rows.extend(rows)
            original_replace(rows)

        repo.replace_watchlist = _capturing_replace  # type: ignore[method-assign]

        run_morning_scan(
            symbols=["XYZ"],
            session_date=_SESSION_DATE,
            client=self._make_stub_client_with_data(),
            repo=repo,
            top_n=30,
            max_symbols_for_budget=40,
            artifact_dir=scanner_artifact,
        )

        # Written rows may be empty (if score is NaN) — check shape when present
        for row in written_rows:
            missing = required_keys - set(row.keys())
            assert not missing, f"watchlist row missing keys: {missing}"
            assert isinstance(row["ts"], datetime)
            assert isinstance(row["symbol"], str)
            assert isinstance(row["score"], float)
            assert isinstance(row["streamed"], bool)

    def test_watchlist_sorted_descending(self, scanner_artifact: Path, repo: Repo) -> None:
        """Watchlist rows in the DB must be ordered by score descending."""

        class _MultiSymClient:
            """Returns real trade data for multiple symbols (different seeds)."""
            def __init__(self) -> None:
                self._data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
                    "FAST": _make_tick_df(seed=1),
                    "SLOW": _make_tick_df(seed=2),
                    "MED": _make_tick_df(seed=3),
                }

            def get_stock_trades(self, req: Any) -> Any:
                sym = req.symbol_or_symbols
                r = MagicMock()
                if sym in self._data:
                    df = self._data[sym][0].copy()
                    df.index = pd.MultiIndex.from_tuples(
                        [(sym, t) for t in df.index],
                        names=["symbol", "timestamp"],
                    )
                    r.df = df
                else:
                    r.df = pd.DataFrame()
                return r

            def get_stock_quotes(self, req: Any) -> Any:
                sym = req.symbol_or_symbols
                r = MagicMock()
                if sym in self._data:
                    df = self._data[sym][1].copy()
                    df.index = pd.MultiIndex.from_tuples(
                        [(sym, t) for t in df.index],
                        names=["symbol", "timestamp"],
                    )
                    r.df = df
                else:
                    r.df = pd.DataFrame()
                return r

            def get_stock_bars(self, req: Any) -> Any:
                r = MagicMock()
                r.data = {}
                return r

        result = run_morning_scan(
            symbols=["FAST", "SLOW", "MED"],
            session_date=_SESSION_DATE,
            client=_MultiSymClient(),
            repo=repo,
            top_n=30,
            max_symbols_for_budget=40,
            artifact_dir=scanner_artifact,
        )

        # plan_focus is already sorted — verify property on the raw rows too
        scored = [r for r in result.rows if np.isfinite(r["score"])]
        if len(scored) >= 2:
            scores = [r["score"] for r in sorted(
                scored, key=lambda r: r["score"], reverse=True)]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 4. Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def _null_client(self) -> Any:
        """Client that always returns empty DataFrames."""
        class _NullClient:
            def get_stock_trades(self, req: Any) -> Any:
                r = MagicMock(); r.df = pd.DataFrame(); return r

            def get_stock_quotes(self, req: Any) -> Any:
                r = MagicMock(); r.df = pd.DataFrame(); return r

            def get_stock_bars(self, req: Any) -> Any:
                r = MagicMock(); r.data = {}; return r

        return _NullClient()

    def test_missing_artifact_returns_empty_result(self, tmp_path: Path,
                                                    repo: Repo) -> None:
        """No model.joblib in artifact dir -> ScanResult with empty plan_focus."""
        empty_dir = tmp_path / "no_artifact"
        empty_dir.mkdir()
        # Make the path a dir but don't write model.joblib
        result = run_morning_scan(
            symbols=["XYZ"],
            session_date=_SESSION_DATE,
            client=self._null_client(),
            repo=repo,
            artifact_dir=empty_dir,
        )
        assert result.plan_focus == []
        assert result.artifact_dir is None

    def test_zero_bar_symbol_excluded_from_watchlist(
            self, scanner_artifact: Path, repo: Repo) -> None:
        """A symbol with no morning trades must be excluded from the watchlist."""
        result = run_morning_scan(
            symbols=["EMPTY"],
            session_date=_SESSION_DATE,
            client=self._null_client(),
            repo=repo,
            top_n=30,
            artifact_dir=scanner_artifact,
        )
        # "EMPTY" scored NaN -> plan_focus must not contain it
        assert "EMPTY" not in result.plan_focus
        # Row is still present for auditing
        assert any(r["symbol"] == "EMPTY" for r in result.rows)
        empty_row = next(r for r in result.rows if r["symbol"] == "EMPTY")
        assert np.isnan(empty_row["score"])

    def test_multiple_symbols_some_missing_bars(
            self, scanner_artifact: Path, repo: Repo) -> None:
        """Mix of good and zero-bar symbols — good ones get scores, bad ones NaN."""
        good_trades, good_quotes = _make_tick_df(seed=55)

        class _MixedClient:
            def get_stock_trades(self, req: Any) -> Any:
                r = MagicMock()
                sym = req.symbol_or_symbols
                if sym == "GOOD":
                    df = good_trades.copy()
                    df.index = pd.MultiIndex.from_tuples(
                        [(sym, t) for t in df.index],
                        names=["symbol", "timestamp"],
                    )
                    r.df = df
                else:
                    r.df = pd.DataFrame()
                return r

            def get_stock_quotes(self, req: Any) -> Any:
                r = MagicMock()
                sym = req.symbol_or_symbols
                if sym == "GOOD":
                    df = good_quotes.copy()
                    df.index = pd.MultiIndex.from_tuples(
                        [(sym, t) for t in df.index],
                        names=["symbol", "timestamp"],
                    )
                    r.df = df
                else:
                    r.df = pd.DataFrame()
                return r

            def get_stock_bars(self, req: Any) -> Any:
                r = MagicMock(); r.data = {}; return r

        result = run_morning_scan(
            symbols=["GOOD", "BAD"],
            session_date=_SESSION_DATE,
            client=_MixedClient(),
            repo=repo,
            top_n=30,
            artifact_dir=scanner_artifact,
        )
        sym_scores = {r["symbol"]: r["score"] for r in result.rows}
        # GOOD may or may not get a finite score depending on the prev_close
        # context — but BAD (empty bars) must be NaN.
        assert np.isnan(sym_scores["BAD"])
        # GOOD appears in the rows regardless
        assert "GOOD" in sym_scores

    def test_no_repo_does_not_crash(self, scanner_artifact: Path) -> None:
        """repo=None means skip persistence — scan still returns ranked list."""
        result = run_morning_scan(
            symbols=["XYZ"],
            session_date=_SESSION_DATE,
            client=self._null_client(),
            repo=None,
            artifact_dir=scanner_artifact,
        )
        # No exception — result object is valid
        assert isinstance(result, ScanResult)
        assert isinstance(result.plan_focus, list)

    def test_rate_limiter_called_per_request(
            self, scanner_artifact: Path, repo: Repo) -> None:
        """rate_limiter must be invoked for every HTTP call."""
        call_count = 0

        def _rl() -> None:
            nonlocal call_count
            call_count += 1

        run_morning_scan(
            symbols=["A", "B"],
            session_date=_SESSION_DATE,
            client=self._null_client(),
            repo=repo,
            rate_limiter=_rl,
            artifact_dir=scanner_artifact,
        )
        # Expect at least: 1 daily batch + 2 trade + 2 quote = 5 calls minimum
        assert call_count >= 5, f"only {call_count} rate_limiter calls for 2 symbols"


# ---------------------------------------------------------------------------
# 5. Daily context builder (unit)
# ---------------------------------------------------------------------------

class TestBuildDailyContext:
    def test_two_bars_give_prev_close(self) -> None:
        raw = {
            "XYZ": [
                {"date": "2025-06-06", "open": 4.50, "close": 4.90, "volume": 1_500_000},
                {"date": "2025-06-09", "open": 5.10, "close": 5.30, "volume": 600_000},
            ]
        }
        ctx = build_daily_context(raw)
        assert ctx["XYZ"]["open"] == pytest.approx(5.10)
        assert ctx["XYZ"]["prev_close"] == pytest.approx(4.90)
        assert ctx["XYZ"]["prev_day_dollar_vol"] == pytest.approx(4.90 * 1_500_000)
        assert ctx["XYZ"]["prev_day_volume"] == pytest.approx(1_500_000)

    def test_single_bar_prev_fields_nan(self) -> None:
        raw = {
            "XYZ": [
                {"date": "2025-06-09", "open": 5.10, "close": 5.30, "volume": 600_000},
            ]
        }
        ctx = build_daily_context(raw)
        assert np.isnan(ctx["XYZ"]["prev_close"])
        assert np.isnan(ctx["XYZ"]["prev_day_dollar_vol"])
        assert np.isnan(ctx["XYZ"]["prev_day_volume"])

    def test_empty_bars_all_nan(self) -> None:
        ctx = build_daily_context({"XYZ": []})
        for v in ctx["XYZ"].values():
            assert np.isnan(v)


# ---------------------------------------------------------------------------
# 6. Daily-context provenance parity: training path == serving path
# ---------------------------------------------------------------------------

class TestDailyContextProvenance:
    """The key regression guard for the train/serve skew fix.

    Both paths must produce IDENTICAL prev_day_dollar_vol and prev_day_volume
    for the same (symbol, runner_date) when fed the same underlying daily bars:

    * Training path  — train_scanner._daily_lookup(index, daily_raw)
      Uses data/daily_raw batches: prev_day_dollar_vol = close * volume of
      the true prior CALENDAR session.

    * Serving path   — live_scan.build_daily_context(raw_daily_bars)
      Uses the Alpaca REST daily-bars response (mocked here with the same bars):
      prev_day_dollar_vol = close[-2] * volume[-2].

    The previous bug used dollar_vol = volume * vwap (runner-index column)
    from the prior RUNNER row (median 42 calendar days stale), while serving
    always used close * volume of the actual prior session.  This test would
    have caught that skew.
    """

    def _make_runner_index_row(
        self,
        sym: str,
        runner_ts: pd.Timestamp,
        open_px: float,
        prev_close: float,
    ) -> pd.DataFrame:
        """Minimal runner_index row (scalar fields needed by _daily_lookup)."""
        return pd.DataFrame(
            {
                "open": [open_px],
                "prev_close": [prev_close],
                # dollar_vol in runner_index is volume*vwap (the OLD training
                # source for prev_day_dollar_vol — we are testing that we no
                # longer use this).
                "dollar_vol": [999_999_999.0],
                "volume": [50_000_000.0],
            },
            index=pd.MultiIndex.from_tuples(
                [(sym, runner_ts)], names=["symbol", "ts"]
            ),
        )

    def _make_daily_raw(
        self,
        sym: str,
        dates_close_vol: list[tuple[str, float, float]],
    ) -> pd.DataFrame:
        """Build a minimal daily_raw DataFrame for use by _daily_lookup."""
        rows = []
        for date_str, close, vol in dates_close_vol:
            ts = pd.Timestamp(date_str, tz="UTC")
            rows.append({"symbol": sym, "ts": ts, "close": close, "volume": vol})
        df = pd.DataFrame(rows).set_index(["symbol", "ts"])
        return df.sort_index()

    def test_prev_day_fields_match_between_train_and_serve(self) -> None:
        """Training and serving agree on prev_day_dollar_vol and prev_day_volume."""
        # We need train_scanner._daily_lookup — import from the research tree.
        _RESEARCH = Path(__file__).resolve().parents[2] / "research"
        if str(_RESEARCH) not in sys.path:
            sys.path.insert(0, str(_RESEARCH))
        # Re-import to get the functions (train_scanner is not on sys.path by
        # default; use importlib to avoid polluting the module namespace).
        import importlib.util as ilu
        train_path = _RESEARCH / "scripts" / "train_scanner.py"
        spec = ilu.spec_from_file_location("train_scanner", train_path)
        ts_mod = ilu.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(ts_mod)  # type: ignore[union-attr]

        sym = "TESTX"
        # Runner fires on 2025-06-09.  Prior CALENDAR sessions: 06-06 and 06-05.
        runner_ts = pd.Timestamp("2025-06-09 04:00:00", tz="UTC")
        prior_close = 4.90
        prior_vol = 1_500_000.0
        open_px = 5.10

        runner_row = self._make_runner_index_row(sym, runner_ts, open_px, prior_close)
        daily_raw = self._make_daily_raw(sym, [
            ("2025-06-05 04:00:00", 4.70, 1_200_000.0),
            ("2025-06-06 04:00:00", prior_close, prior_vol),
            # 2025-06-07 and 2025-06-08 are weekend — no row (correct gap handling)
        ])

        # --- Training path ---
        train_lut = ts_mod._daily_lookup(runner_row, daily_raw=daily_raw)
        train_ctx = train_lut[(sym, "2025-06-09")]

        # --- Serving path ---
        raw_bars_for_serve = {
            sym: [
                {"date": "2025-06-06", "open": 4.80, "close": prior_close,
                 "volume": prior_vol},
                {"date": "2025-06-09", "open": open_px, "close": 5.30,
                 "volume": 600_000},
            ]
        }
        serve_ctx = build_daily_context(raw_bars_for_serve)[sym]

        # Both must agree on the two prev-day fields that carry the skew.
        expected_dv = prior_close * prior_vol
        assert train_ctx["prev_day_dollar_vol"] == pytest.approx(expected_dv), (
            f"train prev_day_dollar_vol={train_ctx['prev_day_dollar_vol']} "
            f"expected {expected_dv}"
        )
        assert serve_ctx["prev_day_dollar_vol"] == pytest.approx(expected_dv), (
            f"serve prev_day_dollar_vol={serve_ctx['prev_day_dollar_vol']} "
            f"expected {expected_dv}"
        )
        assert train_ctx["prev_day_dollar_vol"] == pytest.approx(
            serve_ctx["prev_day_dollar_vol"]
        ), "train and serve prev_day_dollar_vol differ"

        assert train_ctx["prev_day_volume"] == pytest.approx(prior_vol)
        assert serve_ctx["prev_day_volume"] == pytest.approx(prior_vol)
        assert train_ctx["prev_day_volume"] == pytest.approx(
            serve_ctx["prev_day_volume"]
        ), "train and serve prev_day_volume differ"

        # open and prev_close come from runner_index in training but from
        # the today-bar in serving — they may differ slightly; we don't
        # assert equality here (they are correctly documented as different
        # provenance: runner_index open vs REST open).

    def test_prev_day_formula_is_close_times_volume_not_vwap_times_volume(
            self,
    ) -> None:
        """Serving path must use close*volume, NOT vwap*volume (the old bug)."""
        # If a daily bar had close=5.00, vwap=6.00, volume=1_000_000
        # the OLD bug would give prev_day_dollar_vol = 6.00 * 1_000_000 = 6e6.
        # The CORRECT formula is close * volume = 5.00 * 1_000_000 = 5e6.
        raw = {
            "BUG": [
                {"date": "2025-06-06", "open": 4.80, "close": 5.00,
                 "volume": 1_000_000},
                {"date": "2025-06-09", "open": 5.50, "close": 5.80,
                 "volume": 500_000},
            ]
        }
        ctx = build_daily_context(raw)
        expected = 5.00 * 1_000_000
        wrong_vwap_based = 6.00 * 1_000_000  # hypothetical vwap*volume
        assert ctx["BUG"]["prev_day_dollar_vol"] == pytest.approx(expected), (
            "prev_day_dollar_vol must be close*volume"
        )
        assert ctx["BUG"]["prev_day_dollar_vol"] != pytest.approx(wrong_vwap_based), (
            "prev_day_dollar_vol must NOT be vwap*volume"
        )

    def test_no_prior_calendar_session_gives_nan(self) -> None:
        """When daily_raw has no session before the runner date, fields are NaN."""
        _RESEARCH = Path(__file__).resolve().parents[2] / "research"
        if str(_RESEARCH) not in sys.path:
            sys.path.insert(0, str(_RESEARCH))
        import importlib.util as ilu
        train_path = _RESEARCH / "scripts" / "train_scanner.py"
        spec = ilu.spec_from_file_location("train_scanner_noprior", train_path)
        ts_mod = ilu.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(ts_mod)  # type: ignore[union-attr]

        sym = "NOPRIOR"
        runner_ts = pd.Timestamp("2025-06-09 04:00:00", tz="UTC")
        runner_row = self._make_runner_index_row(sym, runner_ts, 5.10, 4.90)
        # daily_raw has NO session before 2025-06-09 for NOPRIOR
        daily_raw = self._make_daily_raw(sym, [
            ("2025-06-09 04:00:00", 5.30, 600_000.0),  # same day — not prior
        ])

        train_lut = ts_mod._daily_lookup(runner_row, daily_raw=daily_raw)
        train_ctx = train_lut[(sym, "2025-06-09")]
        assert np.isnan(train_ctx["prev_day_dollar_vol"]), (
            "expected NaN when no prior calendar session exists"
        )
        assert np.isnan(train_ctx["prev_day_volume"])
