"""One paper cycle, end to end, fully offline.

``run_paper_cycle`` is exercised with a synthetic point-in-time store, the
in-memory ``PaperBroker``, and the ``MockLLMProvider`` — no network, no keys, no
capital at risk — proving score -> risk-limits -> paper execution -> brief wires up.
``run_dry_paper_cycle`` is the same path behind the CLI's ``cycle --dry-run``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_trader.config import Settings
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.execution.broker import Order, OrderSide, OrderStatus, OrderType
from market_trader.execution.paper_broker import PaperBroker
from market_trader.reasoning import MockLLMProvider
from market_trader.runtime import run_dry_paper_cycle, run_live_paper_cycle, run_paper_cycle
from market_trader.storage import InMemoryBitemporalStore

PAPER = Settings(execution_mode="paper", capital_ceiling=1000.0, max_position_weight=0.10)


def _seeded_store(symbols: list[str], n_days: int = 120):
    obs = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=n_days, seed=7
    )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    as_of = max(o.event_time for o in obs)
    prices = {
        o.entity_id: float(o.value["close"]) for o in store.as_of(as_of, dataset=PRICE_DATASET)
    }
    return store, as_of, prices


def test_run_paper_cycle_logs_predictions_when_enabled() -> None:
    from market_trader.feedback.prediction_log import load_predictions

    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)

    run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        prediction_log=True,
        model_version="composite",
    )
    assert load_predictions(store, as_of, model_version="composite")  # logged for later grading


def test_vol_target_weights_scales_a_high_vol_book_to_target() -> None:
    import numpy as np

    from market_trader.portfolio.construction import ledoit_wolf_cov
    from market_trader.runtime.cycle import _vol_target_weights

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=120)
    syms = ["A", "B", "C", "D"]
    # ~3%/day -> ~48% annualised, far above a 10% target, so the book must scale DOWN
    rets = pd.DataFrame(rng.normal(0.0, 0.03, (len(dates), 4)), index=dates, columns=syms)
    base = dict.fromkeys(syms, 0.25)  # equal-weight, gross 1.0

    out = _vol_target_weights(base, rets, target_vol=0.10, max_gross=1.0)
    assert 0.0 < sum(out.values()) < 1.0  # exposure cut into cash

    cov = ledoit_wolf_cov(rets[list(out)])
    wv = pd.Series(out).reindex(cov.columns).fillna(0.0).to_numpy(dtype=float)
    ann_vol = float(np.sqrt(wv @ cov.to_numpy(dtype=float) @ wv) * np.sqrt(252))
    assert abs(ann_vol - 0.10) < 0.01  # book sized to the vol budget


def test_vol_target_weights_no_history_is_a_passthrough() -> None:
    from market_trader.runtime.cycle import _vol_target_weights

    base = {"A": 0.5, "B": 0.5}
    thin = pd.DataFrame({"A": [0.01, -0.02], "B": [0.0, 0.01]})  # 2 rows < 20 -> too little
    assert _vol_target_weights(base, thin, target_vol=0.10, max_gross=1.0) == base


def test_run_paper_cycle_vol_target_mode_stays_inside_the_gross_cap() -> None:
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols, n_days=150)  # enough history for covariance
    broker = PaperBroker(prices, starting_cash=100_000.0)

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        risk_weighting="vol_target",
    )
    gross = sum(abs(w) for w in result.target_weights.values())
    assert result.target_weights and gross <= 1.0 + 1e-9  # a valid, governed (capped) book


def test_run_paper_cycle_size_book_mode_holds_whole_universe_governed() -> None:
    # risk_weighting="size_book" with tilt_strength=0 is the governed equal-weight chassis:
    # it holds the *whole* scored universe (breadth), not a top-N subset, stays inside the
    # gross cap, and weights equally. This is the book we arm on the paper account first.
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols, n_days=150)  # enough history for covariance
    broker = PaperBroker(prices, starting_cash=100_000.0)

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        risk_weighting="size_book",  # tilt_strength defaults to 0 -> governed 1/N
    )

    assert set(result.target_weights) == set(symbols)  # the whole universe, not a top-N subset
    gross = sum(abs(w) for w in result.target_weights.values())
    assert 0.0 < gross <= 1.0 + 1e-9  # governed and gross-capped
    weights = list(result.target_weights.values())
    assert max(weights) - min(weights) < 1e-9  # equal weight (nothing to tilt on at strength 0)


def test_stop_loss_flattens_a_losing_holding_even_when_top_ranked() -> None:
    # A held name that's deep underwater is cut even though the signal loves it —
    # the absolute loss floor overrides the (relative) rank.
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker({**prices, "S3": prices["S3"] * 2}, starting_cash=100_000.0)
    broker.submit_order(Order("seed", "S3", OrderSide.BUY, 10.0))  # entry at 2x the mark

    def score(matrix: pd.DataFrame, _at) -> pd.Series:  # rank the loser #1
        return pd.Series([10.0 if s == "S3" else 1.0 for s in matrix.index], index=matrix.index)

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        score_fn=score,
        top_quantile=0.5,
        stop_loss_pct=0.10,
    )
    assert "S3" not in result.target_weights  # stopped out, not held
    assert any(o.symbol == "S3" and o.side == OrderSide.SELL for o in result.orders)


def test_trailing_stop_cuts_a_name_that_rolled_over_from_its_high() -> None:
    # Exit discipline reacting to performance: a name that ramped up then broke down from
    # its trailing high is cut; a name still near its high is kept.
    from market_trader.collectors.prices import PriceBar, PriceCollector
    from market_trader.execution.broker import Position
    from market_trader.runtime.cycle import _trailing_stops

    days = pd.bdate_range("2024-01-01", periods=70)
    bars = []
    for i, d in enumerate(days):
        loser = 50.0 + i if i < 50 else 100.0 - (i - 50) * 3  # peaks ~99, then falls to ~43
        winner = 50.0 + i * 0.5  # steadily near its high
        bars.append(PriceBar(date=d.date(), symbol="LOSER", close=loser))
        bars.append(PriceBar(date=d.date(), symbol="WINNER", close=winner))
    obs = PriceCollector().normalize(bars)
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    as_of = max(o.knowledge_time for o in obs)
    positions = [Position("LOSER", 10.0, 60.0), Position("WINNER", 10.0, 50.0)]
    prices = {"LOSER": 43.0, "WINNER": 84.5}

    stopped = _trailing_stops(store, as_of, positions, prices, 0.15, window=60)
    assert "LOSER" in stopped  # ~56% below its trailing high -> cut
    assert "WINNER" not in stopped  # still at its high -> kept
    assert _trailing_stops(store, as_of, positions, prices, 0.0) == set()  # 0 disables


def test_reserved_symbols_are_neither_selected_nor_flattened() -> None:
    # A sleeve-owned name must be left completely alone by the daily book: not
    # bought (even if top-ranked) and not flattened (even though it's held).
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)
    broker.submit_order(Order("seed", "S3", OrderSide.BUY, 10.0))  # the sleeve owns S3

    def score(matrix: pd.DataFrame, _at) -> pd.Series:  # rank S3 #1 to tempt the daily book
        return pd.Series([10.0 if s == "S3" else 1.0 for s in matrix.index], index=matrix.index)

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        score_fn=score,
        top_quantile=0.5,
        reserved_symbols=frozenset({"S3"}),
    )
    assert "S3" not in result.target_weights  # reserved -> not selected
    assert all(o.symbol != "S3" for o in result.orders)  # reserved -> not flattened/traded


def test_run_paper_cycle_uses_injected_score_fn() -> None:
    # The pluggable scorer (the seam the forecaster plugs into) must drive
    # selection: a scorer that ranks S3 top makes S3 a winner regardless of features.
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)

    def score(matrix: pd.DataFrame, _at) -> pd.Series:
        return pd.Series([10.0 if s == "S3" else 1.0 for s in matrix.index], index=matrix.index)

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        score_fn=score,
        top_quantile=0.3,
    )
    assert "S3" in result.target_weights


def test_run_paper_cycle_caps_book_size_for_diversification() -> None:
    # A broad universe must produce a diversified book capped at max_positions,
    # not 2-3 names — this is what makes universe breadth useful.
    symbols = [f"S{i}" for i in range(30)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker(prices, starting_cash=1_000_000.0)

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        # A realistic ceiling so each capped name clears one whole share (the default
        # whole-share rounding would floor a $1k-book's tiny per-name slice to zero).
        settings=Settings(execution_mode="paper", capital_ceiling=1_000_000.0),
        top_quantile=0.9,  # 0.9 * 30 = 27 ranked...
        max_positions=5,  # ...but the cap holds the book to 5
    )
    assert len(result.target_weights) == 5
    assert len(result.orders) == 5 and all(o.side == OrderSide.BUY for o in result.orders)


def test_run_paper_cycle_scores_targets_and_fills() -> None:
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)
    llm = MockLLMProvider("BRIEF OK")

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        llm=llm,
    )

    assert len(result.scores) == len(symbols)  # the whole universe got ranked
    assert result.target_weights  # something cleared the top-quantile threshold
    assert all(w <= 0.10 + 1e-9 for w in result.target_weights.values())  # per-name cap held
    assert result.orders and all(o.status == OrderStatus.FILLED for o in result.orders)
    assert broker.get_positions()  # the paper broker actually holds the names
    assert result.brief == "BRIEF OK" and llm.calls  # brief narrated from the PIT context


def test_run_paper_cycle_without_llm_has_no_brief() -> None:
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=PaperBroker(prices),
        settings=PAPER,
    )
    assert result.brief is None  # no provider => deterministic, LLM-free


def test_run_dry_paper_cycle_is_self_contained() -> None:
    result = run_dry_paper_cycle(PAPER)  # no network, no keys
    assert result.target_weights
    assert result.orders
    assert result.brief is None  # the dry path wires no LLM


def test_run_paper_cycle_flattens_holdings_that_drop_out() -> None:
    # A name held from a prior cycle but no longer selected must be sold, or it
    # would linger on the persistent account (rebalance only acts on the target).
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    prices = {**prices, "STALE": 50.0}  # not in the scored universe
    broker = PaperBroker(prices, starting_cash=100_000.0)
    broker.submit_order(Order("seed", "STALE", OrderSide.BUY, 10.0))  # already held
    assert "STALE" in {p.symbol for p in broker.get_positions()}

    result = run_paper_cycle(
        store, as_of=as_of, symbols=symbols, prices=prices, broker=broker, settings=PAPER
    )

    assert "STALE" not in {p.symbol for p in broker.get_positions()}  # flattened to zero
    assert any(o.symbol == "STALE" and o.side == OrderSide.SELL for o in result.orders)
    assert "STALE" not in result.target_weights  # not part of the desired portfolio


def test_run_live_paper_cycle_requires_keys() -> None:
    no_keys = Settings(execution_mode="paper", alpaca_key_id=None, alpaca_secret_key=None)
    with pytest.raises(RuntimeError):  # fails fast, before any network call
        run_live_paper_cycle(no_keys)


def test_run_live_paper_cycle_refuses_live_endpoint_unless_armed() -> None:
    # Keys present + MT_ALPACA_PAPER=false asks for the LIVE endpoint, but live
    # trading isn't armed (execution_mode=paper) => fail closed before any I/O.
    wants_live_endpoint = Settings(
        execution_mode="paper",
        live_trading_enabled=False,
        alpaca_key_id="k",
        alpaca_secret_key="s",
        alpaca_paper=False,
    )
    with pytest.raises(RuntimeError):
        run_live_paper_cycle(wants_live_endpoint)


def test_run_paper_cycle_cancels_stale_open_orders_before_rebalancing() -> None:
    # A still-open order from a prior run is cancelled before the new book is
    # placed (so the broker can't reject the fresh orders as a wash trade against
    # it) — but a sleeve-reserved name's order is left untouched.
    symbols = [f"S{i}" for i in range(8)]
    store, as_of, prices = _seeded_store(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)
    # Resting (non-marketable) limit orders so they sit open rather than fill.
    broker.submit_order(
        Order("stale", "S1", OrderSide.BUY, 1.0, OrderType.LIMIT, limit_price=prices["S1"] * 0.5)
    )
    broker.submit_order(
        Order("sleeve", "S3", OrderSide.BUY, 1.0, OrderType.LIMIT, limit_price=prices["S3"] * 0.5)
    )
    assert {o.client_order_id for o in broker.get_open_orders()} == {"stale", "sleeve"}

    run_paper_cycle(
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=PAPER,
        reserved_symbols=frozenset({"S3"}),
        cancel_stale_orders=True,
    )

    open_ids = {o.client_order_id for o in broker.get_open_orders()}
    assert "stale" not in open_ids  # ordinary stale order cancelled before rebalancing
    assert "sleeve" in open_ids  # reserved (sleeve) name's order left alone
