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
from market_trader.execution.broker import Order, OrderSide, OrderStatus
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
        settings=PAPER,
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
