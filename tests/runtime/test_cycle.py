"""One paper cycle, end to end, fully offline.

``run_paper_cycle`` is exercised with a synthetic point-in-time store, the
in-memory ``PaperBroker``, and the ``MockLLMProvider`` — no network, no keys, no
capital at risk — proving score -> risk-limits -> paper execution -> brief wires up.
``run_dry_paper_cycle`` is the same path behind the CLI's ``cycle --dry-run``.
"""

from __future__ import annotations

from datetime import date

import pytest

from market_trader.config import Settings
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.execution.broker import OrderStatus
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


def test_run_live_paper_cycle_requires_keys() -> None:
    no_keys = Settings(execution_mode="paper", alpaca_key_id=None, alpaca_secret_key=None)
    with pytest.raises(RuntimeError):  # fails fast, before any network call
        run_live_paper_cycle(no_keys)
