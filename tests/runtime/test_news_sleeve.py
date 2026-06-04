"""Event-driven news sleeve: detection, the open/dedup/close lifecycle, and the
loop control flow — all offline (synthetic store, paper broker, a fake feed)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from market_trader.collectors.gdelt import NEWS_DATASET
from market_trader.config import Settings
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.core.time import utcnow
from market_trader.execution.broker import OrderSide
from market_trader.execution.paper_broker import PaperBroker
from market_trader.runtime.news_sleeve import (
    NewsSleeveResult,
    active_sleeve_positions,
    detect_news_events,
    run_news_sleeve_cycle,
    run_news_sleeve_loop,
)
from market_trader.storage import InMemoryBitemporalStore


def _settings() -> Settings:
    return Settings(
        execution_mode="paper",
        capital_ceiling=100_000.0,
        news_sleeve_enabled=True,
        news_sleeve_budget=0.10,
        news_sleeve_max_names=5,
        news_sleeve_min_confidence=0.5,
        news_sleeve_hold_days=5,
        news_sleeve_cooldown_days=3,
        stop_loss_pct=0.0,
        max_position_weight=0.10,
    )


def _store_with_prices(symbols: list[str], n_days: int = 120):
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


def _news(symbol: str, at, n: int, tone: float) -> list[Observation]:
    return [
        Observation(
            source="gdelt",
            dataset=NEWS_DATASET,
            entity_type="equity",
            entity_id=symbol,
            ref=f"{symbol}-url-{i}",
            event_time=at,
            knowledge_time=at,
            value={"title": f"{symbol} headline {i}", "url": f"http://{symbol}/{i}", "tone": tone},
        )
        for i in range(n)
    ]


class _FakeFeed:
    def __init__(self, obs: list[Observation]) -> None:
        self._obs = obs

    def fetch_recent(self, symbols: Sequence[str], *, lookback_minutes: int) -> list[Observation]:
        return list(self._obs)


def test_detect_flags_a_surge_with_direction_not_the_quiet_name() -> None:
    symbols = ["HOT", "COLD"]
    store = InMemoryBitemporalStore()
    as_of = utcnow()
    store.add_many(_news("HOT", as_of, n=5, tone=3.0))  # recent surge, positive
    store.add_many(_news("COLD", as_of - timedelta(days=10), n=1, tone=3.0))  # old, sparse

    events = detect_news_events(
        store, as_of, symbols, count_surge=3.0, tone_min=1.5, baseline_days=14
    )
    by_sym = {e.symbol: e for e in events}

    assert "HOT" in by_sym and "COLD" not in by_sym
    assert by_sym["HOT"].direction == 1 and by_sym["HOT"].confidence > 0


def test_sleeve_opens_on_material_news_then_dedups() -> None:
    symbols = [f"S{i}" for i in range(6)]
    store, as_of, prices = _store_with_prices(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)
    feed = _FakeFeed(_news("S1", as_of, n=4, tone=4.0))

    res = run_news_sleeve_cycle(
        _settings(), feed=feed, store=store, broker=broker, as_of=as_of, watchlist=symbols
    )
    assert res.opened == ["S1"]
    assert any(o.symbol == "S1" and o.side == OrderSide.BUY for o in res.orders)
    assert "S1" in active_sleeve_positions(store, as_of)

    # Same story, still holding -> no second open (no churn).
    res2 = run_news_sleeve_cycle(
        _settings(), feed=feed, store=store, broker=broker, as_of=as_of, watchlist=symbols
    )
    assert res2.opened == [] and res2.orders == []


def test_sleeve_closes_after_the_hold_window() -> None:
    symbols = [f"S{i}" for i in range(6)]
    store, as_of, prices = _store_with_prices(symbols)
    broker = PaperBroker(prices, starting_cash=100_000.0)

    run_news_sleeve_cycle(
        _settings(),
        feed=_FakeFeed(_news("S1", as_of, n=4, tone=4.0)),
        store=store,
        broker=broker,
        as_of=as_of,
        watchlist=symbols,
    )
    later = as_of + timedelta(days=6)  # past hold_days=5
    res = run_news_sleeve_cycle(
        _settings(), feed=_FakeFeed([]), store=store, broker=broker, as_of=later, watchlist=symbols
    )
    assert "S1" in res.closed
    assert any(o.symbol == "S1" and o.side == OrderSide.SELL for o in res.orders)
    assert "S1" not in active_sleeve_positions(store, later)


def test_loop_acts_while_open_and_honors_max_iterations() -> None:
    calls = {"n": 0}

    def run_cycle() -> NewsSleeveResult:
        calls["n"] += 1
        return NewsSleeveResult(as_of=utcnow(), opened=[], closed=[], orders=[])

    ran = run_news_sleeve_loop(
        _settings(),
        is_market_open=lambda: True,
        run_cycle=run_cycle,
        sleep_fn=lambda _s: None,
        max_iterations=3,
    )
    assert ran == 3 and calls["n"] == 3


def test_loop_idles_when_market_closed_and_survives_failures() -> None:
    calls = {"n": 0}

    def run_cycle() -> NewsSleeveResult:
        calls["n"] += 1
        return NewsSleeveResult(as_of=utcnow(), opened=[], closed=[], orders=[])

    run_news_sleeve_loop(
        _settings(),
        is_market_open=lambda: False,
        run_cycle=run_cycle,
        sleep_fn=lambda _s: None,
        max_iterations=3,
    )
    assert calls["n"] == 0  # never acts while closed

    def boom() -> NewsSleeveResult:
        raise RuntimeError("transient blip")

    ran = run_news_sleeve_loop(
        _settings(),
        is_market_open=lambda: True,
        run_cycle=boom,
        sleep_fn=lambda _s: None,
        max_iterations=3,
    )
    assert ran == 3  # a failing pass never kills the loop
