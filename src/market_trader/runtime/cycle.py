"""One paper-trading cycle, end to end.

``run_paper_cycle`` is the pure, injectable core (store + prices + broker + llm) so
it is fully tested offline. ``run_dry_paper_cycle`` wires synthetic data + a local
paper broker (no network). ``run_live_paper_cycle`` wires Alpaca market data +
the Alpaca paper broker + the Claude briefing.

PAPER ONLY. Sizing is capped by ``capital_ceiling`` and the risk limits; live
order routing stays gated by ``Settings.assert_live_allowed()`` in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from market_trader.collectors import IngestionGateway, PriceCollector
from market_trader.config import Settings
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.core.time import utcnow
from market_trader.execution.broker import Broker, Order
from market_trader.execution.engine import ExecutionEngine
from market_trader.execution.paper_broker import PaperBroker
from market_trader.features import FeatureStore, default_features
from market_trader.portfolio import RiskLimits, apply_risk_limits, composite_score, equal_weights
from market_trader.reasoning import LLMProvider, build_briefing_context, generate_llm_brief
from market_trader.storage import InMemoryBitemporalStore
from market_trader.storage.bitemporal import BitemporalStore

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]


@dataclass
class CycleResult:
    as_of: datetime
    scores: dict[str, float]
    target_weights: dict[str, float]
    orders: list[Order]
    brief: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


def _limits_from_settings(settings: Settings) -> RiskLimits:
    return RiskLimits(
        max_position_weight=settings.max_position_weight,
        max_gross_exposure=settings.max_gross_exposure,
        max_net_exposure=settings.max_net_exposure,
    )


def run_paper_cycle(
    store: BitemporalStore,
    *,
    as_of: datetime,
    symbols: list[str],
    prices: dict[str, float],
    broker: Broker,
    settings: Settings,
    limits: RiskLimits | None = None,
    llm: LLMProvider | None = None,
    feature_store: FeatureStore | None = None,
    top_quantile: float = 0.3,
) -> CycleResult:
    """Score the universe, form risk-managed target weights, execute on the broker."""
    limits = limits or _limits_from_settings(settings)
    fs = feature_store or FeatureStore(store, default_features())
    matrix = fs.compute_matrix(as_of, symbols)

    scores = (
        composite_score(matrix, equal_weights(matrix.columns))
        if not matrix.empty
        else pd.Series(dtype=float)
    )
    ranked = scores.dropna().sort_values(ascending=False)

    target: dict[str, float] = {}
    if not ranked.empty:
        k = max(1, int(len(ranked) * top_quantile))
        winners = [str(s) for s in ranked.head(k).index]
        raw = {s: 1.0 / len(winners) for s in winners}
        target = apply_risk_limits(raw, limits)

    engine = ExecutionEngine(broker, settings=settings, limits=limits)
    orders = engine.rebalance(target, prices, as_of=as_of) if target else []

    brief: str | None = None
    if llm is not None:
        context = build_briefing_context(store, as_of)
        brief = generate_llm_brief(context, llm)

    return CycleResult(
        as_of=as_of,
        scores={str(k): float(v) for k, v in ranked.items()},
        target_weights=target,
        orders=orders,
        brief=brief,
    )


def run_dry_paper_cycle(settings: Settings, *, n_syms: int = 8, n_days: int = 120) -> CycleResult:
    """Fully local cycle: synthetic prices + an in-memory paper broker. No network."""
    symbols = [f"S{i}" for i in range(n_syms)]
    observations = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=n_days, seed=7
    )
    store = InMemoryBitemporalStore()
    store.add_many(observations)
    as_of = max(o.event_time for o in observations)
    prices = {
        o.entity_id: float(o.value["close"]) for o in store.as_of(as_of, dataset=PRICE_DATASET)
    }
    broker = PaperBroker(prices, starting_cash=100_000.0)
    return run_paper_cycle(
        store, as_of=as_of, symbols=symbols, prices=prices, broker=broker, settings=settings
    )


def run_live_paper_cycle(
    settings: Settings, *, lookback_days: int = 150, watchlist: list[str] | None = None
) -> CycleResult:
    """Live paper cycle: Alpaca data + Alpaca paper broker + (optional) Claude brief."""
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")

    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.execution.alpaca import AlpacaBroker
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

    watchlist = watchlist or DEFAULT_WATCHLIST
    store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
    store.create_schema()  # idempotent

    end = date.today()
    start = end - timedelta(days=lookback_days)
    data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
    records = data.fetch_daily_bars(watchlist, start=start, end=end)
    IngestionGateway(store).ingest(PriceCollector().normalize(records))

    as_of = utcnow()
    prices = {
        o.entity_id: float(o.value["close"])
        for o in store.as_of(as_of, dataset=PRICE_DATASET)
        if o.entity_id in watchlist
    }

    broker = AlpacaBroker(
        settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
    )
    llm: LLMProvider | None = None
    if settings.anthropic_api_key:
        from market_trader.reasoning import anthropic_provider_from_settings

        llm = anthropic_provider_from_settings(settings)

    return run_paper_cycle(
        store,
        as_of=as_of,
        symbols=watchlist,
        prices=prices,
        broker=broker,
        settings=settings,
        llm=llm,
    )
