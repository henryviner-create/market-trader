"""Insider-cluster event sleeve (PAPER) — the reactive overlay that actually trades.

The event study validated ``insider_cluster_buy`` drift (CAR ~+3%, t~4). ``insider_events``
is the gated *decision* (which fresh clusters to open); this is the *execution*: one pass
that closes expired/stopped sleeve positions, then opens a small, time-boxed position on each
fresh, gated cluster — through :class:`ExecutionEngine`, so it inherits every risk rail
(kill-switch, drawdown breaker, caps, whole-share, rate limit).

It coexists with the daily book exactly like the news sleeve: it manages **only its own
names** via a direct ``rebalance`` (which only touches symbols in the target it's given), on a
separate capital budget, and the daily cycle is told to *reserve* sleeve-owned names so it
never flattens them. Every open is logged as a gradeable ``insider_sleeve`` prediction, so the
sleeve is held to the same evidentiary bar as everything else. And it only trades at all if
``insider_cluster_buy`` cleared the event-study gate this run — if the edge decays out of
significance, the sleeve goes quiet on its own.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from market_trader.config import Settings
from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import to_utc_lenient, utcnow
from market_trader.execution.broker import Broker, Order
from market_trader.execution.engine import ExecutionEngine
from market_trader.feedback.prediction_log import PredictionRecord, log_predictions
from market_trader.memory.event_study import EventOutcomeDistribution
from market_trader.memory.study_runner import significant_event_types
from market_trader.observability import get_logger
from market_trader.portfolio.risk import RiskLimits
from market_trader.runtime.insider_events import insider_cluster_entries
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe.liquid import resolve_universe

_log = get_logger("insider_sleeve")

SLEEVE_DATASET = "sleeve.insider_position"
SLEEVE_MODEL = "insider_sleeve"


@dataclass(frozen=True)
class InsiderSleevePosition:
    symbol: str
    entry_price: float
    exit_by: datetime
    n_buys: int


@dataclass
class InsiderSleeveResult:
    as_of: datetime
    opened: list[str]
    closed: list[str]
    orders: list[Order]


def _records(store: BitemporalStore, as_of: datetime) -> list[Observation]:
    return store.as_of(as_of, dataset=SLEEVE_DATASET)


def _latest_by_symbol(obs: Iterable[Observation]) -> dict[str, Observation]:
    out: dict[str, Observation] = {}
    for o in obs:
        prev = out.get(o.entity_id)
        if prev is None or o.event_time >= prev.event_time:
            out[o.entity_id] = o  # newest state per name wins
    return out


def active_insider_positions(
    store: BitemporalStore, as_of: datetime
) -> dict[str, InsiderSleevePosition]:
    """Symbols the insider sleeve currently owns (latest persisted state is ``open``)."""
    out: dict[str, InsiderSleevePosition] = {}
    for sym, o in _latest_by_symbol(_records(store, as_of)).items():
        if o.value.get("status") == "open":
            out[sym] = InsiderSleevePosition(
                symbol=sym,
                entry_price=float(o.value.get("entry_price", 0.0)),
                exit_by=to_utc_lenient(datetime.fromisoformat(str(o.value["exit_by"]))),
                n_buys=int(o.value.get("n_buys", 0)),
            )
    return out


def _persist(store: BitemporalStore, obs: Observation) -> None:
    store.upsert_many([with_deterministic_id(obs)])


def record_open(
    store: BitemporalStore,
    symbol: str,
    as_of: datetime,
    *,
    entry_price: float,
    exit_by: datetime,
    n_buys: int,
) -> None:
    _persist(
        store,
        Observation(
            source="sleeve",
            dataset=SLEEVE_DATASET,
            entity_type="equity",
            entity_id=symbol,
            ref=f"open:{as_of.date()}",
            event_time=as_of,
            knowledge_time=as_of,
            value={
                "status": "open",
                "entry_price": float(entry_price),
                "exit_by": exit_by.isoformat(),
                "n_buys": int(n_buys),
            },
        ),
    )


def record_close(store: BitemporalStore, symbol: str, as_of: datetime, *, reason: str) -> None:
    _persist(
        store,
        Observation(
            source="sleeve",
            dataset=SLEEVE_DATASET,
            entity_type="equity",
            entity_id=symbol,
            ref=f"close:{reason}:{as_of.date()}",
            event_time=as_of,
            knowledge_time=as_of,
            value={"status": "closed", "reason": reason},
        ),
    )


def _latest_prices(store: BitemporalStore, as_of: datetime, symbols: set[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for o in store.as_of(as_of, dataset=PRICE_DATASET):
        if o.entity_id in symbols and o.value.get("close"):
            prices[o.entity_id] = float(o.value["close"])  # last wins ~ most recent
    return prices


def run_insider_sleeve_cycle(
    settings: Settings,
    *,
    store: BitemporalStore,
    broker: Broker,
    as_of: datetime | None = None,
    gate: dict[str, EventOutcomeDistribution] | None = None,
) -> InsiderSleeveResult:
    """One sleeve pass: close expired/stopped, open fresh gated clusters, execute on the broker.

    ``gate`` is the event-study significance map (computed from the store if not injected);
    the sleeve trades only the event types it contains. Fully injectable (store/broker/gate)
    so the whole open/close lifecycle is tested offline with a PaperBroker.
    """
    as_of = as_of or utcnow()
    universe = set(resolve_universe(settings.universe))
    if gate is None:
        gate = significant_event_types(store, post_days=settings.insider_sleeve_hold_days)

    active = active_insider_positions(store, as_of)
    prices = _latest_prices(store, as_of, universe | set(active))
    target: dict[str, float] = {}
    opened: list[str] = []
    closed: list[str] = []

    # Close positions past their time-box, or stopped out on the hard floor.
    for sym, pos in active.items():
        px = prices.get(sym)
        stopped = bool(
            px
            and pos.entry_price > 0
            and settings.stop_loss_pct > 0
            and px / pos.entry_price - 1.0 <= -settings.stop_loss_pct
        )
        if as_of >= pos.exit_by or stopped:
            target[sym] = 0.0
            record_close(store, sym, as_of, reason="stop" if stopped else "horizon")
            closed.append(sym)

    # Open fresh, gated clusters into the free slots (long-only, time-boxed, deduped).
    live = {s for s in active if s not in closed}
    slots = max(0, settings.insider_sleeve_max_names - len(live))
    if slots:
        per_name = settings.insider_sleeve_budget / max(settings.insider_sleeve_max_names, 1)
        for e in insider_cluster_entries(
            store,
            as_of,
            gate=gate,
            hold_days=settings.insider_sleeve_hold_days,
            max_names=slots,
            held=frozenset(live),
        ):
            if not prices.get(e.symbol):
                continue  # no tradable price -> skip (don't open blind)
            target[e.symbol] = per_name
            record_open(
                store,
                e.symbol,
                as_of,
                entry_price=prices[e.symbol],
                exit_by=e.exit_by,
                n_buys=e.n_buys,
            )
            log_predictions(
                store,
                [
                    PredictionRecord(
                        as_of=as_of,
                        symbol=e.symbol,
                        probability=min(1.0, max(0.0, e.expected_car * 10.0)),
                        horizon_days=settings.insider_sleeve_hold_days,
                        model_version=SLEEVE_MODEL,
                        features={"n_buys": float(e.n_buys), "expected_car": e.expected_car},
                    )
                ],
            )
            opened.append(e.symbol)

    limits = RiskLimits(
        max_position_weight=settings.max_position_weight,
        max_gross_exposure=settings.max_gross_exposure,
        max_net_exposure=settings.max_net_exposure,
    )
    engine = ExecutionEngine(broker, settings=settings, limits=limits)
    orders = engine.rebalance(target, prices, as_of=as_of) if target else []
    if opened or closed:
        _log.info("insider_sleeve", opened=opened, closed=closed, orders=len(orders))
    return InsiderSleeveResult(as_of=as_of, opened=opened, closed=closed, orders=orders)


def run_insider_sleeve_live_pass(
    settings: Settings, *, watchlist: list[str] | None = None, lookback_days: int = 150
) -> InsiderSleeveResult:
    """Standalone live sleeve pass against Alpaca paper: ingest data, then run one cycle.

    The executing counterpart to the read-only ``insider-events`` preview: it pulls fresh
    prices (for entries + stop checks) and recent Form-4 filings (the sleeve's signal), then
    runs :func:`run_insider_sleeve_cycle` through the broker. Paper-gated like every live path
    (reaching the live endpoint requires both arming switches). The daily cycle runs the sleeve
    on already-ingested data; this is the manual/cron entry point that ingests on its own.
    """
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
    if not settings.alpaca_paper:
        settings.assert_live_allowed()  # live endpoint needs both arming switches

    from datetime import date, timedelta

    from market_trader.collectors import IngestionGateway, PriceCollector
    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.collectors.edgar import EdgarClient, Form4Collector
    from market_trader.execution.alpaca import AlpacaBroker
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

    watchlist = watchlist or resolve_universe(settings.universe)
    store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
    store.create_schema()  # idempotent

    end = date.today()
    data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
    bars = data.fetch_daily_bars(
        watchlist,
        start=end - timedelta(days=lookback_days),
        end=end,
        feed=settings.alpaca_data_feed,
    )
    IngestionGateway(store).ingest(PriceCollector().normalize(bars))

    # Form-4 (the sleeve's signal): time-bounded so a slow SEC endpoint can't stall the pass.
    client = EdgarClient(
        user_agent=settings.sec_user_agent,
        timeout_seconds=settings.insider_fetch_timeout_seconds,
        budget_seconds=settings.insider_fetch_budget_seconds,
    )
    filings = client.fetch_for_symbols(watchlist, lookback_days=settings.insider_lookback_days)
    if filings:
        IngestionGateway(store).ingest(Form4Collector().normalize(filings))

    broker = AlpacaBroker(
        settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
    )
    return run_insider_sleeve_cycle(settings, store=store, broker=broker)
