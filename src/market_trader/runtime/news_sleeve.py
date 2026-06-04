"""Event-driven news sleeve (PAPER).

A selective overlay on the daily book: when a name gets *material* news (an
article-count surge with a tone direction), open a small, time-boxed position to
ride the post-news **drift**, then exit after a fixed window. It is **non-churning
by construction** — it acts only on a fresh, deduped story, respects a per-name
cooldown, and otherwise leaves its holdings untouched until they expire (unlike
the timer-driven intraday loop that was removed for shredding the book).

Coexistence with the daily book: the sleeve manages **only its own names** via a
direct ``ExecutionEngine.rebalance`` (which only ever touches symbols in the
target it is given), and ``run_paper_cycle`` is told to *reserve* sleeve-owned
names so the daily cycle never flattens them. Capital is split by a budget.

Every open is logged as a ``news_sleeve`` prediction so it can be graded later
(``grade_predictions``) — the sleeve is held to the same evidentiary bar as
everything else. The decision/sizing here is a deliberately simple v1; the
``memory/event_study`` + ``memory/episodic`` machinery is the seam where a
data-driven expected-drift estimate plugs in next.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from market_trader.collectors.gateway import IngestionGateway
from market_trader.collectors.gdelt import NEWS_DATASET
from market_trader.collectors.news_feed import GdeltNewsFeed, NewsFeed
from market_trader.config import Settings
from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import to_utc_lenient, utcnow
from market_trader.execution.broker import Broker, Order
from market_trader.execution.engine import ExecutionEngine
from market_trader.feedback.prediction_log import PredictionRecord, log_predictions
from market_trader.observability import get_logger
from market_trader.portfolio.risk import RiskLimits
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe.liquid import resolve_universe

_log = get_logger("news_sleeve")

SLEEVE_DATASET = "sleeve.news_position"
SLEEVE_MODEL = "news_sleeve"


@dataclass(frozen=True)
class NewsEvent:
    symbol: str
    direction: int  # +1 positive tone, -1 negative, 0 none
    confidence: float  # [0, 1]
    tone: float
    surge: float  # recent article count / trailing daily mean
    story_hash: str


@dataclass(frozen=True)
class SleevePosition:
    symbol: str
    entry_price: float
    exit_by: datetime
    story_hash: str


@dataclass
class NewsSleeveResult:
    as_of: datetime
    opened: list[str]
    closed: list[str]
    orders: list[Order]


# --- event detection --------------------------------------------------------


def _articles_within(
    store: BitemporalStore, as_of: datetime, symbols: set[str], *, baseline_days: int
) -> dict[str, list[tuple[datetime, dict[str, Any]]]]:
    cutoff = as_of - timedelta(days=baseline_days)
    rows: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for o in store.as_of(as_of, dataset=NEWS_DATASET):
        if o.entity_id in symbols and o.event_time >= cutoff:
            rows[o.entity_id].append((o.event_time, o.value))
    return rows


def _confidence(surge: float, tone: float, count_surge: float, tone_min: float) -> float:
    from_surge = min(1.0, surge / (2.0 * count_surge)) if count_surge > 0 else 0.0
    from_tone = min(1.0, abs(tone) / (2.0 * tone_min)) if tone_min > 0 else 0.0
    return round(max(0.0, min(1.0, 0.5 * from_surge + 0.5 * from_tone)), 4)


def _story_hash(items: list[tuple[datetime, dict[str, Any]]]) -> str:
    latest = max(items, key=lambda tv: tv[0])[1]
    key = str(latest.get("url") or latest.get("title") or "")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def detect_news_events(
    store: BitemporalStore,
    as_of: datetime,
    symbols: Sequence[str],
    *,
    count_surge: float,
    tone_min: float,
    baseline_days: int,
    recent_days: int = 1,
) -> list[NewsEvent]:
    """Flag names whose recent news volume surged past their trailing baseline."""
    rows = _articles_within(store, as_of, set(symbols), baseline_days=baseline_days)
    recent_cut = as_of - timedelta(days=recent_days)
    denom = max(baseline_days - recent_days, 1)
    events: list[NewsEvent] = []
    for sym, items in rows.items():
        recent = [tv for tv in items if tv[0] >= recent_cut]
        if len(recent) < 3:
            continue
        baseline_daily = len([tv for tv in items if tv[0] < recent_cut]) / denom
        surge = len(recent) / max(baseline_daily, 0.5)
        if surge < count_surge:
            continue  # not a genuine attention surge
        tones = [float(v["tone"]) for _t, v in recent if v.get("tone") is not None]
        tone = sum(tones) / len(tones) if tones else 0.0
        direction = 1 if tone > 0 else (-1 if tone < 0 else 0)
        events.append(
            NewsEvent(
                symbol=sym,
                direction=direction,
                confidence=_confidence(surge, tone, count_surge, tone_min),
                tone=round(tone, 4),
                surge=round(surge, 4),
                story_hash=_story_hash(recent),
            )
        )
    return events


# --- sleeve-position state (bitemporal; open/close, dedup, cooldown) --------


def _records(store: BitemporalStore, as_of: datetime) -> list[Observation]:
    return list(store.as_of(as_of, dataset=SLEEVE_DATASET))


def _latest_by_symbol(records: list[Observation]) -> dict[str, Observation]:
    latest: dict[str, Observation] = {}
    for o in records:
        cur = latest.get(o.entity_id)
        if cur is None or o.event_time > cur.event_time:
            latest[o.entity_id] = o
    return latest


def active_sleeve_positions(store: BitemporalStore, as_of: datetime) -> dict[str, SleevePosition]:
    """Symbols the sleeve currently owns (latest state is ``open``)."""
    out: dict[str, SleevePosition] = {}
    for sym, o in _latest_by_symbol(_records(store, as_of)).items():
        if o.value.get("status") == "open":
            out[sym] = SleevePosition(
                symbol=sym,
                entry_price=float(o.value.get("entry_price", 0.0)),
                exit_by=to_utc_lenient(datetime.fromisoformat(str(o.value["exit_by"]))),
                story_hash=str(o.value.get("story_hash", "")),
            )
    return out


def acted_story_hashes(store: BitemporalStore, as_of: datetime) -> set[str]:
    return {
        str(o.value.get("story_hash", ""))
        for o in _records(store, as_of)
        if o.value.get("story_hash")
    }


def last_action_times(store: BitemporalStore, as_of: datetime) -> dict[str, datetime]:
    return {sym: o.event_time for sym, o in _latest_by_symbol(_records(store, as_of)).items()}


def _persist(store: BitemporalStore, obs: Observation) -> None:
    store.upsert_many([with_deterministic_id(obs)])


def record_open(
    store: BitemporalStore,
    symbol: str,
    as_of: datetime,
    *,
    entry_price: float,
    exit_by: datetime,
    story_hash: str,
) -> None:
    _persist(
        store,
        Observation(
            source="sleeve",
            dataset=SLEEVE_DATASET,
            entity_type="equity",
            entity_id=symbol,
            ref=story_hash,
            event_time=as_of,
            knowledge_time=as_of,
            value={
                "status": "open",
                "entry_price": float(entry_price),
                "exit_by": exit_by.isoformat(),
                "story_hash": story_hash,
            },
        ),
    )


def record_close(
    store: BitemporalStore, symbol: str, as_of: datetime, *, story_hash: str, reason: str
) -> None:
    _persist(
        store,
        Observation(
            source="sleeve",
            dataset=SLEEVE_DATASET,
            entity_type="equity",
            entity_id=symbol,
            ref=f"{story_hash}:close:{reason}",
            event_time=as_of,
            knowledge_time=as_of,
            value={"status": "closed", "story_hash": story_hash, "reason": reason},
        ),
    )


def _log_decision(store: BitemporalStore, ev: NewsEvent, as_of: datetime, hold_days: int) -> None:
    """Persist the open as a gradeable prediction (validated like every signal)."""
    log_predictions(
        store,
        [
            PredictionRecord(
                as_of=as_of,
                symbol=ev.symbol,
                probability=ev.confidence,
                horizon_days=hold_days,
                model_version=SLEEVE_MODEL,
                features={"tone": ev.tone, "surge": ev.surge, "confidence": ev.confidence},
            )
        ],
    )


# --- one pass ---------------------------------------------------------------


def _latest_prices(store: BitemporalStore, as_of: datetime, symbols: set[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for o in store.as_of(as_of, dataset=PRICE_DATASET):
        if o.entity_id in symbols and o.value.get("close"):
            prices[o.entity_id] = float(o.value["close"])  # last wins ~ most recent
    return prices


def run_news_sleeve_cycle(
    settings: Settings,
    *,
    feed: NewsFeed,
    store: BitemporalStore,
    broker: Broker,
    as_of: datetime | None = None,
    watchlist: list[str] | None = None,
    limits: RiskLimits | None = None,
) -> NewsSleeveResult:
    """One sleeve pass: ingest news, close expired/stopped, open fresh events."""
    as_of = as_of or utcnow()
    watchlist = list(watchlist or resolve_universe(settings.universe))
    wanted = set(watchlist)

    obs = feed.fetch_recent(watchlist, lookback_minutes=settings.news_sleeve_lookback_minutes)
    if obs:
        IngestionGateway(store).ingest(obs)

    prices = _latest_prices(store, as_of, wanted)
    active = active_sleeve_positions(store, as_of)
    acted = acted_story_hashes(store, as_of)
    last_action = last_action_times(store, as_of)

    target: dict[str, float] = {}
    opened: list[str] = []
    closed: list[str] = []

    # Close expired (past the drift window) or stopped-out positions.
    for sym, pos in active.items():
        price = prices.get(sym)
        stopped = bool(
            price
            and pos.entry_price > 0
            and settings.stop_loss_pct > 0
            and price / pos.entry_price - 1.0 <= -settings.stop_loss_pct
        )
        if as_of >= pos.exit_by or stopped:
            target[sym] = 0.0
            record_close(
                store,
                sym,
                as_of,
                story_hash=pos.story_hash,
                reason="stop" if stopped else "horizon",
            )
            closed.append(sym)

    # Open on fresh, confident, positive events (long-only v1), deduped + cooled down.
    per_name = settings.news_sleeve_budget / max(settings.news_sleeve_max_names, 1)
    cooldown = timedelta(days=settings.news_sleeve_cooldown_days)
    live = {s for s in active if s not in closed}
    events = detect_news_events(
        store,
        as_of,
        watchlist,
        count_surge=settings.news_sleeve_count_surge,
        tone_min=settings.news_sleeve_tone_min,
        baseline_days=settings.news_sleeve_baseline_days,
    )
    for ev in sorted(events, key=lambda e: e.confidence, reverse=True):
        if len(live) >= settings.news_sleeve_max_names:
            break
        price = prices.get(ev.symbol)
        if not price or price <= 0:
            continue
        if ev.direction <= 0 or ev.confidence < settings.news_sleeve_min_confidence:
            continue  # long-only v1: only confident, positive-tone events
        if ev.story_hash in acted or ev.symbol in live:
            continue  # never act on the same story twice / already holding
        prev = last_action.get(ev.symbol)
        if prev is not None and as_of - prev < cooldown:
            continue  # per-name cooldown
        target[ev.symbol] = per_name
        record_open(
            store,
            ev.symbol,
            as_of,
            entry_price=price,
            exit_by=as_of + timedelta(days=settings.news_sleeve_hold_days),
            story_hash=ev.story_hash,
        )
        _log_decision(store, ev, as_of, settings.news_sleeve_hold_days)
        live.add(ev.symbol)
        opened.append(ev.symbol)

    # Execute. ExecutionEngine only touches symbols in `target`, so the daily book
    # is never disturbed. The sleeve's gross is capped at its budget.
    orders: list[Order] = []
    if target:
        sleeve_limits = limits or RiskLimits(
            max_position_weight=max(per_name, settings.max_position_weight),
            max_gross_exposure=settings.news_sleeve_budget,
            max_net_exposure=settings.news_sleeve_budget,
        )
        engine = ExecutionEngine(broker, settings=settings, limits=sleeve_limits)
        orders = engine.rebalance(target, prices, as_of=as_of)

    if opened or closed:
        _log.info("news_sleeve_pass", opened=opened, closed=closed, orders=len(orders))
    return NewsSleeveResult(as_of=as_of, opened=opened, closed=closed, orders=orders)


# --- the loop ---------------------------------------------------------------


def run_news_sleeve_loop(
    settings: Settings,
    *,
    is_market_open: Callable[[], bool] | None = None,
    run_cycle: Callable[[], NewsSleeveResult] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stop: Callable[[], bool] | None = None,
    max_iterations: int | None = None,
) -> int:
    """Poll for news events during market hours and act selectively. Injectable."""
    if is_market_open is None or run_cycle is None:
        if not (settings.alpaca_key_id and settings.alpaca_secret_key):
            raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
        if not settings.alpaca_paper:
            settings.assert_live_allowed()
        from market_trader.execution.alpaca import AlpacaBroker
        from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        broker = AlpacaBroker(
            settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
        )
        feed: NewsFeed = GdeltNewsFeed()
        if is_market_open is None:
            is_market_open = broker.is_market_open
        if run_cycle is None:

            def _run_cycle() -> NewsSleeveResult:
                return run_news_sleeve_cycle(settings, feed=feed, store=store, broker=broker)

            run_cycle = _run_cycle

    interval = float(settings.news_sleeve_interval_seconds)
    iterations = 0
    _log.info("news_sleeve_loop_start", interval_seconds=interval, mode=settings.execution_mode)
    while not (stop and stop()) and not (
        max_iterations is not None and iterations >= max_iterations
    ):
        try:
            if is_market_open():
                result = run_cycle()
                if not (result.opened or result.closed):
                    _log.info("news_sleeve_idle", reason="no_event")
            else:
                _log.info("news_sleeve_idle", reason="market_closed")
        except Exception as exc:  # one bad pass must never kill the loop
            _log.warning("news_sleeve_pass_error", error=str(exc))
        iterations += 1
        if (stop and stop()) or (max_iterations is not None and iterations >= max_iterations):
            break
        sleep_fn(interval)
    _log.info("news_sleeve_loop_stop", iterations=iterations)
    return iterations
