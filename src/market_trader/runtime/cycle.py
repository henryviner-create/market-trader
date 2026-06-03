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
from market_trader.features.news import news_features
from market_trader.observability import get_logger
from market_trader.portfolio import RiskLimits, apply_risk_limits
from market_trader.reasoning import LLMProvider, build_briefing_context, generate_llm_brief
from market_trader.runtime.learning import grade_predictions, log_cycle_predictions
from market_trader.runtime.scoring import ScoreFn, build_scorer, composite_scorer
from market_trader.storage import InMemoryBitemporalStore
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe.liquid import MEGACAP_WATCHLIST, resolve_universe

_log = get_logger("cycle")


def _measured_signal_ic(
    store: BitemporalStore, settings: Settings, as_of: datetime
) -> dict[str, float]:
    """Per-signal IC from this scorer's own graded predictions — the learning loop.

    Grading uses only outcomes whose horizon has fully elapsed, so this is an
    out-of-sample read on what has actually worked, fed back into today's weights.
    """
    graded = grade_predictions(
        store, as_of, model_version=settings.scorer, min_abs_ic=settings.ic_min_abs
    )
    return {str(k): float(v) for k, v in graded.get("ic", {}).items()}


# Re-exported for callers/tests; the live default universe is now the broad
# "liquid" set, chosen via Settings.universe (resolve_universe).
DEFAULT_WATCHLIST = MEGACAP_WATCHLIST


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


def _select_with_hysteresis(
    ranked: pd.Series, held: set[str], enter_k: int, exit_k: int, cap: int | None
) -> list[str]:
    """Top entrants + held names still inside the wider exit band (held kept first).

    Holding a name until it leaves the *exit* band — rather than selling it the
    moment it slips below the *entry* cutoff — is what stops the book churning on
    rank noise. Held names are retained ahead of fresh entrants so the engine
    doesn't sell a position just to rotate into a marginally higher-ranked one.
    """
    keep = set(ranked.head(exit_k).index)
    entrants = list(ranked.head(enter_k).index)
    held_kept = [s for s in ranked.index if s in held and s in keep]
    selected = held_kept + [s for s in entrants if s not in held]
    return [str(s) for s in (selected[:cap] if cap is not None else selected)]


def _stop_losses(positions: list, prices: dict[str, float], stop_loss_pct: float) -> set[str]:
    """Held names down more than ``stop_loss_pct`` from entry — the hard loss floor.

    Absolute (vs the relative rank hysteresis): no matter how strong the signal
    still looks, a position bleeding past the stop is cut. 0 disables it.
    """
    if stop_loss_pct <= 0:
        return set()
    out: set[str] = set()
    for p in positions:
        px = prices.get(p.symbol)
        if px and p.avg_price > 0 and p.qty > 0 and (px / p.avg_price - 1.0) <= -stop_loss_pct:
            out.add(p.symbol)
    return out


def _risk_weights(
    selected: list[str], matrix: pd.DataFrame, mode: str, scores: pd.Series | None = None
) -> dict[str, float]:
    """Size the book: inverse-vol (~equal risk), equal, or conviction (by signal)."""
    n = len(selected)
    if n == 0:
        return {}
    if mode == "conviction" and scores is not None:
        s = scores.reindex(selected).astype(float).fillna(0.0)
        s = s - s.min()  # shift so the weakest selected name anchors at zero
        s = s + (s.max() * 0.10 if s.max() > 0 else 1.0)  # small floor; still held, just light
        total = float(s.sum())
        if total > 0:
            return {str(k): float(v) / total for k, v in s.items()}
    if mode == "inverse_vol":
        vol_cols = [c for c in matrix.columns if str(c).startswith("vol_")]
        if vol_cols:
            vol = matrix.reindex(index=selected)[vol_cols[0]].astype(float)
            inv = 1.0 / vol.where(vol > 0)
            if inv.notna().any():
                inv = inv.fillna(inv.mean())
                total = float(inv.sum())
                if total > 0:
                    return {str(s): float(inv.loc[s]) / total for s in selected}
    return {s: 1.0 / n for s in selected}


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
    score_fn: ScoreFn | None = None,
    top_quantile: float = 0.3,
    max_positions: int | None = None,
    exit_band_multiple: float = 1.0,
    risk_weighting: str = "equal",
    prediction_log: bool = False,
    model_version: str = "composite",
    prediction_horizon: int = 5,
    stop_loss_pct: float = 0.0,
) -> CycleResult:
    """Score the universe, form risk-managed target weights, execute on the broker."""
    limits = limits or _limits_from_settings(settings)
    fs = feature_store or FeatureStore(store, default_features())
    matrix = fs.compute_matrix(as_of, symbols)

    scorer = score_fn or composite_scorer()
    scores = scorer(matrix, as_of) if not matrix.empty else pd.Series(dtype=float)
    ranked = scores.dropna().sort_values(ascending=False)

    if prediction_log and not matrix.empty:
        # Persist this call's ranking + features so it can be graded once outcomes land.
        log_cycle_predictions(
            store,
            ranked,
            matrix,
            as_of,
            model_version=model_version,
            horizon_days=prediction_horizon,
        )

    positions = broker.get_positions()
    held = {p.symbol for p in positions}
    stopped = _stop_losses(positions, prices, stop_loss_pct)  # cut losers past the hard floor
    if stopped:
        _log.info("stop_loss", names=sorted(stopped))
    target: dict[str, float] = {}
    if not ranked.empty:
        enter_k = max(1, int(len(ranked) * top_quantile))
        if max_positions is not None:
            enter_k = min(enter_k, max_positions)  # cap breadth into a diversified book
        exit_k = min(len(ranked), max(enter_k, round(enter_k * exit_band_multiple)))
        winners = _select_with_hysteresis(ranked, held, enter_k, exit_k, max_positions)
        winners = [w for w in winners if w not in stopped]  # stop overrides the signal
        target = apply_risk_limits(_risk_weights(winners, matrix, risk_weighting, ranked), limits)

    # Flatten holdings that left the selection: the engine only acts on symbols in
    # the target, so a dropped name needs an explicit 0 (with a price) or it would
    # linger on the persistent account forever. ``target`` stays the desired book.
    rebalance_target = dict(target)
    for sym in held:
        if sym not in rebalance_target and sym in prices:
            rebalance_target[sym] = 0.0

    engine = ExecutionEngine(broker, settings=settings, limits=limits)
    orders = engine.rebalance(rebalance_target, prices, as_of=as_of) if rebalance_target else []

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


def _ingest_news(store: BitemporalStore, symbols: list[str], settings: Settings) -> None:
    """Best-effort: pull recent GDELT articles for the universe and ingest them."""
    from market_trader.collectors.gdelt import GdeltClient, GdeltNewsCollector

    articles = GdeltClient().fetch_for_symbols(symbols, timespan=settings.news_timespan)
    if articles:
        IngestionGateway(store).ingest(GdeltNewsCollector().normalize(articles))


def run_live_paper_cycle(
    settings: Settings, *, lookback_days: int = 150, watchlist: list[str] | None = None
) -> CycleResult:
    """Live paper cycle: Alpaca data + Alpaca paper broker + (optional) Claude brief."""
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
    if not settings.alpaca_paper:
        # Reaching the *live* Alpaca endpoint requires live trading to be armed
        # (both switches). Without this, MT_ALPACA_PAPER=false would route real
        # orders even though the engine's live gate only keys off execution_mode.
        settings.assert_live_allowed()

    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.execution.alpaca import AlpacaBroker
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

    watchlist = watchlist or resolve_universe(settings.universe)
    store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
    store.create_schema()  # idempotent

    end = date.today()
    start = end - timedelta(days=lookback_days)
    data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
    records = data.fetch_daily_bars(watchlist, start=start, end=end, feed=settings.alpaca_data_feed)
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

    features = default_features()
    if settings.news_enabled:
        _ingest_news(store, watchlist, settings)  # daily-only: per-symbol fetch is heavy
        features = features + news_features(settings.news_window_days)
    fs = FeatureStore(store, features)
    ic: dict[str, float] | None = None
    if settings.scorer.strip().lower() == "composite" and settings.ic_weighting:
        ic = _measured_signal_ic(store, settings, as_of)
        _log.info("ic_weighting", signals=len(ic), ic={k: round(v, 4) for k, v in ic.items()})
    score_fn = build_scorer(settings, store, fs, watchlist, as_of, ic=ic)
    return run_paper_cycle(
        store,
        as_of=as_of,
        symbols=watchlist,
        prices=prices,
        broker=broker,
        settings=settings,
        llm=llm,
        feature_store=fs,
        score_fn=score_fn,
        max_positions=settings.max_positions or None,  # 0 -> uncapped
        exit_band_multiple=settings.exit_band_multiple,
        risk_weighting=settings.risk_weighting,
        prediction_log=True,
        model_version=settings.scorer,
        stop_loss_pct=settings.stop_loss_pct,
    )
