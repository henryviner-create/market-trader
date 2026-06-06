"""One paper-trading cycle, end to end.

``run_paper_cycle`` is the pure, injectable core (store + prices + broker + llm) so
it is fully tested offline. ``run_dry_paper_cycle`` wires synthetic data + a local
paper broker (no network). ``run_live_paper_cycle`` wires Alpaca market data +
the Alpaca paper broker + the Claude briefing.

PAPER ONLY. Sizing is capped by ``capital_ceiling`` and the risk limits; live
order routing stays gated by ``Settings.assert_live_allowed()`` in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from market_trader.portfolio import RiskLimits, apply_risk_limits, size_book
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


def _trailing_stops(
    store: BitemporalStore,
    as_of: datetime,
    positions: list,
    prices: dict[str, float],
    trail_pct: float,
    *,
    window: int = 60,
) -> set[str]:
    """Held names that have fallen more than ``trail_pct`` below their trailing high.

    A *breakdown* stop, complementing the entry-based hard stop: a name that has given back
    its recent gains and rolled over from its ``window``-day high is cut, even if still above
    entry — the book reacts to a position deteriorating instead of holding it blindly. The
    high is read from the point-in-time price panel, so there is no per-position state to
    persist. 0 disables it.
    """
    if trail_pct <= 0:
        return set()
    from market_trader.backtest.pit import observations_to_price_frame

    panel = observations_to_price_frame(store.as_of(as_of, dataset=PRICE_DATASET))
    if panel.empty:
        return set()
    out: set[str] = set()
    for p in positions:
        px = prices.get(p.symbol)
        if p.qty <= 0 or not px or p.symbol not in panel.columns:
            continue
        recent = panel[p.symbol].tail(window).dropna()
        if recent.empty:
            continue
        peak = float(recent.max())
        if peak > 0 and (px / peak - 1.0) <= -trail_pct:
            out.add(p.symbol)
    return out


def _thesis_break_exits(
    store: BitemporalStore, as_of: datetime, positions: list, threshold: float
) -> set[str]:
    """Held names whose Opus news-sentiment has turned strongly negative — a thesis-break exit.

    Reads the point-in-time ``llm_news_sentiment`` (the nightly Opus factory's output, already
    confidence-weighted), so NO LLM call happens in the cycle. A name we *hold* whose sentiment
    drops to ``<= -threshold`` is flagged for exit — the book reacts to material bad news on a
    position rather than holding through it. Defensive (exit-only, never an entry trigger), so
    even an unproven signal is acting only as risk control. 0 disables it.
    """
    if threshold <= 0 or not positions:
        return set()
    from market_trader.features.llm import LLMNewsSentiment

    held = [p.symbol for p in positions if p.qty > 0]
    if not held:
        return set()
    sentiment = LLMNewsSentiment().compute(store, as_of, held)
    return {s for s in held if float(sentiment.get(s, 0.0)) <= -threshold}


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


_VOL_LOOKBACK = 90  # trailing trading days for the covariance estimate


def _trailing_returns(
    store: BitemporalStore, as_of: datetime, symbols: list[str], *, lookback: int = _VOL_LOOKBACK
) -> pd.DataFrame | None:
    """Point-in-time trailing daily returns for ``symbols`` (knowledge_time <= as_of)."""
    from market_trader.backtest.pit import observations_to_price_frame

    panel = observations_to_price_frame(store.as_of(as_of, dataset=PRICE_DATASET))
    if panel.empty:
        return None
    cols = [s for s in symbols if s in panel.columns]
    if not cols:
        return None
    return panel[cols].pct_change().iloc[1:].tail(lookback)


def _vol_target_weights(
    weights: dict[str, float], returns: pd.DataFrame, *, target_vol: float, max_gross: float
) -> dict[str, float]:
    """Scale a long-only book to ``target_vol`` annualised, capped at ``max_gross``.

    This is the drawdown governor the backtest validated to ~-19% max-DD: cut exposure
    (into cash) as volatility rises. Too little return history is a no-op — the input book
    is returned unscaled so the cycle degrades gracefully on a fresh deployment.
    """
    from market_trader.portfolio.construction import ledoit_wolf_cov, volatility_target_weights

    names = [s for s in weights if s in returns.columns]
    window = returns[names].dropna(axis=1, how="any")
    if window.shape[1] < 2 or window.shape[0] < 20:
        return weights
    cov = ledoit_wolf_cov(window)
    w = pd.Series(weights).reindex(cov.columns).fillna(0.0)
    scaled = volatility_target_weights(w, cov, target_vol)
    gross = float(scaled.abs().sum())
    if gross > max_gross and gross > 0:
        scaled = scaled * (max_gross / gross)
    return {str(k): float(v) for k, v in scaled.items() if abs(float(v)) > 1e-9}


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
    tilt_strength: float = 0.0,
    prediction_log: bool = False,
    model_version: str = "composite",
    prediction_horizon: int = 5,
    stop_loss_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
    thesis_exit_threshold: float = 0.0,
    reserved_symbols: frozenset[str] = frozenset(),
    cancel_stale_orders: bool = False,
) -> CycleResult:
    """Score the universe, form risk-managed target weights, execute on the broker."""
    limits = limits or _limits_from_settings(settings)
    if cancel_stale_orders:
        # A fresh daily target supersedes any still-open orders from a prior run.
        # Cancel them (sleeve-reserved names excepted) before rebalancing, so the
        # new book is never rejected as a wash trade against its own stale resting
        # orders — and a leftover order can't distort the position accounting.
        stale = [o for o in broker.get_open_orders() if o.symbol not in reserved_symbols]
        for o in stale:
            broker.cancel_order(o.client_order_id)
        if stale:
            _log.info(
                "cancelled_stale_orders",
                count=len(stale),
                names=sorted({o.symbol for o in stale}),
            )
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

    # Symbols owned by another sleeve (e.g. the news sleeve) are reserved: the
    # daily book neither selects nor flattens them, so the two never fight.
    positions = [p for p in broker.get_positions() if p.symbol not in reserved_symbols]
    held = {p.symbol for p in positions}
    # Exit discipline: cut a name past its hard floor from entry, broken down from its
    # trailing high, OR with materially bad news on it (Opus thesis-break) — the book reacts
    # to deterioration rather than holding regardless.
    stopped = (
        _stop_losses(positions, prices, stop_loss_pct)
        | _trailing_stops(store, as_of, positions, prices, trailing_stop_pct)
        | _thesis_break_exits(store, as_of, positions, thesis_exit_threshold)
    )
    if stopped:
        _log.info("exit_discipline", names=sorted(stopped))
    target: dict[str, float] = {}
    if not ranked.empty and risk_weighting == "size_book":
        # The unified chassis (size_book): hold the *whole* scored universe, vol-governed to
        # the budget and tilted toward higher scores (tilt_strength=0 -> governed 1/N). The
        # top-N selection / hysteresis do not apply here — breadth is the point. size_book
        # applies the hard caps itself, so its output is already the risk-managed target.
        eligible = [s for s in ranked.index if s not in stopped and s not in reserved_symbols]
        returns = _trailing_returns(store, as_of, eligible)
        if returns is not None and not returns.empty:
            target = size_book(
                returns,
                target_vol=settings.target_vol,
                limits=limits,
                scores=ranked if tilt_strength > 0 else None,
                tilt_strength=tilt_strength,
                lookback=_VOL_LOOKBACK,
            )
    elif not ranked.empty:
        enter_k = max(1, int(len(ranked) * top_quantile))
        if max_positions is not None:
            enter_k = min(enter_k, max_positions)  # cap breadth into a diversified book
        exit_k = min(len(ranked), max(enter_k, round(enter_k * exit_band_multiple)))
        winners = _select_with_hysteresis(ranked, held, enter_k, exit_k, max_positions)
        winners = [w for w in winners if w not in stopped and w not in reserved_symbols]
        if risk_weighting == "vol_target":
            # The validated drawdown governor: an equal-risk base book scaled to the
            # vol budget (holding cash as vol rises), then the hard caps. Falls back to
            # the base book when there isn't yet enough price history to estimate cov.
            base = _risk_weights(winners, matrix, "inverse_vol", ranked)
            returns = _trailing_returns(store, as_of, winners)
            weights = (
                base
                if returns is None
                else _vol_target_weights(
                    base,
                    returns,
                    target_vol=settings.target_vol,
                    max_gross=limits.max_gross_exposure,
                )
            )
        else:
            weights = _risk_weights(winners, matrix, risk_weighting, ranked)
        target = apply_risk_limits(weights, limits)

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
        store,
        as_of=as_of,
        symbols=symbols,
        prices=prices,
        broker=broker,
        settings=settings,
        risk_weighting=settings.risk_weighting,  # honour the configured book (e.g. size_book)
        tilt_strength=settings.tilt_strength,
    )


def _ingest_news(store: BitemporalStore, symbols: list[str], settings: Settings) -> None:
    """Best-effort: pull recent GDELT articles for the universe and ingest them.

    Time-bounded (see ``news_fetch_*`` settings) so a slow/throttling GDELT can
    never stall the cycle — the sweep stops at its wall-clock budget and the rest
    is picked up on a later run.
    """
    from market_trader.collectors.gdelt import GdeltClient, GdeltNewsCollector

    client = GdeltClient(
        timeout_seconds=settings.news_fetch_timeout_seconds,
        budget_seconds=settings.news_fetch_budget_seconds,
    )
    articles = client.fetch_for_symbols(symbols, timespan=settings.news_timespan)
    if articles:
        IngestionGateway(store).ingest(GdeltNewsCollector().normalize(articles))


def _ingest_filings(store: BitemporalStore, symbols: list[str], settings: Settings) -> None:
    """Best-effort: pull recent SEC Form-4 insider filings and ingest them.

    Time-bounded (``insider_fetch_*`` settings) so a slow SEC endpoint never stalls
    the cycle. The InsiderNetBuys feature is already in default_features(); this just
    feeds it the data (it returns a neutral 0 while the store is empty).
    """
    from market_trader.collectors.edgar import EdgarClient, Form4Collector

    client = EdgarClient(
        user_agent=settings.sec_user_agent,
        timeout_seconds=settings.insider_fetch_timeout_seconds,
        budget_seconds=settings.insider_fetch_budget_seconds,
    )
    records = client.fetch_for_symbols(symbols, lookback_days=settings.insider_lookback_days)
    if records:
        IngestionGateway(store).ingest(Form4Collector().normalize(records))


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

    if settings.insider_enabled:
        _ingest_filings(store, watchlist, settings)  # Form-4; feeds the InsiderNetBuys feature
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

    # If a sleeve is on, reserve the names it owns (the daily book leaves them alone) and
    # shrink the daily book's gross by the reserved budget so the book + sleeve(s) together
    # stay within the exposure cap. Budgets are additive across sleeves.
    reserved: frozenset[str] = frozenset()
    limits: RiskLimits | None = None
    sleeve_budget = 0.0
    if settings.news_sleeve_enabled:
        from market_trader.runtime.news_sleeve import active_sleeve_positions

        reserved |= frozenset(active_sleeve_positions(store, as_of))
        sleeve_budget += settings.news_sleeve_budget
    if settings.insider_sleeve_enabled:
        from market_trader.runtime.insider_sleeve import active_insider_positions

        reserved |= frozenset(active_insider_positions(store, as_of))
        sleeve_budget += settings.insider_sleeve_budget
    if sleeve_budget > 0:
        base = _limits_from_settings(settings)
        limits = replace(base, max_gross_exposure=base.max_gross_exposure * (1 - sleeve_budget))

    result = run_paper_cycle(
        store,
        as_of=as_of,
        symbols=watchlist,
        prices=prices,
        broker=broker,
        settings=settings,
        limits=limits,
        llm=llm,
        feature_store=fs,
        score_fn=score_fn,
        top_quantile=settings.top_quantile,
        max_positions=settings.max_positions or None,  # 0 -> uncapped
        exit_band_multiple=settings.exit_band_multiple,
        risk_weighting=settings.risk_weighting,
        tilt_strength=settings.tilt_strength,
        prediction_log=True,
        model_version=settings.scorer,
        stop_loss_pct=settings.stop_loss_pct,
        trailing_stop_pct=settings.trailing_stop_pct,
        thesis_exit_threshold=settings.thesis_exit_threshold,
        reserved_symbols=reserved,
        cancel_stale_orders=True,  # clear a prior run's unfilled orders first
    )

    # Run the insider-cluster sleeve on the same fresh store/broker right after the daily
    # book. It manages only its own names (a direct rebalance), so the book is undisturbed;
    # a sleeve error never fails the daily cycle — the daily book has already executed.
    if settings.insider_sleeve_enabled:
        from market_trader.runtime.insider_sleeve import run_insider_sleeve_cycle

        try:
            run_insider_sleeve_cycle(settings, store=store, broker=broker, as_of=as_of)
        except Exception as exc:
            _log.warning("insider_sleeve_error", error=str(exc))
    return result
