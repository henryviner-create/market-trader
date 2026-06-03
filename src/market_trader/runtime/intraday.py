"""Intraday, live-reactive paper trading.

``run_intraday_cycle`` is one pass of the loop: pull recent **minute** bars from
Alpaca, land them in the intraday dataset, then reuse the exact same
``run_paper_cycle`` scoring + risk + execution path — only the signals (computed
on minute bars) and the price snapshot differ. Every dependency is injectable, so
the whole thing is exercised offline with an in-memory store + paper broker.

PAPER ONLY, and only when ``MT_INTRADAY_ENABLED=true``. Live routing stays gated
by ``Settings.assert_live_allowed()`` in the engine, unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from market_trader.backtest.pit import StorePriceView
from market_trader.collectors import IngestionGateway
from market_trader.collectors.intraday import intraday_bars_to_observations
from market_trader.config import Settings
from market_trader.core.synthetic import PRICE_INTRADAY_DATASET
from market_trader.core.time import utcnow
from market_trader.execution.broker import Broker
from market_trader.features import Feature, FeatureStore, MeanReversion, Momentum, Volatility
from market_trader.observability import get_logger
from market_trader.runtime.cycle import CycleResult, run_paper_cycle
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe.liquid import resolve_universe

_log = get_logger("intraday")


def intraday_features(settings: Settings) -> list[Feature]:
    """Technical features pointed at the minute dataset — lookbacks count minutes."""
    ds = PRICE_INTRADAY_DATASET
    return [
        Momentum(lookback=settings.intraday_momentum_lookback, dataset=ds),
        MeanReversion(lookback=settings.intraday_meanrev_lookback, dataset=ds),
        Volatility(window=settings.intraday_vol_window, dataset=ds),
    ]


def _latest_prices(store: BitemporalStore, as_of: datetime) -> dict[str, float]:
    panel = StorePriceView(store, as_of, dataset=PRICE_INTRADAY_DATASET).price_panel()
    if panel.empty:
        return {}
    last = panel.ffill().iloc[-1]
    return {str(s): float(v) for s, v in last.items() if pd.notna(v)}


def run_intraday_cycle(
    settings: Settings,
    *,
    watchlist: list[str] | None = None,
    store: BitemporalStore | None = None,
    broker: Broker | None = None,
    data_client: object | None = None,
    as_of: datetime | None = None,
    lookback_minutes: int | None = None,
    feature_store: FeatureStore | None = None,
) -> CycleResult:
    """One intraday pass: ingest recent minute bars, score, and rebalance on the broker."""
    watchlist = list(watchlist or resolve_universe(settings.universe))

    # Real network/db clients are only built when not injected (tests inject all three).
    if data_client is None or broker is None:
        if not (settings.alpaca_key_id and settings.alpaca_secret_key):
            raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
        if not settings.alpaca_paper:
            settings.assert_live_allowed()  # the live endpoint requires both live switches
    if store is None:
        from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()  # idempotent
    if data_client is None:
        from market_trader.collectors.alpaca import AlpacaDataClient

        data_client = AlpacaDataClient(
            settings.alpaca_key_id or "", settings.alpaca_secret_key or ""
        )
    if broker is None:
        from market_trader.execution.alpaca import AlpacaBroker

        broker = AlpacaBroker(
            settings.alpaca_key_id or "",
            settings.alpaca_secret_key or "",
            paper=settings.alpaca_paper,
        )

    end = as_of or utcnow()
    lookback = lookback_minutes or settings.intraday_lookback_minutes
    start = end - timedelta(minutes=lookback)
    records = data_client.fetch_intraday_bars(  # type: ignore[attr-defined]
        watchlist,
        start=start,
        end=end,
        timeframe=settings.intraday_timeframe,
        feed=settings.alpaca_data_feed,
    )
    IngestionGateway(store).ingest(intraday_bars_to_observations(records))

    prices = _latest_prices(store, end)
    fs = feature_store or FeatureStore(store, intraday_features(settings))
    return run_paper_cycle(
        store,
        as_of=end,
        symbols=watchlist,
        prices=prices,
        broker=broker,
        settings=settings,
        feature_store=fs,
        llm=None,  # no per-minute LLM brief — far under the daily call budget
        top_quantile=settings.intraday_top_quantile,
        max_positions=settings.max_positions or None,  # 0 -> uncapped
        exit_band_multiple=settings.exit_band_multiple,
        risk_weighting=settings.risk_weighting,
        stop_loss_pct=settings.stop_loss_pct,
    )


def run_trading_loop(
    settings: Settings,
    *,
    market_open: Callable[[], bool] | None = None,
    run_cycle: Callable[[], CycleResult] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stop: Callable[[], bool] | None = None,
    max_iterations: int | None = None,
) -> int:
    """Continuously run intraday cycles while the market is open.

    Builds the long-lived Alpaca + store clients **once** and reuses them, so it
    isn't reconnecting every pass. The market-hours check and the per-pass cycle
    are injectable (and ``sleep``/``stop``/``max_iterations`` make the control
    flow unit-testable with no network, DB, or real waiting). One failing pass is
    logged and the loop keeps going — it never dies on a transient error.

    Returns the number of passes executed.
    """
    if market_open is None or run_cycle is None:
        if not (settings.alpaca_key_id and settings.alpaca_secret_key):
            raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
        if not settings.alpaca_paper:
            settings.assert_live_allowed()
        from market_trader.collectors.alpaca import AlpacaDataClient
        from market_trader.execution.alpaca import AlpacaBroker
        from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()  # idempotent; once at startup, not per pass
        broker = AlpacaBroker(
            settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
        )
        data_client = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
        if market_open is None:
            market_open = broker.is_market_open
        if run_cycle is None:

            def _run_cycle() -> CycleResult:
                return run_intraday_cycle(
                    settings, store=store, broker=broker, data_client=data_client
                )

            run_cycle = _run_cycle

    interval = float(settings.intraday_interval_seconds)
    iterations = 0
    _log.info("intraday_loop_start", interval_seconds=interval, mode=settings.execution_mode)
    while not (stop and stop()) and not (
        max_iterations is not None and iterations >= max_iterations
    ):
        try:
            if market_open():
                result = run_cycle()
                _log.info(
                    "intraday_pass",
                    orders=len(result.orders),
                    targets=len(result.target_weights),
                )
            else:
                _log.info("intraday_idle", reason="market_closed")
        except Exception as exc:  # one bad pass must never kill the loop
            _log.warning("intraday_pass_error", error=str(exc))
        iterations += 1
        if (stop and stop()) or (max_iterations is not None and iterations >= max_iterations):
            break
        sleep_fn(interval)
    _log.info("intraday_loop_stop", iterations=iterations)
    return iterations
