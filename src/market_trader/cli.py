"""The ``market-trader`` command-line entrypoint.

Stdlib-only (no new dependencies). Subcommands:

* ``version``      — print the package version.
* ``migrate``      — apply Alembic migrations (run before serving).
* ``serve``        — run the long-running engine process. For now this is a
  minimal health server that keeps the container alive and exposes ``/health``;
  the ingestion/signal/execution loops attach here in later phases. On SIGTERM it
  **stands down cleanly** (the dead-man's-switch posture: cease, don't fire blind).
* ``healthcheck``  — used by Docker's HEALTHCHECK; probes the running ``serve``
  endpoint, falling back to a direct database ping.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import create_engine, text

from market_trader import __version__
from market_trader.config import get_settings
from market_trader.observability import configure_logging, get_logger
from market_trader.observability.metrics import default_registry

if TYPE_CHECKING:
    from market_trader.execution.broker import Account, Position
    from market_trader.runtime import CycleResult


def check_database(url: str, *, timeout: float = 5.0) -> bool:
    """Return True if a trivial query succeeds against ``url``."""
    connect_args: dict[str, Any] = {}
    if url.startswith("postgresql"):
        connect_args["connect_timeout"] = int(timeout)
    try:
        engine = create_engine(url, connect_args=connect_args)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def health_payload(url: str) -> dict[str, Any]:
    db_ok = check_database(url)
    return {"status": "ok" if db_ok else "degraded", "db": db_ok, "version": __version__}


class _HealthServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], db_url: str) -> None:
        super().__init__(address, _HealthHandler)
        self.db_url = db_url


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path in ("", "/health"):
            payload = health_payload(cast(_HealthServer, self.server).db_url)
            self._respond(200 if payload["db"] else 503, "application/json", json.dumps(payload))
        elif path == "/metrics":
            self._respond(200, "text/plain; version=0.0.4", default_registry().render())
        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, status: int, content_type: str, text: str) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # keep the health endpoint quiet
        return


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_migrate(_: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    return 0


def _maybe_start_intraday_loop(settings: Any, log: Any) -> None:
    """Attach the continuous intraday trading loop as a daemon thread, if armed.

    OFF unless ``MT_INTRADAY_ENABLED=true`` and Alpaca keys are present. Runs
    alongside the health server; a crash is logged and never takes the process
    (and thus the health endpoint) down with it.
    """
    if not settings.intraday_enabled:
        return
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        log.warning("intraday_disabled", reason="alpaca keys not set")
        return

    import threading

    from market_trader.runtime import run_trading_loop

    def _loop() -> None:
        try:
            run_trading_loop(settings)
        except Exception as exc:  # surface; keep the health server alive
            log.error("intraday_loop_crashed", error=str(exc))

    threading.Thread(target=_loop, name="intraday-loop", daemon=True).start()
    log.info("intraday_loop_started", interval_seconds=settings.intraday_interval_seconds)


def _maybe_start_daily_schedule(settings: Any, log: Any) -> None:
    """Attach the once-per-trading-day cycle as a daemon thread, if armed.

    OFF unless ``MT_DAILY_CYCLE_ENABLED=true`` and Alpaca keys are present. This is
    the hands-off path: it fires the end-of-day cycle on the market close, feeding
    the learning loop. A crash is logged and never takes the health endpoint down.
    """
    if not settings.daily_cycle_enabled:
        return
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        log.warning("daily_schedule_disabled", reason="alpaca keys not set")
        return

    import threading

    from market_trader.runtime import run_daily_schedule

    def _loop() -> None:
        try:
            run_daily_schedule(settings)
        except Exception as exc:  # surface; keep the health server alive
            log.error("daily_schedule_crashed", error=str(exc))

    threading.Thread(target=_loop, name="daily-schedule", daemon=True).start()
    log.info("daily_schedule_started", poll_seconds=settings.daily_cycle_poll_seconds)


def _maybe_start_news_sleeve_loop(settings: Any, log: Any) -> None:
    """Attach the event-driven news sleeve as a daemon thread, if armed.

    OFF unless ``MT_NEWS_SLEEVE_ENABLED=true`` and Alpaca keys are present. It acts
    only on fresh, material news (non-churning) and goes through ExecutionEngine,
    so it inherits every rail. A crash never takes the health endpoint down.
    """
    if not settings.news_sleeve_enabled:
        return
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        log.warning("news_sleeve_disabled", reason="alpaca keys not set")
        return

    import threading

    from market_trader.runtime.news_sleeve import run_news_sleeve_loop

    def _loop() -> None:
        try:
            run_news_sleeve_loop(settings)
        except Exception as exc:  # surface; keep the health server alive
            log.error("news_sleeve_loop_crashed", error=str(exc))

    threading.Thread(target=_loop, name="news-sleeve", daemon=True).start()
    log.info("news_sleeve_started", interval_seconds=settings.news_sleeve_interval_seconds)


def cmd_serve(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)
    default_registry().gauge("mt_up", "engine process up (1)").set(1.0)
    log = get_logger("engine")
    server = _HealthServer(("0.0.0.0", args.port), settings.database_url)

    def _stand_down(signum: int, _frame: FrameType | None) -> None:
        log.warning("standing_down", signal=signum)  # cease; do not fire blind
        server.shutdown()

    signal.signal(signal.SIGTERM, _stand_down)
    signal.signal(signal.SIGINT, _stand_down)

    _maybe_start_intraday_loop(settings, log)
    _maybe_start_daily_schedule(settings, log)
    _maybe_start_news_sleeve_loop(settings, log)

    log.info(
        "engine_serving",
        mode=settings.execution_mode,
        live_enabled=settings.live_trading_enabled,
        intraday=settings.intraday_enabled,
        daily_cycle=settings.daily_cycle_enabled,
        news_sleeve=settings.news_sleeve_enabled,
        port=args.port,
    )
    server.serve_forever()
    log.info("engine_stopped")
    return 0


def cmd_healthcheck(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/health", timeout=args.timeout
        ) as r:
            ok = r.status == 200
    except Exception:
        ok = check_database(settings.database_url)
    print("ok" if ok else "unhealthy")
    return 0 if ok else 1


def cmd_alpaca_check(_: argparse.Namespace) -> int:
    """Probe the Alpaca (paper) account — confirms keys + connectivity, places nothing."""
    settings = get_settings()
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        print("alpaca: keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
        return 1
    from market_trader.execution.alpaca import AlpacaBroker

    try:
        broker = AlpacaBroker(
            settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
        )
        account = broker.get_account()
    except Exception as exc:  # surface, never swallow
        print(f"alpaca: request failed: {exc}")
        return 1
    endpoint = "paper" if settings.alpaca_paper else "LIVE"
    print(
        f"alpaca ok [{endpoint}]: equity=${account.equity:,.2f} "
        f"cash=${account.cash:,.2f} buying_power=${account.buying_power:,.2f}"
    )
    return 0


def _portfolio_summary(account: Account, positions: list[Position]) -> str:
    """Plain-English P&L snapshot from an account and its open positions."""
    deployed = sum(p.market_value for p in positions)
    unrealized = sum(p.unrealized_pl for p in positions)
    lines = [
        f"portfolio  equity=${account.equity:,.2f}  cash=${account.cash:,.2f}  "
        f"deployed=${deployed:,.2f} across {len(positions)} position(s)"
    ]
    if account.last_equity:
        day_pl = account.equity - account.last_equity
        day_pct = day_pl / account.last_equity * 100
        lines.append(f"  today:          {day_pl:+,.2f} ({day_pct:+.2f}%)")
    lines.append(f"  unrealized P&L: {unrealized:+,.2f}")
    ranked = sorted(positions, key=lambda p: p.unrealized_pl)
    if ranked:
        worst = "  ".join(f"{p.symbol} {p.unrealized_pl:+,.0f}" for p in ranked[:3])
        best = "  ".join(f"{p.symbol} {p.unrealized_pl:+,.0f}" for p in ranked[-3:][::-1])
        lines.append(f"  worst: {worst}")
        lines.append(f"  best:  {best}")
    return "\n".join(lines)


def cmd_status(_: argparse.Namespace) -> int:
    """Read-only portfolio snapshot: equity, today's P&L, deployed cash, win/lose names."""
    settings = get_settings()
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        print("status: keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
        return 1
    from market_trader.execution.alpaca import AlpacaBroker

    try:
        broker = AlpacaBroker(
            settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
        )
        account = broker.get_account()
        positions = broker.get_positions()
    except Exception as exc:  # surface, never swallow
        print(f"status: request failed: {exc}")
        return 1
    print(_portfolio_summary(account, positions))
    return 0


def cmd_llm_check(_: argparse.Namespace) -> int:
    """Probe the hosted Anthropic API — one tiny round-trip to confirm the key works."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("llm: MT_ANTHROPIC_API_KEY not set")
        return 1
    from market_trader.reasoning import anthropic_provider_from_settings

    try:
        provider = anthropic_provider_from_settings(settings)
        reply = provider.complete(
            system="You are a terse connectivity check.",
            prompt="Reply with exactly: OK",
            max_tokens=16,
        )
    except Exception as exc:
        print(f"llm: request failed: {exc}")
        return 1
    print(f"llm ok [{settings.anthropic_model}]: {reply.strip()[:80]}")
    return 0


def cmd_validate_forecaster(_: argparse.Namespace) -> int:
    """Out-of-sample gate: does the trained forecaster beat the equal-weight baseline?

    Ingests daily history for the universe and reports purged-CV AUC for both.
    Only flip MT_SCORER=forecast if the forecaster wins here.
    """
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        print("validate-forecaster: Alpaca keys not set")
        return 1

    from datetime import date, timedelta

    from market_trader.collectors import IngestionGateway, PriceCollector
    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.core.time import utcnow
    from market_trader.runtime.scoring import forecaster_vs_baseline_auc
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore
    from market_trader.universe.liquid import resolve_universe

    universe = resolve_universe(settings.universe)
    try:
        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        end = date.today()
        data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
        records = data.fetch_daily_bars(
            universe, start=end - timedelta(days=400), end=end, feed=settings.alpaca_data_feed
        )
        IngestionGateway(store).ingest(PriceCollector().normalize(records))
        res = forecaster_vs_baseline_auc(store, universe, utcnow())
    except Exception as exc:
        print(f"validate-forecaster failed: {exc}")
        return 1

    fc, bl = res["forecast_cv_auc"], res["baseline_auc"]
    print(f"validate-forecaster [{int(res['n_samples'])} samples, {len(universe)} names]")
    print(f"  forecast  CV AUC: {fc:.4f}")
    print(f"  baseline     AUC: {bl:.4f}")
    beats = fc > bl
    print(
        f"  verdict: forecaster {'BEATS' if beats else 'does NOT beat'} the baseline "
        f"-> {'MT_SCORER=forecast is justified' if beats else 'keep MT_SCORER=composite'}"
    )
    return 0


def cmd_score_predictions(_: argparse.Namespace) -> int:
    """Grade logged predictions against realised outcomes; flag decayed signals."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        print("score-predictions: Alpaca keys not set")
        return 1

    from datetime import date, timedelta

    from market_trader.collectors import IngestionGateway, PriceCollector
    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.core.time import utcnow
    from market_trader.runtime.learning import grade_predictions
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore
    from market_trader.universe.liquid import resolve_universe

    try:
        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        end = date.today()
        data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
        records = data.fetch_daily_bars(
            resolve_universe(settings.universe),
            start=end - timedelta(days=60),
            end=end,
            feed=settings.alpaca_data_feed,
        )
        IngestionGateway(store).ingest(PriceCollector().normalize(records))
        res = grade_predictions(store, utcnow())
    except Exception as exc:
        print(f"score-predictions failed: {exc}")
        return 1

    if res["n"] == 0:
        print("score-predictions: nothing ready to grade yet (run cycles, then wait the horizon)")
        return 0
    print(f"score-predictions [{res['n']} graded]")
    print(f"  brier:    {res['brier']:.4f}  (lower better; 0.25 = coin-flip)")
    print(f"  hit_rate: {res['hit_rate']:.2%}")
    print("  signal IC (vs forward return):")
    for name, val in sorted(res["ic"].items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"    {name:26} {val:+.3f}")
    if res["pruned"]:
        print(f"  decayed -> consider pruning: {', '.join(res['pruned'])}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Self-evaluation: attribute graded decisions to signals + regimes; optional reflection."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        print("evaluate: Alpaca keys not set")
        return 1

    from datetime import date, timedelta

    from market_trader.collectors import IngestionGateway, PriceCollector
    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.core.time import utcnow
    from market_trader.runtime.evaluation import (
        attribute_performance,
        build_trade_journal,
        evaluation_summary_markdown,
        reflect,
    )
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore
    from market_trader.universe.liquid import resolve_universe

    try:
        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        end = date.today()
        data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
        records = data.fetch_daily_bars(
            resolve_universe(settings.universe),
            start=end - timedelta(days=60),
            end=end,
            feed=settings.alpaca_data_feed,
        )
        IngestionGateway(store).ingest(PriceCollector().normalize(records))
        journal = build_trade_journal(store, utcnow(), model_version=args.model)
        report = attribute_performance(journal)
    except Exception as exc:
        print(f"evaluate failed: {exc}")
        return 1

    if args.reflect and settings.anthropic_api_key:
        from market_trader.reasoning import anthropic_provider_from_settings

        print(reflect(report, journal, anthropic_provider_from_settings(settings)))
    else:
        print(evaluation_summary_markdown(report))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Backtest the live strategy over history vs baselines, with a Monte-Carlo downside."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        print("simulate: Alpaca keys not set")
        return 1

    from datetime import date, timedelta

    import pandas as pd

    from market_trader.backtest.engine import buy_and_hold_summary, run_backtest
    from market_trader.backtest.pit import observations_to_price_frame
    from market_trader.backtest.simulation import monte_carlo_report
    from market_trader.backtest.strategies import CompositeBacktestStrategy, EqualWeightStrategy
    from market_trader.collectors import IngestionGateway, PriceCollector
    from market_trader.collectors.alpaca import AlpacaDataClient
    from market_trader.core.synthetic import PRICE_DATASET
    from market_trader.core.time import DISTANT_FUTURE
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore
    from market_trader.universe.liquid import resolve_universe

    try:
        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        end = date.today()
        data = AlpacaDataClient(settings.alpaca_key_id, settings.alpaca_secret_key)
        records = data.fetch_daily_bars(
            resolve_universe(settings.universe),
            start=end - timedelta(days=args.days),
            end=end,
            feed=settings.alpaca_data_feed,
        )
        IngestionGateway(store).ingest(PriceCollector().normalize(records))
        panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET))
        schedule = [ts.to_pydatetime() for ts in pd.DatetimeIndex(panel.index)][60::5]
        if len(schedule) < 5:
            print("simulate: not enough price history (try a larger --days)")
            return 1
        strategy = CompositeBacktestStrategy(max_positions=settings.max_positions or 20)
        result = run_backtest(store, strategy, schedule)  # the candidate, once
        equal_weight = run_backtest(store, EqualWeightStrategy(), schedule)
        summaries = {
            strategy.name: result.summary,
            "equal_weight": equal_weight.summary,
            "buy_and_hold": buy_and_hold_summary(store, start_after=schedule[0]),
        }
        sim = monte_carlo_report(result.net_returns.to_numpy(dtype=float))
    except Exception as exc:
        print(f"simulate failed: {exc}")
        return 1

    print(f"simulate [composite, {len(schedule)} rebalances over ~{args.days}d, net of costs]")
    for name, s in summaries.items():
        print(
            f"  {name:14} ann_return={s.ann_return:+.1%}  sharpe={s.sharpe:+.2f}  "
            f"max_dd={s.max_drawdown:.1%}  hit={s.hit_rate:.0%}"
        )
    print(
        f"  Monte-Carlo ({sim.n_sims} paths): total return q05/q50/q95 = "
        f"{sim.total_return_q05:+.1%} / {sim.total_return_q50:+.1%} / {sim.total_return_q95:+.1%}"
    )
    print(
        f"    worst-5% drawdown {sim.max_drawdown_q05:.1%}; median Sharpe "
        f"{sim.sharpe_q50:+.2f}; P(profit) {sim.prob_positive:.0%}"
    )
    return 0


def cmd_ingest_filings(args: argparse.Namespace) -> int:
    """Backfill SEC Form-4 insider filings for the universe into the store."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)

    from market_trader.collectors import IngestionGateway
    from market_trader.collectors.edgar import EdgarClient, Form4Collector
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore
    from market_trader.universe.liquid import resolve_universe

    try:
        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        client = EdgarClient(
            user_agent=settings.sec_user_agent,
            timeout_seconds=settings.insider_fetch_timeout_seconds,
            budget_seconds=float(args.budget),
        )
        records = client.fetch_for_symbols(
            resolve_universe(settings.universe), lookback_days=args.days
        )
        observations = Form4Collector().normalize(records)
        IngestionGateway(store).ingest(observations)
    except Exception as exc:
        print(f"ingest-filings failed: {exc}")
        return 1
    print(
        f"ingest-filings: {len(records)} Form-4 records over ~{args.days}d "
        f"-> {len(observations)} observations ingested"
    )
    return 0


def cmd_signal_ic(args: argparse.Namespace) -> int:
    """Measure each signal's out-of-sample information coefficient (IC) over history."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)

    from market_trader.core.time import utcnow
    from market_trader.features import FeatureStore, default_features
    from market_trader.runtime.signal_ic import measure_signal_ic
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore
    from market_trader.universe.liquid import resolve_universe

    try:
        store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
        store.create_schema()
        fs = FeatureStore(store, default_features())
        ics = measure_signal_ic(
            store,
            fs,
            resolve_universe(settings.universe),
            utcnow(),
            horizon_days=args.horizon,
            max_dates=args.dates,
        )
    except Exception as exc:
        print(f"signal-ic failed: {exc}")
        return 1
    if not ics:
        print(
            "signal-ic: no gradable history — backfill prices/filings first (simulate, ingest-filings)"
        )
        return 0
    print(f"signal-ic [horizon={args.horizon}d, per-date cross-sectional rank IC]")
    for sig, r in sorted(ics.items(), key=lambda kv: abs(kv[1].mean_ic), reverse=True):
        flag = "  <- significant" if abs(r.ic_t_stat) >= 2.0 else ""
        print(
            f"  {sig:24} IC={r.mean_ic:+.4f}  t={r.ic_t_stat:+.2f}  "
            f"hit={r.hit_rate:.0%}  n_dates={r.n_dates}{flag}"
        )
    return 0


def _print_cycle(result: CycleResult, *, dry: bool) -> None:
    tag = "dry-run" if dry else "live-paper"
    print(f"cycle {result.as_of.isoformat()}  [{tag}]")

    top = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("  scores (top): " + ("  ".join(f"{s}={v:+.2f}" for s, v in top) or "(none)"))

    if result.target_weights:
        weights = "  ".join(f"{s}={w:.2%}" for s, w in sorted(result.target_weights.items()))
        print(f"  targets: {weights}")
    else:
        print("  targets: (none — nothing scored above threshold)")

    if result.orders:
        print(f"  orders ({len(result.orders)}):")
        for o in result.orders:
            fill = f" @ {o.filled_avg_price:.2f}" if o.filled_avg_price is not None else ""
            print(
                f"    {o.side.value.upper():4} {o.symbol:6} "
                f"qty={o.qty:.4f}{fill}  [{o.status.value}]"
            )
    else:
        print("  orders: (none — already at target / dust-only)")

    if result.brief:
        first = next((ln for ln in result.brief.strip().splitlines() if ln.strip()), "")
        print(f"  brief: {first[:120]}")
    else:
        print("  brief: (no LLM configured — set MT_ANTHROPIC_API_KEY for a narrated brief)")


def cmd_cycle(args: argparse.Namespace) -> int:
    """Run one end-to-end paper cycle: score -> risk -> paper execution -> brief."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)

    if args.intraday:
        from market_trader.runtime import run_intraday_cycle

        try:
            result = run_intraday_cycle(settings)
        except Exception as exc:
            print(f"cycle failed: {exc}")
            return 1
        _print_cycle(result, dry=False)
        return 0

    from market_trader.runtime import run_dry_paper_cycle, run_live_paper_cycle

    try:
        result = run_dry_paper_cycle(settings) if args.dry_run else run_live_paper_cycle(settings)
    except Exception as exc:
        print(f"cycle failed: {exc}")
        return 1
    _print_cycle(result, dry=args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-trader", description="market-trader engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the version").set_defaults(func=cmd_version)
    sub.add_parser("migrate", help="apply database migrations").set_defaults(func=cmd_migrate)

    default_port = int(os.environ.get("MT_HEALTH_PORT", "8080"))
    serve = sub.add_parser("serve", help="run the engine (health server)")
    serve.add_argument("--port", type=int, default=default_port)
    serve.set_defaults(func=cmd_serve)

    health = sub.add_parser("healthcheck", help="probe the running engine")
    health.add_argument("--port", type=int, default=default_port)
    health.add_argument("--timeout", type=float, default=5.0)
    health.set_defaults(func=cmd_healthcheck)

    sub.add_parser(
        "status", help="read-only portfolio P&L snapshot (equity, today, win/lose)"
    ).set_defaults(func=cmd_status)
    sub.add_parser("alpaca-check", help="probe the Alpaca paper account").set_defaults(
        func=cmd_alpaca_check
    )
    sub.add_parser("llm-check", help="probe the hosted Anthropic API").set_defaults(
        func=cmd_llm_check
    )
    sub.add_parser(
        "validate-forecaster", help="out-of-sample AUC: forecaster vs the baseline"
    ).set_defaults(func=cmd_validate_forecaster)
    sub.add_parser(
        "score-predictions", help="grade logged predictions vs realised outcomes"
    ).set_defaults(func=cmd_score_predictions)
    evaluate = sub.add_parser(
        "evaluate", help="attribute graded decisions to signals + regimes (self-evaluation)"
    )
    evaluate.add_argument(
        "--model", default="composite", help="model_version (e.g. composite, news_sleeve)"
    )
    evaluate.add_argument("--reflect", action="store_true", help="add an LLM post-mortem")
    evaluate.set_defaults(func=cmd_evaluate)
    simulate = sub.add_parser(
        "simulate", help="backtest the strategy over history vs baselines + Monte-Carlo downside"
    )
    simulate.add_argument("--days", type=int, default=500, help="history window in days")
    simulate.set_defaults(func=cmd_simulate)

    ingest_filings = sub.add_parser(
        "ingest-filings", help="backfill SEC Form-4 insider filings for the universe"
    )
    ingest_filings.add_argument("--days", type=int, default=1095, help="lookback window in days")
    ingest_filings.add_argument(
        "--budget", type=float, default=600.0, help="wall-clock fetch budget (seconds)"
    )
    ingest_filings.set_defaults(func=cmd_ingest_filings)

    signal_ic = sub.add_parser(
        "signal-ic", help="measure each signal's out-of-sample IC over history"
    )
    signal_ic.add_argument("--horizon", type=int, default=5, help="forward-return horizon (days)")
    signal_ic.add_argument("--dates", type=int, default=120, help="max decision dates to sample")
    signal_ic.set_defaults(func=cmd_signal_ic)

    cycle = sub.add_parser("cycle", help="run one paper trading cycle")
    cycle.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic data + local paper broker (no network or keys needed)",
    )
    cycle.add_argument(
        "--intraday",
        action="store_true",
        help="one intraday (minute-bar) cycle instead of the daily one",
    )
    cycle.set_defaults(func=cmd_cycle)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
