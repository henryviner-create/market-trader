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

    log.info(
        "engine_serving",
        mode=settings.execution_mode,
        live_enabled=settings.live_trading_enabled,
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

    sub.add_parser("alpaca-check", help="probe the Alpaca paper account").set_defaults(
        func=cmd_alpaca_check
    )
    sub.add_parser("llm-check", help="probe the hosted Anthropic API").set_defaults(
        func=cmd_llm_check
    )

    cycle = sub.add_parser("cycle", help="run one paper trading cycle")
    cycle.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic data + local paper broker (no network or keys needed)",
    )
    cycle.set_defaults(func=cmd_cycle)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
