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
from typing import Any, cast

from sqlalchemy import create_engine, text

from market_trader import __version__
from market_trader.config import get_settings
from market_trader.observability import configure_logging, get_logger


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
        if self.path.rstrip("/") in ("", "/health"):
            payload = health_payload(cast(_HealthServer, self.server).db_url)
            body = json.dumps(payload).encode()
            self.send_response(200 if payload["db"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
