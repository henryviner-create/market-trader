"""Observability primitives. Structured logging now; metrics/alerts in Phase 7."""

from market_trader.observability.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
