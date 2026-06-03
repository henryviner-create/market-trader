"""Externalised configuration (Pydantic settings). Secrets come from the env."""

from market_trader.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
