"""Application settings.

All configuration is externalised and read from the environment (prefix ``MT_``)
or an optional ``.env`` file. **No secrets live in code.**

The execution-safety fields are planted here in Phase 0 even though the execution
tier is built later: the defaults encode the agreed posture — **paper by
default, live trading disabled** — so the safe state is the one you fall into if
nobody touches anything.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "ci", "prod"] = "dev"
    log_level: str = "INFO"
    json_logs: bool = False

    # --- storage ---------------------------------------------------------
    database_url: str = "postgresql+psycopg://market:market@localhost:5432/market_trader"
    # When set (e.g. in CI), integration tests run against this database.
    test_database_url: str | None = None

    # --- execution safety (enforced in Phase 5; defaults are the safe state) ---
    execution_mode: Literal["paper", "live"] = "paper"
    live_trading_enabled: bool = False
    max_gross_exposure: float = 1.0
    max_position_weight: float = 0.10
    max_drawdown_halt: float = 0.20

    def assert_live_allowed(self) -> None:
        """Fail closed: live order routing requires *both* explicit switches.

        The execution tier (Phase 5) calls this before any real-money path. It
        exists now so the invariant is impossible to forget later.
        """
        if not (self.execution_mode == "live" and self.live_trading_enabled):
            raise RuntimeError(
                "Live trading is disabled. Set MT_EXECUTION_MODE=live and "
                "MT_LIVE_TRADING_ENABLED=true to arm it (intentionally two switches)."
            )


def get_settings() -> Settings:
    """Construct settings from the environment. Call at the edges, not in hot loops."""
    return Settings()
