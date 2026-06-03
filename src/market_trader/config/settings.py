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

    # --- Execution & trading safety (PAPER-FIRST; enforced in Phase 8/9) ---
    # `execution_mode` is the single TRADING_MODE control. It defaults to
    # "paper" and is intentionally inconvenient to flip: arming live requires
    # BOTH MT_EXECUTION_MODE=live AND MT_LIVE_TRADING_ENABLED=true, and the
    # execution tier (Phase 8) additionally demands a startup confirmation and
    # a passing paper->live graduation-gate checklist. Nothing here touches a
    # broker until Phase 8 (paper) / Phase 9 (gated live). Defaults are safe.
    execution_mode: Literal["paper", "live"] = "paper"
    live_trading_enabled: bool = False
    live_dry_run: bool = True  # in live mode: compute orders but log-only, submit nothing

    # Guardrail caps (enforced by the risk + execution tiers; see DECISIONS D10).
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_position_weight: float = 0.10
    max_drawdown_halt: float = 0.20
    max_daily_loss: float = 0.02  # fraction of capital
    max_orders_per_interval: int = 50
    capital_ceiling: float = 1000.0  # hard cap on deployable capital; low by default

    # --- Universe & portfolio breadth -----------------------------------
    # `universe` selects what to scan each cycle: "liquid" (broad, ~110 names
    # across all sectors; default), "watchlist" (the 8 megacaps), or a
    # comma-separated custom list. `max_positions` caps how many names the book
    # holds, so breadth yields a diversified portfolio rather than 2-3 megacaps.
    universe: str = "liquid"
    max_positions: int = 20

    # --- Broker (Alpaca; paper-first) -----------------------------------
    # Paper keys from https://app.alpaca.markets/ (Paper). Env-only, never
    # committed. alpaca_paper=true uses the paper endpoints.
    alpaca_key_id: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_paper: bool = True
    # Market-data feed: "iex" is the free feed; the paid consolidated "sip" feed
    # 403s on free plans. Override with MT_ALPACA_DATA_FEED=sip once subscribed.
    alpaca_data_feed: str = "iex"

    # --- Intraday live loop (PAPER; OFF by default) ----------------------
    # The continuous, market-reactive loop only runs when explicitly enabled
    # (MT_INTRADAY_ENABLED=true) AND the market is open. Signals are computed on
    # minute bars, so the lookbacks below count *minutes*. It stays paper-gated
    # exactly like every other execution path.
    intraday_enabled: bool = False
    intraday_timeframe: str = "1Min"
    intraday_interval_seconds: int = 60  # how often the loop wakes during market hours
    intraday_lookback_minutes: int = 180  # minute-bar history fetched each pass
    intraday_top_quantile: float = 0.3
    intraday_momentum_lookback: int = 30
    intraday_meanrev_lookback: int = 10
    intraday_vol_window: int = 30

    # --- Reasoning / LLM (hosted Anthropic API in production; see DECISIONS D12) ---
    # Claude Code is a dev-time tool; the deployed engine calls the hosted API
    # itself, on schedule. The key is a managed, rotatable secret — never committed.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    llm_daily_call_budget: int = 200  # cadence/cost gate; enforced in Phase 2+

    def assert_live_allowed(self) -> None:
        """Fail closed: live order routing requires *both* explicit switches.

        The execution tier (Phase 8) calls this before any real-money path —
        and even then only after the paper->live graduation gates are met and a
        human has confirmed. The system must never flip itself to live.
        """
        if not (self.execution_mode == "live" and self.live_trading_enabled):
            raise RuntimeError(
                "Live trading is disabled. Set MT_EXECUTION_MODE=live and "
                "MT_LIVE_TRADING_ENABLED=true to arm it (intentionally two switches)."
            )


def get_settings() -> Settings:
    """Construct settings from the environment. Call at the edges, not in hot loops."""
    return Settings()
