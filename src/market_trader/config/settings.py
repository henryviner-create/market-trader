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
    # Daily-loss kill switch: halt (and require a human re-arm) once equity falls
    # this far below the previous close. 0 = off. Now ENFORCED in the engine — it
    # was previously inert. Set it deliberately for an aggressive book (e.g. 0.08).
    max_daily_loss: float = 0.0
    max_orders_per_interval: int = 50
    capital_ceiling: float = 1000.0  # hard cap on deployable capital; low by default
    # Order in whole shares (default) rather than fractional. Many small/mid-cap names are
    # NOT fractionable on the broker, and a fractional order on one is rejected (HTTP 403
    # "not fractionable"), which aborts the whole rebalance. Whole-share rounding sidesteps
    # that; the sub-share remainder is immaterial on a diversified book. Set
    # MT_FRACTIONAL_SHARES=true only on a universe you know is fully fractionable.
    fractional_shares: bool = False
    # No-trade band (turnover control). Skip rebalancing an existing position when the
    # adjustment is smaller than this fraction of its target — so a daily book doesn't churn
    # (and pay costs) on small drifts, especially the vol-governor's per-cycle rescaling. A
    # full entry (no current position) or exit (target 0) always executes. 0 = off (rebalance
    # to the exact target every cycle). 0.2 is a sane starting band for a daily book.
    rebalance_band: float = 0.0
    # Risk-based sizing (the drawdown governor; see portfolio/sizing.py). The DD cap is
    # translated into a portfolio volatility target (halved for fat tails); a fractional
    # -Kelly tilt sets relative conviction; regime_derisk_factor shrinks the target in
    # risk-off. Inert until the cycle routes sizing through `size_book` (a later phase).
    target_vol: float = 0.10  # annualized portfolio volatility target
    kelly_fraction: float = 0.25  # fraction of full Kelly (heavy haircut; noisy mu)
    regime_derisk_factor: float = 0.5  # multiply target_vol/net by this when risk-off

    # --- Universe & portfolio breadth -----------------------------------
    # `universe` selects what to scan each cycle: "liquid" (broad, ~110 names
    # across all sectors; default), "watchlist" (the 8 megacaps), or a
    # comma-separated custom list. `max_positions` caps how many names the book
    # holds, so breadth yields a diversified portfolio rather than 2-3 megacaps.
    # Set max_positions=0 to remove the cap entirely (hold everything selected).
    universe: str = "liquid"
    max_positions: int = 20
    # Entry breadth: the top `top_quantile` of the ranked universe are eligible,
    # then capped at max_positions. Raise it to fill a larger cap from a big
    # universe (e.g. 0.4 of a ~140-name "global" set ~= 56 candidates -> up to 50).
    top_quantile: float = 0.3
    # Ranking model: "composite" (transparent equal-weight z-score; default) or
    # "forecast" (the trained, calibrated ensemble). Keep it on "composite" until
    # the forecaster clears the equal-weight baseline out-of-sample (the
    # `validate-forecaster` command measures this). Forecast = the daily cycle only.
    scorer: str = "composite"
    # Close the learning loop: weight the composite's signals by their *measured*
    # IC (from graded past predictions) instead of equal-weighting, and drop
    # signals whose |IC| has decayed below ic_min_abs. Sign-aware, so a signal
    # that predicts the wrong way is inverted, not blindly trusted. Falls back to
    # equal weights until there is enough graded out-of-sample data, so it is a
    # no-op on a cold start. Composite scorer only.
    ic_weighting: bool = True
    ic_min_abs: float = 0.02
    # Orthogonalize the composite: down-weight signals that are redundant (correlated with
    # each other across the cross-section) so a cluster of similar signals can't dominate the
    # score (Grinold "combining alphas"). Composite/IC-weighted scorer only; OFF by default so
    # the graded prediction history is unchanged. Harmless on the governed-1/N book (which
    # ignores scores) -- enable it (MT_ORTHOGONALIZE_SIGNALS=true) alongside a score tilt.
    orthogonalize_signals: bool = False
    # Holding discipline + sizing. exit_band_multiple keeps a held name until it
    # leaves the top (entry_count * multiple) — so the book holds winners instead
    # of churning on rank noise (your "it sells too quickly"). risk_weighting:
    # "inverse_vol" sizes each name to ~equal risk; "equal" splits evenly;
    # "conviction" bets more on the strongest signals (aggressive offense);
    # "size_book" is the unified chassis (see tilt_strength below).
    exit_band_multiple: float = 2.0
    risk_weighting: str = "inverse_vol"
    # The unified chassis (portfolio.sizing.size_book). When risk_weighting="size_book"
    # the cycle ignores top-N selection and holds the *whole* scored universe, vol-governed
    # to target_vol and tilted toward higher scores by tilt_strength. tilt_strength=0 is
    # governed equal-weight (1/N) — the validated, mandate-compliant book we deploy first;
    # a signal that later earns its place out-of-sample gets a small positive tilt here.
    tilt_strength: float = 0.0
    # Per-name hard stop: flatten a holding once it is this far below its entry
    # price, regardless of how good its signal still looks — the trader's "cut
    # your losers" rail. 0 = off. Complements (does not replace) the rank
    # hysteresis: hysteresis is relative, this is an absolute loss floor.
    stop_loss_pct: float = 0.0
    # News signal (daily cycle only; OFF by default). When on, the cycle pulls
    # recent GDELT articles for the universe and adds news-flow + sentiment
    # features to the ranking. Per-symbol fetch is heavy, so it's daily, not
    # intraday. Like every signal, it must earn its keep out-of-sample.
    news_enabled: bool = False
    news_window_days: int = 7
    news_timespan: str = "3d"
    # Bound the per-cycle GDELT sweep so a slow/throttling free API can never stall
    # a trading cycle: each request times out fast, and the whole per-symbol sweep
    # is capped by a wall-clock budget (remaining names are skipped that cycle).
    news_fetch_timeout_seconds: float = 10.0
    news_fetch_budget_seconds: float = 45.0

    # SEC EDGAR insider (Form-4) flow signal. The InsiderNetBuys feature is already
    # in default_features(); this fetches the data that feeds it. SEC requires a
    # descriptive User-Agent — set MT_SEC_USER_AGENT to "Your Name your@email". OFF
    # by default; bounded like the news sweep so a slow SEC endpoint can't stall.
    insider_enabled: bool = False
    sec_user_agent: str = "market-trader research contact@example.com"
    insider_fetch_timeout_seconds: float = 15.0
    insider_fetch_budget_seconds: float = 90.0
    insider_lookback_days: int = 400

    # Massive market-data (Polygon-compatible REST). An optional, cleaner EOD price source
    # than the free IEX feed (which silently drops thin small-caps). Free tier = EOD US
    # equities, 5 calls/min, ~2y history — enough for the daily book's coverage. Key is
    # env-only (MT_MASSIVE_API_KEY); base URL is overridable to match the account's paths.
    # Used only by the opt-in `ingest-prices-massive` backfill, never automatically.
    massive_api_key: str | None = None
    massive_base_url: str = "https://api.massive.com"

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
    # How often the loop wakes during market hours. 5 min gives a broad-universe
    # pass time to finish before the next one starts (60s could not keep up), and
    # the signals are daily-horizon so nothing is lost by waking less often.
    intraday_interval_seconds: int = 300
    intraday_lookback_minutes: int = 180  # minute-bar history fetched each pass
    intraday_top_quantile: float = 0.3
    intraday_momentum_lookback: int = 30
    intraday_meanrev_lookback: int = 10
    intraday_vol_window: int = 30

    # --- Daily scheduler (PAPER; OFF by default) -------------------------
    # Runs the end-of-day cycle automatically once per *trading day*, triggered by
    # the market open->closed transition (so weekends/holidays are skipped for
    # free). This is what feeds the learning loop: every session logs predictions
    # that grade themselves once their horizon elapses. Enable with
    # MT_DAILY_CYCLE_ENABLED=true; it runs inside `serve` alongside the health
    # server. Prefer this over the intraday loop for hands-off operation.
    daily_cycle_enabled: bool = False
    daily_cycle_poll_seconds: int = 300  # how often to check for the close

    # --- Event-driven news sleeve (PAPER; OFF by default) ----------------
    # A selective, event-triggered overlay: when a name gets *material* news, open
    # a small, time-boxed position to ride the post-news drift, then exit. Runs in
    # `serve` when MT_NEWS_SLEEVE_ENABLED=true. Non-churning by design — it acts
    # only on a fresh, deduped story, then holds for a fixed window. It coexists
    # with the daily book via a reserved capital budget (the daily cycle leaves
    # sleeve-owned names alone). GDELT feed (~15-min latency, so it trades drift,
    # not the instant move); the feed is pluggable for a real-time provider later.
    news_sleeve_enabled: bool = False
    news_sleeve_budget: float = 0.10  # fraction of gross reserved for the sleeve
    news_sleeve_max_names: int = 5  # cap on concurrent sleeve positions
    news_sleeve_interval_seconds: int = 300
    news_sleeve_hold_days: int = 5  # time-boxed drift window before exit
    news_sleeve_cooldown_days: int = 3  # min gap between trades on one name
    news_sleeve_min_confidence: float = 0.5
    news_sleeve_lookback_minutes: int = 120  # news poll window per pass
    # Event detection (what counts as "material" news):
    news_sleeve_count_surge: float = 3.0  # recent count >= this x trailing daily mean
    news_sleeve_tone_min: float = 1.5  # min |mean tone| to assign a direction
    news_sleeve_baseline_days: int = 14  # trailing window for the count baseline

    # --- Reasoning / LLM (hosted Anthropic API in production; see DECISIONS D12) ---
    # Claude Code is a dev-time tool; the deployed engine calls the hosted API
    # itself, on schedule. The key is a managed, rotatable secret — never committed.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"
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
