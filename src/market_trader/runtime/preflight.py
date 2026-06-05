"""Preflight config-doctor — catch a misconfigured engine *before* it trades.

The live cycle has several settings that fail *silently* into the wrong behaviour:
``capital_ceiling`` far below equity deploys a token book; ``risk_weighting`` other
than the validated chassis trades a different strategy; ``target_vol`` above the
drawdown budget breaches the mandate at the tail; ``daily_cycle_enabled=false`` means
``serve`` never trades at all. ``preflight_checks`` is a pure function over the
settings (plus a little injected live state) returning a structured report, so the CLI
can warn loudly and a test can assert the rules without a broker or a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from market_trader.config import Settings


class Level(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    level: Level
    name: str
    detail: str


def preflight_checks(
    settings: Settings,
    *,
    equity: float | None,
    db_ok: bool,
    universe_size: int,
) -> list[Check]:
    """Audit the live configuration. ``equity`` is the broker account equity (``None`` if
    keys are unset / unreachable); ``db_ok`` is a database ping; ``universe_size`` is the
    resolved universe length. Returns one :class:`Check` per rule, worst-first by level."""
    checks: list[Check] = []

    # --- connectivity ---------------------------------------------------
    if settings.alpaca_key_id and settings.alpaca_secret_key:
        checks.append(Check(Level.OK, "alpaca_keys", "Alpaca keys are set"))
    else:
        checks.append(
            Check(Level.FAIL, "alpaca_keys", "MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY not set")
        )
    checks.append(
        Check(
            Level.OK if db_ok else Level.FAIL, "database", "reachable" if db_ok else "unreachable"
        )
    )
    if universe_size <= 0:
        checks.append(Check(Level.FAIL, "universe", "resolves to 0 names (check MT_UNIVERSE)"))
    elif universe_size < 10:
        checks.append(Check(Level.WARN, "universe", f"only {universe_size} names — thin breadth"))
    else:
        checks.append(Check(Level.OK, "universe", f"{universe_size} names"))

    # --- capital deployment --------------------------------------------
    if equity is not None and equity > 0:
        frac = settings.capital_ceiling / equity
        if frac < 0.5:
            checks.append(
                Check(
                    Level.WARN,
                    "capital_ceiling",
                    f"deploys only ${settings.capital_ceiling:,.0f} of ${equity:,.0f} equity "
                    f"({frac:.0%}) — raise MT_CAPITAL_CEILING to use the account",
                )
            )
        else:
            checks.append(
                Check(
                    Level.OK, "capital_ceiling", f"${settings.capital_ceiling:,.0f} (~{frac:.0%})"
                )
            )

    # --- strategy / governor -------------------------------------------
    if settings.risk_weighting == "size_book":
        tilt = settings.tilt_strength
        kind = "governed 1/N" if tilt <= 0 else f"tilted (strength {tilt:g})"
        checks.append(Check(Level.OK, "book", f"size_book chassis — {kind}"))
    else:
        checks.append(
            Check(
                Level.WARN,
                "book",
                f"risk_weighting={settings.risk_weighting!r} — not the validated size_book chassis",
            )
        )
    if settings.target_vol > 0.10:
        checks.append(
            Check(
                Level.WARN,
                "target_vol",
                f"{settings.target_vol:.0%}: Monte-Carlo tail DD can exceed the 25% mandate "
                "(0.08 keeps it inside)",
            )
        )
    else:
        checks.append(Check(Level.OK, "target_vol", f"{settings.target_vol:.0%} (mandate-safe)"))

    # --- hands-off operation -------------------------------------------
    if not settings.daily_cycle_enabled:
        checks.append(
            Check(
                Level.WARN,
                "daily_schedule",
                "MT_DAILY_CYCLE_ENABLED=false — `serve` runs a health server but never trades",
            )
        )
    else:
        checks.append(Check(Level.OK, "daily_schedule", "fires once per trading day at the close"))

    # --- wasted per-cycle fetches --------------------------------------
    ignores_signals = settings.risk_weighting == "size_book" and settings.tilt_strength <= 0
    if ignores_signals and (settings.insider_enabled or settings.news_enabled):
        on = [
            n
            for n, e in (("insider", settings.insider_enabled), ("news", settings.news_enabled))
            if e
        ]
        checks.append(
            Check(
                Level.WARN,
                "wasted_fetch",
                f"{'/'.join(on)} fetched every cycle but the governed-1/N book ignores all signals "
                "(tilt_strength=0) — disable to speed the cycle, or backfill via `ingest-*`",
            )
        )

    # --- live-trading guard --------------------------------------------
    if settings.execution_mode == "live":
        checks.append(
            Check(
                Level.WARN,
                "execution_mode",
                "LIVE — real orders once armed; confirm this is intended",
            )
        )
    else:
        checks.append(Check(Level.OK, "execution_mode", "paper"))

    order = {Level.FAIL: 0, Level.WARN: 1, Level.OK: 2}
    return sorted(checks, key=lambda c: order[c.level])


def worst_level(checks: list[Check]) -> Level:
    if any(c.level is Level.FAIL for c in checks):
        return Level.FAIL
    if any(c.level is Level.WARN for c in checks):
        return Level.WARN
    return Level.OK
