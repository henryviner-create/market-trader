"""The preflight config-doctor rules (pure, no broker/db needed)."""

from __future__ import annotations

from market_trader.config import Settings
from market_trader.runtime.preflight import Level, preflight_checks, worst_level


def _levels(checks) -> dict[str, Level]:
    return {c.name: c.level for c in checks}


def test_clean_governed_config_passes() -> None:
    s = Settings(
        alpaca_key_id="k",
        alpaca_secret_key="s",
        capital_ceiling=100_000.0,
        risk_weighting="size_book",
        tilt_strength=0.0,
        target_vol=0.08,
        daily_cycle_enabled=True,
        insider_enabled=False,
        news_enabled=False,
    )
    checks = preflight_checks(s, equity=100_000.0, db_ok=True, universe_size=120)
    assert worst_level(checks) is Level.OK
    assert all(c.level is Level.OK for c in checks)


def test_flags_the_known_footguns() -> None:
    s = Settings(
        alpaca_key_id="k",
        alpaca_secret_key="s",
        capital_ceiling=1_000.0,  # under-deploys
        risk_weighting="inverse_vol",  # not the chassis
        target_vol=0.15,  # tail breaches mandate
        daily_cycle_enabled=False,  # serve won't trade
    )
    levels = _levels(preflight_checks(s, equity=100_000.0, db_ok=True, universe_size=120))
    assert levels["capital_ceiling"] is Level.WARN
    assert levels["book"] is Level.WARN
    assert levels["target_vol"] is Level.WARN
    assert levels["daily_schedule"] is Level.WARN


def test_missing_keys_and_db_are_failures() -> None:
    s = Settings(alpaca_key_id=None, alpaca_secret_key=None)
    checks = preflight_checks(s, equity=None, db_ok=False, universe_size=0)
    levels = _levels(checks)
    assert levels["alpaca_keys"] is Level.FAIL
    assert levels["database"] is Level.FAIL
    assert levels["universe"] is Level.FAIL
    assert worst_level(checks) is Level.FAIL


def test_warns_on_wasted_signal_fetch_under_governed_1n() -> None:
    # The governed-1/N book ignores all signals, so fetching insider/news each cycle is waste.
    s = Settings(
        alpaca_key_id="k",
        alpaca_secret_key="s",
        risk_weighting="size_book",
        tilt_strength=0.0,
        insider_enabled=True,
        news_enabled=True,
    )
    levels = _levels(preflight_checks(s, equity=100_000.0, db_ok=True, universe_size=120))
    assert levels["wasted_fetch"] is Level.WARN


def test_no_wasted_fetch_warning_when_tilted() -> None:
    # With a tilt, the signals ARE used, so fetching them is not waste.
    s = Settings(
        alpaca_key_id="k",
        alpaca_secret_key="s",
        risk_weighting="size_book",
        tilt_strength=1.0,
        insider_enabled=True,
    )
    names = {c.name for c in preflight_checks(s, equity=100_000.0, db_ok=True, universe_size=120)}
    assert "wasted_fetch" not in names
