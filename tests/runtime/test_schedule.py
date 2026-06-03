"""The daily scheduler's control flow: fire on the market-close edge, once per
trading day, resilient to a bad cycle, stoppable — all through injected callables,
so it's verified with no network, DB, or real waiting.
"""

from __future__ import annotations

from collections.abc import Callable

from market_trader.config import Settings
from market_trader.core.time import utcnow
from market_trader.runtime.cycle import CycleResult
from market_trader.runtime.schedule import run_daily_schedule


def _result() -> CycleResult:
    return CycleResult(as_of=utcnow(), scores={}, target_weights={}, orders=[])


def _settings() -> Settings:
    return Settings(execution_mode="paper", daily_cycle_poll_seconds=300)


def _clock(states: list[bool]) -> Callable[[], bool]:
    """An is_market_open that walks ``states`` then holds the final value."""
    i = {"n": 0}

    def is_open() -> bool:
        v = states[min(i["n"], len(states) - 1)]
        i["n"] += 1
        return v

    return is_open


def test_fires_once_on_market_close() -> None:
    runs = {"n": 0}

    def run_cycle() -> CycleResult:
        runs["n"] += 1
        return _result()

    ran = run_daily_schedule(
        _settings(),
        is_market_open=_clock([True, True, False, False]),
        run_cycle=run_cycle,
        sleep_fn=lambda _s: None,
        max_iterations=4,
    )
    assert ran == 4
    assert runs["n"] == 1  # exactly one end-of-day cycle, on the open->closed edge


def test_does_not_fire_while_market_stays_open() -> None:
    runs = {"n": 0}

    def run_cycle() -> CycleResult:
        runs["n"] += 1
        return _result()

    run_daily_schedule(
        _settings(),
        is_market_open=lambda: True,  # never closes -> no edge
        run_cycle=run_cycle,
        sleep_fn=lambda _s: None,
        max_iterations=5,
    )
    assert runs["n"] == 0


def test_open_edge_does_not_fire() -> None:
    runs = {"n": 0}

    def run_cycle() -> CycleResult:
        runs["n"] += 1
        return _result()

    run_daily_schedule(
        _settings(),
        is_market_open=_clock([False, True, True]),  # closed->open is not a close
        run_cycle=run_cycle,
        sleep_fn=lambda _s: None,
        max_iterations=3,
    )
    assert runs["n"] == 0


def test_survives_a_failing_cycle() -> None:
    calls = {"n": 0}

    def boom() -> CycleResult:
        calls["n"] += 1
        raise RuntimeError("transient alpaca blip")

    ran = run_daily_schedule(
        _settings(),
        is_market_open=_clock([True, False, True, False]),  # two close edges
        run_cycle=boom,
        sleep_fn=lambda _s: None,
        max_iterations=4,
    )
    assert ran == 4 and calls["n"] == 2  # both closes attempted; loop never died


def test_stops_on_stop_signal() -> None:
    state = {"i": 0}

    def stop() -> bool:
        state["i"] += 1
        return state["i"] > 2

    ran = run_daily_schedule(
        _settings(),
        is_market_open=lambda: True,
        run_cycle=_result,
        sleep_fn=lambda _s: None,
        stop=stop,
        max_iterations=100,
    )
    assert ran < 100  # halted by stop(), not by max_iterations
