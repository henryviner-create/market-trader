"""The intraday trading loop's control flow: market gating, resilience, stop.

Driven entirely through injected callables (market-open check, per-pass cycle,
sleep, stop), so the loop is verified with no network, DB, or real waiting.
"""

from __future__ import annotations

from market_trader.config import Settings
from market_trader.core.time import utcnow
from market_trader.runtime.cycle import CycleResult
from market_trader.runtime.intraday import run_trading_loop


def _result() -> CycleResult:
    return CycleResult(as_of=utcnow(), scores={}, target_weights={}, orders=[])


def _settings() -> Settings:
    return Settings(execution_mode="paper", intraday_interval_seconds=60)


def test_loop_runs_each_pass_while_market_open() -> None:
    passes = {"n": 0}
    sleeps: list[float] = []

    def run_cycle() -> CycleResult:
        passes["n"] += 1
        return _result()

    ran = run_trading_loop(
        _settings(),
        market_open=lambda: True,
        run_cycle=run_cycle,
        sleep_fn=sleeps.append,
        max_iterations=3,
    )
    assert ran == 3
    assert passes["n"] == 3
    assert sleeps == [60.0, 60.0]  # sleeps between passes, not after the last


def test_loop_idles_when_market_closed() -> None:
    passes = {"n": 0}

    def run_cycle() -> CycleResult:
        passes["n"] += 1
        return _result()

    run_trading_loop(
        _settings(),
        market_open=lambda: False,
        run_cycle=run_cycle,
        sleep_fn=lambda _s: None,
        max_iterations=3,
    )
    assert passes["n"] == 0  # never traded while closed


def test_loop_survives_a_failing_pass() -> None:
    calls = {"n": 0}

    def boom() -> CycleResult:
        calls["n"] += 1
        raise RuntimeError("transient alpaca blip")

    ran = run_trading_loop(
        _settings(),
        market_open=lambda: True,
        run_cycle=boom,
        sleep_fn=lambda _s: None,
        max_iterations=3,
    )
    assert ran == 3 and calls["n"] == 3  # kept going despite every pass raising


def test_loop_stops_on_stop_signal() -> None:
    state = {"i": 0}

    def stop() -> bool:
        state["i"] += 1
        return state["i"] > 2  # ask to stop on the third check

    ran = run_trading_loop(
        _settings(),
        market_open=lambda: True,
        run_cycle=_result,
        sleep_fn=lambda _s: None,
        stop=stop,
        max_iterations=100,
    )
    assert ran < 100  # halted by stop(), not by max_iterations
