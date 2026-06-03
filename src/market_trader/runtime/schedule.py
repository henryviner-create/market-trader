"""Daily scheduler: run the end-of-day cycle once per trading day.

``run_daily_schedule`` polls the market clock and fires the daily cycle on the
open->closed transition — the moment the session ends. That runs it once per
*trading* day and skips weekends/holidays for free (no session, no transition).
It is what makes the prediction/learning loop advance on its own: every close
logs a fresh cohort of predictions that grade themselves once their horizon
elapses, and the next cycle re-weights its signals on what actually worked.

PAPER, and only inside ``serve`` when ``MT_DAILY_CYCLE_ENABLED=true``. Every
dependency is injectable, so the control flow is unit-tested with no network, DB,
or real waiting — exactly like the intraday loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from market_trader.config import Settings
from market_trader.observability import get_logger
from market_trader.runtime.cycle import CycleResult, run_live_paper_cycle

_log = get_logger("schedule")


def run_daily_schedule(
    settings: Settings,
    *,
    is_market_open: Callable[[], bool] | None = None,
    run_cycle: Callable[[], CycleResult] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stop: Callable[[], bool] | None = None,
    max_iterations: int | None = None,
) -> int:
    """Fire the end-of-day cycle on each market open->closed transition.

    Builds the Alpaca broker (for its market clock) and binds the cycle to
    ``run_live_paper_cycle`` only when they aren't injected. Returns the number of
    poll iterations executed.
    """
    if is_market_open is None or run_cycle is None:
        if not (settings.alpaca_key_id and settings.alpaca_secret_key):
            raise RuntimeError("Alpaca keys not set (MT_ALPACA_KEY_ID / MT_ALPACA_SECRET_KEY)")
        if not settings.alpaca_paper:
            settings.assert_live_allowed()
        from market_trader.execution.alpaca import AlpacaBroker

        broker = AlpacaBroker(
            settings.alpaca_key_id, settings.alpaca_secret_key, paper=settings.alpaca_paper
        )
        if is_market_open is None:
            is_market_open = broker.is_market_open
        if run_cycle is None:

            def _run_cycle() -> CycleResult:
                return run_live_paper_cycle(settings)

            run_cycle = _run_cycle

    poll = float(settings.daily_cycle_poll_seconds)
    was_open: bool | None = None
    iterations = 0
    _log.info("daily_schedule_start", poll_seconds=poll, mode=settings.execution_mode)
    while not (stop and stop()) and not (
        max_iterations is not None and iterations >= max_iterations
    ):
        try:
            now_open = is_market_open()
        except Exception as exc:  # a clock blip must not look like a close
            _log.warning("market_clock_error", error=str(exc))
            now_open = bool(was_open)
        fire = bool(was_open) and not now_open  # only the close edge triggers
        was_open = now_open  # advance first, so a failed cycle is not retried in a loop
        if fire:
            try:
                result = run_cycle()
                _log.info(
                    "daily_cycle_ran",
                    orders=len(result.orders),
                    targets=len(result.target_weights),
                )
            except Exception as exc:  # try again next session, not next poll
                _log.warning("daily_cycle_error", error=str(exc))
        iterations += 1
        if (stop and stop()) or (max_iterations is not None and iterations >= max_iterations):
            break
        sleep_fn(poll)
    _log.info("daily_schedule_stop", iterations=iterations)
    return iterations
