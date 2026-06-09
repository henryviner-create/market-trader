"""Locked single-look holdout: the first verdict is sealed; you can't re-shop it."""

from __future__ import annotations

from datetime import date, timedelta

from market_trader.backtest.holdout import (
    HoldoutLook,
    confirm_on_holdout,
    holdout_start,
    prior_look,
)
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.core.time import DISTANT_FUTURE, day_close
from market_trader.storage import InMemoryBitemporalStore


def _price_store(n_days: int = 50) -> InMemoryBitemporalStore:
    store = InMemoryBitemporalStore()
    store.add_many(
        synthetic_price_observations(symbols=["A", "B"], start=date(2023, 1, 2), n_days=n_days)
    )
    return store


def test_holdout_start_is_in_the_recent_tail() -> None:
    store = _price_store(50)
    dates = sorted({o.event_time for o in store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET)})
    start = holdout_start(store, DISTANT_FUTURE, frac=0.2)
    assert start is not None
    assert dates[int(len(dates) * 0.8) - 1] <= start <= dates[-1]  # last ~20% of history


def test_prior_look_is_none_before_any_confirmation() -> None:
    assert prior_look(InMemoryBitemporalStore(), "mom_20") is None


def test_confirm_is_single_look_and_cannot_be_reshopped() -> None:
    store = InMemoryBitemporalStore()
    t = day_close(date(2024, 6, 1))

    first = HoldoutLook("mom_20", t, n_dates=12, mean_ic=0.05, t_stat=3.5, passed=True)
    look, repeat = confirm_on_holdout(store, first)
    assert not repeat and look.t_stat == 3.5  # first look recorded

    # A later, "shopped" look with a flashier t must NOT overwrite the sealed first verdict.
    shopped = HoldoutLook(
        "mom_20", t + timedelta(days=10), n_dates=14, mean_ic=0.09, t_stat=6.0, passed=True
    )
    look2, repeat2 = confirm_on_holdout(store, shopped)
    assert repeat2  # sealed
    assert look2.t_stat == 3.5  # the ORIGINAL verdict is returned, not the re-shopped 6.0
