"""Event-study engine: market-model recovery and CAR significance vs a null."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from market_trader.core.synthetic import business_days
from market_trader.core.time import day_close
from market_trader.memory import aggregate_event_study, estimate_market_model


def test_market_model_recovers_alpha_and_beta() -> None:
    rng = np.random.default_rng(1)
    market = rng.normal(0.0, 0.01, 500)
    stock = 0.0002 + 1.5 * market + rng.normal(0.0, 0.0005, 500)
    model = estimate_market_model(stock, market)
    assert abs(model.beta - 1.5) < 0.05
    assert abs(model.alpha - 0.0002) < 0.001


def _panel_with_events(seed: int = 0) -> tuple[pd.DataFrame, pd.Series, list]:
    rng = np.random.default_rng(seed)
    n = 400
    idx = pd.DatetimeIndex([day_close(d) for d in business_days(date(2021, 1, 4), n)])
    market = rng.normal(0.0003, 0.008, n)
    evt = market + rng.normal(0.0, 0.002, n)
    ctrl = market + rng.normal(0.0, 0.002, n)
    anchor_pos = [50, 90, 130, 170, 210, 250, 290, 330, 370]
    for p in anchor_pos:  # +2% abnormal on each of the 3 days after the event
        for d in (1, 2, 3):
            evt[p + d] += 0.02
    panel = pd.DataFrame({"EVT": evt, "CTRL": ctrl}, index=idx)
    return panel, pd.Series(market, index=idx), [idx[p] for p in anchor_pos]


def test_event_study_finds_significant_positive_abnormal_return() -> None:
    panel, market, anchors = _panel_with_events()
    dist = aggregate_event_study(
        [("EVT", a) for a in anchors],
        panel,
        market_returns=market,
        label="evt",
        estimation_days=30,
        gap_days=3,
        pre=0,
        post=3,
    )
    assert dist.n == len(anchors)
    assert dist.mean_car > 0.03  # ~3 x 2% abnormal
    assert dist.share_positive > 0.8
    assert dist.significant()


def test_null_events_are_not_significant() -> None:
    panel, market, anchors = _panel_with_events()
    dist = aggregate_event_study(
        [("CTRL", a) for a in anchors],
        panel,
        market_returns=market,
        label="ctrl",
        estimation_days=30,
        gap_days=3,
        pre=0,
        post=3,
    )
    assert not dist.significant()
