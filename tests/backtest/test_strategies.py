"""CompositeBacktestStrategy's optional insider blend — the simulate A/B candidate."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from market_trader.backtest.pit import PanelPriceView
from market_trader.backtest.strategies import (
    CompositeBacktestStrategy,
    EqualWeightStrategy,
    VolTargetedStrategy,
)
from market_trader.portfolio.construction import ledoit_wolf_cov


def _flat_view() -> tuple[PanelPriceView, list[str], datetime]:
    """Identical price path for every symbol -> the price composite is a flat tie (all
    z-scores 0), so the insider blend is the only thing that can break it."""
    dates = pd.bdate_range("2022-01-03", periods=70)
    syms = [f"S{i}" for i in range(6)]
    path = 100.0 * (1.0 + pd.Series(range(len(dates)), dtype=float) * 0.001)
    prices = pd.DataFrame({s: path.to_numpy() for s in syms}, index=dates)
    t = dates[-1].to_pydatetime()
    return PanelPriceView(prices, t), syms, t


def test_insider_scores_break_the_price_tie_and_select_favored_names() -> None:
    view, syms, t = _flat_view()

    w_base = CompositeBacktestStrategy(max_positions=2, top_quantile=0.4).target_weights(view, t)
    assert len(w_base) == 2

    # concentrate insider buys on two names the price-tie baseline did not pick
    favored = [s for s in syms if s not in w_base][:2]
    scores = pd.Series(0.0, index=syms)
    scores[favored] = 5.0
    w_tilt = CompositeBacktestStrategy(
        max_positions=2, top_quantile=0.4, insider_scores={t: scores}, name="composite+insider"
    ).target_weights(view, t)

    assert set(w_tilt) == set(favored)  # insider pulled the favored names in
    assert set(w_base) != set(w_tilt)


def test_absent_insider_scores_leave_the_price_only_baseline_unchanged() -> None:
    view, _, t = _flat_view()
    baseline = CompositeBacktestStrategy(max_positions=2, top_quantile=0.4).target_weights(view, t)
    # an empty / date-less map blends nothing -> byte-identical to the price-only book
    empty = CompositeBacktestStrategy(
        max_positions=2, top_quantile=0.4, insider_scores={}
    ).target_weights(view, t)
    assert empty == baseline


def test_vol_targeted_strategy_scales_a_high_vol_book_to_target() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=120)
    syms = [f"S{i}" for i in range(5)]
    # ~3%/day -> ~48% annualised, far above a 10% target, so the book must be scaled DOWN
    rets = rng.normal(0.0, 0.03, (len(dates), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=syms)
    t = dates[-1].to_pydatetime()
    view = PanelPriceView(prices, t)

    w = VolTargetedStrategy(EqualWeightStrategy(), target_vol=0.10, lookback=90).target_weights(
        view, t
    )

    assert 0.0 < sum(w.values()) < 1.0  # exposure cut (target below realised vol)
    window = view.returns_panel()[list(w)].tail(90)
    cov = ledoit_wolf_cov(window)
    wv = pd.Series(w).reindex(cov.columns).fillna(0.0).to_numpy(dtype=float)
    ann_vol = float(np.sqrt(wv @ cov.to_numpy(dtype=float) @ wv) * np.sqrt(252))
    assert abs(ann_vol - 0.10) < 0.01  # book now sized to the target volatility


def test_vol_targeted_strategy_does_not_lever_past_max_gross() -> None:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2022-01-03", periods=120)
    syms = [f"S{i}" for i in range(4)]
    # ~0.3%/day -> ~5% annualised, below the 10% target: scaling up is capped at max_gross
    rets = rng.normal(0.0, 0.003, (len(dates), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=syms)
    t = dates[-1].to_pydatetime()
    view = PanelPriceView(prices, t)

    w = VolTargetedStrategy(
        EqualWeightStrategy(), target_vol=0.10, max_gross=1.0, lookback=90
    ).target_weights(view, t)

    assert abs(sum(w.values()) - 1.0) < 1e-6  # would lever up, but held at max_gross
