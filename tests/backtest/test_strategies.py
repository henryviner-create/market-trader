"""CompositeBacktestStrategy's optional insider blend — the simulate A/B candidate."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_trader.backtest.pit import PanelPriceView
from market_trader.backtest.strategies import CompositeBacktestStrategy


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
