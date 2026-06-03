"""Holding discipline: hysteresis keeps winners; inverse-vol sizes by risk."""

from __future__ import annotations

import pandas as pd

from market_trader.runtime.cycle import _risk_weights, _select_with_hysteresis


def test_hysteresis_holds_names_inside_the_exit_band() -> None:
    ranked = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}).sort_values(
        ascending=False
    )
    # Enter the top 2 (A, B); tolerate holds down to the top 4 (A..D).
    # A name held at rank #4 (D) is inside the band -> kept, not churned out.
    kept = _select_with_hysteresis(ranked, held={"D"}, enter_k=2, exit_k=4, cap=None)
    assert set(kept) == {"A", "B", "D"}
    # A name held at rank #5 (E) is outside the band -> dropped (will be sold).
    dropped = _select_with_hysteresis(ranked, held={"E"}, enter_k=2, exit_k=4, cap=None)
    assert "E" not in dropped and set(dropped) == {"A", "B"}


def test_no_hysteresis_when_bands_are_equal() -> None:
    ranked = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}).sort_values(ascending=False)
    # exit_k == enter_k -> a held name below the cutoff is not retained.
    assert _select_with_hysteresis(ranked, held={"C"}, enter_k=1, exit_k=1, cap=None) == ["A"]


def test_inverse_vol_underweights_the_riskier_name() -> None:
    matrix = pd.DataFrame({"mom_60": [0.0, 0.0], "vol_20": [0.01, 0.04]}, index=["CALM", "WILD"])
    w = _risk_weights(["CALM", "WILD"], matrix, "inverse_vol")
    assert w["CALM"] > w["WILD"]  # lower vol -> larger weight
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert _risk_weights(["CALM", "WILD"], matrix, "equal") == {"CALM": 0.5, "WILD": 0.5}
