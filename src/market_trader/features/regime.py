"""Macro regime, read point-in-time from FRED series.

Regime is not per-symbol; it conditions weighting and model selection later. The
starter signal is the yield-curve slope (10y minus 2y): a positive slope is the
risk-on default, inversion flags risk-off. Deliberately simple — it earns more
nuance only if it proves useful out-of-sample.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from market_trader.collectors.fred import FRED_DATASET
from market_trader.storage.bitemporal import BitemporalStore

LONG_RATE = "DGS10"
SHORT_RATE = "DGS2"


def macro_regime(store: BitemporalStore, as_of: datetime) -> dict[str, Any]:
    latest: dict[str, float] = {}
    for o in store.as_of(as_of, dataset=FRED_DATASET):  # sorted by knowledge_time → last wins
        v = o.value.get("value")
        if v is not None:
            latest[o.entity_id] = float(v)

    slope: float | None = None
    if LONG_RATE in latest and SHORT_RATE in latest:
        slope = latest[LONG_RATE] - latest[SHORT_RATE]

    risk_on = slope is None or slope > 0.0
    return {
        "yield_curve_slope": slope,
        "risk_on": risk_on,
        "label": "risk_on" if risk_on else "risk_off",
        "n_series": len(latest),
    }
