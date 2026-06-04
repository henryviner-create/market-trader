"""Liquidity screen for a data-driven small/mid-cap universe.

Insider buying's abnormal returns are concentrated in smaller companies, but free
market data is thin there — so we screen the SEC-filer population (the names that can
*have* insider filings) by realised dollar volume, keeping a liquid small/mid tier that
is both tradable and has usable bars. Reproducible from data, not a hand-picked or
scraped list (which would bake in selection/survivorship bias).

NOTE: dollar volume here is whatever feed produced the bars. On the free IEX feed that
is a *fraction* of consolidated volume, so the band thresholds are IEX-relative — treat
the first screen as calibration and tune the band to the size tier you actually want.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any


def screen_for_liquidity(
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    exclude: Collection[str] = (),
    min_price: float = 5.0,
    min_dollar_volume: float = 2e5,
    max_dollar_volume: float = 1e7,
    min_days: int = 15,
    top_n: int = 400,
) -> list[str]:
    """The most-liquid symbols whose average dollar volume sits in the small/mid band.

    ``bars_by_symbol`` maps a ticker to recent daily bars (each with ``close`` and
    ``volume``). Names below ``min_price`` (penny-stock data is unreliable), outside the
    ``[min_dollar_volume, max_dollar_volume]`` band, with fewer than ``min_days`` usable
    bars, or in ``exclude`` (e.g. the large-cap set) are dropped; the survivors are
    ranked by average dollar volume and the top ``top_n`` returned.
    """
    excluded = {s.upper() for s in exclude}
    ranked: list[tuple[str, float]] = []
    for symbol, bars in bars_by_symbol.items():
        sym = symbol.upper()
        if sym in excluded:
            continue
        closes = [float(b["close"]) for b in bars if b.get("close")]
        dvs = [
            float(b["close"]) * float(b["volume"])
            for b in bars
            if b.get("close") and b.get("volume")
        ]
        if len(dvs) < min_days or not closes:
            continue
        adv = sum(dvs) / len(dvs)
        if closes[-1] < min_price or not (min_dollar_volume <= adv <= max_dollar_volume):
            continue
        ranked.append((sym, adv))
    ranked.sort(key=lambda r: r[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]
