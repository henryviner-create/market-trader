"""screen_for_liquidity: the data-driven small/mid-cap universe screen."""

from __future__ import annotations

from market_trader.universe.screen import screen_for_liquidity


def _bars(price: float, volume: float, n: int = 20) -> list[dict[str, float]]:
    return [{"close": price, "volume": volume} for _ in range(n)]


def test_screen_keeps_in_band_liquid_names_and_drops_the_rest() -> None:
    data = {
        "BIG": _bars(100.0, 5_000_000),  # $500M/day -> above the band (large cap)
        "MID": _bars(40.0, 1_500_000),  # $60M/day -> in band
        "SMALL": _bars(20.0, 300_000),  # $6M/day -> in band
        "PENNY": _bars(2.0, 5_000_000),  # price < min_price -> dropped
        "ILLIQ": _bars(30.0, 10_000),  # $300k/day -> below the band
        "THIN": _bars(25.0, 1_000_000, n=5),  # too few bars -> dropped
        "EXCL": _bars(35.0, 1_000_000),  # in band but explicitly excluded
    }
    out = screen_for_liquidity(
        data,
        exclude={"EXCL"},
        min_price=5.0,
        min_dollar_volume=3e6,
        max_dollar_volume=100e6,
        min_days=15,
        top_n=10,
    )
    assert out == ["MID", "SMALL"]  # ranked by average dollar volume, descending
    for dropped in ("BIG", "PENNY", "ILLIQ", "THIN", "EXCL"):
        assert dropped not in out


def test_screen_caps_at_top_n_by_dollar_volume() -> None:
    data = {f"S{i}": _bars(10.0, (i + 1) * 1_000_000) for i in range(5)}  # S4 most liquid
    out = screen_for_liquidity(data, min_dollar_volume=1e6, max_dollar_volume=1e9, top_n=2)
    assert out == ["S4", "S3"]
