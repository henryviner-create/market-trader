"""FRED: the vintage realtime_start becomes our knowledge time; missing values skipped."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import FredSeriesCollector


def test_realtime_start_is_knowledge_time_and_missing_skipped() -> None:
    raw = {
        "observations": [
            {"date": "2023-01-01", "realtime_start": "2023-02-01", "value": "3.5"},
            {
                "date": "2023-02-01",
                "realtime_start": "2023-03-01",
                "value": ".",
            },  # missing -> skipped
        ]
    }
    obs = FredSeriesCollector("dgs10").normalize(raw)

    assert len(obs) == 1
    (o,) = obs
    assert o.entity_id == "DGS10"
    assert o.entity_type == "macro_series"
    assert o.event_time.date() == date(2023, 1, 1)  # the period the value describes
    assert o.knowledge_time.date() == date(2023, 2, 1)  # when it became knowable
    assert o.value["value"] == 3.5


def test_accepts_bare_list_of_records() -> None:
    obs = FredSeriesCollector("CPIAUCSL").normalize(
        [{"date": "2023-03-01", "realtime_start": "2023-04-12", "value": "300.1"}]
    )
    assert len(obs) == 1
    assert obs[0].knowledge_time.date() == date(2023, 4, 12)
