"""Metrics rendering, data-freshness, and heartbeat."""

from __future__ import annotations

from datetime import date, timedelta

from market_trader.core.synthetic import synthetic_price_observations
from market_trader.core.time import day_close, utcnow
from market_trader.observability import MetricsRegistry, data_freshness, ping_heartbeat
from market_trader.storage import InMemoryBitemporalStore


def test_metrics_render_prometheus_format() -> None:
    reg = MetricsRegistry()
    orders = reg.counter("mt_orders_total", "orders submitted")
    orders.inc()
    orders.inc(2.0)
    reg.gauge("mt_drawdown", "current drawdown").set(-0.1)

    text = reg.render()
    assert "# TYPE mt_orders_total counter" in text
    assert "mt_orders_total 3.0" in text
    assert "mt_drawdown -0.1" in text


def test_metric_get_or_create_is_idempotent() -> None:
    reg = MetricsRegistry()
    assert reg.counter("x") is reg.counter("x")


def test_data_freshness_flags_stale_and_missing() -> None:
    store = InMemoryBitemporalStore()
    store.add_many(
        synthetic_price_observations(symbols=["A"], start=date(2023, 1, 2), n_days=2, seed=1)
    )

    stale = data_freshness(store, ["synthetic"], now=utcnow(), max_age_hours=24.0)
    assert stale[0].stale is True  # 2023 data is ancient vs now

    missing = data_freshness(store, ["edgar"], now=utcnow())
    assert missing[0].last_knowledge_time is None
    assert missing[0].stale is True


def test_data_freshness_fresh_when_recent() -> None:
    store = InMemoryBitemporalStore()
    store.add_many(
        synthetic_price_observations(symbols=["A"], start=date(2023, 1, 2), n_days=4, seed=1)
    )
    now = day_close(date(2023, 1, 5)) + timedelta(hours=1)
    fresh = data_freshness(store, ["synthetic"], now=now, max_age_hours=48.0)
    assert fresh[0].stale is False


def test_ping_heartbeat_handles_missing_url() -> None:
    assert ping_heartbeat(None) is False
    assert ping_heartbeat("") is False
