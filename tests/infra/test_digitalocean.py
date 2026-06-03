"""DigitalOcean client request shapes (offline; injected transport)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from market_trader.infra import DigitalOceanClient, DigitalOceanError


def _recorder(status: int = 200, payload: dict[str, Any] | None = None):
    calls: list[dict[str, Any]] = []

    def transport(method, url, headers, body):
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": json.loads(body) if body else None,
            }
        )
        return status, (payload or {"policies": [], "firewalls": [], "snapshots": []})

    return transport, calls


def test_requires_a_token() -> None:
    with pytest.raises(DigitalOceanError):
        DigitalOceanClient("")


def test_create_alert_policy_request_shape() -> None:
    transport, calls = _recorder()
    client = DigitalOceanClient("tok", transport=transport)
    client.create_alert_policy(
        alert_type="v1/insights/droplet/cpu",
        value=80,
        emails=["me@example.com"],
        description="cpu high",
    )
    call = calls[-1]
    assert call["method"] == "POST" and call["url"].endswith("/v2/monitoring/alerts")
    assert call["headers"]["Authorization"] == "Bearer tok"
    assert call["body"]["type"] == "v1/insights/droplet/cpu"
    assert call["body"]["value"] == 80
    assert call["body"]["alerts"]["email"] == ["me@example.com"]


def test_snapshot_request_shape() -> None:
    transport, calls = _recorder()
    DigitalOceanClient("tok", transport=transport).snapshot_droplet(12345, name="pre-migration")
    call = calls[-1]
    assert call["url"].endswith("/v2/droplets/12345/actions")
    assert call["body"] == {"type": "snapshot", "name": "pre-migration"}


def test_create_firewall_restricts_ssh_to_given_ips() -> None:
    transport, calls = _recorder()
    DigitalOceanClient("tok", transport=transport).create_firewall(
        name="mt", ssh_source_ips=["1.2.3.4/32"], droplet_ids=[1]
    )
    body = calls[-1]["body"]
    ssh_rule = next(r for r in body["inbound_rules"] if r["ports"] == "22")
    assert ssh_rule["sources"]["addresses"] == ["1.2.3.4/32"]
    assert body["droplet_ids"] == [1]


def test_non_2xx_raises() -> None:
    transport, _ = _recorder(status=422, payload={"message": "bad"})
    with pytest.raises(DigitalOceanError):
        DigitalOceanClient("tok", transport=transport).list_firewalls()


def test_list_helpers_unwrap_collections() -> None:
    transport, _ = _recorder()
    client = DigitalOceanClient("tok", transport=transport)
    assert client.list_firewalls() == []
    assert client.list_alert_policies() == []
    assert client.list_snapshots() == []
