"""DigitalOcean management API client (stdlib urllib; token from env).

Scoped to the non-provisioning operations chosen for this deployment:

* **monitoring alert policies** (CPU/memory/disk/bandwidth -> email),
* **on-demand Droplet snapshots** (e.g. before a migration),
* **Cloud Firewall** management (SSH restricted to your IP, plus 80/443).

There is deliberately **no** Droplet create/resize/destroy here. The token
(``DO_API_TOKEN``) is a powerful secret — env / secret-manager only, never
committed. The HTTP transport is injectable, so this is fully unit-tested offline.

Note: a Cloud Firewall can lock you out of SSH if ``ssh_source_ips`` is wrong;
keep the DigitalOcean web Droplet Console available as a fallback.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

DEFAULT_BASE_URL = "https://api.digitalocean.com"

# (method, url, headers, body) -> (status_code, json_payload)
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, dict[str, Any]]]


class DigitalOceanError(RuntimeError):
    pass


def _urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            message = json.loads(raw).get("message", "")
        except Exception:
            message = raw.decode("utf-8", "ignore")[:200]
        raise DigitalOceanError(f"HTTP {exc.code}: {message}") from exc
    except Exception as exc:
        raise DigitalOceanError(f"request failed: {exc}") from exc


class DigitalOceanClient:
    def __init__(
        self, token: str, *, base_url: str = DEFAULT_BASE_URL, transport: Transport | None = None
    ) -> None:
        if not token:
            raise DigitalOceanError("DO_API_TOKEN is required")
        self._token = token
        self._base = base_url.rstrip("/")
        self._transport: Transport = transport or _urllib_transport

    @classmethod
    def from_env(cls, *, transport: Transport | None = None) -> DigitalOceanClient:
        return cls(os.environ.get("DO_API_TOKEN", ""), transport=transport)

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        status, payload = self._transport(method, self._base + path, headers, data)
        if status >= 300:
            raise DigitalOceanError(f"HTTP {status}: {payload}")
        return payload

    # --- monitoring alert policies -------------------------------------
    def create_alert_policy(
        self,
        *,
        alert_type: str,  # e.g. "v1/insights/droplet/cpu"
        value: float,
        emails: Sequence[str],
        description: str,
        window: str = "5m",
        compare: str = "GreaterThan",
        entities: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        body = {
            "alerts": {"email": list(emails)},
            "compare": compare,
            "description": description,
            "type": alert_type,
            "value": value,
            "window": window,
            "enabled": True,
            "entities": list(entities or []),
        }
        return self._request("POST", "/v2/monitoring/alerts", body)

    def list_alert_policies(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/monitoring/alerts").get("policies", [])

    # --- snapshots ------------------------------------------------------
    def snapshot_droplet(self, droplet_id: int, *, name: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v2/droplets/{droplet_id}/actions", {"type": "snapshot", "name": name}
        )

    def list_snapshots(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/snapshots?resource_type=droplet").get("snapshots", [])

    # --- cloud firewall -------------------------------------------------
    def create_firewall(
        self,
        *,
        name: str,
        ssh_source_ips: Sequence[str],
        droplet_ids: Sequence[int] | None = None,
        allow_web: bool = True,
    ) -> dict[str, Any]:
        inbound: list[dict[str, Any]] = [
            {"protocol": "tcp", "ports": "22", "sources": {"addresses": list(ssh_source_ips)}}
        ]
        if allow_web:
            inbound += [
                {"protocol": "tcp", "ports": "80", "sources": {"addresses": ["0.0.0.0/0", "::/0"]}},
                {
                    "protocol": "tcp",
                    "ports": "443",
                    "sources": {"addresses": ["0.0.0.0/0", "::/0"]},
                },
            ]
        outbound = [
            {
                "protocol": "tcp",
                "ports": "all",
                "destinations": {"addresses": ["0.0.0.0/0", "::/0"]},
            },
            {
                "protocol": "udp",
                "ports": "all",
                "destinations": {"addresses": ["0.0.0.0/0", "::/0"]},
            },
        ]
        body = {
            "name": name,
            "inbound_rules": inbound,
            "outbound_rules": outbound,
            "droplet_ids": list(droplet_ids or []),
        }
        return self._request("POST", "/v2/firewalls", body)

    def list_firewalls(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/firewalls").get("firewalls", [])
