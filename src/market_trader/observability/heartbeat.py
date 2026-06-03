"""External heartbeat ping (Healthchecks.io / UptimeRobot).

A fully-dead box can't alert on itself, so the engine pings an external monitor on
each healthy cycle; if the pings stop, that service pages the operator.
"""

from __future__ import annotations

import urllib.request


def ping_heartbeat(url: str | None, *, timeout: float = 10.0) -> bool:
    """Best-effort GET of a heartbeat URL. Returns True on success, never raises."""
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 400
    except Exception:
        return False
