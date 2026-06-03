"""Infrastructure helpers (optional, ops-time).

Thin DigitalOcean management-API client for the chosen, **non-provisioning**
operations: monitoring alert policies, on-demand Droplet snapshots, and Cloud
Firewall management. The token is a managed secret (env only). No Droplet
create/resize/destroy here — the destructive surface stays out of code.
"""

from market_trader.infra.digitalocean import DigitalOceanClient, DigitalOceanError

__all__ = ["DigitalOceanClient", "DigitalOceanError"]
