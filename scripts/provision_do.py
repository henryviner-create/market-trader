#!/usr/bin/env python3
"""Provision a DigitalOcean Droplet that self-installs the paper stack on boot.

Reads ``DO_API_TOKEN`` from the environment (never committed). Fills the cloud-init
script (``deploy/cloud-init.sh``) with your SSH public key + repo clone URL,
registers the key, and creates the Droplet. Use ``--dry-run`` to preview the exact
user-data and plan without calling the API.

WARNING: without --dry-run this creates a Droplet (costs money) and is gated only
by the live token. Destroy any old Droplet separately to avoid double charges.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from market_trader.infra import DigitalOceanClient, DigitalOceanError

REPO_DEFAULT = "https://github.com/henryviner-create/market-trader.git"
CLOUD_INIT = Path(__file__).resolve().parent.parent / "deploy" / "cloud-init.sh"


def build_user_data(pubkey: str, repo_url: str) -> str:
    text = CLOUD_INIT.read_text(encoding="utf-8")
    return text.replace("__SSH_PUBKEY__", pubkey.strip()).replace("__REPO_URL__", repo_url.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision the market-trader Droplet")
    parser.add_argument(
        "--pubkey", required=True, help="SSH public key string, or @path/to/key.pub"
    )
    parser.add_argument("--region", required=True, help="e.g. lon1, nyc3, ams3")
    parser.add_argument("--size", default="s-4vcpu-8gb")
    parser.add_argument("--image", default="ubuntu-24-04-x64")
    parser.add_argument("--name", default="market-trader")
    parser.add_argument(
        "--repo-url", default=REPO_DEFAULT, help="clone URL (embed a read-only token if private)"
    )
    parser.add_argument("--no-backups", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pubkey = (
        Path(args.pubkey[1:]).read_text(encoding="utf-8")
        if args.pubkey.startswith("@")
        else args.pubkey
    )
    user_data = build_user_data(pubkey, args.repo_url)

    if args.dry_run:
        print("=== PLAN (dry-run, no API calls) ===")
        print(
            f"register SSH key, then create droplet name={args.name} region={args.region} "
            f"size={args.size} image={args.image} backups={not args.no_backups}"
        )
        print("=== cloud-init user-data that will run on first boot ===")
        print(user_data)
        return 0

    client = DigitalOceanClient.from_env()
    key = client.create_ssh_key(name=f"{args.name}-key", public_key=pubkey)
    fingerprint = key.get("fingerprint")
    if not fingerprint:
        raise DigitalOceanError("could not register SSH key with DigitalOcean")

    droplet = client.create_droplet(
        name=args.name,
        region=args.region,
        size=args.size,
        image=args.image,
        ssh_key_fingerprints=[fingerprint],
        user_data=user_data,
        backups=not args.no_backups,
        tags=["market-trader"],
    )
    droplet_id = droplet.get("id")
    print(f"created droplet id={droplet_id}; waiting for it to become active...")

    public_ip = None
    for _ in range(60):
        current = client.get_droplet(int(droplet_id))
        for net in current.get("networks", {}).get("v4", []):
            if net.get("type") == "public":
                public_ip = net.get("ip_address")
        if current.get("status") == "active" and public_ip:
            break
        time.sleep(5)

    print(f"droplet active. public IP: {public_ip}")
    print("cloud-init is installing the stack (a few minutes). Then:")
    print(f"  ssh trader@{public_ip}")
    print('  curl -s localhost:8080/health   # expect {"status":"ok",...}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
